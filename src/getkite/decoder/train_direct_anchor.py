"""Learned-anchor diagnostic A1 (2026-11-19): a bounded 10k-step diagnostic for the direct decoder.

Question: does giving intent an explicit spatial slot, expressed in the start-wrist frame, let the motion network use it?

    frozen psi z --(z standardisation as stored)--> FROZEN intent trunk from direct_worldaux_v1 (eval mode) --> h
        h --> FRESH anchor head --> M = 6 anchors: world-frame point a_m (metres) + free feature f_m (3-vector)
          --> exact rigid transform with the deployment-time start wrist pose:  a_m' = R0w^T (a_m - t0w),  f_m' = R0w^T f_m
          --> FRESH anchor-token encoder (as J1's contact-token encoder: position * pos_scale, f_m' in the normal slot,
              Fourier features of the transformed anchor POSITION only) --> 6 intent tokens --> FRESH GET --> motion
        h --> frozen auxiliary contact head (diagnostics only; never read by the motion path)

What is the same as Track D (direct_frozentrunk_d1): the frozen trunk / aux head / z statistics (v1 best, step 35k);
the GET and its construction order and seed (same initial GET weights as v1 and D); motion loss only; data, family
sampling, start perturbations, normal jitter, residual loss; optimiser group (AdamW 3e-4, wd 1e-4, clip 1.0); warmup
500 + cosine with the 50k horizon, stopped at 10k; evaluation pools and selection rule.

The ONE change: the intent-token construction. Track D's abstract tokens (intent_tok(h) + intent_pose(world wrist pose))
are replaced by the anchor tokens above. D's intent_tok / intent_pose modules are still constructed (so every other
module gets D's initial values) but are not used and not trained.

Not in the anchors' training signal: contact labels, set matching, existence, masks, counts. The anchors are free latents
trained only through the motion loss.
CAVEAT (stated in the report): the frozen trunk was trained in direct_worldaux_v1 together with a world-frame contact
loss, so contact supervision is present upstream. This tests the combined interface (frozen contact-informed features +
learned spatial slots + exact wrist-frame transform + positional encoding), not frame alignment alone. A failure does
not rule out the alignment hypothesis.

Logged: anchor positions (world and wrist frame), their spread within an example and across examples, distance from
each true contact to its nearest anchor (diagnostic only, labels never used for training), gradient norms into the
anchor head / anchor encoder / GET, training motion loss; at evaluation: success, contact error, wrist error and how far
the wrist moved, on source val, training goals and held-out LEAP.

    python -m getkite.decoder.train_direct_anchor --config configs/decoder/scratch/direct_anchor_a1.json [--checks_only] [--steps N]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .get_scratch import ScratchConfig, fourier, mlp, target_residuals
from .multifamily import BalancedSampler, draw_start, fixed_start
from .rotations import geodesic_angle
from .spatial_bridge import SetBridgeConfig
from .train_direct_frozentrunk import FROZEN_PREFIXES, FrozenTrunkDirect, frozen_state_sha
from .train_direct_worldaux import gnorm
from .train_joint import contact_eval
from .train_scratch import (FrozenPsi, build_bank, evaluate, git_state, jitter_normals, load_config, load_pool,
                            residual_loss, sha16, start_noise)

UNUSED_PREFIXES = ("intent_tok.", "intent_pose.")


class AnchorDirect(FrozenTrunkDirect):
    def __init__(self, decoder_cfg: ScratchConfig, set_cfg: SetBridgeConfig, n_anchors=6, f_dim=3, out_scale=0.1):
        super().__init__(decoder_cfg, set_cfg)  # trunk, aux, intent_tok, intent_pose, GET: same order/values as D
        c, d = self.c, self.c.d_model
        assert f_dim == 3, "f_m occupies J1's normal slot (rotated like a direction)"
        self.n_anchors, self.f_dim, self.anchor_scale = n_anchors, f_dim, out_scale
        self.anchor_head = nn.Linear(self.trunk.hidden, n_anchors * (3 + f_dim))  # built after the GET
        self.anchor_in = mlp(3 + f_dim + 3 * 2 * c.pos_fourier, d, d)  # = J1's contact_in layout (6 + 3*2*cf)
        self._last = None

    def anchors(self, z):
        h = self.trunk(z)
        o = self.anchor_head(h).view(len(z), self.n_anchors, 3 + self.f_dim)
        return o[..., :3] * self.anchor_scale, o[..., 3:], h

    def intent(self, z, R0w, t0w):
        c = self.c
        a, f, h = self.anchors(z)
        aw = (a - t0w[:, None]) @ R0w  # R0w^T (a - t0w), row-vector form (as the raw-contact wrist transform)
        fw = f @ R0w
        pos = aw * c.pos_scale
        tok = self.anchor_in(torch.cat([pos, fw, fourier(pos, c.pos_fourier, c.fourier_base)], -1))
        self._last = (a, aw)
        return tok, h

    def trainable(self):
        return [p for n, p in self.named_parameters() if not n.startswith(FROZEN_PREFIXES + UNUSED_PREFIXES)]


@torch.no_grad()
def anchor_stats(a, aw, c_pos=None, c_valid=None):
    """a, aw (B, M, 3) metres. Spread within an example: mean over examples of the RMS distance of anchors to their
    centroid. Across examples: mean over anchors of the RMS distance of that anchor to its batch mean."""
    s = {"anchor_world_norm_mm": float(a.norm(dim=-1).mean() * 1000),
         "anchor_wrist_norm_mm": float(aw.norm(dim=-1).mean() * 1000),
         "spread_within_example_mm": float(((a - a.mean(1, keepdim=True)) ** 2).sum(-1).mean(1).sqrt().mean() * 1000),
         "spread_across_examples_mm": float(((a - a.mean(0, keepdim=True)) ** 2).sum(-1).mean(0).sqrt().mean() * 1000),
         "spread_across_examples_wrist_mm": float(((aw - aw.mean(0, keepdim=True)) ** 2).sum(-1).mean(0).sqrt().mean() * 1000)}
    if c_pos is not None:  # diagnostic only: labels are never used for training
        D = torch.cdist(c_pos.float(), a.float())  # (B, K, M)
        v = c_valid.float()
        s["contact_to_nearest_anchor_mm"] = float(((D.amin(-1) * v).sum(1) / v.sum(1)).mean() * 1000)
        cen = (c_pos * v[..., None]).sum(1) / v.sum(1, keepdim=True)
        s["contact_centroid_to_anchor_centroid_mm"] = float((cen - a.mean(1)).norm(dim=-1).mean() * 1000)
    return s


def run_checks(model, ref_state, psi, bank, train, mc, noise, dev):
    C = {}
    sd = model.state_dict()
    C["frozen_weights_equal_v1"] = all(torch.equal(sd[k].cpu(), ref_state[k].cpu()) for k in ref_state if k.startswith(FROZEN_PREFIXES))
    C["z_stats_equal_v1"] = torch.equal(sd["trunk.z_mu"].cpu(), ref_state["trunk.z_mu"].cpu()) and torch.equal(sd["trunk.z_sd"].cpu(), ref_state["trunk.z_sd"].cpu())
    model.train()
    C["frozen_no_grad"] = not any(p.requires_grad for n, p in model.named_parameters() if n.startswith(FROZEN_PREFIXES))
    C["frozen_eval_mode_under_train"] = (not model.trunk.training) and (not model.contacts.training) and model.decoder.training
    C["psi_frozen_eval"] = (not any(p.requires_grad for p in psi.parameters())) and not psi.training
    C["surface_sampling"] = bank.surface_sampling
    C["surf_idx_sha16"] = hashlib.sha256(bank.surf_idx.cpu().numpy().tobytes()).hexdigest()[:16]
    with torch.no_grad():
        C["fresh_heads_zero_init"] = all(float(m[-1].weight.abs().max()) == 0.0 for m in (model.decoder.root_head, model.decoder.joint_head))
    tp = {id(p) for p in model.trainable()}
    C["trainable_excludes_frozen_and_unused"] = not any(id(p) in tp for n, p in model.named_parameters() if n.startswith(FROZEN_PREFIXES + UNUSED_PREFIXES))
    m = copy.deepcopy(model)  # path checks with small random output heads (zero-init heads block motion grads)
    g0 = torch.Generator(device="cpu").manual_seed(7)
    with torch.no_grad():
        for h in (m.decoder.root_head, m.decoder.joint_head):
            h[-1].weight.copy_(torch.randn(h[-1].weight.shape, generator=g0) * 1e-2)
    m.decoder.amp = False
    m.eval()
    gen = torch.Generator().manual_seed(123)
    rb = train.index(BalancedSampler(train.to("cpu"), bank.family, seed=123).sample(64).to(dev))
    sg = draw_start(bank, rb, gen, **noise)
    z = psi(rb.c_pos, rb.c_normal, rb.c_valid)
    fam = sorted(set(bank.family)); fi = torch.tensor([fam.index(f) for f in bank.family], device=dev)
    with torch.no_grad():
        o1 = m(bank, rb.r, sg.qc1, sg.R1, sg.t1, z)
        a, aw = m._last
        a1, aw1 = a.clone(), aw.clone()
        C["anchor_transform_max_abs"] = float((aw - torch.einsum("bji,bmj->bmi", sg.R1, a - sg.t1[:, None])).abs().max())
        rb2 = copy.copy(rb)
        for f in ("c_pos", "c_normal", "t2", "e2", "R2", "qc2", "active", "c_valid", "c_tpl"):
            x = getattr(rb2, f)
            setattr(rb2, f, ~x if x.dtype == torch.bool else x + (3 if x.dtype == torch.long else 0.37))
        o2 = m(bank, rb2.r, sg.qc1, sg.R1, sg.t1, z)
        C["labels_not_inputs_max_abs"] = max(float((o1[k] - o2[k]).abs().max()) for k in ("qc", "R", "t"))
        o3 = m(bank, rb.r, sg.qc1, sg.R1, sg.t1, z, aux=False)
        m2 = copy.deepcopy(m)
        m2.contacts.head.weight.normal_(); m2.contacts.head.bias.normal_()
        m2.intent_tok.weight.normal_(); [p.normal_() for p in m2.intent_pose.parameters()]
        o4 = m2(bank, rb.r, sg.qc1, sg.R1, sg.t1, z)
    C["aux_removed_motion_max_abs"] = max(float((o1[k] - o3[k]).abs().max()) for k in ("qc", "R", "t", "res_root", "res_coord"))
    C["aux_and_unused_replaced_motion_max_abs"] = max(float((o1[k] - o4[k]).abs().max()) for k in ("qc", "R", "t", "res_root", "res_coord"))
    # rigid-motion consistency: moving the start wrist and the world anchors together leaves the wrist-frame anchors fixed
    with torch.no_grad():
        Rg = torch.linalg.qr(torch.randn(3, 3, generator=g0))[0].to(dev); Rg = Rg * torch.sign(torch.det(Rg))
        tg = torch.tensor([0.1, -0.2, 0.05], device=dev)
        a_, _, _ = m.anchors(z)
        R1g, t1g = Rg @ sg.R1, (sg.t1 @ Rg.T) + tg
        awg = ((a_ @ Rg.T + tg) - t1g[:, None]) @ R1g
        C["wrist_frame_invariance_max_abs"] = float((awg - aw).abs().max())
    out = m(bank, rb.r, sg.qc1, sg.R1, sg.t1, z)
    tr_root, tr_j = target_residuals(bank, rb.r, sg.qc1, sg.R1, sg.t1, sg.qcg, sg.Rg, sg.tg, mc)
    lm, _, _ = residual_loss(out, tr_root, tr_j, bank.coord_valid[rb.r], fi[rb.r], len(fam))
    lm.backward()
    C["grad_motion_to_anchor_head"] = gnorm([p.grad for p in m.anchor_head.parameters()])
    C["grad_motion_to_anchor_in"] = gnorm([p.grad for p in m.anchor_in.parameters()])
    C["grad_motion_to_graph_blocks"] = gnorm([p.grad for p in m.decoder.blocks.parameters()])
    C["unused_and_frozen_grads_none"] = all(p.grad is None for n, p in m.named_parameters() if n.startswith(FROZEN_PREFIXES + UNUSED_PREFIXES))
    with torch.no_grad():
        err = 0.0
        for rid in rb.r.unique():
            s = (rb.r == rid).nonzero().squeeze(1)
            oi = m(bank, rb.r[s], sg.qc1[s], sg.R1[s], sg.t1[s], z[s])
            err = max(err, max(float((oi[k] - o1[k][s]).abs().max()) for k in ("qc", "R", "t")))
        C["mixed_batch_vs_single_max_abs"] = err
        d = m.decoder.decode(bank, rb.r, sg.qc1, sg.R1, sg.t1, tr_root, tr_j)
        cv = bank.coord_valid[rb.r]
        C["decode_roundtrip_max_abs"] = max(float((d["t"] - sg.tg).abs().max()), float((d["R"] - sg.Rg).abs().max()),
                                            float(((d["qc"] - sg.qcg) * cv).abs().max()))
        C["anchors_at_init"] = anchor_stats(a1, aw1, rb.c_pos, rb.c_valid)
    C["pass"] = bool(C["frozen_weights_equal_v1"] and C["z_stats_equal_v1"] and C["frozen_no_grad"] and C["frozen_eval_mode_under_train"]
                     and C["psi_frozen_eval"] and C["surface_sampling"] == "hand_frame" and C["fresh_heads_zero_init"]
                     and C["trainable_excludes_frozen_and_unused"] and C["anchor_transform_max_abs"] < 1e-5
                     and C["labels_not_inputs_max_abs"] == 0.0 and C["aux_removed_motion_max_abs"] == 0.0
                     and C["aux_and_unused_replaced_motion_max_abs"] == 0.0 and C["wrist_frame_invariance_max_abs"] < 1e-5
                     and C["grad_motion_to_anchor_head"] > 0 and C["grad_motion_to_anchor_in"] > 0
                     and C["grad_motion_to_graph_blocks"] > 0 and C["unused_and_frozen_grads_none"]
                     and C["mixed_batch_vs_single_max_abs"] < 1e-4 and C["decode_roundtrip_max_abs"] < 1e-4)
    return C


@torch.no_grad()
def motion_eval(model, psi, bank, rec, sg, bs=1024):
    """Movement and wrist error on a pool (fixed starts) + anchor statistics."""
    model.eval()
    mv, md, wm, wd, A = [], [], [], [], []
    for s in range(0, len(rec), bs):
        i = torch.arange(s, min(s + bs, len(rec)), device=rec.r.device)
        rb = rec.index(i)
        g = type(sg)(**{k: getattr(sg, k)[i] for k in sg.__dataclass_fields__})
        z = psi(rb.c_pos, rb.c_normal, rb.c_valid)
        o = model(bank, rb.r, g.qc1, g.R1, g.t1, z, aux=False)
        mv.append((o["t"] - g.t1).norm(dim=-1) * 1000); md.append(torch.rad2deg(geodesic_angle(o["R"], g.R1)))
        wm.append((o["t"] - g.tg).norm(dim=-1) * 1000); wd.append(torch.rad2deg(geodesic_angle(o["R"], g.Rg)))
        A.append((len(i), anchor_stats(*model._last, rb.c_pos, rb.c_valid)))
    n = sum(k for k, _ in A)
    out = {"move_mm": float(torch.cat(mv).mean()), "move_deg": float(torch.cat(md).mean()),
           "wrist_mm": float(torch.cat(wm).mean()), "wrist_deg": float(torch.cat(wd).mean()),
           "noop_wrist_mm": float(((sg.t1 - sg.tg).norm(dim=-1) * 1000).mean())}
    out |= {f"anchor/{k}": sum(w * a[k] for w, a in A) / n for k in A[0][1]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--checks_only", action="store_true")
    ap.add_argument("--steps", type=int, default=None, help="stop after N steps (schedule horizon stays total_steps)")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    stop_at = a.steps or cfg["stop_steps"]
    A = cfg["anchor"]
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])  # identical construction order to v1 / D
    bank = build_bank(cfg, dev)
    assert bank.surface_sampling == "hand_frame"
    families = sorted(set(bank.family))
    fam_index = torch.tensor([families.index(f) for f in bank.family], device=dev)
    train = load_pool(cfg["train"], bank).to(dev)
    val = load_pool(cfg["val"], bank).to(dev)
    extra = {k: load_pool(v, bank).to(dev) for k, v in cfg.get("extra_eval", {}).items()}
    mc = ScratchConfig.from_dict(cfg["model"])
    ref = torch.load(cfg["set_config_from"], map_location="cpu", weights_only=False)
    scfg = SetBridgeConfig(**ref["config"])
    model = AnchorDirect(mc, scfg, n_anchors=A["n_anchors"], f_dim=A["f_dim"], out_scale=A["out_scale"]).to(dev)
    src = torch.load(cfg["trunk_from"], map_location="cpu", weights_only=False)
    assert src["kind"] == "direct_worldaux_v1"
    assert src["model_config"] == mc.to_dict() and src["set_config"] == scfg.to_dict(), "architecture must equal v1"
    ref_state = {k: v for k, v in src["model"].items() if k.startswith(FROZEN_PREFIXES)}
    missing = model.load_state_dict(ref_state, strict=False)
    assert not missing.unexpected_keys and all(not k.startswith(FROZEN_PREFIXES) for k in missing.missing_keys)
    model.freeze_trunk()
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    noise = start_noise(cfg)
    params = model.trainable()
    n_params = {"total": sum(p.numel() for p in model.parameters()), "trainable": sum(p.numel() for p in params),
                "anchor_head": sum(p.numel() for p in model.anchor_head.parameters()),
                "anchor_in": sum(p.numel() for p in model.anchor_in.parameters()),
                "get": sum(p.numel() for p in model.decoder.parameters()),
                "frozen_trunk": sum(p.numel() for p in model.trunk.parameters()),
                "unused_intent_tok_pose": sum(p.numel() for n, p in model.named_parameters() if n.startswith(UNUSED_PREFIXES))}
    checks = run_checks(model, ref_state, psi, bank, train, mc, noise, dev)
    checks["n_params"] = n_params
    checks["frozen_sha16"] = frozen_state_sha(model)
    checks["frozen_sha16_equals_trackD"] = checks["frozen_sha16"] == cfg.get("expected_frozen_sha16")
    checks["pass"] = checks["pass"] and checks["frozen_sha16_equals_trackD"]
    (out / "checks.json").write_text(json.dumps(checks, indent=1))
    print("checks", json.dumps(checks), flush=True)
    assert checks["pass"], "anchor diagnostic implementation checks failed"
    if a.checks_only:
        return

    opt = torch.optim.AdamW([{"params": params, "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]}])
    W, Tt = cfg["warmup_steps"], cfg["total_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / W) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / Tt))))
    val_sg = fixed_start(bank, val, cfg["val_start_seed"], **noise)
    val_sg = type(val_sg)(**{k: getattr(val_sg, k).to(dev) for k in val_sg.__dataclass_fields__})
    extra_sg = {}
    for k, v in extra.items():
        g = fixed_start(bank, v, cfg["val_start_seed"], **noise)
        extra_sg[k] = type(g)(**{f: getattr(g, f).to(dev) for f in g.__dataclass_fields__})
    step, best = 0, None
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step, best = ck["step"], ck.get("best")
        assert frozen_state_sha(model) == checks["frozen_sha16"]
        print(f"resumed from step {step}")
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    data_files = sorted({s["npz"] for s in cfg["train"] + cfg.get("val", []) + sum(cfg.get("extra_eval", {}).values(), [])})
    prov = {"kind": "direct_anchor_a1", "trunk_from": cfg["trunk_from"], "trunk_from_sha16": sha16(cfg["trunk_from"]),
            "trunk_from_step": src.get("step"), "trunk_history": "direct_worldaux_v1 best.pt, trained jointly with a world-frame "
            "contact loss: contact supervision is present upstream of the anchors", "anchor": A,
            "frozen_sha16": checks["frozen_sha16"], "surface_sampling": bank.surface_sampling, "n_params": n_params,
            "psi_ckpt_sha16": psi.ckpt_sha, "schedule": {"warmup": W, "cosine_horizon": Tt, "stop_at": stop_at},
            "optimizer": {"lr": cfg["lr"], "wd": cfg["weight_decay"], "clip": cfg["grad_clip"]},
            "model_repo": git_state(Path(__file__).parents[3]), "data_sha16": {f: sha16(f) for f in data_files},
            "n_train": len(train), "n_val": len(val), "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else None}
    (out / "provenance.json").write_text(json.dumps(prov, indent=1))
    print(f"params {n_params} on {dev}; train {len(train)} / val {len(val)}", flush=True)

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    sampler = BalancedSampler(train.to("cpu"), bank.family, seed=cfg["seed"])
    log = open(out / "log.jsonl", "a")
    sgen = torch.Generator().manual_seed(cfg["seed"] + 1000 + step)

    def save(path, extra_=None):
        tmp = Path(str(path) + ".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "best": best,
                    "config": cfg, "model_config": mc.to_dict(), "set_config": scfg.to_dict(), "anchor": A,
                    "kind": "direct_anchor_a1"} | (extra_ or {}), tmp)
        os.replace(tmp, path)

    def do_eval():
        res = evaluate(model, psi, bank, val, val_sg, cfg, families)
        for k, v_ in extra.items():
            res |= {f"{k}:{m_}": x for m_, x in evaluate(model, psi, bank, v_, extra_sg[k], cfg, families).items()}
        res["motion/val"] = motion_eval(model, psi, bank, val, val_sg)
        res |= {f"motion/{k}": motion_eval(model, psi, bank, v_, extra_sg[k]) for k, v_ in extra.items()}
        res["contacts/val"] = contact_eval(model, psi, val)
        model.train()
        return res

    t_last, n_last, t0 = time.time(), 0, time.time()
    if step == 0:
        res = do_eval()
        log.write(json.dumps({"step": 0, "eval": res}) + "\n"); log.flush()
        print(json.dumps({"step": 0, "motion/val": res["motion/val"]}), flush=True)
    model.train()
    while step < stop_at and not stop["flag"]:
        idx = sampler.sample(cfg["batch_size"]).to(dev)
        rb = train.index(idx)
        sg = draw_start(bank, rb, sgen, **noise)
        z = psi(rb.c_pos, jitter_normals(rb.c_normal, cfg["normal_noise_rad"], sgen), rb.c_valid)
        out_m = model(bank, rb.r, sg.qc1, sg.R1, sg.t1, z, aux=False)
        tr_root, tr_j = target_residuals(bank, rb.r, sg.qc1, sg.R1, sg.t1, sg.qcg, sg.Rg, sg.tg, mc)
        lm, lr_, lj_ = residual_loss(out_m, tr_root, tr_j, bank.coord_valid[rb.r], fam_index[rb.r], len(families))
        opt.zero_grad(set_to_none=True)
        lm.backward()
        log_now = (step + 1) % cfg["log_every"] == 0
        if log_now:
            g_ah = gnorm([p.grad for p in model.anchor_head.parameters()])
            g_ai = gnorm([p.grad for p in model.anchor_in.parameters()])
            g_get = gnorm([p.grad for p in model.decoder.parameters()])
            ast = anchor_stats(*[x.detach() for x in model._last], rb.c_pos, rb.c_valid)
            with torch.no_grad():
                mv = float(((out_m["t"] - sg.t1).norm(dim=-1) * 1000).mean())
        gn = torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        opt.step(); sched.step()
        step += 1
        n_last += len(idx)
        if log_now:
            row = {"step": step, "lr": sched.get_last_lr()[0], "train/motion": lm.item(), "train/root": lr_.mean().item(),
                   "train/joint": lj_.mean().item(), "train/move_mm": mv, "grad/trainable": float(gn), "grad/anchor_head": g_ah,
                   "grad/anchor_in": g_ai, "grad/get": g_get, "clip": float(gn) > cfg["grad_clip"],
                   **{f"anchor/{k}": v for k, v in ast.items()},
                   "examples_per_s": n_last / (time.time() - t_last), "wall_s": time.time() - t0,
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0.0}
            log.write(json.dumps(row) + "\n"); log.flush()
            print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in row.items()}), flush=True)
            t_last, n_last = time.time(), 0
        if step % cfg["eval_every"] == 0 or step == stop_at:
            res = do_eval()
            key = (res["all/model/success"], -res["all/model/contact_mm"])
            log.write(json.dumps({"step": step, "eval": res}) + "\n"); log.flush()
            brief = {k: round(x, 3) for k, x in res.items() if isinstance(x, float) and k.split("/")[-1] in ("contact_mm", "success")
                     and "robot_macro" not in k and "/wrist_only/" not in k}
            print(json.dumps({"step": step, **brief, "motion/val": res["motion/val"]}), flush=True)
            if best is None or key > tuple(best["key"]):
                best = {"key": list(key), "step": step}
                save(out / "best.pt", {"selected_on": "val all success, then all contact_mm (J1 rule)"})
        if step % cfg["ckpt_every"] == 0:
            save(out / "last.pt")
    save(out / "last.pt")
    if step >= stop_at:
        fs = frozen_state_sha(model)
        (out / "DONE").write_text(json.dumps({"best": best, "wall_end": time.time(), "frozen_sha16_end": fs,
                                              "frozen_unchanged": fs == checks["frozen_sha16"]}))


if __name__ == "__main__":
    main()
