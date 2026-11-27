"""Track D (2026-11-19): bounded frozen-trunk diagnostic for the direct decoder.

Question: does competition from contact supervision (shared-trunk gradient dominance / feature drift) prevent the motion
network from using a spatially informative intent representation?

    frozen psi z --(z standardisation, exactly as stored)--> FROZEN intent trunk from direct_worldaux_v1 (eval mode) --> h
        h --> FRESH intent-token projection + FRESH start-wrist world-pose conditioning --> FRESH GET --> motion
        h --> frozen auxiliary contact head (diagnostics only; never read by the motion path)

Trained parameters: intent_tok, intent_pose, decoder (GET) only. Loss: motion residual loss only (no contact loss).
Everything else = direct_worldaux_v1 / J1: data, family sampling, start perturbations, normal jitter, residual loss,
decoder optimiser group (AdamW lr 3e-4, wd 1e-4, clip 1.0), warmup 500 + cosine schedule with the SAME 50k horizon
(so the learning rate at every step equals v1's and J1's; the run stops at --steps, default 10k), eval every 2500 on the
same validation / held-out / training-goal pools, same selection rule. The fresh modules are initialised with the
original seed in the original construction order (manual_seed -> build_bank -> pools -> DirectWorldAux), i.e. the same
initial motion-network weights v1 started from; the trunk/aux weights and z statistics are then loaded from the v1 checkpoint.

    python -m getkite.decoder.train_direct_frozentrunk --config configs/decoder/scratch/direct_frozentrunk_d1.json \
        [--checks_only] [--steps N] [--set k=v ...]
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

from .direct_worldaux import DirectWorldAux
from .get_scratch import ScratchConfig, target_residuals
from .multifamily import BalancedSampler, draw_start, fixed_start
from .spatial_bridge import SetBridgeConfig
from .train_direct_worldaux import gnorm
from .train_joint import contact_eval
from .train_scratch import (FrozenPsi, build_bank, evaluate, git_state, jitter_normals, load_config, load_pool,
                            residual_loss, sha16, start_noise)

FROZEN_PREFIXES = ("trunk.", "contacts.")


def frozen_state_sha(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        if k.startswith(FROZEN_PREFIXES):
            h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


class FrozenTrunkDirect(DirectWorldAux):
    """DirectWorldAux whose trunk + auxiliary head stay frozen and in eval mode even when .train() is called."""

    def freeze_trunk(self):
        for m in (self.trunk, self.contacts):
            m.requires_grad_(False); m.eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        self.trunk.eval(); self.contacts.eval()
        return self

    def trainable(self):
        return [p for n, p in self.named_parameters() if not n.startswith(FROZEN_PREFIXES)]


def run_checks(model, ref_state, psi, bank, train, mc, noise, dev, zs):
    C = {}
    sd = model.state_dict()
    C["frozen_weights_equal_v1"] = all(torch.equal(sd[k].cpu(), ref_state[k].cpu()) for k in ref_state if k.startswith(FROZEN_PREFIXES))
    C["n_frozen_tensors"] = sum(k.startswith(FROZEN_PREFIXES) for k in ref_state)
    C["z_stats_equal_v1"] = torch.equal(sd["trunk.z_mu"].cpu(), ref_state["trunk.z_mu"].cpu()) and torch.equal(sd["trunk.z_sd"].cpu(), ref_state["trunk.z_sd"].cpu())
    C["z_stats_recomputed_max_abs_diff"] = max(float((zs.mean(0).cpu() - ref_state["trunk.z_mu"].cpu()).abs().max()),
                                               float((zs.std(0).clamp_min(1e-4).cpu() - ref_state["trunk.z_sd"].cpu()).abs().max()))
    model.train()
    C["frozen_no_grad"] = not any(p.requires_grad for n, p in model.named_parameters() if n.startswith(FROZEN_PREFIXES))
    C["frozen_eval_mode_under_train"] = (not model.trunk.training) and (not model.contacts.training) and model.decoder.training
    C["psi_frozen_eval"] = (not any(p.requires_grad for p in psi.parameters())) and not psi.training
    C["surface_sampling"] = bank.surface_sampling
    C["surf_idx_sha16"] = hashlib.sha256(bank.surf_idx.cpu().numpy().tobytes()).hexdigest()[:16]
    with torch.no_grad():
        C["fresh_heads_zero_init"] = all(float(m[-1].weight.abs().max()) == 0.0 for m in (model.decoder.root_head, model.decoder.joint_head))
    m = copy.deepcopy(model)  # path checks on a copy with small random output heads (zero-init heads block motion grads)
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
        rb2 = copy.copy(rb)
        for f in ("c_pos", "c_normal", "t2", "e2", "R2", "qc2", "active", "c_valid", "c_tpl"):
            x = getattr(rb2, f)
            setattr(rb2, f, ~x if x.dtype == torch.bool else x + (3 if x.dtype == torch.long else 0.37))
        o2 = m(bank, rb2.r, sg.qc1, sg.R1, sg.t1, z)
        C["labels_not_inputs_max_abs"] = max(float((o1[k] - o2[k]).abs().max()) for k in ("qc", "R", "t"))
        o3 = m(bank, rb.r, sg.qc1, sg.R1, sg.t1, z, aux=False)
        m2 = copy.deepcopy(m)
        m2.contacts.head.weight.normal_(); m2.contacts.head.bias.normal_()
        o4 = m2(bank, rb.r, sg.qc1, sg.R1, sg.t1, z)
    C["aux_removed_motion_max_abs"] = max(float((o1[k] - o3[k]).abs().max()) for k in ("qc", "R", "t", "res_root", "res_coord"))
    C["aux_replaced_motion_max_abs"] = max(float((o1[k] - o4[k]).abs().max()) for k in ("qc", "R", "t", "res_root", "res_coord"))
    out = m(bank, rb.r, sg.qc1, sg.R1, sg.t1, z)
    tr_root, tr_j = target_residuals(bank, rb.r, sg.qc1, sg.R1, sg.t1, sg.qcg, sg.Rg, sg.tg, mc)
    lm, _, _ = residual_loss(out, tr_root, tr_j, bank.coord_valid[rb.r], fi[rb.r], len(fam))
    lm.backward()
    C["grad_motion_to_intent_tok"] = gnorm([p.grad for p in list(m.intent_tok.parameters()) + list(m.intent_pose.parameters())])
    C["grad_motion_to_graph_blocks"] = gnorm([p.grad for p in m.decoder.blocks.parameters()])
    C["frozen_grads_none"] = all(p.grad is None for n, p in m.named_parameters() if n.startswith(FROZEN_PREFIXES))
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
    C["pass"] = bool(C["frozen_weights_equal_v1"] and C["z_stats_equal_v1"] and C["frozen_no_grad"] and C["frozen_eval_mode_under_train"]
                     and C["psi_frozen_eval"] and C["surface_sampling"] == "hand_frame" and C["fresh_heads_zero_init"]
                     and C["labels_not_inputs_max_abs"] == 0.0 and C["aux_removed_motion_max_abs"] == 0.0
                     and C["aux_replaced_motion_max_abs"] == 0.0 and C["grad_motion_to_intent_tok"] > 0
                     and C["grad_motion_to_graph_blocks"] > 0 and C["frozen_grads_none"]
                     and C["mixed_batch_vs_single_max_abs"] < 1e-4 and C["decode_roundtrip_max_abs"] < 1e-4)
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--checks_only", action="store_true")
    ap.add_argument("--steps", type=int, default=None, help="stop after N steps (schedule horizon stays total_steps)")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    stop_at = a.steps or cfg["stop_steps"]
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- identical construction order to train_direct_worldaux (same initial motion-network weights) ---
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
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
    model = FrozenTrunkDirect(mc, scfg).to(dev)
    # --- load the trained trunk + aux head (incl. z standardisation buffers) from direct_worldaux_v1 ---
    src = torch.load(cfg["trunk_from"], map_location="cpu", weights_only=False)
    assert src["kind"] == "direct_worldaux_v1"
    assert src["model_config"] == mc.to_dict() and src["set_config"] == scfg.to_dict(), "architecture must equal v1"
    ref_state = {k: v for k, v in src["model"].items() if k.startswith(FROZEN_PREFIXES)}
    missing = model.load_state_dict(ref_state, strict=False)
    assert not missing.unexpected_keys and all(not k.startswith(FROZEN_PREFIXES) for k in missing.missing_keys)
    model.freeze_trunk()
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    noise = start_noise(cfg)
    with torch.no_grad():
        zs = torch.cat([psi(train.c_pos[s:s + 4096], train.c_normal[s:s + 4096], train.c_valid[s:s + 4096]) for s in range(0, len(train), 4096)])
    params = model.trainable()
    n_params = {"total": sum(p.numel() for p in model.parameters()), "trainable": sum(p.numel() for p in params),
                "frozen_trunk": sum(p.numel() for p in model.trunk.parameters()), "frozen_aux_head": sum(p.numel() for p in model.contacts.parameters()),
                "deploy": sum(p.numel() for p in model.deploy_parameters())}
    checks = run_checks(model, ref_state, psi, bank, train, mc, noise, dev, zs)
    checks["n_params"] = n_params
    checks["trainable_param_ids_exclude_frozen"] = not ({id(p) for p in params} & {id(p) for n, p in model.named_parameters() if n.startswith(FROZEN_PREFIXES)})
    checks["pass"] = checks["pass"] and checks["trainable_param_ids_exclude_frozen"]
    checks["frozen_sha16"] = frozen_state_sha(model)
    checks["aux_contacts_val_at_load"] = contact_eval(model, psi, val)
    (out / "checks.json").write_text(json.dumps(checks, indent=1))
    print("checks", json.dumps(checks), flush=True)
    assert checks["pass"], "frozen-trunk implementation checks failed"
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
    prov = {"kind": "direct_frozentrunk_d1", "trunk_from": cfg["trunk_from"], "trunk_from_sha16": sha16(cfg["trunk_from"]),
            "trunk_from_step": src.get("step"), "trunk_history": "direct_worldaux_v1 best.pt (selected on source val), trained "
            "jointly (motion + world-contact loss) for that many steps x 256 examples (v1 total run: 50k steps, ~1.5 GPU-h)",
            "frozen_sha16": checks["frozen_sha16"], "surface_sampling": bank.surface_sampling, "surf_idx_sha16": checks["surf_idx_sha16"],
            "n_params": n_params, "psi_ckpt_sha16": psi.ckpt_sha, "schedule": {"warmup": W, "cosine_horizon": Tt, "stop_at": stop_at},
            "optimizer": {"lr": cfg["lr"], "wd": cfg["weight_decay"], "clip": cfg["grad_clip"]},
            "model_repo": git_state(Path(__file__).parents[3]),
            "data_sha16": {f: sha16(f) for f in data_files}, "n_train": len(train), "n_val": len(val),
            "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else None}
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
                    "config": cfg, "model_config": mc.to_dict(), "set_config": scfg.to_dict(), "kind": "direct_frozentrunk_d1"} | (extra_ or {}), tmp)
        os.replace(tmp, path)

    def do_eval():
        res = evaluate(model, psi, bank, val, val_sg, cfg, families)
        for k, v_ in extra.items():
            res |= {f"{k}:{m_}": x for m_, x in evaluate(model, psi, bank, v_, extra_sg[k], cfg, families).items()}
        res["contacts/val"] = contact_eval(model, psi, val)
        res |= {f"contacts/{k}": contact_eval(model, psi, v_) for k, v_ in extra.items()}
        model.train()
        return res

    t_last, n_last, t0 = time.time(), 0, time.time()
    if step == 0:  # step-0 reference (fresh motion network, zero-init heads = no movement)
        res = do_eval()
        log.write(json.dumps({"step": 0, "eval": res}) + "\n"); log.flush()
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
            g_tok = gnorm([p.grad for p in list(model.intent_tok.parameters()) + list(model.intent_pose.parameters())])
            g_get = gnorm([p.grad for p in model.decoder.parameters()])
        gn = torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        opt.step(); sched.step()
        step += 1
        n_last += len(idx)
        if log_now:
            row = {"step": step, "lr": sched.get_last_lr()[0], "train/motion": lm.item(), "train/root": lr_.mean().item(),
                   "train/joint": lj_.mean().item(), "grad/trainable": float(gn), "grad/intent_tok": g_tok, "grad/get": g_get,
                   "clip": float(gn) > cfg["grad_clip"], "examples_per_s": n_last / (time.time() - t_last), "wall_s": time.time() - t0,
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
            print(json.dumps({"step": step, **brief, "contacts/val": res["contacts/val"]}), flush=True)
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
