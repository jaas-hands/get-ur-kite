"""Train direct_worldaux_v1 (DirectWorldAux): frozen psi z -> shared z-only intent trunk -> intent tokens (+ start
world pose) -> GET motion; training-only world-frame contact-set head on the trunk features.

    python -m getkite.decoder.train_direct_worldaux --config configs/decoder/scratch/direct_worldaux_v1.json \
        [--set k=v ...] [--checks_only] [--overfit N]

Same data, seed, start sampling, motion loss, budget, checkpoint-selection rule and decoder optimiser settings as J1
(j1_joint50k_hf). Parameter groups: "decoder" = GET backbone/heads + intent-token projection + intent pose
conditioning (lr 3e-4, wd 1e-4, clip 1.0: the reference decoder's); "trunk" = intent trunk + auxiliary contact head
(lr 1e-3, wd 1e-4, clip 1.0: the reference translator's / J1 contact group's). One warmup+cosine schedule.
Loss = motion residual loss + contact_weight * world-frame contact-set loss (historical set loss, unchanged; labels from
training contacts, supervision only). Implementation checks run first (out_dir/checks.json; abort on failure).
--overfit N: tiny fixed-example learnability check (N fixed records, fixed starts, no jitter, no eval).
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
from .train_joint import contact_eval, grad_norm, world_labels
from .train_scratch import (FrozenPsi, build_bank, evaluate, git_state, jitter_normals, load_config, load_pool,
                            residual_loss, sha16, start_noise)

DEFAULTS = dict(contact_weight=1.0, trunk_lr=1e-3, trunk_grad_clip=1.0, trunk_weight_decay=1e-4,
                expected_surf_idx_sha16="6e9d3d5a912a0086")  # J1's corrected hand-frame surface selection


def param_groups(model):
    trunk = list(model.trunk.parameters()) + list(model.contacts.parameters())
    ids = {id(p) for p in trunk}
    dec = [p for p in model.parameters() if id(p) not in ids]
    return dec, trunk


def gnorm(grads):
    g = [x.flatten() for x in grads if x is not None]
    return float(torch.cat(g).norm()) if g else 0.0


def run_checks(model, psi, bank, train, cfg, mc, noise, dev, expected_hash):
    """Implementation checks on real data. Returns dict with 'pass'."""
    C = {}
    C["psi_frozen_eval"] = (not any(p.requires_grad for p in psi.parameters())) and not psi.training
    C["surface_sampling"] = bank.surface_sampling
    C["surf_idx_sha16"] = hashlib.sha256(bank.surf_idx.cpu().numpy().tobytes()).hexdigest()[:16]
    C["surf_idx_matches_j1"] = C["surf_idx_sha16"] == expected_hash
    with torch.no_grad():
        C["trained_model_heads_zero_init"] = all(float(m[-1].weight.abs().max()) == 0.0
                                                 for m in (model.decoder.root_head, model.decoder.joint_head))
    m = copy.deepcopy(model)  # path checks on a copy with small random output heads (zero-init heads block motion grads)
    g0 = torch.Generator(device="cpu").manual_seed(7)
    with torch.no_grad():
        for h in (m.decoder.root_head, m.decoder.joint_head):
            h[-1].weight.copy_(torch.randn(h[-1].weight.shape, generator=g0) * 1e-2)
    m.decoder.amp = False
    m.eval()
    gen = torch.Generator().manual_seed(123)
    sampler = BalancedSampler(train.to("cpu"), bank.family, seed=123)
    rb = train.index(sampler.sample(64).to(dev))
    sg = draw_start(bank, rb, gen, **noise)
    z = psi(rb.c_pos, rb.c_normal, rb.c_valid)
    fam = sorted(set(bank.family)); fi = torch.tensor([fam.index(f) for f in bank.family], device=dev)
    # 1. no goal/contact labels in the prediction inputs: forward() takes (bank, r, q, R, t, z) only; altering every
    #    label field of the records (contacts, goals, correspondences) with z held fixed leaves the output unchanged.
    with torch.no_grad():
        o1 = m(bank, rb.r, sg.qc1, sg.R1, sg.t1, z)
        rb2 = copy.copy(rb)
        for f in ("c_pos", "c_normal", "t2", "e2", "R2", "qc2", "active", "c_valid", "c_tpl"):
            x = getattr(rb2, f)
            setattr(rb2, f, ~x if x.dtype == torch.bool else x + (3 if x.dtype == torch.long else 0.37))
        o2 = m(bank, rb2.r, sg.qc1, sg.R1, sg.t1, z)
    C["labels_not_inputs_max_abs"] = max(float((o1[k] - o2[k]).abs().max()) for k in ("qc", "R", "t"))
    # 2. replacing/removing the auxiliary predictions does not change motion
    with torch.no_grad():
        o3 = m(bank, rb.r, sg.qc1, sg.R1, sg.t1, z, aux=False)
        m2 = copy.deepcopy(m)
        m2.contacts.head.weight.normal_(); m2.contacts.head.bias.normal_()
        o4 = m2(bank, rb.r, sg.qc1, sg.R1, sg.t1, z)
    C["aux_removed_motion_max_abs"] = max(float((o1[k] - o3[k]).abs().max()) for k in ("qc", "R", "t", "res_root", "res_coord"))
    C["aux_replaced_motion_max_abs"] = max(float((o1[k] - o4[k]).abs().max()) for k in ("qc", "R", "t", "res_root", "res_coord"))
    # 3./4. gradient paths
    out = m(bank, rb.r, sg.qc1, sg.R1, sg.t1, z)
    tr_root, tr_j = target_residuals(bank, rb.r, sg.qc1, sg.R1, sg.t1, sg.qcg, sg.Rg, sg.tg, mc)
    lm, _, _ = residual_loss(out, tr_root, tr_j, bank.coord_valid[rb.r], fi[rb.r], len(fam))
    CL, _ = m.contacts.set_loss(*out["contact_set"], *world_labels(rb))
    groups = {"trunk": list(m.trunk.parameters()), "aux_head": list(m.contacts.parameters()),
              "intent_tok": list(m.intent_tok.parameters()) + list(m.intent_pose.parameters()),
              "graph_blocks": list(m.decoder.blocks.parameters())}
    for lname, L in (("motion", lm), ("contact", CL["total"])):
        for gname, ps in groups.items():
            C[f"grad_{lname}_to_{gname}"] = gnorm(torch.autograd.grad(L, ps, retain_graph=True, allow_unused=True))
    # 5. mixed joint counts / padding: each example's output is the same alone (single-robot batch) as in the mixed batch
    with torch.no_grad():
        err = 0.0
        for rid in rb.r.unique():
            s = (rb.r == rid).nonzero().squeeze(1)
            oi = m(bank, rb.r[s], sg.qc1[s], sg.R1[s], sg.t1[s], z[s])
            err = max(err, max(float((oi[k] - o1[k][s]).abs().max()) for k in ("qc", "R", "t")))
        C["mixed_batch_vs_single_max_abs"] = err
        C["n_robots_in_batch"] = int(len(rb.r.unique())); C["joint_counts"] = sorted({int(x) for x in bank.coord_valid[rb.r].sum(1)})
    # 6. output conventions: decode(target_residuals(goal)) reproduces the goal
    with torch.no_grad():
        d = m.decoder.decode(bank, rb.r, sg.qc1, sg.R1, sg.t1, tr_root, tr_j)
        cv = bank.coord_valid[rb.r]
        C["decode_roundtrip_max_abs"] = max(float((d["t"] - sg.tg).abs().max()), float((d["R"] - sg.Rg).abs().max()),
                                            float(((d["qc"] - sg.qcg) * cv).abs().max()))
    # 7. contact-loss matching on known examples: permuted exact prediction -> pos/normal 0; +1 cm x-shift -> pos 1.0
    with torch.no_grad():
        gp, gn, gv = world_labels(rb)
        S = m.contacts.c.slots
        K = gp.shape[1]
        P = torch.nn.functional.pad(gp, (0, 0, 0, S - K)); N = torch.nn.functional.pad(gn, (0, 0, 0, S - K))
        V = torch.nn.functional.pad(gv, (0, S - K), value=False)
        N = torch.where(V[..., None], N, torch.tensor([0.0, 0.0, 1.0], device=dev))
        perm = torch.randperm(S, generator=torch.Generator().manual_seed(1)).to(dev)
        lg = torch.where(V, 20.0, -20.0)
        L0, a0 = m.contacts.set_loss(P[:, perm], N[:, perm], lg[:, perm], gp, gn, gv)
        L1, _ = m.contacts.set_loss(P[:, perm] + torch.tensor([0.01, 0, 0], device=dev), N[:, perm], lg[:, perm], gp, gn, gv)
    C["setloss_exact"] = {k: float(v) for k, v in L0.items()}
    C["setloss_shift1cm_pos"] = float(L1["pos"])
    C["pass"] = bool(C["psi_frozen_eval"] and C["surface_sampling"] == "hand_frame" and C["surf_idx_matches_j1"]
                     and C["labels_not_inputs_max_abs"] == 0.0 and C["aux_removed_motion_max_abs"] == 0.0
                     and C["aux_replaced_motion_max_abs"] == 0.0
                     and C["grad_contact_to_trunk"] > 0 and C["grad_contact_to_aux_head"] > 0
                     and C["grad_contact_to_intent_tok"] == 0.0 and C["grad_contact_to_graph_blocks"] == 0.0
                     and C["grad_motion_to_trunk"] > 0 and C["grad_motion_to_graph_blocks"] > 0
                     and C["grad_motion_to_intent_tok"] > 0 and C["grad_motion_to_aux_head"] == 0.0
                     and C["mixed_batch_vs_single_max_abs"] < 1e-4 and C["decode_roundtrip_max_abs"] < 1e-4
                     and C["setloss_exact"]["pos"] < 1e-8 and C["setloss_exact"]["normal"] < 1e-6
                     and abs(C["setloss_shift1cm_pos"] - 1.0) < 1e-4)
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--checks_only", action="store_true")
    ap.add_argument("--overfit", type=int, default=0)
    a = ap.parse_args()
    cfg = DEFAULTS | load_config(a.config, a.set)
    out = Path(cfg["out_dir"] + (f"_overfit{a.overfit}" if a.overfit else "")); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    bank = build_bank(cfg, dev)
    assert bank.surface_sampling == "hand_frame", "direct_worldaux uses the corrected surface sampling"
    families = sorted(set(bank.family))
    fam_index = torch.tensor([families.index(f) for f in bank.family], device=dev)
    train = load_pool(cfg["train"], bank).to(dev)
    val = load_pool(cfg["val"], bank).to(dev)
    extra = {k: load_pool(v, bank).to(dev) for k, v in cfg.get("extra_eval", {}).items()}
    mc = ScratchConfig.from_dict(cfg["model"])
    ref = torch.load(cfg["set_config_from"], map_location="cpu", weights_only=False)  # config only; weights NOT loaded
    scfg = SetBridgeConfig(**ref["config"])
    model = DirectWorldAux(mc, scfg).to(dev)
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    noise = start_noise(cfg)
    with torch.no_grad():  # trunk z standardisation from the training pool (clean normals, as J1 / the translator)
        zs = torch.cat([psi(train.c_pos[s:s + 4096], train.c_normal[s:s + 4096], train.c_valid[s:s + 4096]) for s in range(0, len(train), 4096)])
        model.trunk.set_z_stats(zs)
    dec_p, trunk_p = param_groups(model)
    n_params = {"total": sum(p.numel() for p in model.parameters()), "deploy": sum(p.numel() for p in model.deploy_parameters()),
                "trunk": sum(p.numel() for p in model.trunk.parameters()), "aux_head": sum(p.numel() for p in model.contacts.parameters()),
                "intent_tok+pose": sum(p.numel() for p in list(model.intent_tok.parameters()) + list(model.intent_pose.parameters())),
                "get": sum(p.numel() for p in model.decoder.parameters())}
    checks = run_checks(model, psi, bank, train, cfg, mc, noise, dev, cfg["expected_surf_idx_sha16"])
    checks["n_params"] = n_params
    (out / "checks.json").write_text(json.dumps(checks, indent=1))
    print("checks", json.dumps(checks), flush=True)
    assert checks["pass"], "direct_worldaux implementation checks failed"
    if a.checks_only:
        return

    opt = torch.optim.AdamW([{"params": dec_p, "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]},
                             {"params": trunk_p, "lr": cfg["trunk_lr"], "weight_decay": cfg["trunk_weight_decay"]}])
    W, Tt = cfg["warmup_steps"], cfg["total_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / W) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / Tt))))

    if a.overfit:  # tiny fixed-example learnability check
        g = torch.Generator().manual_seed(5)
        rb = train.index(BalancedSampler(train.to("cpu"), bank.family, seed=5).sample(a.overfit).to(dev))
        sg = draw_start(bank, rb, g, **noise)
        z = psi(rb.c_pos, rb.c_normal, rb.c_valid)
        tr_root, tr_j = target_residuals(bank, rb.r, sg.qc1, sg.R1, sg.t1, sg.qcg, sg.Rg, sg.tg, mc)
        hist = []
        model.train()
        for step in range(1, Tt + 1):
            o = model(bank, rb.r, sg.qc1, sg.R1, sg.t1, z)
            lm, _, _ = residual_loss(o, tr_root, tr_j, bank.coord_valid[rb.r], fam_index[rb.r], len(families))
            CL, _ = model.contacts.set_loss(*o["contact_set"], *world_labels(rb))
            opt.zero_grad(set_to_none=True); (lm + cfg["contact_weight"] * CL["total"]).backward()
            torch.nn.utils.clip_grad_norm_(dec_p, cfg["grad_clip"]); torch.nn.utils.clip_grad_norm_(trunk_p, cfg["trunk_grad_clip"])
            opt.step(); sched.step()
            if step % 100 == 0 or step == 1:
                model.eval()
                with torch.no_grad():
                    from .train_scratch import pose_metrics
                    oe = model(bank, rb.r, sg.qc1, sg.R1, sg.t1, z)
                    pm = pose_metrics(bank, rb, rb.r, oe["qc"], oe["t"], oe["R"], sg, cfg)
                    ce = contact_eval(model, psi, rb)
                model.train()
                row = {"step": step, "motion": lm.item(), "contact": CL["total"].item(), "contact_mm": float(pm["contact_mm"].mean()),
                       "success": float(pm["success"].mean()), "aux_pos_mm": ce["contact_pos_mm"], "aux_count_acc": ce["count_acc"]}
                hist.append(row); print(json.dumps(row), flush=True)
        (out / "overfit.json").write_text(json.dumps({"n": a.overfit, "steps": Tt, "history": hist}, indent=1))
        return

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
        print(f"resumed from step {step}")
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    data_files = sorted({s["npz"] for s in cfg["train"] + cfg.get("val", []) + sum(cfg.get("extra_eval", {}).values(), [])})
    prov = {"kind": "direct_worldaux_v1", "surface_sampling": bank.surface_sampling, "surf_idx_sha16": checks["surf_idx_sha16"],
            "n_params": n_params, "psi_ckpt_sha16": psi.ckpt_sha, "set_config_from": cfg["set_config_from"],
            "contact_weight": cfg["contact_weight"],
            "param_groups": {"decoder": {"lr": cfg["lr"], "wd": cfg["weight_decay"], "clip": cfg["grad_clip"]},
                             "trunk": {"lr": cfg["trunk_lr"], "wd": cfg["trunk_weight_decay"], "clip": cfg["trunk_grad_clip"]}},
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
                    "config": cfg, "model_config": mc.to_dict(), "set_config": scfg.to_dict(), "kind": "direct_worldaux_v1"} | (extra_ or {}), tmp)
        os.replace(tmp, path)

    t_last, n_last, t0 = time.time(), 0, time.time()
    model.train()
    while step < Tt and not stop["flag"]:
        idx = sampler.sample(cfg["batch_size"]).to(dev)
        rb = train.index(idx)
        sg = draw_start(bank, rb, sgen, **noise)
        z = psi(rb.c_pos, jitter_normals(rb.c_normal, cfg["normal_noise_rad"], sgen), rb.c_valid)
        out_m = model(bank, rb.r, sg.qc1, sg.R1, sg.t1, z)
        tr_root, tr_j = target_residuals(bank, rb.r, sg.qc1, sg.R1, sg.t1, sg.qcg, sg.Rg, sg.tg, mc)
        lm, lr_, lj_ = residual_loss(out_m, tr_root, tr_j, bank.coord_valid[rb.r], fam_index[rb.r], len(families))
        CL, _ = model.contacts.set_loss(*out_m["contact_set"], *world_labels(rb))
        loss = lm + cfg["contact_weight"] * CL["total"]
        log_now = (step + 1) % cfg["log_every"] == 0
        if log_now:  # separate gradient contributions into the shared intent trunk (diagnostic, every log step)
            tp = list(model.trunk.parameters())
            g_motion = gnorm(torch.autograd.grad(lm, tp, retain_graph=True, allow_unused=True))
            g_contact = gnorm(torch.autograd.grad(CL["total"], tp, retain_graph=True, allow_unused=True))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn_d = torch.nn.utils.clip_grad_norm_(dec_p, cfg["grad_clip"])
        gn_t = torch.nn.utils.clip_grad_norm_(trunk_p, cfg["trunk_grad_clip"])
        opt.step(); sched.step()
        step += 1
        n_last += len(idx)
        if log_now:
            with torch.no_grad():
                pv = out_m["contact_set"][2].sigmoid() > 0.5
                pv[:, 0] |= ~pv.any(1)
            row = {"step": step, "lr": sched.get_last_lr()[0], "lr_trunk": sched.get_last_lr()[1], "train/loss": loss.item(),
                   "train/motion": lm.item(), "train/root": lr_.mean().item(), "train/joint": lj_.mean().item(),
                   "train/contact": CL["total"].item(), "train/contact_pos": CL["pos"].item(), "train/contact_normal": CL["normal"].item(),
                   "train/contact_exist": CL["exist"].item(), "train/contact_weighted": cfg["contact_weight"] * CL["total"].item(),
                   "grad/decoder": float(gn_d), "grad/trunk": float(gn_t), "grad/trunk_from_motion": g_motion,
                   "grad/trunk_from_contact": g_contact,
                   "clip/decoder": float(gn_d) > cfg["grad_clip"], "clip/trunk": float(gn_t) > cfg["trunk_grad_clip"],
                   "count/pred_mean": float(pv.sum(1).float().mean()), "count/true_mean": float(rb.c_valid.sum(1).float().mean()),
                   "count/acc": float((pv.sum(1) == rb.c_valid.sum(1)).float().mean()),
                   "examples_per_s": n_last / (time.time() - t_last), "wall_s": time.time() - t0,
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0.0}
            log.write(json.dumps(row) + "\n"); log.flush()
            print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in row.items()}), flush=True)
            t_last, n_last = time.time(), 0
        if step % cfg["eval_every"] == 0 or step == Tt:
            res = evaluate(model, psi, bank, val, val_sg, cfg, families)
            key = (res["all/model/success"], -res["all/model/contact_mm"])
            for k, v_ in extra.items():
                res |= {f"{k}:{m}": x for m, x in evaluate(model, psi, bank, v_, extra_sg[k], cfg, families).items()}
            res["contacts/val"] = contact_eval(model, psi, val)
            res |= {f"contacts/{k}": contact_eval(model, psi, v_) for k, v_ in extra.items()}
            log.write(json.dumps({"step": step, "eval": res}) + "\n"); log.flush()
            brief = {k: round(x, 3) for k, x in res.items() if isinstance(x, float) and k.split("/")[-1] in ("contact_mm", "success")
                     and "robot_macro" not in k and "/wrist_only/" not in k and "/noop/" not in k}
            print(json.dumps({"step": step, **brief, "contacts/val": res["contacts/val"]}), flush=True)
            if best is None or key > tuple(best["key"]):
                best = {"key": list(key), "step": step}
                save(out / "best.pt", {"selected_on": "val all success, then all contact_mm (J1 rule)"})
        if step % cfg["ckpt_every"] == 0:
            save(out / "last.pt")
    save(out / "last.pt")
    if step >= Tt:
        (out / "DONE").write_text(json.dumps({"best": best, "wall_end": time.time()}))


if __name__ == "__main__":
    main()
