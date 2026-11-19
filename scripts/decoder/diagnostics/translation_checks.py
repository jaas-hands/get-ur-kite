"""Direct checks on the wrist-translation path (2026-11-15).

    python scripts/decoder/diagnostics/translation_checks.py --config CONFIG --runs RUN_DIR [RUN_DIR ...] [--out FILE]

1. Input shift. Hold goal, start rotation and start joints fixed; translate the start wrist by a known world
   displacement d (3 mm, random direction per example). Capture the exact tensors entering the model (inputs of
   contact_in / root_in / joint_in / surf_in, and nbr_enc) with forward hooks, and compare:
     contact xyz feature should change by -R1^T d * pos_scale (valid contacts), normals and padding unchanged;
     root / joint / surface inputs should be unchanged (they live in the start wrist frame);
     target translation residual should change by -R1^T d / res_xyz_m; rotation and joint targets unchanged.
   Also reports how much the trained model's root-token feature (input of the root head's last layer) moves under
   this shift versus under a 3 deg start-rotation change (does the network represent the shift at all?).
2. Preprocessing: printed audit of every operation between raw contacts and the attention blocks.
3. Correspondence oracle. Given the true goal rotation and goal joints plus the known contact-to-template
   correspondence, t_goal = mean_valid(c_pos - hand_point(q_goal, R_goal, t=0)). Reported error in mm against the
   stored goal, and against the wrist-frame translation residual target, on the config's training and validation pools.
4. Gradient cancellation, per run: over N training-style batches, the gradient of the training loss w.r.t. the root
   head's last-layer parameters, split into translation rows (0:3) and rotation rows (3:9).
   coherence = ||mean_b g_b|| / mean_b ||g_b||  (1 = all batches agree, ~1/sqrt(N) = directions cancel).
   These are parameter gradients, not output gradients.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from getkite.decoder.get_scratch import GETScratch, ScratchConfig, target_residuals
from getkite.decoder.multifamily import BalancedSampler, draw_start, fixed_start
from getkite.decoder.rotations import euler_xyz_to_matrix
from getkite.decoder.train_scratch import (RawContacts, build_bank, jitter_normals, load_config, load_pool, residual_loss,
                                           start_noise)

PRE_AUDIT = """\
raw contacts -> model inputs (raw_contacts, frame=wrist), from get_scratch.py / train_scratch.py:
  RawContacts: z = [pos * valid, normal * valid, valid]            (no centering, no scaling, no pooling)
  forward:     pos_w = (pos - t0) @ R0, n_w = normal @ R0, both * valid   (rigid transform only)
               token input = [pos_w * pos_scale, n_w, Fourier(pos_w * pos_scale, contact_fourier)] -> contact_in MLP
               one token per contact; invalid contacts masked as attention keys
  hand:        FK with identity base at q1 (= hand in the start wrist frame); joint tokens get positions/rotations,
               surface tokens get centre position + normal + max-pool of an MLP over neighbour offsets (p - centre)
               (that pool is local surface shape; the centre's position is passed separately, not removed)
  blocks:      LayerNorm per token over its feature vector (does not subtract positions across tokens)
  root token:  constant identity pose in the wrist frame; reads contacts/hand only through attention
No step centres contacts on their own centroid or subtracts a per-example mean."""


def capture_inputs(model, bank, r, qc, R, t, z):
    cap = {}
    hooks = []
    for name in ("contact_in", "root_in", "joint_in", "surf_in", "nbr_enc"):
        mod = getattr(model, name, None)
        if mod is not None:
            hooks.append(mod.register_forward_hook(lambda m, i, o, n=name: cap.__setitem__(n, i[0].detach().clone())))
    hooks.append(model.root_head[-1].register_forward_hook(lambda m, i, o: cap.__setitem__("root_feat", i[0].detach().clone())))
    with torch.no_grad():
        out = model(bank, r, qc, R, t, z)
    for h in hooks:
        h.remove()
    return cap, out


def check_input_shift(model, bank, rec, sg, enc, mc, g):
    B = len(rec)
    d = torch.nn.functional.normalize(torch.randn(B, 3, generator=g), dim=-1).to(sg.t1.device) * 0.003
    z = enc(rec.c_pos, rec.c_normal, rec.c_valid)
    a, _ = capture_inputs(model, bank, rec.r, sg.qc1, sg.R1, sg.t1, z)
    b, _ = capture_inputs(model, bank, rec.r, sg.qc1, sg.R1, sg.t1 + d, z)
    v = rec.c_valid
    exp_xyz = -(d[:, None] @ sg.R1).squeeze(1) * mc.pos_scale  # -R1^T d, as a row vector
    dx = b["contact_in"][..., :3] - a["contact_in"][..., :3]
    res = {"contact_xyz_shift_max_err": float((dx - exp_xyz[:, None]).abs()[v].max()),
           "contact_xyz_shift_expected_rms": float(exp_xyz.pow(2).mean().sqrt()),
           "contact_normal_change_max": float((b["contact_in"][..., 3:6] - a["contact_in"][..., 3:6]).abs().max()),
           "contact_padding_change_max": float((b["contact_in"] - a["contact_in"]).abs()[~v].max()) if (~v).any() else 0.0}
    for n in ("root_in", "joint_in", "surf_in", "nbr_enc"):
        if n in a:
            res[f"{n}_change_max"] = float((b[n] - a[n]).abs().max())
    tr_a, tj_a = target_residuals(bank, rec.r, sg.qc1, sg.R1, sg.t1, sg.qcg, sg.Rg, sg.tg, mc)
    tr_b, tj_b = target_residuals(bank, rec.r, sg.qc1, sg.R1, sg.t1 + d, sg.qcg, sg.Rg, sg.tg, mc)
    exp_res = -(d[:, None] @ sg.R1).squeeze(1) / mc.res_xyz_m
    res["target_trans_change_max_err"] = float(((tr_b[:, :3] - tr_a[:, :3]) - exp_res).abs().max())
    res["target_trans_change_rms"] = float(exp_res.pow(2).mean().sqrt())
    res["target_rot_change_max"] = float((tr_b[:, 3:] - tr_a[:, 3:]).abs().max())
    res["target_joint_change_max"] = float((tj_b - tj_a).abs().max())
    # predicted translation response to the shift vs the expected change of the correction (-R1^T d / 5 mm in wrist
    # frame): least-squares slope through the origin (1 = responds exactly as needed, 0 = ignores the shift) and corr
    with torch.no_grad():
        oa = model(bank, rec.r, sg.qc1, sg.R1, sg.t1, z)["res_root"][:, :3]
        ob = model(bank, rec.r, sg.qc1, sg.R1, sg.t1 + d, z)["res_root"][:, :3]
    resp, want = (ob - oa).flatten(), (tr_b[:, :3] - tr_a[:, :3]).flatten()
    res["pred_response_slope"] = float((resp * want).sum() / want.pow(2).sum())
    res["pred_response_corr"] = float(torch.corrcoef(torch.stack([resp, want]))[0, 1])
    # does the trained network represent the shift? root feature movement: 3 mm shift vs 3 deg rotation
    dR = euler_xyz_to_matrix(torch.nn.functional.normalize(torch.randn(B, 3, generator=g), dim=-1).to(d.device) * np.radians(3))
    c, _ = capture_inputs(model, bank, rec.r, sg.qc1, sg.R1 @ dR, sg.t1, z)
    nf = a["root_feat"].norm(dim=-1).mean()
    res["root_feat_norm"] = float(nf)
    res["root_feat_change_3mm_shift"] = float((b["root_feat"] - a["root_feat"]).norm(dim=-1).mean())
    res["root_feat_change_3deg_rot"] = float((c["root_feat"] - a["root_feat"]).norm(dim=-1).mean())
    return res


def oracle(bank, rec, sg, mc):
    r = rec.r
    Rs, ts = bank.fk(r, bank.expand(r, sg.qcg), sg.Rg, torch.zeros_like(sg.tg))
    p, _ = bank.points(r, Rs, ts, rec.c_tpl)
    v = rec.c_valid[..., None].float()
    t_est = ((rec.c_pos - p) * v).sum(1) / v.sum(1)
    tr, _ = target_residuals(bank, r, sg.qc1, sg.R1, sg.t1, sg.qcg, sg.Rg, sg.tg, mc)
    res_est = (((t_est - sg.t1)[:, None] @ sg.R1).squeeze(1) / mc.res_xyz_m) if mc.frame == "wrist" else (t_est - sg.t1) / mc.res_xyz_m
    err_mm = (t_est - sg.tg).norm(dim=-1) * 1000
    return {"n": len(rec), "t_err_mm_mean": float(err_mm.mean()), "t_err_mm_max": float(err_mm.max()),
            "residual_err_max": float((res_est - tr[:, :3]).abs().max()), "target_rms": float(tr[:, :3].pow(2).mean().sqrt())}


def coherence(model, bank, train, cfg, families, fam_index, enc, n_batches=50, seed=9):
    g = torch.Generator().manual_seed(seed)
    sampler = BalancedSampler(train.to("cpu"), bank.family, seed=seed)
    gt, grt, gtw, gtb, grw, grb = [], [], [], [], [], []
    model.train()
    for _ in range(n_batches):
        idx = sampler.sample(cfg["batch_size"]).to(train.r.device)
        rb = train.index(idx)
        sg = draw_start(bank, rb, g, **start_noise(cfg))
        z = enc(rb.c_pos, jitter_normals(rb.c_normal, cfg["normal_noise_rad"], g), rb.c_valid)
        model.zero_grad(set_to_none=True)
        out = model(bank, rb.r, sg.qc1, sg.R1, sg.t1, z)
        tr, tj = target_residuals(bank, rb.r, sg.qc1, sg.R1, sg.t1, sg.qcg, sg.Rg, sg.tg, model.c)
        loss, _, _ = residual_loss(out, tr, tj, bank.coord_valid[rb.r], fam_index[rb.r], len(families))
        loss.backward()
        W, b = model.root_head[-1].weight.grad, model.root_head[-1].bias.grad
        gt.append(torch.cat([W[:3].flatten(), b[:3]]).clone())
        grt.append(torch.cat([W[3:].flatten(), b[3:]]).clone())
        gtw.append(W[:3].flatten().clone()); gtb.append(b[:3].clone())
        grw.append(W[3:].flatten().clone()); grb.append(b[3:].clone())
    model.eval()
    out = {}
    for name, gs in (("trans_rows", gt), ("rot_rows", grt), ("trans_weight", gtw), ("trans_bias", gtb),
                     ("rot_weight", grw), ("rot_bias", grb)):
        G = torch.stack(gs)
        out[name] = {"mean_norm": float(G.norm(dim=1).mean()), "norm_of_mean": float(G.mean(0).norm()),
                     "coherence": float(G.mean(0).norm() / G.norm(dim=1).mean())}
    out["random_direction_coherence"] = float(1 / np.sqrt(n_batches))
    out["n_batches"], out["batch_size"] = n_batches, cfg["batch_size"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--oracle-configs", nargs="*", default=[])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"device": str(dev), "preprocessing_audit": PRE_AUDIT}
    print(PRE_AUDIT)
    cfg = load_config(a.config, [])
    bank = build_bank(cfg, dev)
    families = sorted(set(bank.family))
    fam_index = torch.tensor([families.index(f) for f in bank.family], device=dev)
    train = load_pool(cfg["train"], bank).to(dev)
    val = load_pool(cfg["val"], bank).to(dev) if cfg.get("val") else train  # C2: same goals, fresh starts
    # 3. oracle on this config's pools and any extra configs (e.g. the 10x pool)
    report["oracle"] = {}
    for cpath in [a.config] + a.oracle_configs:
        c2 = load_config(cpath, [])
        b2 = bank if cpath == a.config else build_bank(c2, dev)
        mc2 = ScratchConfig.from_dict(c2["model"])
        for pool in ("train", "val"):
            if not c2.get(pool):
                continue
            rec = load_pool(c2[pool], b2).to(dev)
            sg = fixed_start(b2, rec, 777, **start_noise(c2))
            sg = type(sg)(**{k: getattr(sg, k).to(dev) for k in sg.__dataclass_fields__})
            report["oracle"][f"{Path(cpath).stem}/{pool}"] = oracle(b2, rec, sg, mc2)
    print("oracle:", json.dumps(report["oracle"], indent=1))
    # 1 + 4 per run
    report["runs"] = {}
    for run in a.runs:
        ck = torch.load(Path(run) / "last.pt", map_location=dev, weights_only=False)
        mc = ScratchConfig.from_dict(ck["model_config"])
        assert mc.intent_source == "raw_contacts"
        model = GETScratch(mc).to(dev)
        model.load_state_dict(ck["model"])
        model.eval()
        enc = RawContacts()
        sg = fixed_start(bank, val, cfg["val_start_seed"], **start_noise(cfg))
        sg = type(sg)(**{k: getattr(sg, k).to(dev) for k in sg.__dataclass_fields__})
        g = torch.Generator().manual_seed(21)
        rr = {"step": ck["step"], "input_shift": check_input_shift(model, bank, val, sg, enc, mc, g),
              "grad_coherence_trained": coherence(model, bank, train, cfg, families, fam_index, enc)}
        torch.manual_seed(cfg["seed"])
        init = GETScratch(mc).to(dev)
        init.set_z_stats(model.z_mu, model.z_sd)
        rr["grad_coherence_init"] = coherence(init, bank, train, cfg, families, fam_index, enc)
        report["runs"][Path(run).name] = rr
        print(f"\n== {Path(run).name}\n" + json.dumps(rr, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
