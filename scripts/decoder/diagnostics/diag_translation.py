"""Translation-path diagnostic for scratch decoder runs (wrist-frame C2 tests, 2026-11-15).

    python scripts/decoder/diagnostics/diag_translation.py RUN_DIR [RUN_DIR ...] [--out FILE]

For each run's last.pt, on the run's validation records with its fixed evaluation start pose, per family:
  - loss terms separately: translation (mean of 3 normalised dims), rotation (6), joints (valid coords), and the same
    for the no-movement prediction;
  - wrist translation error (mm) and rotation error (deg);
  - output vs target magnitude of the translation residual (RMS), and their cosine; same for rotation;
  - gradient norms of the training loss (one training-style batch: start redrawn) w.r.t. the root head's last layer,
    split into translation rows (0:3) and rotation rows (3:9), the joint head's last layer, and the root token entering
    the head; also at initialisation (same seed, untrained) for reference;
  - dtypes along the path (block output, final norm, heads, targets, loss), to confirm fp32 when model.amp is False.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from getkite.decoder.get_scratch import GETScratch, ScratchConfig, target_residuals
from getkite.decoder.multifamily import BalancedSampler, draw_start, fixed_start
from getkite.decoder.train_scratch import (FrozenPsi, RawContacts, build_bank, jitter_normals, load_config, load_pool,
                                           pose_metrics, residual_loss, start_noise)


def terms(rr, rj, tr, tj, cv):
    m = cv.float()
    return {"trans": ((rr[:, :3] - tr[:, :3]) ** 2).mean(-1), "rot": ((rr[:, 3:] - tr[:, 3:]) ** 2).mean(-1),
            "joint": (((rj - tj) ** 2) * m).sum(-1) / m.sum(-1).clamp_min(1)}


def grads(model, enc, bank, train, families, fam_index, cfg, seed=5):
    """Gradient norms for one training-style batch (start redrawn, training normal noise)."""
    g = torch.Generator().manual_seed(seed)
    idx = BalancedSampler(train.to("cpu"), bank.family, seed=seed).sample(256).to(train.r.device)
    rb = train.index(idx)
    sg = draw_start(bank, rb, g, **start_noise(cfg))
    z = enc(rb.c_pos, jitter_normals(rb.c_normal, cfg["normal_noise_rad"], g), rb.c_valid)
    feats = {}
    h = model.root_head[-1].register_forward_hook(lambda m, i, o: feats.__setitem__("root_in", i[0]))
    model.train()
    model.zero_grad(set_to_none=True)
    out = model(bank, rb.r, sg.qc1, sg.R1, sg.t1, z)
    feats["root_in"].retain_grad()
    tr, tj = target_residuals(bank, rb.r, sg.qc1, sg.R1, sg.t1, sg.qcg, sg.Rg, sg.tg, model.c)
    loss, _, _ = residual_loss(out, tr, tj, bank.coord_valid[rb.r], fam_index[rb.r], len(families))
    loss.backward()
    h.remove()
    W = model.root_head[-1].weight.grad
    b = model.root_head[-1].bias.grad
    res = {"loss": float(loss), "dtype_loss": str(loss.dtype), "dtype_target": str(tr.dtype),
           "dtype_res_root": str(out["res_root"].dtype),
           "grad_root_trans_rows": float(torch.cat([W[:3].flatten(), b[:3]]).norm()),
           "grad_root_trans_weight": float(W[:3].norm()), "grad_root_trans_bias": float(b[:3].norm()),
           "grad_root_rot_weight": float(W[3:].norm()), "grad_root_rot_bias": float(b[3:].norm()),
           "grad_root_rot_rows": float(torch.cat([W[3:].flatten(), b[3:]]).norm()),
           "grad_root_rot_rows_per_row": float(torch.cat([W[3:].flatten(), b[3:]]).norm() / np.sqrt(2)),
           "grad_joint_head_last": float(torch.cat([p.grad.flatten() for p in model.joint_head[-1].parameters()]).norm()),
           "grad_root_token_into_head": float(feats["root_in"].grad.norm(dim=-1).mean()) if feats["root_in"].grad is not None else None,
           "out_trans_rms": float(out["res_root"][:, :3].pow(2).mean().sqrt()),
           "target_trans_rms": float(tr[:, :3].pow(2).mean().sqrt())}
    model.eval()
    return res


def dtype_trace(model, bank, rec, sg, z):
    seen = {}
    hooks = [model.blocks[0].register_forward_hook(lambda m, i, o: seen.__setitem__("block0_out", str(o.dtype))),
             model.blocks[-1].register_forward_hook(lambda m, i, o: seen.__setitem__("blockN_out", str(o.dtype))),
             model.ln_out.register_forward_hook(lambda m, i, o: seen.__setitem__("ln_out", str(o.dtype))),
             model.root_head.register_forward_hook(lambda m, i, o: seen.__setitem__("root_head", str(o.dtype)))]
    with torch.no_grad():
        out = model(bank, rec.r[:64], sg.qc1[:64], sg.R1[:64], sg.t1[:64], z[:64])
    for h in hooks:
        h.remove()
    seen["decoded_t"] = str(out["t"].dtype)
    seen["inputs_t0_R0_contacts"] = f"{sg.t1.dtype}/{sg.R1.dtype}/{z.dtype}"
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    allres = {}
    for run in a.runs:
        ck = torch.load(Path(run) / "last.pt", map_location=dev, weights_only=False)
        cfg = load_config(str(Path(run) / "config.json"), [])
        mc = ScratchConfig.from_dict(ck["model_config"])
        bank = build_bank(cfg, dev)
        families = sorted(set(bank.family))
        fam_index = torch.tensor([families.index(f) for f in bank.family], device=dev)
        fam = np.array([str(x) for x in bank.family], dtype=object)
        train = load_pool(cfg["train"], bank).to(dev)
        val = load_pool(cfg["val"], bank).to(dev) if cfg.get("val") else train
        enc = RawContacts() if mc.intent_source == "raw_contacts" else FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev)
        sg = fixed_start(bank, val, cfg["val_start_seed"], **start_noise(cfg))
        sg = type(sg)(**{k: getattr(sg, k).to(dev) for k in sg.__dataclass_fields__})
        model = GETScratch(mc).to(dev)
        model.load_state_dict(ck["model"])
        model.eval()
        z = enc(val.c_pos, val.c_normal, val.c_valid)
        with torch.no_grad():
            out = model(bank, val.r, sg.qc1, sg.R1, sg.t1, z)
        tr, tj = target_residuals(bank, val.r, sg.qc1, sg.R1, sg.t1, sg.qcg, sg.Rg, sg.tg, mc)
        cv = bank.coord_valid[val.r]
        T = terms(out["res_root"], out["res_coord"], tr, tj, cv)
        N = terms(torch.zeros_like(tr), torch.zeros_like(tj), tr, tj, cv)
        pm = pose_metrics(bank, val, val.r, out["qc"], out["t"], out["R"], sg, cfg)
        res = {"step": ck["step"], "model_config": mc.to_dict(), "per_family": {}}
        for f in families:
            sel = torch.as_tensor(fam[val.r.cpu().numpy()] == f, device=dev)
            pt, tt = out["res_root"][sel, :3], tr[sel, :3]
            prr, trr = out["res_root"][sel, 3:], tr[sel, 3:]
            res["per_family"][f] = {
                **{f"loss_{k}": round(float(v[sel].mean()), 4) for k, v in T.items()},
                **{f"noop_loss_{k}": round(float(v[sel].mean()), 4) for k, v in N.items()},
                "wrist_mm": round(float(pm["wrist_mm"][sel].mean()), 3), "wrist_deg": round(float(pm["wrist_deg"][sel].mean()), 3),
                "contact_mm": round(float(pm["contact_mm"][sel].mean()), 3), "success": round(float(pm["success"][sel].mean()), 3),
                "trans_out_rms": round(float(pt.pow(2).mean().sqrt()), 4), "trans_target_rms": round(float(tt.pow(2).mean().sqrt()), 4),
                "trans_cos": round(float(torch.nn.functional.cosine_similarity(pt, tt, dim=-1).mean()), 4),
                "trans_corr_per_axis": [round(float(torch.corrcoef(torch.stack([pt[:, k], tt[:, k]]))[0, 1]), 4) for k in range(3)],
                "trans_slope": round(float((pt * tt).sum() / tt.pow(2).sum()), 4),
                "trans_err_rel_noop": round(float(((pt - tt).norm(dim=-1).mean()) / tt.norm(dim=-1).mean()), 4),
                "rot_out_rms": round(float(prr.pow(2).mean().sqrt()), 4), "rot_target_rms": round(float(trr.pow(2).mean().sqrt()), 4),
                "rot_cos": round(float(torch.nn.functional.cosine_similarity(prr, trr, dim=-1).mean()), 4)}
        res["grad_trained"] = grads(model, enc, bank, train, families, fam_index, cfg)
        torch.manual_seed(cfg["seed"])
        init = GETScratch(mc).to(dev)
        res["grad_init"] = grads(init, enc, bank, train, families, fam_index, cfg)
        res["dtypes"] = dtype_trace(model, bank, val, sg, z)
        res["root_head_last_weight_norm"] = {"trans_rows": float(model.root_head[-1].weight[:3].norm()),
                                             "rot_rows": float(model.root_head[-1].weight[3:].norm())}
        allres[Path(run).name] = res
        print(f"\n== {Path(run).name} (step {ck['step']}, amp={mc.amp}, contact_fourier={mc.contact_fourier}, frame={mc.frame})")
        for f, r in res["per_family"].items():
            print(f"  {f}: " + " ".join(f"{k}={v}" for k, v in r.items()))
        for k in ("grad_trained", "grad_init", "dtypes", "root_head_last_weight_norm"):
            print(f"  {k}: {res[k]}")
    if a.out:
        Path(a.out).write_text(json.dumps(allres, indent=1))


if __name__ == "__main__":
    main()
