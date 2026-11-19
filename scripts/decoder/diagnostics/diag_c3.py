"""Step 1 of the C3 generalisation diagnosis: loss-term check and train/val preprocessing parity.

    python scripts/decoder/diagnostics/diag_c3.py --out $GETKITE_RUNS/c3_diag_step1.json

For each family and record set (training goals with a training-style start draw + normal noise, training goals with
the fixed evaluation start, validation goals with the fixed evaluation start) it reports, separately for wrist
translation, wrist rotation and joints:
  - the supervised loss of q_init (zero residual), of q_goal (exact target residual), of the half-way prediction
    (0.5 x target) and of both trained 50k models (last.pt),
  - contact error of each of those poses (goal must be ~0 mm: data / FK / goal-mix consistency),
  - decode(target_residuals(goal)) == goal,
  - input statistics used for the parity check (contacts per example, contact spread, wrist position, residual
    target mean/std per component, active-joint fraction).
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

from getkite.decoder.get_scratch import GETScratch, ScratchConfig, target_residuals
from getkite.decoder.multifamily import draw_start, fixed_start
from getkite.decoder.rotations import geodesic_angle
from getkite.decoder.train_scratch import (FrozenPsi, RawContacts, build_bank, jitter_normals, load_config, load_pool,
                                      pose_metrics)

from getkite.decoder import paths  # noqa: E402
RUNS_DIR = paths.get("GETKITE_RUNS")
REPO = str(paths.REPO)
RUNS = {"psi": f"{RUNS_DIR}/scratch_c3_long50k", "raw": f"{RUNS_DIR}/scratch_c3_raw50k"}


def terms(res_root, res_coord, tr_root, tr_j, cv):
    m = cv.float()
    return {"trans": ((res_root[:, :3] - tr_root[:, :3]) ** 2).mean(-1),
            "rot": ((res_root[:, 3:] - tr_root[:, 3:]) ** 2).mean(-1),
            "joint": (((res_coord - tr_j) ** 2) * m).sum(-1) / m.sum(-1).clamp_min(1),
            "loss": ((res_root - tr_root) ** 2).mean(-1) + (((res_coord - tr_j) ** 2) * m).sum(-1) / m.sum(-1).clamp_min(1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=f"{REPO}/configs/decoder/scratch/c3_raw50k.json")
    ap.add_argument("--out", default=f"{RUNS_DIR}/c3_diag_step1.json")
    a = ap.parse_args()
    cfg = load_config(a.config, [])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    bank = build_bank(cfg, dev)
    fam = np.array([str(x) for x in bank.family], dtype=object)
    train = load_pool(cfg["train"], bank).to(dev)
    val = load_pool(cfg["val"], bank).to(dev)
    held = load_pool(cfg["extra_eval"]["leap_heldout"], bank).to(dev)
    mvs = {}
    enc = {"psi": FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev), "raw": RawContacts()}
    for k, run in RUNS.items():
        ck = torch.load(f"{run}/last.pt", map_location=dev, weights_only=False)
        m = GETScratch(ScratchConfig.from_dict(ck["model_config"])).to(dev)
        m.load_state_dict(ck["model"])
        m.amp = False
        mvs[k] = (m.eval(), ck["step"])

    g_train = torch.Generator().manual_seed(4242)
    sets = {
        "train_goals/train_draw": (train, draw_start(bank, train, g_train), cfg["normal_noise_rad"]),
        "train_goals/eval_start": (train, fixed_start(bank, train, cfg["val_start_seed"]), 0.0),
        "val_goals/eval_start": (val, fixed_start(bank, val, cfg["val_start_seed"]), 0.0),
        "heldout_leap/eval_start": (held, fixed_start(bank, held, cfg["val_start_seed"]), 0.0),
    }
    out = {"_model_steps": {k: v[1] for k, v in mvs.items()}}
    ngen = torch.Generator().manual_seed(77)
    for sname, (rec, sg, noise) in sets.items():
        sg = type(sg)(**{f: getattr(sg, f).to(dev) for f in sg.__dataclass_fields__})
        r = rec.r
        cv = bank.coord_valid[r]
        tr_root, tr_j = target_residuals(bank, r, sg.qc1, sg.R1, sg.t1, sg.qcg, sg.Rg, sg.tg, ScratchConfig())
        zero_r, zero_j = torch.zeros_like(tr_root), torch.zeros_like(tr_j)
        preds = {"q_init": (zero_r, zero_j), "q_goal": (tr_root, tr_j), "halfway": (0.5 * tr_root, 0.5 * tr_j)}
        nrm = jitter_normals(rec.c_normal, noise, ngen) if noise > 0 else rec.c_normal
        with torch.no_grad():
            for k, (m, _) in mvs.items():
                rr, rj = [], []
                for s in range(0, len(rec), 1024):
                    i = slice(s, s + 1024)
                    z = enc[k](rec.c_pos[i], nrm[i], rec.c_valid[i])
                    o = m(bank, r[i], sg.qc1[i], sg.R1[i], sg.t1[i], z)
                    rr.append(o["res_root"]); rj.append(o["res_coord"])
                preds[f"model_{k}"] = (torch.cat(rr), torch.cat(rj))
        dec_m = GETScratch(ScratchConfig()).to(dev)
        for pname, (pr, pj) in preds.items():
            d = dec_m.decode(bank, r, sg.qc1, sg.R1, sg.t1, pr, pj)
            T = terms(pr, pj, tr_root, tr_j, cv)
            pm = pose_metrics(bank, rec, r, d["qc"], d["t"], d["R"], sg, {"success_pos_mm": 10, "success_normal_deg": 15})
            for f in sorted(set(fam)):
                sel = torch.as_tensor(fam[r.cpu().numpy()] == f, device=dev)
                if not sel.any():
                    continue
                row = out.setdefault(sname, {}).setdefault(f, {}).setdefault(pname, {})
                for kk, v in list(T.items()) + [(x, pm[x]) for x in ("contact_mm", "success", "wrist_mm", "wrist_deg")]:
                    row[kk] = round(float(v[sel].mean()), 4)
                if pname == "q_goal":
                    row["decode_err_t_mm"] = round(float(((d["t"] - sg.tg).norm(dim=-1)[sel]).max() * 1000), 6)
                    row["decode_err_R_deg"] = round(float(torch.rad2deg(geodesic_angle(d["R"], sg.Rg))[sel].max()), 6)
                    row["decode_err_q"] = round(float(((d["qc"] - sg.qcg).abs() * cv)[sel].max()), 7)
                    row["contact_mm_max"] = round(float(pm["contact_mm"][sel].max()), 4)
        # parity statistics
        for f in sorted(set(fam)):
            sel = torch.as_tensor(fam[r.cpu().numpy()] == f, device=dev)
            if not sel.any():
                continue
            v = rec.c_valid[sel].float()
            cp = rec.c_pos[sel]
            cen = (cp * v[..., None]).sum(1) / v.sum(1, keepdim=True)
            spread = (((cp - cen[:, None]).norm(dim=-1) * v).sum(1) / v.sum(1)).mean()
            out[sname][f]["_stats"] = {
                "n": int(sel.sum()), "k_mean": round(float(v.sum(1).mean()), 3),
                "contact_spread_mm": round(float(spread * 1000), 2),
                "contact_pos_std_m": [round(float(x), 4) for x in (cp * v[..., None]).reshape(-1, 3)[v.reshape(-1) > 0].std(0)],
                "wrist_pos_std_m": [round(float(x), 4) for x in sg.tg[sel].std(0)],
                "contact_to_wrist_mm": round(float((((cp - sg.tg[sel][:, None]).norm(dim=-1) * v).sum(1) / v.sum(1)).mean() * 1000), 1),
                "active_frac": round(float((rec.active[sel] & cv[sel]).float().sum() / cv[sel].float().sum()), 3),
                "target_mean": {"trans": round(float(tr_root[sel, :3].mean()), 3), "rot": round(float(tr_root[sel, 3:].mean()), 3),
                                "joint": round(float((tr_j[sel] * cv[sel]).sum() / cv[sel].sum()), 3)},
                "target_rms": {"trans": round(float(tr_root[sel, :3].pow(2).mean().sqrt()), 3),
                               "rot": round(float(tr_root[sel, 3:].pow(2).mean().sqrt()), 3),
                               "joint": round(float(((tr_j[sel] ** 2 * cv[sel]).sum() / cv[sel].sum()).sqrt()), 3)},
                "normal_noise_rad": noise}
    # leakage: identical goal records between training and validation
    def keys(rec):
        return {(int(a), *np.round(b, 5).tolist()) for a, b in zip(rec.r.cpu().numpy(), rec.t2.cpu().numpy())}
    out["_overlap_train_val_goals"] = len(keys(train) & keys(val))
    json.dump(out, open(a.out, "w"), indent=1)
    for sname in sets:
        print(f"\n== {sname}")
        for f, rows in out[sname].items():
            print(f"  {f}  stats {rows['_stats']}")
            for p, row in rows.items():
                if p != "_stats":
                    print(f"    {p:9s} " + " ".join(f"{k}={v}" for k, v in row.items()))
    print("overlap train/val goals:", out["_overlap_train_val_goals"])


if __name__ == "__main__":
    main()
