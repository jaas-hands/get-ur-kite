"""Wrist-error reference-point check for CF-O vs CF2 (evaluation only; user-approved 2026-11-27; docs/cf3f_proposal.md).

CF-O failed its gate partly on wrist_mm (Wuji / Shadow +0.5..+1.45 mm vs CF2). wrist_mm is measured at each hand's BASE ORIGIN, so a
model whose rotation error is unchanged can score differently when the reference point moves (lever arm). Here the same
predictions are re-scored with the wrist position error measured at a common point on the hand:
  base  = base origin (the standard wrist_mm; reproduces the benchmark),
  O1    = palm-box centre (CF-O's own landmark, docs/canonical_origin_proposal.md),
  O2    = mean of the palm template points (landmark-free).
error_p = |(t + R p) - (t_goal + R_goal p)| in mm, p in the native base frame. Contact error (corr_mm) does not depend on p.

Predeclared reading (fixed before running): CF-O's wrist deficit is a measurement effect if, at BOTH O1 and O2, the paired
CF-O - CF2 wrist difference is not significantly positive on Wuji and on Shadow in >= 2 of 3 seeds, while at base it is.
Otherwise: not a measurement effect. CF-3e is reported for reference.

    PYTHONPATH=src python scripts/decoder/benchmark/origin_measure_eval.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from getkite.decoder import paths
from getkite.decoder.multifamily import fixed_start
from getkite.decoder.robots import canonical_bank, canonical_origin, palm_points
from getkite.decoder.train_scratch import FrozenPsi, build_bank, load_config, load_pool, start_noise

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "diagnostics"))
from crossbody_heldout_eval import HD, WUJI, sg_index, sg_to  # noqa: E402
from transfer_benchmark import SHADOW, SHADOW_AD, load_system  # noqa: E402
from wuji_per_example import paired_boot  # noqa: E402
from external_hands_eval import EXT, HANDS, spec  # noqa: E402

R = os.environ["GETKITE_RUNS"]
RUNS = {"cf2": "cf2_long_hf", "cfo": "cfo_long_hf", "cf3e": "frozen_cf3e_2026-11-27/cf3e_long_hf"}
POINTS = ("base", "O1", "O2")


def main():
    dev = torch.device("cuda")
    out = Path(f"{R}/origin_measure_v1"); out.mkdir(parents=True, exist_ok=True)
    B = json.loads(Path("configs/decoder/benchmark/transfer_v1_leapval105.json").read_text())
    cfg = load_config("configs/decoder/scratch/c3_psi50k_10x_wrist_leapval105.json", []); cfg["surface_sampling"] = B["surface_sampling"]
    torch.manual_seed(0); noise = start_noise(cfg)
    ext = [spec(h) for h in HANDS]
    bank = build_bank({**cfg, "robots": cfg["robots"] + [WUJI, SHADOW] + ext}, dev)
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    _, Q, _ = canonical_bank(bank)
    Qn = Q.double().cpu().numpy()
    P = {"base": np.zeros((len(bank.names), 3)),
         "O1": np.stack([canonical_origin(d, q) for d, q in zip(bank.descs, Qn)]),
         "O2": np.stack([palm_points(d).mean(0) for d in bank.descs])}
    Pt = {k: torch.as_tensor(v, dtype=torch.float32, device=dev) for k, v in P.items()}
    systems = {(a, s): load_system(f"{a}_s{s}", {"kind": "j1", "ckpt": f"{R}/{RUNS[a]}{'' if s == 0 else f'_s{s}'}/best.pt"}, dev,
                                   B["surface_sampling"])[0] for a in RUNS for s in range(3)}
    pools = {"wuji": [{"kind": "kite", "robot": "wujihand_right", "npz": f"{HD}/wujihand_right_eval.npz"}],
             "shadow": [{"kind": "kite", "robot": "shadow_hand_right", "npz": str(SHADOW_AD / "shadow_eval.npz")}],
             **{h: [{"kind": "kite", "robot": f"{h}_right", "npz": str(EXT / f"{h}_adapter" / f"{h}_eval.npz")}] for h in HANDS}}
    res = {}
    for pn, sp in pools.items():
        rec = load_pool(sp, bank).to(dev)
        sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
        chunks = [slice(b, min(b + 256, len(rec))) for b in range(0, len(rec), 256)]
        with torch.no_grad():
            z = torch.cat([psi(rec.c_pos[i], rec.c_normal[i], rec.c_valid[i]) for i in chunks])
        err = {}
        for key, S in systems.items():
            e = {p: [] for p in POINTS}
            with torch.no_grad():
                for i in chunks:
                    g = sg_index(sg, i)
                    o = S.predict(bank, rec.r[i], g, z[i])
                    for p in POINTS:
                        pp = Pt[p][rec.r[i]]
                        a = o["t"] + (o["R"] @ pp[..., None]).squeeze(-1)
                        b = g.tg + (g.Rg @ pp[..., None]).squeeze(-1)
                        e[p].append(((a - b).norm(dim=-1) * 1000).cpu().numpy())
            err[key] = {p: np.concatenate(v) for p, v in e.items()}
        rid = int(rec.r[0])
        D = {"n": len(rec), "point_offset_cm": {p: round(float(np.linalg.norm(P[p][rid])) * 100, 2) for p in POINTS},
             "means": {f"{a}_s{s}": {p: round(float(err[(a, s)][p].mean()), 4) for p in POINTS} for a, s in systems},
             "cfo_minus_cf2": {p: [paired_boot(err[("cfo", s)][p], err[("cf2", s)][p]) for s in range(3)] for p in POINTS},
             "cf3e_minus_cf2": {p: [paired_boot(err[("cf3e", s)][p], err[("cf2", s)][p]) for s in range(3)] for p in POINTS}}
        res[pn] = D
        print(pn, {p: [round(x["mean_diff"], 3) for x in D["cfo_minus_cf2"][p]] for p in POINTS}, flush=True)

    def sig_pos(pn, p):
        return sum(x["ci95"][0] > 0 for x in res[pn]["cfo_minus_cf2"][p])
    base_def = all(sig_pos(pn, "base") >= 2 for pn in ("wuji", "shadow"))
    gone = all(sig_pos(pn, p) < 2 for pn in ("wuji", "shadow") for p in ("O1", "O2"))
    res["predeclared"] = {"base_deficit_reproduced": bool(base_def), "deficit_gone_at_O1_and_O2": bool(gone),
                          "measurement_effect": bool(base_def and gone)}
    (out / "origin_measure.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["predeclared"]))
    print("wrote", out / "origin_measure.json")


if __name__ == "__main__":
    main()
