"""Readout for the 105-hand held-out LEAP evaluation (transfer_benchmark_v1_leapval105; evaluation only).

Per seed, paired over the same records: arm A - arm B contact error (corr_mm), wrist mm / deg, and a hand-level view
(per-hand mean differences; sign test over hands; bootstrap over hands, so the CI reflects body-to-body variation, not only
record noise). Groups: number of fingers (topology code letters with a non-zero digit), and the canonical-frame angle of each
hand to the mean LEAP training frame (A's 48 hands). No seed is chosen; nothing here feeds training or selection.

    python scripts/decoder/benchmark/leapval105_summary.py --a cf2 --b d1long
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import torch
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())

sys.path.insert(0, "src")
from getkite.decoder.robots import canonical_frame, leap_desc  # noqa: E402

R = os.environ["GETKITE_RUNS"]; DSET = os.environ.get("GETKITE_DATASET", str(_REPO.parent / "get-ur-kite-dataset" / "data"))
ap = argparse.ArgumentParser(); ap.add_argument("--a", default="cf2"); ap.add_argument("--b", default="d1long")
ap.add_argument("--pe", default=f"{R}/transfer_benchmark_v1_leapval105/per_example.pt"); args = ap.parse_args()
X = torch.load(args.pe, weights_only=False)["leap_heldout"]
names = np.array(X["names"]); P = X["per"]; hands = sorted(set(names))
ORIG = {"I0-M2-R2-T3_g0", "I0-M4-R3-T2_g0", "I2-M3-R2-T2_g0", "I2-M3-R3-T2_g4"}
rng = np.random.default_rng(0)

# canonical-frame angle to the mean training LEAP frame (deployment-time input: description + template only)
cache = f"{R}/transfer_benchmark_v1_leapval105/leap_frame_angles.json"  # same 105 hands for any per_example.pt
if os.path.exists(cache):
    angle = json.load(open(cache))
else:
    fr = json.load(open(f"{R}/cf_bv_long_hf/canonical_frames.json"))
    A = json.load(open("configs/decoder/scratch/cf2_long_hf.json"))
    a_tr = [os.path.basename(t["npz"])[:-4] for t in A["train"] if t["kind"] == "leap" and "/prim/" not in t["npz"]]
    U, _, Vt = np.linalg.svd(sum(np.array(fr[h]["Q"]) for h in a_tr)); Mean = U @ Vt
    angle = {}
    for h in hands:
        Q = np.array(fr[h]["Q"]) if h in fr else canonical_frame(leap_desc(f"{DSET}/hands/{h}"))[0]
        angle[h] = float(np.degrees(np.arccos(np.clip((np.trace(Mean.T @ Q) - 1) / 2, -1, 1))))
    json.dump(angle, open(cache, "w"), indent=1)


def nfing(h):
    return sum(int(d) > 0 for d in re.findall(r"[IMRT](\d)", h.split("_g")[0]))


print(f"held-out LEAP: {len(hands)} hands, {len(names)} records; {args.a} - {args.b} (negative = {args.a} better)")
print("no movement corr_mm %.3f" % np.mean(P["no_movement"]["corr_mm"]))
gap = {}
for s in range(3):
    a, b = P[f"{args.a}_s{s}"], P[f"{args.b}_s{s}"]
    d = {m: np.asarray(a[m]) - np.asarray(b[m]) for m in ("corr_mm", "wrist_mm", "wrist_deg", "compat_success")}
    ph = np.array([d["corr_mm"][names == h].mean() for h in hands]); gap[s] = dict(zip(hands, ph))
    boot = [ph[rng.integers(0, len(ph), len(ph))].mean() for _ in range(4000)]
    o = np.array([h in ORIG for h in hands])
    print(f"\nseed {s}: {args.a} corr {np.mean(a['corr_mm']):.3f}  {args.b} corr {np.mean(b['corr_mm']):.3f}")
    print(f"  corr diff {d['corr_mm'].mean():+.3f}  hand-bootstrap 95% CI [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]"
          f"  hands {args.a} lower {int((ph < 0).sum())}/{len(ph)}")
    print(f"  wrist_mm {d['wrist_mm'].mean():+.3f}  wrist_deg {d['wrist_deg'].mean():+.3f}  compat_success {d['compat_success'].mean():+.3f}")
    print(f"  original 4 hands {ph[o].mean():+.3f}   other {(~o).sum()} hands {ph[~o].mean():+.3f}")
    for k in sorted({nfing(h) for h in hands}):
        m = np.array([nfing(h) == k for h in hands])
        print(f"  {k} fingers: {m.sum():3d} hands  corr diff {ph[m].mean():+.3f}  {args.a} lower {int((ph[m] < 0).sum())}/{m.sum()}")
    ang = np.array([angle[h] for h in hands])
    print(f"  corr diff vs frame angle: r={np.corrcoef(ang, ph)[0, 1]:+.2f}  (angle median {np.median(ang):.1f}, max {ang.max():.1f};"
          f" >15deg: {int((ang > 15).sum())} hands, diff {ph[ang > 15].mean() if (ang > 15).any() else float('nan'):+.3f})")
consist = np.array([[gap[s][h] for s in range(3)] for h in hands])
print(f"\nhands where {args.a} is higher in all 3 seeds: {int((consist > 0).all(1).sum())}/{len(hands)}; lower in all 3: {int((consist < 0).all(1).sum())}")
worst = np.argsort(-consist.mean(1))[:8]
print("largest mean gaps:", [(hands[i], round(float(consist[i].mean()), 2), round(angle[hands[i]], 1), nfing(hands[i])) for i in worst])
