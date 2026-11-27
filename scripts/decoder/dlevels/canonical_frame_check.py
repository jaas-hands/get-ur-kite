"""CF step 1: geometric check of the canonical palm frame (docs/cf_canonical_palm_frame_proposal.md; CPU, no training).

Generated bodies have a known native convention (x = palmar normal, y = across, z = distal), so the angle between their
canonical frame and the native frame checks the rule directly (generator metadata used for checking only). Named hands:
Q, the conditioning of the curl axis, and the native axis closest to each canonical axis. Also verifies exact invariance
to relabelling (canonical frame of a relabelled description = the same physical frame).

    PYTHONPATH=src python scripts/decoder/dlevels/canonical_frame_check.py --out configs/decoder/dlevels/cf_frame_check.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from getkite.decoder import paths
from getkite.decoder.robots import canonical_frame, leap_desc, relabel_base
from getkite.decoder.train_scratch import build_bank, load_config

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "diagnostics")); sys.path.insert(0, str(HERE.parent / "benchmark"))
from crossbody_heldout_eval import WUJI  # noqa: E402
from transfer_benchmark import SHADOW  # noqa: E402

REPO = Path(paths.REPO)


def ang(Q):
    return float(np.degrees(Rotation.from_matrix(Q).magnitude()))


def nearest(Q):
    """For each canonical axis (n, a, d), the closest signed native axis and its angle."""
    out = {}
    for k, lab in enumerate("nad"):
        v = Q[:, k]; i = int(np.argmax(np.abs(v)))
        out[lab] = f"{'+' if v[i] > 0 else '-'}{'xyz'[i]} ({np.degrees(np.arccos(min(1.0, abs(v[i])))):.1f} deg)"
    return out


def prim_one(p):
    import torch
    torch.set_num_threads(1)
    Q, info = canonical_frame(leap_desc(p))
    return Path(p).name, ang(Q), info["curl_perp_norm_m"], nearest(Q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/decoder/scratch/d1_prim50_hf.json")
    ap.add_argument("--prim_root", default="${GETKITE_DATASET}/prim/hands"); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    cfg = load_config(str(REPO / a.config), [])
    bank = build_bank({**cfg, "robots": cfg["robots"] + [WUJI, SHADOW]}, "cpu")
    R = {"prim": {}, "named": {}, "invariance": {}}
    # generated bodies (all 900 of draw A)
    roots = sorted(p for p in Path(paths.expand(a.prim_root)).iterdir() if p.is_dir())
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        res = pool.map(prim_one, [str(p) for p in roots], chunksize=4)
    angs = []
    for name, an, cn, nr in res:
        angs.append(an)
        R["prim"][name] = {"angle_to_native_deg": round(an, 2), "curl_perp_norm_mm": round(1000 * cn, 2), "nearest": nr}
    angs = np.array(angs)
    R["prim_summary"] = {"n": len(angs), "frac_within_30deg": float((angs <= 30).mean()), "median_deg": float(np.median(angs)),
                         "p90_deg": float(np.percentile(angs, 90)), "max_deg": float(angs.max())}
    print("prim", json.dumps(R["prim_summary"]), flush=True)
    # named hands: one per family (LEAP: every variant in the bank)
    seen = set()
    for d in bank.descs:
        if d.family == "prim" or (d.family in seen and d.family != "leap"):
            continue
        seen.add(d.family)
        Q, info = canonical_frame(d)
        R["named"][d.name] = {"family": d.family, "Q": np.round(Q, 4).tolist(), "angle_to_native_deg": round(ang(Q), 2),
                              "nearest": nearest(Q), **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in info.items()}}
        print(d.family, d.name, round(ang(Q), 1), nearest(Q), "curl_perp_mm", round(1000 * info["curl_perp_norm_m"], 1), flush=True)
    # exact invariance to relabelling (a few robots x a few random rotations)
    for d in [bank.descs[0], bank.descs[-1], bank.descs[-2]]:
        Q0, _ = canonical_frame(d)
        errs = []
        for s in range(3):
            Rr = Rotation.random(random_state=s).as_matrix()
            Q1, _ = canonical_frame(relabel_base(d, Rr))
            errs.append(float(np.abs(Rr @ Q1 - Q0).max()))  # physical frame: native axes of the relabelled robot = Rr
        R["invariance"][d.name] = max(errs)
    print("invariance", R["invariance"], flush=True)
    Path(a.out).write_text(json.dumps(R, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
