"""Per-hand frame choice with K = 64 calibration records (evaluation only, CPU; frozen CF-3e; user-approved 2026-11-28).
Why: check 3 of frame_robustness_eval.py FAILED at K = 32 (palm-normal picks on the LEAP flip hands changed with the calibration
set; finger-axis picks were stable). Question: does a larger calibration set (64 sample requests per new hand) make the picks stable?
Recomputed from frame_robustness_v1/scores_cache.pt (per-record metrics of every candidate frame; no new model runs).
Rules fixed before running (this file is committed before it is run).

CHECK 3-K64 (primary): C6 (rule E; 90/180/270 deg about d; +-90 deg about n), margin 0.25, CF-3e s0/s1/s2, on the 9 flip hands of
  the 360 pool and MetaHand: default = first 64 records of each hand; 5 random calibration sets of 64 records per hand (rng seed
  2000 + i, drawn from all of the hand's records). PASS if, in every seed, at least 80 % of these 10 hands get the same pick in
  >= 4 of the 5 random sets as with the first-64 default. (Same rule as check 3 with 64 in place of 32.)
Reported (descriptive, not part of the pass rule):
  * the same statistic at K = 32 (reproduces check 3) and for option (a), finger-axis candidates only (E, d90, d180, d270), at K = 32
    and K = 64;
  * records per hand (K = 64 leaves fewer scored records);
  * corr_mm and wrist_mm of per-hand choice minus rule E on the 9 flip hands and on the other 351 hands, for C6 and option (a), at
    K = 32 and K = 64, all scored on the SAME records (each hand's records from index 64 on), paired bootstrap, seeds not pooled;
  * the 105 pool and source-validation LEAP, same comparison on records from index 64 on.

    PYTHONPATH=src python scripts/decoder/benchmark/frame_calib64_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "diagnostics"))
from frame_robustness_eval import C6, choose, first_k, MARGIN  # noqa: E402
from frame_select_eval import FLIP_DEG  # noqa: E402
from wuji_per_example import paired_boot  # noqa: E402
from getkite.decoder import paths  # noqa: E402

FINGER = [c for c in C6 if c[0] in ("E", "d90", "d180", "d270")]
K_MAIN = 64


def from_k(names, k):
    """records from index k on, per hand (the common scored set)."""
    return ~first_k(names, k)


def random_calib(names, k, seed):
    g = np.random.default_rng(seed)
    calib = np.zeros(len(names), bool)
    for h in np.unique(names):
        ix = np.flatnonzero(names == h)
        calib[g.choice(ix, min(k, len(ix)), replace=False)] = True
    return calib


def stability(pools, flips, cands, k, seed0, s):
    agree = {}
    for pn, hands in (("leap360", flips), ("metahand", None)):
        P = pools[pn]; n = P["names"]
        default, _ = choose(P["per"], ("cf3e", s), cands, n, first_k(n, k))
        rnd = [choose(P["per"], ("cf3e", s), cands, n, random_calib(n, k, seed0 + i))[0] for i in range(5)]
        for h in (hands if hands is not None else sorted(set(n))):
            agree[h] = {"default": default[h], "random_picks": [r[h] for r in rnd],
                        "random_same": int(sum(r[h] == default[h] for r in rnd)), "n_records": int((n == h).sum())}
    frac = sum(v["random_same"] >= 4 for v in agree.values()) / len(agree)
    return {"hands": agree, "stable_frac": round(frac, 3), "pass": bool(frac >= 0.8)}


def gains(pools, flips, cands, k, s):
    out = {}
    for pn in ("leap360", "leap105", "val", "metahand"):
        P = pools[pn]; n = P["names"]
        _, sel = choose(P["per"], ("cf3e", s), cands, n, first_k(n, k))
        base = P["per"][("cf3e", s)]["E"]
        common = from_k(n, K_MAIN)
        if pn == "leap360":
            groups = {"flip": np.isin(n, flips), "rest": ~np.isin(n, flips)}
        elif pn == "val":
            groups = {"source_leap": P["fam"] == "leap"}
        else:
            groups = {"all": np.ones(len(n), bool)}
        for g, m in groups.items():
            mm = m & common
            out[f"{pn}:{g}"] = {"n": int(mm.sum()), **{q: paired_boot(sel[q][mm], base[q][mm]) for q in ("corr_mm", "wrist_mm")}}
    return out


def main():
    src = Path(paths.expand("${GETKITE_RUNS}/frame_robustness_v1"))
    out = Path(paths.expand("${GETKITE_RUNS}/frame_calib64_v1")); out.mkdir(parents=True, exist_ok=True)
    C = torch.load(src / "scores_cache.pt", weights_only=False, map_location="cpu")
    pools = C["pools"]
    ang = json.load(open(paths.expand("${GETKITE_RUNS}/transfer_benchmark_v1_leap360/leap_frame_angles.json")))
    flips = sorted(h for h, a in ang.items() if a > FLIP_DEG)
    assert len(flips) == 9
    R = {"margin": MARGIN, "k_main": K_MAIN, "flip_hands": flips, "primary": {}, "descriptive": {}}
    for s in range(3):
        R["primary"][f"s{s}"] = stability(pools, flips, C6, 64, 2000, s)
        R["descriptive"][f"s{s}"] = {
            "C6_k32_stability": stability(pools, flips, C6, 32, 1000, s),
            "finger_k32_stability": stability(pools, flips, FINGER, 32, 1000, s),
            "finger_k64_stability": stability(pools, flips, FINGER, 64, 2000, s),
            "gains": {f"{nm}_k{k}": gains(pools, flips, cands, k, s)
                      for nm, cands in (("C6", C6), ("finger", FINGER)) for k in (32, 64)},
        }
        print(f"s{s} primary stable_frac", R["primary"][f"s{s}"]["stable_frac"], flush=True)
    R["pass"] = all(v["pass"] for v in R["primary"].values())
    (out / "frame_calib64.json").write_text(json.dumps(R, indent=1, default=str))
    print("CHECK 3-K64 PASS", R["pass"], flush=True)


if __name__ == "__main__":
    main()
