"""CF-3s readout (recovery from lag, Step 2). Implements the gate fixed in scripts/decoder/dlevels/make_cf3s.py (committed before
launch); this file is committed before any CF-3s result exists.
  (a) near-goal non-inferiority at x1: no seed with arm - CF-3e point estimate above the anchor on source val Allegro / Barrett /
      LEAP (0.20 mm), 105 pool (0.20), Wuji (0.54), Shadow (0.82), held-out bodies (0.14; training-log eval).
  (b) lag: at x3 start noise (lag_robustness_cf3swide), CF-3e corr_mm minus arm corr_mm > 0.5 mm on the 105 pool AND on
      source-val LEAP, in >= 2 of 3 seeds.
  PASS = (a) and (b).

    PYTHONPATH=src python scripts/decoder/benchmark/cf3s_readout.py
Writes ${GETKITE_RUNS}/cf3s_wide_readout.json.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cf3f_gate import prim_heldout  # noqa: E402
from cf3m_readout import transfer_rows  # noqa: E402

R = os.environ["GETKITE_RUNS"]
SEEDS = (0, 1, 2)
ANCH = {"Allegro": 0.20, "Barrett": 0.20, "LEAP": 0.20, "105": 0.20, "Wuji": 0.54, "Shadow": 0.82}
SYS = "cf3swide"


def main():
    M = json.load(open(f"{R}/transfer_benchmark_v1_{SYS}/metrics.json"))
    H = json.load(open(f"{R}/transfer_benchmark_v1_{SYS}_leapval105/metrics.json"))
    T = transfer_rows(M, H, None, SYS)
    prim = (prim_heldout("cf3s_wide_hf"), prim_heldout("cf3e_long_hf"))
    pd = [round(x - y, 3) for x, y in zip(*prim)]
    bad = {p: [T[p]["corr_mm"]["diff"][s] > a for s in SEEDS] for p, a in ANCH.items()}
    bad["heldout_bodies"] = [d > 0.14 for d in pd]
    qa = not any(any(v) for v in bad.values())
    L = json.load(open(f"{R}/lag_robustness_{SYS}/lag_robustness.json"))["pools"]
    lag = {}
    for pool, key in (("105", "leap105:all"), ("LEAP", "val:leap")):
        lag[pool] = {f"k{k}": [round(L[key][f"cf3e_s{s}"][f"k{k}"]["call1"]["corr_mm"] - L[key][f"{SYS}_s{s}"][f"k{k}"]["call1"]["corr_mm"], 4)
                               for s in SEEDS] for k in (1, 2, 3, 4, 6)}
    ok3 = [lag["105"]["k3"][s] > 0.5 and lag["LEAP"]["k3"][s] > 0.5 for s in SEEDS]
    qb = sum(ok3) >= 2
    out = {"pass": qa and qb, "a_neargoal_noninferior": qa, "a_violations": bad, "b_lag": qb, "b_seed_ok": ok3,
           "lag_cf3e_minus_arm_corr": lag, "transfer": T, "heldout_bodies_diff": pd}
    json.dump(out, open(f"{R}/cf3s_wide_readout.json", "w"), indent=1)
    print(f"CF-3s wide: PASS={out['pass']}  (a) near-goal {qa}  (b) lag {qb} {ok3}")
    for p in ANCH:
        print(f"  {p:8s} corr {T[p]['corr_mm']['diff']}  wrist {T[p]['wrist_mm']['diff']}")
    print(f"  held-out bodies {pd}")
    for p in lag:
        print(f"  lag {p}: CF-3e - arm by k {lag[p]}")


if __name__ == "__main__":
    main()
