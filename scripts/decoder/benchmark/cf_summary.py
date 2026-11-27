"""CF readout (docs/cf_canonical_palm_frame_proposal.md, Step 2; audit §63).

Per pool and seed: no movement, D1, CF, W2-b (G1 for reference). Paired within-seed differences from the benchmark's _paired
entries (paired bootstrap over the same evaluation records; no pooling across runs, no seed selected). Two questions are
reported separately:
  * improvement over D1        (CF - D1)
  * transfer beyond no movement (CF - no movement, D1 - no movement)
"sig" = the 95% CI excludes 0. Then the predeclared primaries.

    python scripts/decoder/benchmark/cf_summary.py --metrics $GETKITE_RUNS/transfer_benchmark_v1_cf/metrics.json
"""
import argparse
import json

import numpy as np

SEEDS = (0, 1, 2)
ARMS = ["d0_g1", "d1", "w2b", "cf"]
POOLS = [("val", "allegro", "source"), ("val", "barrett", "source"), ("val", "leap", "source"),
         ("leap_heldout", "leap", "held-out LEAP"), ("wuji", "wuji", "Wuji (unseen by our decoder, seen by KITE's encoder psi)"),
         ("shadow", "shadow", "Shadow (unseen by decoder and psi; reserved target, evaluation only)")]
METRICS = ["corr_mm", "corr_success", "compat_success", "wrist_mm", "wrist_deg"]
EXTRA = ["trans_R2_id", "rot_R2_id"]
LOWER = {"corr_mm", "compat_mm", "wrist_mm", "wrist_deg"}
PAIRS = [("cf", "d1"), ("cf", "no_movement"), ("d1", "no_movement"), ("cf", "w2b")]


def sys_(arm, s):
    return arm if arm == "no_movement" else f"{arm}_s{s}"


def paired(F, a, b, m):
    P = F["_paired"]
    if f"{a}-{b}/{m}" in P:
        p = P[f"{a}-{b}/{m}"]; return p["mean_diff"], p["ci95"]
    p = P[f"{b}-{a}/{m}"]; return -p["mean_diff"], [-p["ci95"][1], -p["ci95"][0]]


def verdict(m, ci):
    """+1 if A significantly better, -1 if significantly worse, 0 otherwise."""
    if m in LOWER:
        return 1 if ci[1] < 0 else (-1 if ci[0] > 0 else 0)
    return 1 if ci[0] > 0 else (-1 if ci[1] < 0 else 0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--metrics", required=True); a = ap.parse_args()
    M = json.load(open(a.metrics))
    prim1, prim2 = [], []
    for pool, rob, label in POOLS:
        F = M[pool][rob]
        print(f"\n=== {pool}/{rob}: {label} (n={F['n']}) ===")
        nm = F["no_movement"]
        print("no movement   " + "  ".join(f"{m}={nm[m]:.3f}" for m in METRICS))
        for arm in ARMS:
            print(f"{arm:13s} " + "  ".join(f"{m}=[" + " ".join(f"{F[f'{arm}_s{s}'][m]:.3f}" for s in SEEDS) + "]" for m in METRICS + EXTRA))
        for A, B in PAIRS:
            row = f"  {A}-{B:<12s}"
            for m in METRICS:
                d, w, l = [], 0, 0
                for s in SEEDS:
                    md, ci = paired(F, sys_(A, s), sys_(B, s), m); d.append(md); v = verdict(m, ci); w += v > 0; l += v < 0
                    if A == "cf" and B == "d1":
                        if pool in ("val", "leap_heldout") and m in ("corr_mm", "compat_success"):
                            prim1.append((f"{pool}/{rob}", m, s, v))
                        if pool == "shadow" and m == "wrist_mm":
                            prim2.append(("shadow", m, s, v))
                row += f"  {m} {np.mean(d):+.3f} [" + " ".join(f"{x:+.3f}" for x in d) + f"] better{w}/worse{l}"
            print(row)
        if pool == "shadow":
            for s in SEEDS:
                prim2.append(("shadow", "trans_R2_id", s, 1 if F[f"cf_s{s}"]["trans_R2_id"] > F[f"d1_s{s}"]["trans_R2_id"] else -1))
    print("\n=== Predeclared readout ===")
    worse = [p for p in prim1 if p[3] < 0]
    print(f"Primary 1 (CF not significantly worse than D1 on source + held-out LEAP corr_mm / compat_success, every seed): "
          f"{'HOLDS' if not worse else 'FAILS'}; significant losses: {worse}")
    sh = {(m, s): v for _, m, s, v in prim2}
    ok2 = all(sh[("wrist_mm", s)] > 0 for s in SEEDS) and all(sh[("trans_R2_id", s)] > 0 for s in SEEDS)
    print(f"Primary 2 (Shadow wrist_mm significantly better and trans_R2_id higher, 3/3): {'HOLDS' if ok2 else 'FAILS'}; {sh}")
    print(f"Promotion: {'YES' if (not worse and ok2) else 'NO'}")


if __name__ == "__main__":
    main()
