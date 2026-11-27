"""Summary of the morphology-diversity follow-up benchmark (transfer_benchmark_v1_followups; docs/decoder_audit.md §59).

Per pool: run-level means for every arm by seed, mean ± SD over the 3 training seeds (ddof=1; evaluation examples are never
pooled across runs), and paired within-seed differences A-B from the benchmark's _paired entries (paired bootstrap over the
same evaluation records). "sig" = the 95% CI excludes 0 in favour of A; counts are out of 3 seeds. No seed is selected.

    python scripts/decoder/benchmark/followup_summary.py --metrics $GETKITE_RUNS/transfer_benchmark_v1_followups/metrics.json
"""
import argparse
import json

import numpy as np

SEEDS = (0, 1, 2)
ARMS = ["d0_g1", "c1", "c5", "n5", "d1", "d1b", "r4"]
POOLS = [("val", "allegro"), ("val", "barrett"), ("val", "leap"), ("leap_heldout", "leap"), ("wuji", "wuji"), ("shadow", "shadow")]
METRICS = ["corr_mm", "compat_success", "corr_success", "wrist_mm"]
LOWER_BETTER = {"corr_mm", "wrist_mm", "compat_mm", "wrist_deg"}
PAIRS = [("d1b", "d1"), ("d1b", "d0_g1"), ("d1b", "c1"), ("c5", "c1"), ("d1", "c5"), ("n5", "c1"), ("n5", "c5"), ("d1", "n5"),
         ("r4", "d1")]


def paired(F, a, b, m):
    """A-B within one seed pair."""
    if f"{a}-{b}/{m}" in F["_paired"]:
        p = F["_paired"][f"{a}-{b}/{m}"]; return p["mean_diff"], p["ci95"]
    p = F["_paired"][f"{b}-{a}/{m}"]; return -p["mean_diff"], [-p["ci95"][1], -p["ci95"][0]]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--metrics", required=True); a = ap.parse_args()
    M = json.load(open(a.metrics))
    for pool, rob in POOLS:
        F = M[pool][rob]
        print(f"\n=== {pool}/{rob} (n={F['n']}) ===")
        print("no movement:", {m: F["no_movement"][m] for m in METRICS})
        print(f"{'arm':7s}" + "".join(f"{m:>34s}" for m in METRICS))
        for arm in ARMS:
            row = f"{arm:7s}"
            for m in METRICS:
                v = np.array([F[f"{arm}_s{s}"][m] for s in SEEDS])
                row += f"  {v.mean():7.3f}±{v.std(ddof=1):.3f} [" + " ".join(f"{x:.3f}" for x in v) + "]"
            print(row)
        print("paired A-B (mean diff per seed; sig = CI excludes 0 favouring A / favouring B)")
        for A, B in PAIRS:
            row = f"  {A:>5s}-{B:<5s}"
            for m in METRICS:
                d, wins, losses = [], 0, 0
                for s in SEEDS:
                    md, ci = paired(F, f"{A}_s{s}", f"{B}_s{s}", m); d.append(md)
                    better = (ci[1] < 0) if m in LOWER_BETTER else (ci[0] > 0)
                    worse = (ci[0] > 0) if m in LOWER_BETTER else (ci[1] < 0)
                    wins += better; losses += worse
                row += f"  {m}: {np.mean(d):+.3f} [" + " ".join(f"{x:+.3f}" for x in d) + f"] A{wins}/B{losses}"
            print(row)


if __name__ == "__main__":
    main()
