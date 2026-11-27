"""Seed-repeat summary for transfer_benchmark_v1_seeds (J1 vs G1, training seeds 0/1/2, paired by seed).

Each seed is one training run evaluated on the identical frozen pools. Across-seed statistics are over the 3 run-level
means (SD with ddof=1, n=3); evaluation examples are never pooled across runs. G1-J1 comparisons are within a seed pair
(paired bootstrap over the same evaluation records, from the benchmark's _paired entries, sign flipped to G1-J1).
"""
import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = (0, 1, 2)
J1 = {0: "j1", 1: "j1_s1", 2: "j1_s2"}
G1 = {0: "g1_s0", 1: "g1_s1", 2: "g1_s2"}
POOLS = [("val", "allegro"), ("val", "barrett"), ("val", "leap"), ("leap_heldout", "leap"), ("wuji", "wuji")]
METRICS = ["corr_mm", "corr_success", "compat_success", "wrist_mm", "wrist_deg"]
WRIST = ["trans_slope", "trans_R2_lin", "rot_slope", "rot_R2_lin"]
PAIRED = ["corr_mm", "corr_success", "compat_success", "wrist_mm"]


def paired(F, g, j, m):
    """G1-J1 within a seed pair. Benchmark key is f'{a}-{b}' with b < a (string order)."""
    k1, k2 = f"{g}-{j}/{m}", f"{j}-{g}/{m}"
    if k1 in F["_paired"]:
        p = F["_paired"][k1]
        return p["mean_diff"], p["ci95"]
    p = F["_paired"][k2]
    return -p["mean_diff"], [-p["ci95"][1], -p["ci95"][0]]


def ms(v):
    v = np.asarray(v, float)
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="runs/transfer_benchmark_v1_seeds/metrics.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    M = json.load(open(a.metrics))
    L, S = [], {}
    for pool, fam in POOLS:
        F = M[pool][fam]
        L += [f"\n### {pool}/{fam} (n={F['n']} records per run)\n",
              "| metric | no movement | " + " | ".join(f"J1 s{s}" for s in SEEDS) + " | J1 mean ± SD | "
              + " | ".join(f"G1 s{s}" for s in SEEDS) + " | G1 mean ± SD |", "|" + "---|" * 10]
        mets = METRICS + (WRIST if fam == "wuji" or pool == "val" else [])
        for m in mets:
            if m not in F[J1[0]]:
                continue
            jv = [F[J1[s]][m] for s in SEEDS]
            gv = [F[G1[s]][m] for s in SEEDS]
            nm = F["no_movement"].get(m, float("nan"))
            S[f"{pool}/{fam}/{m}"] = {"j1": jv, "g1": gv, "no_movement": nm}
            L.append(f"| {m} | {nm:.3f} | " + " | ".join(f"{x:.3f}" for x in jv) + f" | {ms(jv)} | "
                     + " | ".join(f"{x:.3f}" for x in gv) + f" | {ms(gv)} |")
        L += ["", "G1 − J1 within each seed pair (paired bootstrap over the same records; 95% CI):", "",
              "| metric | s0 | s1 | s2 | mean ± SD of the 3 paired diffs | pairs with CI excluding 0 (sign) |", "|---|---|---|---|---|---|"]
        for m in PAIRED:
            ds, cs = zip(*[paired(F, G1[s], J1[s], m) for s in SEEDS])
            sig = [("+" if c[0] > 0 else "−") for c in cs if c[0] > 0 or c[1] < 0]
            S[f"{pool}/{fam}/paired/{m}"] = {"diff": ds, "ci95": cs}
            L.append(f"| {m} | " + " | ".join(f"{d:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}]" for d, c in zip(ds, cs))
                     + f" | {ms(ds)} | {len(sig)}/3 {''.join(sig)} |")
        if "_across_morphology_sd" in F:
            L += ["", "Across-morphology SD (per run): " + ", ".join(
                f"{n} {F['_across_morphology_sd'][n]['corr_success']:.3f}" for n in [J1[s] for s in SEEDS] + [G1[s] for s in SEEDS]),
                  ]
    if "crossbody" in M:
        A = M["crossbody"]["aggregate"]["all"]
        jv = [A[J1[s]]["success"] for s in SEEDS]
        gv = [A[G1[s]]["success"] for s in SEEDS]
        S["crossbody/all/success"] = {"j1": jv, "g1": gv}
        L += ["\n### crossbody (all 768 cases; success)\n", f"J1: {', '.join(f'{x:.4f}' for x in jv)} → {ms(jv)}",
              f"G1: {', '.join(f'{x:.4f}' for x in gv)} → {ms(gv)}"]
    txt = "\n".join(L)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n")
        Path(a.out).with_suffix(".json").write_text(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
