"""CF2 / B readout (docs/cf2_longer_training_proposal.md, "Predeclared readout", Primaries 1-3).

Compares a candidate arm with a reference arm, paired by seed, in one benchmark's metrics.json; adds no-movement transfer
and Primary 3 (contact error lower than the reference on every family, every seed). Generated-body families come from the
trainer's validation log at the selected checkpoint (best step in DONE): no paired CI there, point estimate only.

    python scripts/decoder/benchmark/cf2_summary.py --metrics $GETKITE_RUNS/transfer_benchmark_v1_cf2/metrics.json \
        --cand cf2 --ref d1long --cand_runs cf2_long_hf --ref_runs d1_long_hf
"""
import argparse
import json
import os

import numpy as np

SEEDS = (0, 1, 2)
POOLS = [("val", "allegro", "Allegro (source)"), ("val", "barrett", "Barrett (source)"), ("val", "leap", "LEAP (source)"),
         ("leap_heldout", "leap", "held-out LEAP"), ("wuji", "wuji", "Wuji (unseen by our decoder, seen by KITE's encoder psi)"),
         ("shadow", "shadow", "Shadow (unseen by decoder and psi; evaluation only)")]
METRICS = ["corr_mm", "corr_success", "compat_success", "wrist_mm", "wrist_deg"]
LOWER = {"corr_mm", "compat_mm", "wrist_mm", "wrist_deg"}
LOG_FAMS = [("prim/model/contact_mm", "generated bodies (source val, log)"),
            ("prim_heldout:prim/model/contact_mm", "held-out generated bodies (log)")]


def sname(arm, s):
    return arm if arm == "no_movement" else f"{arm}_s{s}"


def paired(F, a, b, m):
    P = F["_paired"]
    if f"{a}-{b}/{m}" in P:
        p = P[f"{a}-{b}/{m}"]; return p["mean_diff"], p["ci95"]
    p = P[f"{b}-{a}/{m}"]; return -p["mean_diff"], [-p["ci95"][1], -p["ci95"][0]]


def verdict(m, ci):
    if m in LOWER:
        return 1 if ci[1] < 0 else (-1 if ci[0] > 0 else 0)
    return 1 if ci[0] > 0 else (-1 if ci[1] < 0 else 0)


def log_at_best(run_dir, key):
    best = json.load(open(os.path.join(run_dir, "DONE")))["best"]["step"]
    for line in open(os.path.join(run_dir, "log.jsonl")):
        if '"eval"' in line:
            d = json.loads(line)
            if d["step"] == best:
                return d["eval"].get(key, float("nan"))
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True); ap.add_argument("--cand", required=True); ap.add_argument("--ref", required=True)
    ap.add_argument("--cand_runs", required=True, help="run dir stem, e.g. cf2_long_hf (seeds add _s1/_s2)")
    ap.add_argument("--ref_runs", required=True)
    a = ap.parse_args()
    M = json.load(open(a.metrics)); runs = os.path.expandvars("$GETKITE_RUNS")
    C, Rf = a.cand, a.ref
    p1, p2, p3 = [], {}, {}
    for pool, rob, label in POOLS:
        F = M[pool][rob]
        print(f"\n=== {label} (n={F['n']}) ===")
        nm = F["no_movement"]
        print("no movement   " + "  ".join(f"{m}={nm[m]:.3f}" for m in METRICS))
        for arm in (Rf, C):
            print(f"{arm:13s} " + "  ".join(f"{m}=[" + " ".join(f"{F[sname(arm, s)][m]:.3f}" for s in SEEDS) + "]"
                                            for m in METRICS + ["trans_R2_id"]))
        for A, B in ((C, Rf), (C, "no_movement"), (Rf, "no_movement")):
            row = f"  {A}-{B:<12s}"
            for m in METRICS:
                d, w, l = [], 0, 0
                for s in SEEDS:
                    md, ci = paired(F, sname(A, s), sname(B, s), m); d.append(md); v = verdict(m, ci); w += v > 0; l += v < 0
                    if (A, B) == (C, Rf):
                        if pool in ("val", "leap_heldout") and m in ("corr_mm", "compat_success"):
                            p1.append((label, m, s, v))
                        if m == "corr_mm":
                            p3.setdefault(label, []).append((md < 0, v))
                        if pool == "shadow" and m == "wrist_mm":
                            p2[("wrist_mm", s)] = v
                row += f"  {m} {np.mean(d):+.3f} [" + " ".join(f"{x:+.3f}" for x in d) + f"] better{w}/worse{l}"
            print(row)
        if pool == "shadow":
            for s in SEEDS:
                p2[("trans_R2_id", s)] = 1 if F[sname(C, s)]["trans_R2_id"] > F[sname(Rf, s)]["trans_R2_id"] else -1
    print("\n=== generated-body families (training-log validation at the selected checkpoint; point estimates) ===")
    for key, label in LOG_FAMS:
        vals = []
        for s, sfx in zip(SEEDS, ("", "_s1", "_s2")):
            c = log_at_best(f"{runs}/{a.cand_runs}{sfx}", key); r = log_at_best(f"{runs}/{a.ref_runs}{sfx}", key)
            vals.append((c, r)); p3.setdefault(label, []).append((c < r, 0))
        print(f"  {label:38s} {C}=[" + " ".join(f"{c:.3f}" for c, _ in vals) + f"] {Rf}=[" + " ".join(f"{r:.3f}" for _, r in vals)
              + "] diff=[" + " ".join(f"{c - r:+.3f}" for c, r in vals) + "]")
    print(f"\n=== Predeclared readout: {C} vs {Rf} ===")
    worse = [x for x in p1 if x[3] < 0]
    print(f"Primary 1 (no significant loss on source + held-out LEAP corr_mm / compat_success, any seed): "
          f"{'HOLDS' if not worse else 'FAILS'} {worse}")
    ok2 = all(v > 0 for v in p2.values())
    print(f"Primary 2 (Shadow wrist_mm significantly better and trans_R2_id higher, 3/3): {'HOLDS' if ok2 else 'FAILS'} {p2}")
    print("Primary 3 (contact error lower than the reference in 3/3 seeds and never significantly higher, every family):")
    ok3 = True
    for fam, xs in p3.items():
        lower = sum(x[0] for x in xs); sig_lower = sum(x[1] > 0 for x in xs); sig_higher = sum(x[1] < 0 for x in xs)
        ok = lower == 3 and sig_higher == 0; ok3 &= ok
        print(f"  {fam:55s} lower {lower}/3, significantly lower {sig_lower}/3, significantly higher {sig_higher}/3 -> {'ok' if ok else 'NOT MET'}")
    print(f"Primary 3: {'HOLDS' if ok3 else 'FAILS'}")
    print(f"All three: {'YES' if (not worse and ok2 and ok3) else 'NO'}")


if __name__ == "__main__":
    main()
