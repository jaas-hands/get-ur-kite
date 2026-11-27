"""CF-O readout and launch gate (docs/canonical_origin_proposal.md, "Predeclared readout"; launch condition approved 2026-11-25).

Launch condition for the combined run (CF-O + B-scale body variety) = implementation checks AND Primary 1 AND Primary 2:
  * implementation: CF-O's relabelling spread and origin-shift spread <= 0.01 (mm for corr / wrist, absolute for compat);
  * Primary 1: CF-O vs CF2, paired within seed, no significant loss on source Allegro / Barrett / LEAP (transfer_v1_cfo) or the 105
    held-out LEAP hands (transfer_v1_cfo_leapval105) in corr_mm or compat_success, any seed;
  * Primary 2: on Wuji and on Shadow, wrist_mm lower than CF2 in 3/3 seeds and significantly lower in >= 2/3, and corr_mm not
    significantly higher in any seed.
Primary 3 (delivery bar vs matched D1) is reported but does not gate the launch.
Writes ${GETKITE_RUNS}/cfo_gate.json with "launch": true/false.
"""
import json
import os
import sys

import numpy as np

R = os.environ["GETKITE_RUNS"]
SEEDS = (0, 1, 2)
LOWER = {"corr_mm", "wrist_mm", "wrist_deg", "compat_mm"}


def paired(F, a, b, m):
    P = F["_paired"]
    if f"{a}-{b}/{m}" in P:
        p = P[f"{a}-{b}/{m}"]; return p["mean_diff"], p["ci95"]
    p = P[f"{b}-{a}/{m}"]; return -p["mean_diff"], [-p["ci95"][1], -p["ci95"][0]]


def verdict(m, ci):  # +1 = a significantly better, -1 = significantly worse
    if m in LOWER:
        return 1 if ci[1] < 0 else (-1 if ci[0] > 0 else 0)
    return 1 if ci[0] > 0 else (-1 if ci[1] < 0 else 0)


def spread_relabel(path, systems):
    d = json.load(open(path)); mx = 0.0
    for pool in d["pools"].values():
        for rob in pool.values():
            for s in systems:
                for m in ("corr_mm", "wrist_mm", "compat_success"):
                    v = [x[m] for x in rob[s]]; mx = max(mx, max(v) - min(v))
    return mx


def spread_origin(path, systems):
    d = json.load(open(path)); mx = 0.0
    for pool in d["pools"].values():
        for rob in pool.values():
            for s in systems:
                base = rob[s]["none"]
                for c, e in rob[s].items():
                    mx = max(mx, abs(e["corr_mm"] - base["corr_mm"]), abs(e["wrist_mm"] - base["wrist_mm"]))
    return mx


def main():
    M = json.load(open(f"{R}/transfer_benchmark_v1_cfo/metrics.json"))
    H = json.load(open(f"{R}/transfer_benchmark_v1_cfo_leapval105/metrics.json"))
    out = {"checks": {}, "primary1": [], "primary2": {}, "primary3": {}}
    cfo = [f"cfo_s{s}" for s in SEEDS]
    out["checks"]["relabel_spread"] = spread_relabel(f"{R}/cfo_frame_relabel/frame_relabel.json", cfo)
    out["checks"]["origin_shift_spread"] = spread_origin(f"{R}/origin_shift_cfo/origin_shift.json", cfo)
    impl = out["checks"]["relabel_spread"] <= 0.01 and out["checks"]["origin_shift_spread"] <= 0.01
    p1_ok = True
    for label, F in [("Allegro", M["val"]["allegro"]), ("Barrett", M["val"]["barrett"]), ("LEAP", M["val"]["leap"]),
                     ("held-out LEAP (105)", H["leap_heldout"]["leap"])]:
        for m in ("corr_mm", "compat_success"):
            for s in SEEDS:
                d, ci = paired(F, f"cfo_s{s}", f"cf2_s{s}", m); v = verdict(m, ci)
                out["primary1"].append({"pool": label, "metric": m, "seed": s, "diff": d, "ci": ci, "verdict": v})
                p1_ok &= v >= 0
    p2_ok = True
    for fam in ("wuji", "shadow"):
        F = M[fam][fam]; w = [paired(F, f"cfo_s{s}", f"cf2_s{s}", "wrist_mm") for s in SEEDS]
        c = [paired(F, f"cfo_s{s}", f"cf2_s{s}", "corr_mm") for s in SEEDS]
        lower = sum(d < 0 for d, _ in w); sig = sum(verdict("wrist_mm", ci) > 0 for _, ci in w)
        corr_bad = sum(verdict("corr_mm", ci) < 0 for _, ci in c)
        ok = lower == 3 and sig >= 2 and corr_bad == 0; p2_ok &= ok
        out["primary2"][fam] = {"wrist_diff": [d for d, _ in w], "wrist_lower": lower, "wrist_sig_lower": sig,
                                "corr_diff": [d for d, _ in c], "corr_sig_higher": corr_bad, "ok": ok}
    for label, F in [("Allegro", M["val"]["allegro"]), ("Barrett", M["val"]["barrett"]), ("LEAP", M["val"]["leap"]),
                     ("held-out LEAP (105)", H["leap_heldout"]["leap"]), ("Wuji", M["wuji"]["wuji"]), ("Shadow", M["shadow"]["shadow"])]:
        r = [paired(F, f"cfo_s{s}", f"d1long_s{s}", "corr_mm") for s in SEEDS]
        out["primary3"][label] = {"lower": sum(d < 0 for d, _ in r), "sig_higher": sum(verdict("corr_mm", ci) < 0 for _, ci in r),
                                  "diff": [d for d, _ in r]}
    out["implementation_ok"] = impl; out["primary1_ok"] = p1_ok; out["primary2_ok"] = p2_ok
    out["primary3_ok"] = all(v["lower"] == 3 and v["sig_higher"] == 0 for v in out["primary3"].values())
    out["launch"] = bool(impl and p1_ok and p2_ok)
    json.dump(out, open(f"{R}/cfo_gate.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("checks", "primary2", "implementation_ok", "primary1_ok", "primary2_ok", "primary3_ok", "launch")}, indent=1))
    print("primary1 failures:", [x for x in out["primary1"] if x["verdict"] < 0])
    print("primary3:", out["primary3"])
    sys.exit(0)


if __name__ == "__main__":
    main()
