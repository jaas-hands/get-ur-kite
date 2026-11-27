"""CF-3d readout (docs/cf3_distillation_proposal.md, "Predeclared readout"). Pass = P1 and P2 and P3:
  * P1: CF-3d vs CF2, paired within seed: no significant loss in corr_mm or compat_success on source Allegro / Barrett / LEAP, any seed;
  * P2: 105 held-out LEAP hands: corr_mm lower than CF2 in 3/3 seeds and significantly lower in >= 2/3;
  * P3: Wuji: corr_mm and wrist_mm not significantly higher than CF2 in any seed.
Reported, not gating: Shadow; vs D1-bv seeds; vs matched D1-long on every pool (delivery bar); the 360-hand second held-out pool.
Writes ${GETKITE_RUNS}/cf3d_gate.json.
CF-3e (same predeclared P1-P3 vs CF2; also reported vs CF-3d): `cf3d_gate.py cf3e` -> ${GETKITE_RUNS}/cf3e_gate.json.
"""
import json
import os
import sys

R = os.environ["GETKITE_RUNS"]
ARM = sys.argv[1] if len(sys.argv) > 1 else "cf3d"
REFS = ("cf2", "d1bv", "d1long") + (("cf3d",) if ARM != "cf3d" else ())
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


def row(F, ref, m):
    r = [paired(F, f"{ARM}_s{s}", f"{ref}_s{s}", m) for s in SEEDS]
    return {"diff": [round(d, 4) for d, _ in r], "ci": [[round(x, 4) for x in c] for _, c in r], "verdict": [verdict(m, c) for _, c in r]}


def main():
    M = json.load(open(f"{R}/transfer_benchmark_v1_{ARM}/metrics.json"))
    H = json.load(open(f"{R}/transfer_benchmark_v1_{ARM}_leapval105/metrics.json"))
    p360 = f"{R}/transfer_benchmark_v1_{ARM}_leap360/metrics.json"
    H2 = json.load(open(p360)) if os.path.exists(p360) else None
    pools = {"Allegro": M["val"]["allegro"], "Barrett": M["val"]["barrett"], "LEAP": M["val"]["leap"],
             "held-out LEAP (105)": H["leap_heldout"]["leap"], "Wuji": M["wuji"]["wuji"], "Shadow": M["shadow"]["shadow"]}
    if H2:
        pools["held-out LEAP (360)"] = H2["leap_heldout"]["leap"]
    out = {f"vs_{ref}": {} for ref in REFS}
    for name, F in pools.items():
        for ref in REFS:
            out[f"vs_{ref}"][name] = {m: row(F, ref, m) for m in ("corr_mm", "compat_success", "wrist_mm")}
    C = out["vs_cf2"]
    p1 = all(v >= 0 for p in ("Allegro", "Barrett", "LEAP") for m in ("corr_mm", "compat_success") for v in C[p][m]["verdict"])
    h = C["held-out LEAP (105)"]["corr_mm"]
    p2 = sum(d < 0 for d in h["diff"]) == 3 and sum(v > 0 for v in h["verdict"]) >= 2
    p3 = all(v >= 0 for m in ("corr_mm", "wrist_mm") for v in C["Wuji"][m]["verdict"])
    out |= {"P1": p1, "P2": p2, "P3": p3, "pass": bool(p1 and p2 and p3)}
    json.dump(out, open(f"{R}/{ARM}_gate.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("P1", "P2", "P3", "pass")}))
    for ref in REFS:
        print(f"\n{ARM} - {ref} (corr_mm / wrist_mm diffs per seed, verdict +1 better / -1 worse)")
        for name in pools:
            r = out[f"vs_{ref}"][name]
            print(f"  {name:22s} corr {r['corr_mm']['diff']} {r['corr_mm']['verdict']}   wrist {r['wrist_mm']['diff']} {r['wrist_mm']['verdict']}")


if __name__ == "__main__":
    main()
