"""CF-3f readout (docs/cf3f_proposal.md, predeclared before any CF-3f result). Each arm vs frozen CF-3e of the same seed; paired
bootstrap over records within seed (from the benchmark's _paired entries); seeds never pooled. Pass = Q1..Q5:
  * Q1 source: no significant loss in corr_mm or compat_success on Allegro / Barrett / LEAP validation, any seed;
  * Q2 105 held-out LEAP: corr_mm lower than CF-3e in 3/3 seeds, significantly lower in >= 2/3, significantly higher in none;
  * Q3 Wuji: corr_mm and wrist_mm not significantly higher in any seed;
  * Q4 held-out generated bodies (P0800-P0899; training-log eval at each run's selected checkpoint, point estimate per seed):
       arm - CF-3e <= +0.10 mm in every seed;
  * Q5 Shadow guard (soft): for corr_mm and wrist_mm, mean of the 3 per-seed differences <= 0 and significantly higher in <= 1 seed.
Extra gate for the contact arm only (predeclared 2026-11-28, docs/cf3f_extra_arms.md, before any result):
  * Q6 wrist (contact): wrist_mm and wrist_deg not significantly higher than CF-3e on Allegro / Barrett / LEAP validation and on
       the 105 held-out LEAP pool, in any seed.
Reported, not gating: vs CF2 / D1-bv; the 360-hand pool (long arm only); Shadow / Wuji success rates; wrist_mm and wrist_deg
on every pool; limit_violation_preclamp per pool (arm and CF-3e).

    PYTHONPATH=src python scripts/decoder/benchmark/cf3f_gate.py ARM        # ARM in leap long teach big rec big300k contact widestart
    PYTHONPATH=src python scripts/decoder/benchmark/cf3f_gate.py --selftest # reproduces cf3e_gate.json's CF-3e-vs-CF-3d rows
Writes ${GETKITE_RUNS}/cf3f_{ARM}_gate.json.
"""
import json
import os
import sys

R = os.environ["GETKITE_RUNS"]
SEEDS = (0, 1, 2)
LOWER = {"corr_mm", "wrist_mm", "wrist_deg", "compat_mm"}
REF = "cf3e"


def paired(F, a, b, m):
    if a == b:
        return 0.0, [0.0, 0.0]
    P = F["_paired"]
    if f"{a}-{b}/{m}" in P:
        p = P[f"{a}-{b}/{m}"]; return p["mean_diff"], p["ci95"]
    p = P[f"{b}-{a}/{m}"]; return -p["mean_diff"], [-p["ci95"][1], -p["ci95"][0]]


def verdict(m, ci):  # +1 = arm significantly better, -1 = significantly worse
    if m in LOWER:
        return 1 if ci[1] < 0 else (-1 if ci[0] > 0 else 0)
    return 1 if ci[0] > 0 else (-1 if ci[1] < 0 else 0)


def row(F, sysname, ref, m):
    r = [paired(F, f"{sysname}_s{s}", f"{ref}_s{s}", m) for s in SEEDS]
    return {"diff": [round(d, 4) for d, _ in r], "ci": [[round(x, 4) for x in c] for _, c in r], "verdict": [verdict(m, c) for _, c in r]}


def prim_heldout(run):
    """held-out generated-body contact error at the run's selected checkpoint (training-log eval)."""
    out = []
    for s in SEEDS:
        d = f"{R}/{run}{'' if s == 0 else f'_s{s}'}"
        step = json.load(open(f"{d}/DONE"))["best"]["step"]
        ev = {}
        for line in open(f"{d}/log.jsonl"):
            r = json.loads(line)
            if "eval" in r:
                ev[r["step"]] = r["eval"]["prim_heldout:all/model/contact_mm"]
        out.append(ev[step])
    return out


def readout(M, H, H2, sysname, ref, prim=None):
    pools = {"Allegro": M["val"]["allegro"], "Barrett": M["val"]["barrett"], "LEAP": M["val"]["leap"],
             "held-out LEAP (105)": H["leap_heldout"]["leap"], "Wuji": M["wuji"]["wuji"], "Shadow": M["shadow"]["shadow"]}
    if H2:
        pools["held-out LEAP (360)"] = H2["leap_heldout"]["leap"]
    refs = [ref] + [x for x in ("cf2", "d1bv") if f"{x}_s0" in M["val"]["allegro"]]
    out = {f"vs_{r}": {n: {m: row(F, sysname, r, m) for m in ("corr_mm", "compat_success", "wrist_mm", "wrist_deg")}
                       for n, F in pools.items()} for r in refs}
    out["limit_violation_preclamp"] = {n: {x: [F[f"{x}_s{s}"].get("limit_violation_preclamp") for s in SEEDS] for x in (sysname, ref)}
                                       for n, F in pools.items()}
    C = out[f"vs_{ref}"]
    q1 = all(v >= 0 for p in ("Allegro", "Barrett", "LEAP") for m in ("corr_mm", "compat_success") for v in C[p][m]["verdict"])
    h = C["held-out LEAP (105)"]["corr_mm"]
    q2 = sum(d < 0 for d in h["diff"]) == 3 and sum(v > 0 for v in h["verdict"]) >= 2 and all(v >= 0 for v in h["verdict"])
    q3 = all(v >= 0 for m in ("corr_mm", "wrist_mm") for v in C["Wuji"][m]["verdict"])
    sh = C["Shadow"]
    q5 = all(sum(sh[m]["diff"]) / 3 <= 0 and sum(v < 0 for v in sh[m]["verdict"]) <= 1 for m in ("corr_mm", "wrist_mm"))
    res = {"Q1": q1, "Q2": q2, "Q3": q3, "Q5": q5}
    if sysname == "cf3fwidestart":  # predeclared (docs/wide_start_proposal.md): at KITE-noise starts the arm must not lose
        # (Q2 becomes non-inferiority: 105 pool not significantly higher in any seed); its gain is judged by W (gate_widestart)
        res["Q2"] = all(v >= 0 for v in h["verdict"])
    if sysname == "cf3fcontact":  # predeclared extra gate for the contact arm (docs/cf3f_extra_arms.md)
        res["Q6"] = all(v >= 0 for p in ("Allegro", "Barrett", "LEAP", "held-out LEAP (105)") for m in ("wrist_mm", "wrist_deg")
                        for v in C[p][m]["verdict"])
    if prim is not None:
        a, b = prim
        out["prim_heldout_mm"] = {"arm": [round(x, 3) for x in a], ref: [round(x, 3) for x in b],
                                  "diff": [round(x - y, 3) for x, y in zip(a, b)]}
        res["Q4"] = all(x - y <= 0.10 for x, y in zip(a, b))
    res["pass"] = all(res[k] for k in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6") if k in res) and "Q4" in res
    out |= res
    return out, pools, refs


def show(out, pools, refs, sysname):
    print(json.dumps({k: out[k] for k in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "pass") if k in out}))
    if "prim_heldout_mm" in out:
        print("held-out generated bodies (log, selected ckpt):", out["prim_heldout_mm"])
    for r in refs:
        print(f"\n{sysname} - {r} (corr_mm / wrist_mm diffs per seed, verdict +1 better / -1 worse)")
        for n in pools:
            x = out[f"vs_{r}"][n]
            print(f"  {n:22s} corr {x['corr_mm']['diff']} {x['corr_mm']['verdict']}   wrist {x['wrist_mm']['diff']} {x['wrist_mm']['verdict']}"
                  f"   wrist_deg {x['wrist_deg']['diff']} {x['wrist_deg']['verdict']}")


def main():
    if sys.argv[1] == "--selftest":
        # (a) the arm compared with itself gives all-zero differences; (b) CF-3d vs CF-3e rows equal cf3e_gate.json's
        # CF-3e-vs-CF-3d rows negated (same benchmark files, opposite sign).
        M = json.load(open(f"{R}/transfer_benchmark_v1_cf3e/metrics.json"))
        H = json.load(open(f"{R}/transfer_benchmark_v1_cf3e_leapval105/metrics.json"))
        z, _, _ = readout(M, H, None, "cf3e", REF)
        assert all(d == 0 for p in z[f"vs_{REF}"].values() for m in p.values() for d in m["diff"]), "self-comparison not zero"
        assert z["Q2"] is False and z["Q1"] and z["Q3"] and z["Q5"]
        o, _, _ = readout(M, H, None, "cf3d", REF)
        G = json.load(open(f"{R}/cf3e_gate.json"))["vs_cf3d"]
        for n in ("Allegro", "Barrett", "LEAP", "held-out LEAP (105)", "Wuji", "Shadow"):
            for m in ("corr_mm", "wrist_mm"):
                a, b = o[f"vs_{REF}"][n][m], G[n][m]
                assert all(abs(x + y) < 1e-3 for x, y in zip(a["diff"], b["diff"])), (n, m, a["diff"], b["diff"])
                assert [-v for v in a["verdict"]] == b["verdict"], (n, m)
        print("selftest OK: self-comparison zero; CF-3d-vs-CF-3e reproduces cf3e_gate.json (negated);",
              "CF-3d as a mock arm ->", {k: o[k] for k in ("Q1", "Q2", "Q3", "Q5")})
        return
    arm = sys.argv[1]; sysname = f"cf3f{arm}"
    M = json.load(open(f"{R}/transfer_benchmark_v1_{sysname}/metrics.json"))
    H = json.load(open(f"{R}/transfer_benchmark_v1_{sysname}_leapval105/metrics.json"))
    p360 = f"{R}/transfer_benchmark_v1_{sysname}_leap360/metrics.json"
    H2 = json.load(open(p360)) if os.path.exists(p360) else None
    prim = (prim_heldout(f"cf3f_{arm}_hf"), prim_heldout("cf3e_long_hf"))
    out, pools, refs = readout(M, H, H2, sysname, REF, prim)
    if arm == "widestart":  # W: the predeclared gain, on starts drawn at 3x KITE noise (same records, same scoring)
        Mw = json.load(open(f"{R}/transfer_benchmark_v1_{sysname}_wide3x/metrics.json"))
        Hw = json.load(open(f"{R}/transfer_benchmark_v1_{sysname}_leapval105_wide3x/metrics.json"))
        wo, _, _ = readout(Mw, Hw, None, sysname, REF)
        Cw = wo[f"vs_{REF}"]
        hw = Cw["held-out LEAP (105)"]["corr_mm"]
        W1 = sum(d < 0 for d in hw["diff"]) == 3 and sum(v > 0 for v in hw["verdict"]) >= 2 and all(v >= 0 for v in hw["verdict"])
        W2 = all(v >= 0 for p in ("Wuji", "Shadow") for v in Cw[p]["corr_mm"]["verdict"])
        out["wide3x"] = {"vs_cf3e": Cw, "W1_105_better": W1, "W2_new_families_not_worse": W2}
        out["W"] = W1 and W2
        out["pass"] = out["pass"] and out["W"]
        print("wide3x:", json.dumps({"W1": W1, "W2": W2, "W": out["W"]}))
        for n in Cw:
            print(f"  wide3x {n:22s} corr {Cw[n]['corr_mm']['diff']} {Cw[n]['corr_mm']['verdict']}")
    json.dump(out, open(f"{R}/cf3f_{arm}_gate.json", "w"), indent=1)
    show(out, pools, refs, sysname)


if __name__ == "__main__":
    main()
