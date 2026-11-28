"""CF-3m readout (teacher memorisation v2, Stage A). Implements the rules fixed in scripts/decoder/dlevels/make_cf3m.py (committed
before launch); this file is committed before any CF-3m result exists. Each arm vs CF-3e of the same seed, re-scored in the same
benchmark / external-hands jobs; paired bootstrap over scored records within seed; seeds never pooled.

  new-family score per seed = mean over the 9 new-family hands (Wuji, Shadow, ability, inspire, xhand, schunk, metahand, sharpa,
    aero) of the per-hand corr_mm difference (arm - CF-3e).
  "hurts"  (memorised targets hurt the student): score < 0 in >= 2/3 seeds AND >= 5 of 9 hands significantly better (95 % CI
           upper < 0) in >= 2/3 seeds AND no seed where source val (Allegro, Barrett, LEAP) or the 105 pool is worse by more than
           the anchor (point estimate).
  "helps": score > 0 in >= 2/3 seeds AND >= 5 of 9 hands significantly worse (CI lower > 0) in >= 2/3 seeds.
  otherwise "no detectable effect".
Anchors (CF-3e seed spread, plan_2026-11-28): 105 pool 0.20 mm; held-out bodies 0.14 mm (reported); Wuji 0.54, Shadow 0.82
(reported). Clarification written here before any result: make_cf3m.py names no separate anchor for the source-val pools, so the
105-pool anchor (0.20 mm) is used for them too.
Reported, not deciding: wrist_mm, compat_success, the 360 pool, held-out bodies (training-log eval at the selected checkpoint;
per-record frame bands are not available from the logs), vs CF2 / D1-bv, MetaHand (frame-limited) shown in the per-hand table.

    PYTHONPATH=src python scripts/decoder/benchmark/cf3m_readout.py ARM          # ARM in leap body
    PYTHONPATH=src python scripts/decoder/benchmark/cf3m_readout.py --selftest   # transfer part on cf3fbig vs cf3f_big_gate.json
Writes ${GETKITE_RUNS}/cf3m_{ARM}_readout.json.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cf3f_gate import paired, prim_heldout, verdict  # noqa: E402

R = os.environ["GETKITE_RUNS"]
SEEDS = (0, 1, 2)
EXT = ["ability", "inspire", "xhand", "schunk", "metahand", "sharpa", "aero"]
ANCHOR_105 = 0.20


def transfer_rows(M, H, H2, sysname, ref="cf3e"):
    pools = {"Allegro": M["val"]["allegro"], "Barrett": M["val"]["barrett"], "LEAP": M["val"]["leap"],
             "105": H["leap_heldout"]["leap"], "Wuji": M["wuji"]["wuji"], "Shadow": M["shadow"]["shadow"]}
    if H2:
        pools["360"] = H2["leap_heldout"]["leap"]
    out = {}
    for n, F in pools.items():
        out[n] = {}
        for m in ("corr_mm", "wrist_mm", "compat_success"):
            r = [paired(F, f"{sysname}_s{s}", f"{ref}_s{s}", m) for s in SEEDS]
            out[n][m] = {"diff": [round(d, 4) for d, _ in r], "ci": [[round(x, 4) for x in c] for _, c in r]}
    return out


def main():
    if sys.argv[1] == "--selftest":
        M = json.load(open(f"{R}/transfer_benchmark_v1_cf3fbig/metrics.json"))
        H = json.load(open(f"{R}/transfer_benchmark_v1_cf3fbig_leapval105/metrics.json"))
        t = transfer_rows(M, H, None, "cf3fbig")
        G = json.load(open(f"{R}/cf3f_big_gate.json"))["vs_cf3e"]
        for mine, theirs in (("Wuji", "Wuji"), ("Shadow", "Shadow"), ("105", "held-out LEAP (105)"), ("LEAP", "LEAP")):
            for m in ("corr_mm", "wrist_mm"):
                assert all(abs(a - b) < 1e-3 for a, b in zip(t[mine][m]["diff"], G[theirs][m]["diff"])), (mine, m)
        print("selftest OK: transfer rows reproduce cf3f_big_gate.json (Wuji, Shadow, 105, LEAP)")
        return
    arm = sys.argv[1]; sysname = f"cf3m{arm}"
    M = json.load(open(f"{R}/transfer_benchmark_v1_{sysname}/metrics.json"))
    H = json.load(open(f"{R}/transfer_benchmark_v1_{sysname}_leapval105/metrics.json"))
    p360 = f"{R}/transfer_benchmark_v1_{sysname}_leap360/metrics.json"
    H2 = json.load(open(p360)) if os.path.exists(p360) else None
    X = json.load(open(f"{R}/external_hands_{sysname}_v1/external_hands.json"))["pools"]
    T = transfer_rows(M, H, H2, sysname)
    hands = {"Wuji": T["Wuji"]["corr_mm"], "Shadow": T["Shadow"]["corr_mm"]}
    for h in EXT:
        mc = X[h][sysname]["minus_cf3e"]["corr_mm"]
        hands[h] = {"diff": [round(x["mean_diff"], 4) for x in mc], "ci": [x["ci95"] for x in mc]}
    score = [round(sum(hands[h]["diff"][s] for h in hands) / len(hands), 4) for s in SEEDS]
    better = [sum(hands[h]["ci"][s][1] < 0 for h in hands) for s in SEEDS]
    worse = [sum(hands[h]["ci"][s][0] > 0 for h in hands) for s in SEEDS]
    guard_bad = [any(T[p]["corr_mm"]["diff"][s] > ANCHOR_105 for p in ("Allegro", "Barrett", "LEAP", "105")) for s in SEEDS]
    hurts = sum(x < 0 for x in score) >= 2 and sum(b >= 5 for b in better) >= 2 and not any(guard_bad)
    helps = sum(x > 0 for x in score) >= 2 and sum(w >= 5 for w in worse) >= 2
    reading = "memorisation hurts the student" if hurts else ("memorisation helps the student" if helps else "no detectable effect")
    prim = (prim_heldout(f"cf3m_{arm}_hf"), prim_heldout("cf3e_long_hf"))
    out = {"arm": arm, "reading": reading, "new_family_score": score, "hands_sig_better": better, "hands_sig_worse": worse,
           "source_or_105_worse_than_anchor": guard_bad, "per_hand_corr": hands, "transfer": T,
           "heldout_bodies_mm": {"arm": [round(x, 3) for x in prim[0]], "cf3e": [round(x, 3) for x in prim[1]],
                                 "diff": [round(x - y, 3) for x, y in zip(*prim)]},
           "external_wrist": {h: [round(x["mean_diff"], 4) for x in X[h][sysname]["minus_cf3e"]["wrist_mm"]] for h in EXT}}
    json.dump(out, open(f"{R}/cf3m_{arm}_readout.json", "w"), indent=1)
    print(f"CF-3m {arm}: READING = {reading}")
    print(f"  new-family score per seed (mean corr diff over 9 hands): {score}; sig better {better}; sig worse {worse}")
    print(f"  source/105 worse than anchor per seed: {guard_bad}; held-out bodies diff: {out['heldout_bodies_mm']['diff']}")
    for n, x in T.items():
        print(f"  {n:8s} corr {x['corr_mm']['diff']}  wrist {x['wrist_mm']['diff']}")
    for h in EXT:
        print(f"  {h:8s} corr {hands[h]['diff']}")


if __name__ == "__main__":
    main()
