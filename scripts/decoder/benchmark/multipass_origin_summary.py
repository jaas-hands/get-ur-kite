"""Readout for docs/multipass_origin_checks_proposal.md (rules predeclared there)."""
import json, os
import numpy as np
R = os.environ["GETKITE_RUNS"]
ARMS = ["d1", "d1long", "cf2", "d1bv"]; SRC = ["allegro", "barrett", "leap"]
M = json.load(open(f"{R}/multipass_v1/multipass.json"))
print("=== Item 1: multi-pass (corr_mm per pass; sig = paired 95% CI vs previous pass excludes 0) ===")
for arm in ARMS:
    K = 1
    for k in (2, 3):
        ok = all(M["pools"]["val"][f][f"{arm}_s{s}"][k - 1]["vs_prev"]["corr_mm"]["ci95"][1] < 0 for f in SRC for s in range(3))
        if ok and K == k - 1: K = k
    print(f"\n{arm}: K = {K} -> {'helps' if K >= 2 else 'does not help'}")
    for pool, f in [("val", x) for x in SRC] + [("leap_heldout", "leap"), ("wuji", "wuji"), ("shadow", "shadow")]:
        E = M["pools"][pool][f]
        row = f"  {pool}/{f:8s} no-mvt {E['no_movement']['corr_mm']:.2f} |"
        for s in range(3):
            e = E[f"{arm}_s{s}"]
            sig2 = e[1]["vs_prev"]["corr_mm"]["ci95"]; s2 = "+" if sig2[0] > 0 else ("-" if sig2[1] < 0 else "0")
            row += f" s{s} {e[0]['corr_mm']:.2f}/{e[1]['corr_mm']:.2f}/{e[2]['corr_mm']:.2f} ({s2})"
        row += " | wrist_mm p1->p2 " + " ".join(f"{E[f'{arm}_s{s}'][0]['wrist_mm']:.2f}->{E[f'{arm}_s{s}'][1]['wrist_mm']:.2f}" for s in range(3))
        print(row)
    warn = [(f, s) for f in ("wuji", "shadow") for s in range(3)
            if K >= 2 and M["pools"][f][f][f"{arm}_s{s}"][K - 1]["vs_pass1"]["corr_mm"]["ci95"][0] > 0]
    if K >= 2: print("  warning sign (pass K significantly worse than pass 1 on Wuji/Shadow):", warn)
O = json.load(open(f"{R}/origin_shift_v1/origin_shift.json"))
print("\n=== Item 2: base-origin shift (mean corr_mm increase over 6 directions vs no shift) ===")
print("checks: cf rotation diff", O["checks"]["cf_rotation_max_diff"], "| max world error m",
      max(v for c, d in O["checks"].items() if isinstance(d, dict) for k, v in d.items() if k.startswith("world")),
      "| surf_idx equal", all(d.get("surf_idx_equal", True) for d in O["checks"].values() if isinstance(d, dict)))
sens_any = False
for arm in ARMS:
    print(f"\n{arm}")
    for pool, f in [("val", x) for x in SRC] + [("leap_heldout", "leap"), ("wuji", "wuji"), ("shadow", "shadow")]:
        E = O["pools"][pool][f]; row = f"  {pool}/{f:8s}"
        for cm in (1, 2, 5):
            d = [np.mean([E[f"{arm}_s{s}"][f"{a}{g}{cm}cm"]["corr_mm"] - E[f"{arm}_s{s}"]["none"]["corr_mm"] for a in "xyz" for g in "+-"]) for s in range(3)]
            row += f"  {cm}cm [" + " ".join(f"{x:+.2f}" for x in d) + "]"
        print(row)
    src2 = [np.mean([np.mean([O["pools"]["val"][f][f"{arm}_s{s}"][f"{a}{g}2cm"]["corr_mm"] - O["pools"]["val"][f][f"{arm}_s{s}"]["none"]["corr_mm"] for a in "xyz" for g in "+-"]) for f in SRC]) for s in range(3)]
    sens = all(x >= 0.3 for x in src2); sens_any |= sens
    print(f"  source (mean of 3 families) 2 cm increase per seed: {[round(x, 3) for x in src2]} -> {'SENSITIVE' if sens else 'not sensitive'}")
print(f"\nItem 2 rule: {'SENSITIVE (item 3 becomes a candidate; separate go-ahead)' if sens_any else 'not sensitive (item 3 dropped)'}")
pc = O.get("palm_centroid_cm_from_origin_baselink_points", {})
print("palm-point centroid distance from native origin (cm):", {k: v for k, v in pc.items() if not k.startswith("I")}, "LEAP median", np.median([v for k, v in pc.items() if k.startswith("I")]) if any(k.startswith("I") for k in pc) else None)
