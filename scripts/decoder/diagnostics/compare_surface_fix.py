"""Old (link_local_legacy) vs corrected (hand_frame) raw-contact decoder on identical records/starts/translator."""
import json, sys
old, new = (json.load(open(p)) for p in sys.argv[1:3])
M = ["corr_contact_mm", "corr_normal_deg", "corr_success", "compat_mm", "success", "move_mm", "limit_violation"]
def row(name, o, n):
    for src in ("true", "predicted"):
        if src not in o: continue
        cells = [f"{o[src][m]:.3g}→{n[src][m]:.3g}" if m in o[src] else "–" for m in M]
        print(f"| {name} | {src} | " + " | ".join(cells) + " |")
print("| pool | contacts | " + " | ".join(M) + " |"); print("|" + "---|" * (len(M) + 2))
for b in ("allegro", "barrett", "leap"):
    row(f"source val {b}", old["A_same_body_val"][b], new["A_same_body_val"][b])
row("held-out LEAP", old["A_same_body_leap_heldout"]["leap"], new["A_same_body_leap_heldout"]["leap"])
row("Wuji near-goal (unseen by decoder, seen by ψ)", old["A_same_body_wuji"]["wuji"], new["A_same_body_wuji"]["wuji"])
nm = old["A_same_body_wuji"]["wuji"]["no_movement"]; print(f"\nWuji no-movement reference: corr {nm['corr_contact_mm']:.3g} mm, success {nm['success']:.3g}\n")
print("Cross-body (closest-example start):"); print("| target | contacts | " + " | ".join(M) + " |"); print("|" + "---|" * (len(M) + 2))
for t, o in old["B_crossbody_aggregate"].items():
    n = new["B_crossbody_aggregate"][t]
    row(t, o, n); row(t + " @oracle-feasible", {"true": o["true@feasible"], "predicted": o["predicted@feasible"]},
                      {"true": n["true@feasible"], "predicted": n["predicted@feasible"]})
    print(f"| {t} oracle solutions found | | {o['oracle_feasible_frac']:.3g}→{n['oracle_feasible_frac']:.3g} |")
