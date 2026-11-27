"""Gap vs canonical-frame angle on the 105 held-out LEAP hands (bins, Spearman, outliers removed). Evaluation only."""
import json, os, sys
import numpy as np, torch
from scipy.stats import spearmanr
R = os.environ["GETKITE_RUNS"]; D = f"{R}/transfer_benchmark_v1_leapval105"
PE = sys.argv[1] if len(sys.argv) > 1 else f"{D}/per_example.pt"
PAIRS = [tuple(x.split("-")) for x in sys.argv[2:]] or [("cf2", "d1long"), ("cf2", "d1bv"), ("cfbv", "d1bv")]
X = torch.load(PE, weights_only=False)["leap_heldout"]; names = np.array(X["names"]); P = X["per"]
ang = json.load(open(f"{D}/leap_frame_angles.json")); hands = sorted(ang); a = np.array([ang[h] for h in hands])
bins = [(0, 3), (3, 6), (6, 10), (10, 15), (15, 90)]
for arm, ref in PAIRS:
    print(f"\n{arm} - {ref}")
    for s in range(3):
        g = np.array([(np.asarray(P[f"{arm}_s{s}"]["corr_mm"]) - np.asarray(P[f"{ref}_s{s}"]["corr_mm"]))[names == h].mean() for h in hands])
        own = np.array([np.asarray(P[f"{arm}_s{s}"]["corr_mm"])[names == h].mean() for h in hands])
        rf = np.array([np.asarray(P[f"{ref}_s{s}"]["corr_mm"])[names == h].mean() for h in hands])
        m = a <= 15
        row = f" s{s} spearman {spearmanr(a, g)[0]:+.2f}  r(<=15deg) {np.corrcoef(a[m], g[m])[0, 1]:+.2f} | "
        row += "  ".join(f"[{lo},{hi}) n={((a >= lo) & (a < hi)).sum()} {g[(a >= lo) & (a < hi)].mean():+.3f}" for lo, hi in bins)
        row += f" | own-err vs angle r {np.corrcoef(a[m], own[m])[0, 1]:+.2f}, ref-err vs angle r {np.corrcoef(a[m], rf[m])[0, 1]:+.2f}"
        print(row)
