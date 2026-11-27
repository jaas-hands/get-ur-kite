"""Why does body variety (B) help D1 but not CF? Per-hand breakdown on the 105 held-out LEAP hands (evaluation data only).

Gain from variety, per hand and seed: D1: d1bv - d1long; CF: cfbv - cf2 (corr_mm, negative = variety helped).
Hand properties (description + template only): finger count, topology distance to the nearest training topology in A (48 hands)
and in B (138 topologies), palm extent (palm principal variances), canonical-origin distance from the native origin, and the
canonical-frame tilt to the mean LEAP frame. Reads existing results; no new evaluation.
"""
import json, os, re
import numpy as np
import torch
from scipy.stats import spearmanr

R = os.environ["GETKITE_RUNS"]
X = torch.load(f"{R}/transfer_benchmark_v1_leapval105/per_example.pt", weights_only=False)["leap_heldout"]
names = np.array(X["names"]); P = X["per"]; hands = sorted(set(names))
ang = json.load(open(f"{R}/transfer_benchmark_v1_leapval105/leap_frame_angles.json"))
oc = json.load(open("configs/decoder/dlevels/cfo_origin_check.json"))["leap"]
ax = json.load(open("configs/decoder/dlevels/steady_axis_check.json"))["leap"]
man = json.load(open("configs/decoder/dlevels/bodyvar_manifest.json"))
A = json.load(open("configs/decoder/scratch/d1_long_hf.json"))
a_tr = [os.path.basename(t["npz"])[:-4] for t in A["train"] if t["kind"] == "leap" and "/prim/" not in t["npz"]]


def code(h):
    return np.array([int(d) for d in re.findall(r"[IMRT](\d)", h.split("_g")[0])])


def nearest(h, pool):
    c = code(h)
    return min(int(np.abs(code(p) - c).sum()) for p in pool)


props = {}
for h in hands:
    eig = np.array(ax[h]["eig"])  # ascending palm variances
    props[h] = {"fingers": int((code(h) > 0).sum()), "joints": int(code(h).sum()),
                "topo_dist_A": nearest(h, a_tr), "topo_dist_B": nearest(h, man["leap_train"]),
                "palm_area": float(np.sqrt(eig[1] * eig[2]) * 1e4), "origin_cm": float(np.linalg.norm(oc[h]["o1"]) * 100),
                "tilt_deg": ang[h]}


def per_hand(arm, s):
    v = np.asarray(P[f"{arm}_s{s}"]["corr_mm"])
    return np.array([v[names == h].mean() for h in hands])


print("held-out LEAP, 105 hands. gain = with variety - without (mm; negative = variety helped)")
G = {}
for s in range(3):
    gd = per_hand("d1bv", s) - per_hand("d1long", s); gc = per_hand("cfbv", s) - per_hand("cf2", s)
    G[s] = (gd, gc)
    print(f"seed {s}: D1 gain {gd.mean():+.3f} (helped on {int((gd < 0).sum())}/105)   CF gain {gc.mean():+.3f} (helped on {int((gc < 0).sum())}/105)"
          f"   corr(D1 gain, CF gain) across hands r={np.corrcoef(gd, gc)[0, 1]:+.2f}")
print("\nSpearman correlation of gain with hand property (per seed: D1 | CF)")
for k in ("fingers", "joints", "topo_dist_A", "topo_dist_B", "palm_area", "origin_cm", "tilt_deg"):
    x = np.array([props[h][k] for h in hands])
    print(f"  {k:12s} " + "  ".join(f"s{s} {spearmanr(x, G[s][0])[0]:+.2f} | {spearmanr(x, G[s][1])[0]:+.2f}" for s in range(3)))
print("\nGain by group (mean over 3 seeds; D1 / CF / CF2-minus-D1-long baseline gap / CF-bv-minus-D1-bv gap)")
def grp(k, bins):
    x = np.array([props[h][k] for h in hands])
    for lo, hi in bins:
        m = (x >= lo) & (x < hi)
        if not m.any(): continue
        gd = np.mean([G[s][0][m].mean() for s in range(3)]); gc = np.mean([G[s][1][m].mean() for s in range(3)])
        base = np.mean([(per_hand("cf2", s) - per_hand("d1long", s))[m].mean() for s in range(3)])
        after = np.mean([(per_hand("cfbv", s) - per_hand("d1bv", s))[m].mean() for s in range(3)])
        print(f"  {k} [{lo},{hi}) n={m.sum():3d}: D1 gain {gd:+.3f}  CF gain {gc:+.3f}  | CF-D1 gap before {base:+.3f} -> after {after:+.3f}")
grp("fingers", [(2, 3), (3, 4), (4, 5)])
grp("topo_dist_A", [(0, 2), (2, 3), (3, 4), (4, 20)])
grp("topo_dist_B", [(0, 1), (1, 2), (2, 20)])
grp("tilt_deg", [(0, 3), (3, 6), (6, 10), (10, 90)])
grp("palm_area", [(0, 8.5), (8.5, 9.5), (9.5, 100)])
print("\nproperty ranges:", {k: (round(min(p[k] for p in props.values()), 2), round(max(p[k] for p in props.values()), 2)) for k in ("palm_area", "origin_cm", "topo_dist_A", "topo_dist_B")})
json.dump({"props": props, "gain": {s: {"d1": G[s][0].tolist(), "cf": G[s][1].tolist()} for s in range(3)}, "hands": hands},
          open("configs/decoder/dlevels/cf2_readout/variety_gain.json", "w"), indent=1)
