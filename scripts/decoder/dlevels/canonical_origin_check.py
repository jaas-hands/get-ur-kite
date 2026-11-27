"""CF-O Step 1 (docs/canonical_origin_proposal.md): geometric check of the canonical origin rule. CPU only, no training.

Rule O1: o = centre of the bounding box of the palm points P (CF's palm points: template points on links rigidly attached to
the base, posed at q0) measured along the canonical axes Q (CF rule E). Fallback O2: o = mean of P. Both in the native base
frame. Inputs: robot description + template only (deployment-time inputs).
Checks (predeclared in the proposal):
  * generated bodies: O1 within 5 mm of the palm-box centre (0, 0, length/2), known by construction (generator metadata,
    checking only), for >= 95% of bodies;
  * LEAP variants (train + val split hands only; test splits not read): spread of o in the canonical frame across geometry
    draws of the same topology; variants > 1 cm from their topology's median are flagged;
  * named hands (Allegro, Barrett, Wuji, Shadow): o and its distance from the native origin;
  * invariance: o of a rotated + shifted description is the same physical point (a few robots).

    PYTHONPATH=src python scripts/decoder/dlevels/canonical_origin_check.py --workers 32
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from getkite.decoder import paths
from getkite.decoder.robots import canonical_frame, leap_desc, relabel_base
from getkite.decoder.train_scratch import build_bank, load_config

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "diagnostics")); sys.path.insert(0, str(HERE.parent / "benchmark"))
from crossbody_heldout_eval import WUJI  # noqa: E402
from transfer_benchmark import SHADOW  # noqa: E402
from origin_shift_eval import shift_origin  # noqa: E402

REPO = Path(paths.REPO)
DS = Path(paths.expand("${GETKITE_DATASET}"))


def palm_points(desc):
    """CF's palm points P at q0 in the native base frame (same definition as robots.canonical_frame)."""
    import torch
    from getkite.decoder.robots import RobotBank
    bank = RobotBank([desc], device="cpu", dtype=torch.float64, surface_sampling="hand_frame")
    names = list(desc.tpl_link_names); tl = np.asarray(desc.tpl_link)
    children = {}
    for j in desc.joints:
        children.setdefault(j.parent, []).append(j)
    palm, stack = {desc.base_link}, [desc.base_link]
    while stack:
        for j in children.get(stack.pop(), []):
            if j.type == "fixed" and j.child not in palm:
                palm.add(j.child); stack.append(j.child)
    iP = np.flatnonzero(np.isin(np.array(names, dtype=object)[tl], list(palm)))
    r = torch.zeros(1, dtype=torch.long)
    cl, cu = bank.coords_of(r, bank.lower[r])[0], bank.coords_of(r, bank.upper[r])[0]
    qc = torch.where(bank.coord_valid[0], torch.zeros_like(cl).clamp(cl, cu), 0.0)[None]
    Rs, ts = bank.fk(r, bank.expand(r, qc), torch.eye(3, dtype=torch.float64)[None], torch.zeros(1, 3, dtype=torch.float64))
    Rs, ts = Rs[0].numpy(), ts[0].numpy()
    anc = bank.tpl_anchor[0].numpy(); loc = bank.tpl_pos[0].numpy()
    return np.einsum("pij,pj->pi", Rs[anc[iP]], loc[iP]) + ts[anc[iP]]


def origins(desc):
    Q, _ = canonical_frame(desc)
    P = palm_points(desc)
    c = P @ Q  # canonical coordinates
    o1 = Q @ ((c.min(0) + c.max(0)) / 2)
    return Q, o1, P.mean(0)


def one(args):
    import torch
    torch.set_num_threads(1)
    kind, p = args
    try:
        Q, o1, o2 = origins(leap_desc(p))
    except Exception as e:  # report, don't hide
        return kind, Path(p).name, None, repr(e)
    out = {"o1": o1.tolist(), "o2": o2.tolist(), "o1_can": (o1 @ Q).tolist()}
    if kind == "prim":
        L = json.loads((Path(p) / "hand.json").read_text())["spec"]["palm_size"][1]
        true = np.array([0.0, 0.0, L / 2])
        out["err_o1_mm"] = float(np.linalg.norm(o1 - true) * 1000); out["err_o2_mm"] = float(np.linalg.norm(o2 - true) * 1000)
    return kind, Path(p).name, out, None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="configs/decoder/dlevels/cfo_origin_check.json")
    a = ap.parse_args()
    splits = json.loads((DS / "splits.json").read_text())["hands"]
    leap = sorted(splits["train"]) + sorted(splits["val"])  # test splits not read
    prims = sorted(p.name for p in (DS / "prim/hands").iterdir() if p.is_dir())
    jobs = [("prim", str(DS / "prim/hands" / h)) for h in prims] + [("leap", str(DS / "hands" / h)) for h in leap]
    with Pool(a.workers) as pool:
        res = pool.map(one, jobs, chunksize=4)
    R = {"rule": "O1 = palm-point bounding-box centre along CF axes; O2 = palm-point mean", "prim": {}, "leap": {}, "errors": {}}
    for kind, name, out, err in res:
        if err:
            R["errors"][name] = err
        else:
            R[kind][name] = out
    e1 = np.array([v["err_o1_mm"] for v in R["prim"].values()]); e2 = np.array([v["err_o2_mm"] for v in R["prim"].values()])
    R["prim_summary"] = {"n": len(e1), "o1_within_5mm": float((e1 <= 5).mean()), "o1_median_mm": float(np.median(e1)),
                         "o1_p95_mm": float(np.percentile(e1, 95)), "o1_max_mm": float(e1.max()),
                         "o2_within_5mm": float((e2 <= 5).mean()), "o2_median_mm": float(np.median(e2)),
                         "gate_o1_pass": bool((e1 <= 5).mean() >= 0.95)}
    by_topo = {}
    for h, v in R["leap"].items():
        by_topo.setdefault(h.rsplit("_g", 1)[0], []).append((h, np.array(v["o1_can"])))
    dev, flagged = [], []
    for t, xs in by_topo.items():
        med = np.median(np.stack([x for _, x in xs]), 0)
        for h, x in xs:
            d = float(np.linalg.norm(x - med) * 1000); dev.append(d)
            if d > 10:
                flagged.append((h, round(d, 1)))
    dist = [float(np.linalg.norm(v["o1"]) * 100) for v in R["leap"].values()]
    R["leap_summary"] = {"n_hands": len(R["leap"]), "n_topologies": len(by_topo),
                         "dev_from_topology_median_mm": {"median": float(np.median(dev)), "p90": float(np.percentile(dev, 90)),
                                                         "max": float(np.max(dev))},
                         "flagged_gt_10mm": sorted(flagged, key=lambda x: -x[1]),
                         "o1_dist_from_native_origin_cm": {"median": float(np.median(dist)), "min": float(np.min(dist)), "max": float(np.max(dist))}}
    cfg = load_config(str(REPO / "configs/decoder/scratch/d1_prim50_hf.json"), [])
    bank = build_bank({**cfg, "robots": [r for r in cfg["robots"] if r["kind"] == "kite"] + [WUJI, SHADOW]}, "cpu")
    R["named"] = {}
    inv = []
    rng = np.random.default_rng(0)
    from scipy.spatial.transform import Rotation
    for d in bank.descs:
        Q, o1, o2 = origins(d)
        R["named"][d.name] = {"o1": [round(x, 4) for x in o1], "o1_dist_cm": round(float(np.linalg.norm(o1) * 100), 2),
                              "o2_dist_cm": round(float(np.linalg.norm(o2) * 100), 2)}
        # invariance: rotate axes by G and shift origin by s; the new o1, mapped back to the native frame, must coincide
        G = Rotation.random(random_state=rng).as_matrix(); s = rng.normal(size=3) * 0.03
        d2 = relabel_base(shift_origin(d, s), G)
        _, o1b, _ = origins(d2)
        inv.append(float(np.linalg.norm((G @ o1b + s) - o1)))
    R["invariance_max_m"] = max(inv)
    Path(a.out).write_text(json.dumps(R, indent=1))
    print(json.dumps({"prim": R["prim_summary"], "leap": {k: v for k, v in R["leap_summary"].items() if k != "flagged_gt_10mm"},
                      "n_flagged": len(R["leap_summary"]["flagged_gt_10mm"]), "flagged_top": R["leap_summary"]["flagged_gt_10mm"][:10],
                      "named": R["named"], "invariance_max_m": R["invariance_max_m"], "errors": len(R["errors"])}, indent=1))


if __name__ == "__main__":
    main()
