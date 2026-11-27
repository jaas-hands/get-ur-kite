"""Steady axis rule check, rule K (knuckle line) (docs/steady_axis_rule_proposal.md, second candidate; CPU only).

Rule E (current CF) vs rule K (a = principal direction of the finger-root positions; d = mean finger direction minus its
component along a; n = a x d signed to the curl side; single-finger hands fall back to rule E). Earlier header kept below:
Rule E (current CF) vs rule G (n = smallest-variance palm axis signed by curl; d = mean finger direction projected onto the
palm plane; a = d x n). frame_parts() is robots.canonical_frame's code up to the palm points (copied verbatim, so rule E here
equals robots.canonical_frame; checked on every hand).

    PYTHONPATH=src python scripts/decoder/dlevels/steady_axis_check.py --workers 32
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from getkite.decoder import paths
from getkite.decoder.robots import RobotBank, RobotDesc, canonical_frame, leap_desc, relabel_base
from getkite.decoder.train_scratch import build_bank, load_config

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "diagnostics")); sys.path.insert(0, str(HERE.parent / "benchmark"))
from crossbody_heldout_eval import WUJI  # noqa: E402
from transfer_benchmark import SHADOW  # noqa: E402

REPO = Path(paths.REPO)
DS = Path(paths.expand("${GETKITE_DATASET}"))


def frame_parts(desc, n_samples: int = 512, seed: int = 0):
    """CF canonical palm frame (docs/cf_canonical_palm_frame_proposal.md, rule E), from the robot description + template only.
    Palm points P: template points on links rigidly attached to the base (fixed joints only). Distal links: links with
    points and no descendant link with points. Fingers: movable joints attached to the palm whose subtree holds distal
    links. q0 = zero coordinates clamped to the limits; workspace = n_samples uniform draws within the limits (fixed seed;
    mimics follow their sources). All in the native base frame:
      f    = mean over fingers of unit(mean distal points of the finger at q0 - the finger's root joint origin);
      curl = mean_S(mean distal points) - mean distal points at q0;
      V    = principal axes of P;  d = the axis of V best aligned with f (sign of f);  n = the remaining axis best aligned
             with curl (sign of curl);  a = d x n;  Q = [n a d] (columns in the native base frame).
    relabel_base(desc, Q) then gives x = palmar, y = across, z = distal. Returns (Q (3, 3) float64, info dict)."""
    bank = RobotBank([desc], device="cpu", dtype=torch.float64, surface_sampling="hand_frame")
    names = list(desc.tpl_link_names)
    tl = np.asarray(desc.tpl_link)
    has_pts = {names[i] for i in np.unique(tl)}
    children = {}
    for j in desc.joints:
        children.setdefault(j.parent, []).append(j)
    palm, stack = {desc.base_link}, [desc.base_link]
    while stack:
        for j in children.get(stack.pop(), []):
            if j.type == "fixed" and j.child not in palm:
                palm.add(j.child); stack.append(j.child)

    def subtree(link):
        out = {link}
        for c in children.get(link, []):
            out |= subtree(c.child)
        return out

    def desc_has_pts(link):
        return any(c.child in has_pts or desc_has_pts(c.child) for c in children.get(link, []))
    distal = {l for l in has_pts if not desc_has_pts(l)}
    by_child = {j.child: j for j in desc.joints}

    def depth(j):  # slot order as robot_arrays
        d, link = 0, j.parent
        while link != desc.base_link:
            p = by_child[link]
            d, link = d + (p.type != "fixed"), p.parent
        return d
    slot = {j.name: i + 1 for i, j in enumerate(sorted([j for j in desc.joints if j.type != "fixed"], key=depth))}
    lname = np.array(names, dtype=object)[tl]
    iP = np.flatnonzero(np.isin(lname, list(palm))); iD = np.flatnonzero(np.isin(lname, list(distal)))
    roots = [j for p in sorted(palm) for j in children.get(p, []) if j.type != "fixed"]
    fingers = [(j, [l for l in subtree(j.child) if l in distal]) for j in roots]
    fingers = [(j, dl) for j, dl in fingers if dl]
    assert len(iP) and len(iD) and fingers, f"{desc.name}: palm {len(iP)} / distal {len(iD)} points / {len(fingers)} fingers"
    r = torch.zeros(1, dtype=torch.long)
    cl, cu = bank.coords_of(r, bank.lower[r])[0], bank.coords_of(r, bank.upper[r])[0]
    cv = bank.coord_valid[0]
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(n_samples, cl.shape[0], generator=g, dtype=torch.float64)
    qc = torch.cat([torch.where(cv, torch.zeros_like(cl).clamp(cl, cu), 0.0)[None],
                    torch.where(cv, cl + u * (cu - cl), 0.0)])
    rr = torch.zeros(len(qc), dtype=torch.long)
    Rs, ts = bank.fk(rr, bank.expand(rr, qc), torch.eye(3, dtype=torch.float64).expand(len(qc), 3, 3),
                     torch.zeros(len(qc), 3, dtype=torch.float64))
    Rs, ts = Rs.numpy(), ts.numpy()
    anc = bank.tpl_anchor[0].numpy(); loc = bank.tpl_pos[0].numpy()

    def gmean(ii):  # mean of a point group over all configurations = sum_k (n_k / N)(R_k mean_local_k + t_k)  (exact)
        sl = np.unique(anc[ii])
        w = np.array([(anc[ii] == k).sum() for k in sl], dtype=np.float64) / len(ii)
        ml = np.stack([loc[ii][anc[ii] == k].mean(0) for k in sl])
        return (w[None, :, None] * (np.einsum("skij,kj->ski", Rs[:, sl], ml) + ts[:, sl])).sum(1)

    def unit(v):
        return v / np.linalg.norm(v)
    D = gmean(iD)
    curl = D[1:].mean(0) - D[0]
    units = np.stack([unit(gmean(np.flatnonzero(np.isin(lname, dl)))[0] - ts[0, slot[j.name]]) for j, dl in fingers])
    f = unit(units.mean(0))
    pw = np.einsum("pij,pj->pi", Rs[0][anc[iP]], loc[iP]) + ts[0][anc[iP]]
    roots_pos = np.stack([ts[0, slot[j.name]] for j, _ in fingers])
    return f, curl, pw, roots_pos


def unit(v):
    return v / np.linalg.norm(v)


def rule_E(f, curl, pw):
    ev, V = np.linalg.eigh(np.cov((pw - pw.mean(0)).T))
    kd = int(np.argmax(np.abs(V.T @ f)))
    d = V[:, kd] * np.sign(V[:, kd] @ f)
    rest = [k for k in range(3) if k != kd]
    kn = rest[int(np.argmax(np.abs(V[:, rest].T @ curl)))]
    n = V[:, kn] * np.sign(V[:, kn] @ curl)
    return np.stack([n, np.cross(d, n), d], 1)


def rule_G(f, curl, pw):
    ev, V = np.linalg.eigh(np.cov((pw - pw.mean(0)).T))  # ascending: V[:, 0] = smallest variance = palm normal
    n = V[:, 0] * np.sign(V[:, 0] @ curl)
    d = unit(f - (f @ n) * n)
    return np.stack([n, np.cross(d, n), d], 1), ev


def rule_K(f, curl, pw, roots):
    if len(roots) < 2:
        return rule_E(f, curl, pw), True
    c = roots - roots.mean(0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    a = Vt[0]
    d = unit(f - (f @ a) * a)
    n = np.cross(a, d)
    if n @ curl < 0:
        a, n = -a, -n
    return np.stack([n, a, d], 1), False


def ang(A, B=np.eye(3)):
    return float(np.degrees(Rotation.from_matrix(B.T @ A).magnitude()))


def one(args):
    torch.set_num_threads(1)
    kind, p = args
    try:
        desc = leap_desc(p)
        f, curl, pw, roots = frame_parts(desc)
        QE = rule_E(f, curl, pw); QK, fb = rule_K(f, curl, pw, roots)
        ref = canonical_frame(desc)[0] if kind.startswith("prim") and Path(p).name in ("P0000", "P0001") else None
        return kind, Path(p).name, {"QE": QE.tolist(), "QK": QK.tolist(), "fallback": fb, "n_fingers": len(roots),
                                    "E_matches_lib": None if ref is None else float(np.abs(ref - QE).max())}, None
    except Exception as e:
        return kind, Path(p).name, None, repr(e)


def stats(R, rule, kind):
    pa = np.array([ang(np.array(v[f"Q{rule}"])) for v in R[kind].values()])
    return {"within_30": float((pa <= 30).mean()), "median": float(np.median(pa)), "p90": float(np.percentile(pa, 90))}


def leap_stats(R, rule):
    bt = {}
    for h, v in R["leap"].items():
        bt.setdefault(h.rsplit("_g", 1)[0], []).append(np.array(v[f"Q{rule}"]))
    dev = []
    for t, Qs in bt.items():
        U, _, Vt = np.linalg.svd(sum(Qs)); M = U @ Vt
        dev += [ang(Q, M) for Q in Qs]
    dev = np.array(dev)
    allQ = [np.array(v[f"Q{rule}"]) for v in R["leap"].values()]
    U, _, Vt = np.linalg.svd(sum(allQ)); M = U @ Vt
    dm = np.array([ang(Q, M) for Q in allQ])
    return {"flip_gt30_vs_topology": float((dev > 30).mean()), "median_vs_topology": float(np.median(dev)),
            "p90_vs_topology": float(np.percentile(dev, 90)), "median_vs_family_mean": float(np.median(dm)),
            "gt30_vs_family_mean": float((dm > 30).mean())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="configs/decoder/dlevels/steady_axis_check_K.json")
    a = ap.parse_args()
    sp = json.loads((DS / "splits.json").read_text())["hands"]
    leap = sorted(sp["train"]) + sorted(sp["val"])  # test splits not read
    pa_ = sorted(p.name for p in (DS / "prim/hands").iterdir() if p.is_dir())
    pb_ = sorted(p.name for p in (DS / "prim_b/hands").iterdir() if p.is_dir())
    jobs = ([("prim", str(DS / "prim/hands" / h)) for h in pa_] + [("prim_b", str(DS / "prim_b/hands" / h)) for h in pb_]
            + [("leap", str(DS / "hands" / h)) for h in leap])
    with Pool(a.workers) as pool:
        res = pool.map(one, jobs, chunksize=4)
    R = {"prim": {}, "prim_b": {}, "leap": {}, "errors": {}}
    for kind, name, out, err in res:
        (R["errors"].__setitem__(f"{kind}/{name}", err) if err else R[kind].__setitem__(name, out))
    S = {r: {"prim_b": stats(R, r, "prim_b"), "prim_A": stats(R, r, "prim"), "leap": leap_stats(R, r)} for r in ("E", "K")}
    S["fallback_counts"] = {k: int(sum(v["fallback"] for v in R[k].values())) for k in ("prim", "prim_b", "leap")}
    S["E_matches_library"] = [v["E_matches_lib"] for k in ("prim", "prim_b") for v in R[k].values() if v["E_matches_lib"] is not None]
    cfg = load_config(str(REPO / "configs/decoder/scratch/d1_prim50_hf.json"), [])
    bank = build_bank({**cfg, "robots": [r for r in cfg["robots"] if r["kind"] == "kite"] + [WUJI, SHADOW]}, "cpu")
    named, inv = {}, []
    rng = np.random.default_rng(0)
    for d in bank.descs:
        f, curl, pw, roots = frame_parts(d)
        QE = rule_E(f, curl, pw); QK, fb = rule_K(f, curl, pw, roots)
        named[d.name] = {"angle_K_vs_E_deg": round(ang(QK, QE), 1), "f_dot_dK": round(float(f @ QK[:, 2]), 3),
                         "curl_dot_nK": round(float(curl @ QK[:, 0]), 4), "n_fingers": len(roots), "fallback": fb}
        G = Rotation.random(random_state=rng).as_matrix()
        f2, c2, p2, r2 = frame_parts(relabel_base(d, G))
        QK2, _ = rule_K(f2, c2, p2, r2)
        inv.append(float(np.abs(G @ QK2 - QK).max()))
    S["named"] = named; S["invariance_max"] = max(inv)
    K, E = S["K"], S["E"]
    S["gate"] = {"1_prim_b_within_30_ge_0.90": K["prim_b"]["within_30"] >= 0.90,
                 "2a_leap_flip_le_1pct": K["leap"]["flip_gt30_vs_topology"] <= 0.01,
                 "2b_leap_median_topology_not_worse": K["leap"]["median_vs_topology"] <= E["leap"]["median_vs_topology"],
                 "3_leap_family_mean_not_worse": K["leap"]["median_vs_family_mean"] <= E["leap"]["median_vs_family_mean"],
                 "4_named_signs": all(v["f_dot_dK"] > 0 and v["curl_dot_nK"] > 0 for v in named.values()),
                 "5_invariance": S["invariance_max"] < 1e-6}
    S["gate_pass"] = all(S["gate"].values())
    R["summary"] = S
    Path(a.out).write_text(json.dumps(R, indent=1))
    print(json.dumps(S, indent=1)); print("errors", len(R["errors"]), list(R["errors"].items())[:3])


if __name__ == "__main__":
    main()
