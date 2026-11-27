"""Steady axis rule check (docs/steady_axis_rule_proposal.md; CPU only, no training).

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
    return f, curl, pw


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


def ang(A, B=np.eye(3)):
    return float(np.degrees(Rotation.from_matrix(B.T @ A).magnitude()))


def one(args):
    torch.set_num_threads(1)
    kind, p = args
    try:
        desc = leap_desc(p)
        f, curl, pw = frame_parts(desc)
        QE = rule_E(f, curl, pw); QG, ev = rule_G(f, curl, pw)
        ref = canonical_frame(desc)[0] if kind == "prim" and Path(p).name in ("P0000", "P0001") else None
        return kind, Path(p).name, {"QE": QE.tolist(), "QG": QG.tolist(), "eig": ev.tolist(), "f_dot_dG": float(f @ QG[:, 2]),
                                    "curl_dot_nG": float(curl @ QG[:, 0]),
                                    "E_matches_lib": None if ref is None else float(np.abs(ref - QE).max())}, None
    except Exception as e:
        return kind, Path(p).name, None, repr(e)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="configs/decoder/dlevels/steady_axis_check.json")
    a = ap.parse_args()
    sp = json.loads((DS / "splits.json").read_text())["hands"]
    leap = sorted(sp["train"]) + sorted(sp["val"])  # test splits not read
    prims = sorted(p.name for p in (DS / "prim/hands").iterdir() if p.is_dir())
    jobs = [("prim", str(DS / "prim/hands" / h)) for h in prims] + [("leap", str(DS / "hands" / h)) for h in leap]
    with Pool(a.workers) as pool:
        res = pool.map(one, jobs, chunksize=4)
    R = {"prim": {}, "leap": {}, "errors": {}}
    for kind, name, out, err in res:
        (R["errors"].__setitem__(name, err) if err else R[kind].__setitem__(name, out))
    S = {}
    for rule in ("E", "G"):
        pa = np.array([ang(np.array(v[f"Q{rule}"])) for v in R["prim"].values()])
        bt = {}
        for h, v in R["leap"].items():
            bt.setdefault(h.rsplit("_g", 1)[0], []).append(np.array(v[f"Q{rule}"]))
        dev = []
        for t, Qs in bt.items():
            U, _, Vt = np.linalg.svd(sum(Qs)); M = U @ Vt
            if np.linalg.det(M) < 0:
                U[:, -1] *= -1; M = U @ Vt
            dev += [ang(Q, M) for Q in Qs]
        dev = np.array(dev)
        allQ = [np.array(v[f"Q{rule}"]) for v in R["leap"].values()]
        U, _, Vt = np.linalg.svd(sum(allQ)); M = U @ Vt
        dm = np.array([ang(Q, M) for Q in allQ])
        S[rule] = {"prim_within_30": float((pa <= 30).mean()), "prim_median": float(np.median(pa)), "prim_p90": float(np.percentile(pa, 90)),
                   "leap_flip_gt30_vs_topology": float((dev > 30).mean()), "leap_n_flip": int((dev > 30).sum()),
                   "leap_median_vs_topology": float(np.median(dev)), "leap_p90_vs_topology": float(np.percentile(dev, 90)),
                   "leap_vs_family_mean_median": float(np.median(dm)), "leap_vs_family_mean_p90": float(np.percentile(dm, 90)),
                   "leap_vs_family_mean_gt30": float((dm > 30).mean())}
    eig = np.array([v["eig"] for v in R["leap"].values()])
    S["leap_eig_ratio_small_over_mid_median"] = float(np.median(eig[:, 0] / eig[:, 1]))
    S["leap_eig_ratio_mid_over_large_median"] = float(np.median(eig[:, 1] / eig[:, 2]))
    S["E_matches_library"] = [v["E_matches_lib"] for v in R["prim"].values() if v["E_matches_lib"] is not None]
    cfg = load_config(str(REPO / "configs/decoder/scratch/d1_prim50_hf.json"), [])
    bank = build_bank({**cfg, "robots": [r for r in cfg["robots"] if r["kind"] == "kite"] + [WUJI, SHADOW]}, "cpu")
    named, inv = {}, []
    rng = np.random.default_rng(0)
    for d in bank.descs:
        f, curl, pw = frame_parts(d)
        QE = rule_E(f, curl, pw); QG, ev = rule_G(f, curl, pw)
        named[d.name] = {"angle_G_vs_E_deg": round(ang(QG, QE), 1), "f_dot_dG": round(float(f @ QG[:, 2]), 3),
                         "curl_dot_nG": round(float(curl @ QG[:, 0]), 4), "palm_eig": [float(x) for x in ev],
                         "E_matches_library": float(np.abs(canonical_frame(d)[0] - QE).max())}
        G = Rotation.random(random_state=rng).as_matrix()
        f2, c2, p2 = frame_parts(relabel_base(d, G))
        QG2, _ = rule_G(f2, c2, p2)
        inv.append(float(np.abs(G @ QG2 - QG).max()))
    S["named"] = named; S["invariance_max"] = max(inv)
    S["gate"] = {"1_prim_within_30_ge_0.90": S["G"]["prim_within_30"] >= 0.90,
                 "2a_leap_flip_le_1pct": S["G"]["leap_flip_gt30_vs_topology"] <= 0.01,
                 "2b_leap_median_not_worse": S["G"]["leap_median_vs_topology"] <= S["E"]["leap_median_vs_topology"],
                 "3_named_d_and_n_signs": all(v["f_dot_dG"] > 0 and v["curl_dot_nG"] > 0 for v in named.values()),
                 "4_invariance": S["invariance_max"] < 1e-6}
    S["gate_pass"] = all(S["gate"].values())
    R["summary"] = S
    Path(a.out).write_text(json.dumps(R, indent=1))
    print(json.dumps(S, indent=1)); print("errors", len(R["errors"]), list(R["errors"].items())[:3])


if __name__ == "__main__":
    main()
