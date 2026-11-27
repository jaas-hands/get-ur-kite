"""Per-hand validation of the primitive bodies (morphology-diversity test, 2026-11-19). Same checks as
scripts/decoder/g1/validate_hands.py, with the native model being the body's own MJCF (hand.xml):
 1. decoder adapter (robots.leap_desc / RobotBank) FK == generator FK (kite_data.prim_hand) on all 4096 template points,
    at 5 random joint / wrist poses;
 2. native model: MuJoCo body frames of hand.xml == generator FK (5 random poses, every link);
 3. records (every split file present for the hand): goal joints within limits; FK at the goal pose reproduces the stored
    contacts through the decoder's RobotBank;
 4. surface coverage (hand_frame sampling): every finger subtree with >= 5% of the template has a token; covering radius;
 5. coupled joints (kite_data.prim_hand.sample_coupling; 2026-11-21): the decoder's mimic joints are exactly the
    generator's coupled joints (count = spec.n_coupled, 0 for uncoupled bodies), each follows an independent joint of
    the same finger, its limits are multiplier x the source's, and hand.xml's <equality> constraint matches.
    PYTHONPATH=src python scripts/decoder/dlevels/validate_prim.py --hands <dir> --records <examples dir> --out <json>
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests/decoder"))
from getkite.decoder.get_scratch import J_MAX  # noqa: E402
from getkite.decoder.multifamily import load_leap_records  # noqa: E402
from getkite.decoder.robots import RobotBank, leap_desc, robot_arrays  # noqa: E402
from getkite.decoder.rotations import quat_to_matrix  # noqa: E402
from kite_data.prim_hand import load_prim  # noqa: E402
from test_surface_sampling import _finger_of_slot, _hand_points  # noqa: E402


def check(hd: Path, rec_dir: Path | None, cover_max_mm: float):
    R = {}
    hand, d = load_prim(hd), leap_desc(hd)
    tpl = np.load(hd / "template.npz")
    bank = RobotBank([d])
    g = torch.Generator().manual_seed(0)
    n, err = hand.n_dof, 0.0
    for _ in range(5):
        q = hand.lower + torch.rand(n, generator=g).double().numpy() * (hand.upper - hand.lower)
        Rw = quat_to_matrix(torch.randn(1, 4, generator=g)).double()
        tw = (torch.rand(1, 3, generator=g) * 0.8 - 0.4).double()
        W = np.eye(4); W[:3, :3], W[:3, 3] = Rw[0].numpy(), tw[0].numpy()
        T = hand.fk(q, W)
        Pg = np.stack([T[hand.links[l]][:3, :3] @ p + T[hand.links[l]][:3, 3] for l, p in zip(tpl["link"], tpl["pos"])])
        qc = torch.zeros(1, J_MAX); qc[0, :n] = torch.as_tensor(q)
        h = torch.zeros(1, dtype=torch.long)
        Rr, trr = bank.fk(h, bank.expand(h, qc), Rw.float(), tw.float())
        Pr, _ = bank.points(h, Rr, trr, torch.arange(len(Pg))[None])
        err = max(err, float(np.abs(Pr[0].double().numpy() - Pg).max()))
    R["adapter_fk_max_abs_m"] = err
    m = mujoco.MjModel.from_xml_path(str(hd / "hand.xml")); dd = mujoco.MjData(m)
    rng, merr = np.random.default_rng(1), 0.0
    for _ in range(5):
        q = rng.uniform(hand.lower, hand.upper)
        for j in hand.joints:
            dd.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j.name)]] = (
                q[j.dof] if j.mimic is None else j.mimic[1] * q[j.mimic[0]] + j.mimic[2])
        mujoco.mj_kinematics(m, dd); T = hand.fk(q)
        for l in hand.links:
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, l)
            merr = max(merr, float(np.abs(dd.xpos[b] - T[l][:3, 3]).max()), float(np.abs(dd.xmat[b].reshape(3, 3) - T[l][:3, :3]).max()))
    R["mjcf_native_fk_max_abs"] = merr
    a = robot_arrays(d, "hand_frame")
    P = _hand_points(a); finger = _finger_of_slot(a.parent_slot[: int(a.n_joint)])
    f_of_pt = np.array([finger[int(s)] for s in a.tpl_anchor]); sel = set(f_of_pt[a.surf_idx].tolist())
    R["fingers_uncovered"] = [int(f) for f, c in zip(*np.unique(f_of_pt, return_counts=True)) if f != 0 and c >= 0.05 * len(P) and f not in sel]
    R["cover_radius_mm"] = float(np.sqrt(((P[:, None] - P[a.surf_idx][None]) ** 2).sum(-1)).min(1).max() * 1000)
    R["mimic_joints"] = sum(j.mimic is not None for j in d.joints)
    R["n_coupled_spec"] = hand.spec.n_coupled
    cerr = []
    byname = {j.name: j for j in hand.joints}
    for jd in d.joints:
        if jd.mimic is None:
            continue
        jg, src = byname[jd.name], byname[jd.mimic[0]]
        if jg.mimic is None or src.mimic is not None or src.finger != jg.finger or hand.dof_joints[jg.mimic[0]].name != src.name:
            cerr.append(f"{jd.name}: source mismatch")
        if abs(jd.mimic[1] - jg.mimic[1]) > 1e-9 or abs(jd.mimic[2]) > 0:
            cerr.append(f"{jd.name}: multiplier/offset mismatch")
        lo, hi = sorted((jg.mimic[1] * src.lower, jg.mimic[1] * src.upper))
        if abs(jg.lower - lo) > 1e-9 or abs(jg.upper - hi) > 1e-9:
            cerr.append(f"{jd.name}: limits are not multiplier x source limits")
    # hand.xml writes polycoef with 9 significant digits: compare joint pairs exactly, coefficients to 1e-7
    eqs = {(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.eq_obj1id[e])), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.eq_obj2id[e]))):
           (float(m.eq_data[e][0]), float(m.eq_data[e][1])) for e in range(m.neq) if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT}
    want = {(j.name, hand.dof_joints[j.mimic[0]].name): (j.mimic[2], j.mimic[1]) for j in hand.joints if j.mimic is not None}
    if set(eqs) != set(want) or any(abs(eqs[k][0] - want[k][0]) > 1e-7 or abs(eqs[k][1] - want[k][1]) > 1e-7 for k in want):
        cerr.append(f"mjcf equality mismatch: {sorted(eqs.items())} vs {sorted(want.items())}")
    R["coupling_errors"] = cerr
    R["n_dof"] = n
    R["family"] = d.family
    if rec_dir is not None:
        for f in sorted(rec_dir.glob(f"*/{hd.name}.npz")):
            rb = load_leap_records(f, _bank1(d))
            lo, hi = bank.coords_of(rb.r, bank.lower[rb.r]), bank.coords_of(rb.r, bank.upper[rb.r])
            cv = bank.coord_valid[rb.r]
            oob = int((((rb.qc2 < lo - 1e-6) | (rb.qc2 > hi + 1e-6)) & cv).any(1).sum())
            Rs, ts = bank.fk(rb.r, bank.expand(rb.r, rb.qc2), rb.R2, rb.t2)
            P2, _ = bank.points(rb.r, Rs, ts, rb.c_tpl.clamp_min(0))
            rep = float(((P2 - rb.c_pos).norm(dim=-1) * rb.c_valid).max())
            R[f"records/{f.parent.name}"] = {"n": len(rb), "goal_out_of_limits": oob, "contact_reproduction_max_m": rep}
    ok = (R["adapter_fk_max_abs_m"] < 1e-5 and R["mjcf_native_fk_max_abs"] < 1e-6 and not R["fingers_uncovered"]
          and R["cover_radius_mm"] < cover_max_mm and R["mimic_joints"] == R["n_coupled_spec"] and not R["coupling_errors"]
          and R["family"] == "prim"
          and all(v["goal_out_of_limits"] == 0 and v["contact_reproduction_max_m"] < 1e-5 for k, v in R.items() if k.startswith("records/")))
    R["pass"] = bool(ok)
    return R


_B = {}


def _bank1(d):
    """load_leap_records needs bank.index[hand_id]; a one-robot bank keyed by the hand id."""
    if d.name not in _B:
        _B.clear(); _B[d.name] = RobotBank([d])
    return _B[d.name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", required=True)
    ap.add_argument("--records", default=None)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--cover_max_mm", type=float, default=45.0, help="G1's LEAP threshold")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    hd = Path(a.hands)
    ids = a.ids or sorted(p.name for p in hd.iterdir() if (p / "hand.json").exists())
    out, ok_all = {}, True
    for i in ids:
        R = check(hd / i, Path(a.records) if a.records else None, a.cover_max_mm)
        out[i] = R; ok_all &= R["pass"]
        if not R["pass"]:
            print(i, json.dumps(R), flush=True)
    cr = np.array([r["cover_radius_mm"] for r in out.values()])
    res = {"all_pass": bool(ok_all), "n_hands": len(out), "n_fail": sum(not r["pass"] for r in out.values()),
           "n_coupled_bodies": sum(r["mimic_joints"] > 0 for r in out.values()), "n_coupled_joints": sum(r["mimic_joints"] for r in out.values()),
           "adapter_fk_max_abs_m": max(r["adapter_fk_max_abs_m"] for r in out.values()),
           "mjcf_native_fk_max_abs": max(r["mjcf_native_fk_max_abs"] for r in out.values()),
           "cover_radius_mm": {"median": float(np.median(cr)), "max": float(cr.max())},
           "contact_reproduction_max_m": max([v["contact_reproduction_max_m"] for r in out.values() for k, v in r.items() if k.startswith("records/")] or [0.0]),
           "hands": out}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "hands"}, indent=1))


if __name__ == "__main__":
    main()
