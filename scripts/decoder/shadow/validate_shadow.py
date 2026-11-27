"""Validate the reserved Shadow adapter (shadow_hand_right_extended.urdf) against the native MJCF. 2026-11-19.
Kinematic checks only; no decoder, no performance numbers (Shadow stays reserved: no training, selection or tuning).

  1 our FK path (urdf_desc -> RobotBank.expand -> RobotBank.fk) vs MuJoCo on the native MJCF (wrist fixed at 0),
    all 23 link frames (origin + 3 axis points, palm frame), at zero / lower / upper / 256 random configurations
  2 the URDF compiled by MuJoCo itself vs the native MJCF (independent of our parser), same configurations
  3 coupling: our mimic expansion gives J1 = J2; the MJCF tendon {FF,MF,RF,LF}J0 length = J1 + J2 = 2*J2 lies inside its
    actuator range for every coordinate in limits
  4 limits: our coordinate limits = MJCF joint ranges; 18 coordinates, 22 physical joints, <= J_MAX
Not yet (needs the KITE surface template): surface coverage, contact reproduction, zero-pose self-collision.

FK uses a placeholder template (4 points per link: origin, +1 cm x/y/z) written next to the URDF; it is NOT the
surface template.
    python scripts/decoder/shadow/validate_shadow.py
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from getkite.decoder.multifamily import J_MAX
from getkite.decoder.robots import RobotBank, urdf_desc

AD = Path.home() / "code/uni-kite/external_assets/shadow_adapter"
MJCF = Path.home() / "code/uni-kite/external_assets/mujoco_menagerie/shadow_hand/right_hand.xml"
URDF = AD / "shadow_hand_right_extended.urdf"
COUPLED = {f"rh_{f}J1": f"rh_{f}J2" for f in ("FF", "MF", "RF", "LF")}
OFF = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]], float)


def placeholder_template(links):
    P = np.concatenate([OFF for _ in links])
    path = AD / "fkcheck_placeholder_template.pt"
    torch.save({"link_names": list(links), "surface_template_link_indices": torch.arange(len(links)).repeat_interleave(len(OFF)),
                "surface_template_points_local": torch.tensor(P, dtype=torch.float32),
                "surface_template_normals_local": torch.tensor(np.tile([0, 0, 1.0], (len(P), 1)), dtype=torch.float32),
                "meta": {"note": "FK-check placeholder (origin + 1 cm axis points per link); NOT a surface template"}}, path)
    return path


def mj_points(m, d, qmap, links, rel_body):
    d.qpos[:] = 0
    for n, v in qmap.items():
        d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
    mujoco.mj_fwdPosition(m, d)  # kinematics + tendon lengths
    Rb, tb = d.xmat[m.body(rel_body).id].reshape(3, 3), d.xpos[m.body(rel_body).id]
    out = []
    for L in links:
        b = m.body(L).id
        R, t = d.xmat[b].reshape(3, 3), d.xpos[b]
        out.append(((R @ OFF.T).T + t - tb) @ Rb)  # palm frame
    return np.concatenate(out)


def main():
    import xml.etree.ElementTree as ET
    root = ET.parse(URDF).getroot()
    links = [l.get("name") for l in root.findall("link") if not l.get("name").startswith("virtual_")]
    tpl = placeholder_template(links)
    desc = urdf_desc(URDF, tpl, "shadow_hand_right", "shadow", coords=None, ignore_mimic=False)
    bank = RobotBank([desc], dtype=torch.float64)
    coords = desc.coords
    nc = len(coords)
    r = torch.zeros(1, dtype=torch.long)
    lo = bank.coords_of(r, bank.lower[r])[0, :nc].numpy()
    hi = bank.coords_of(r, bank.upper[r])[0, :nc].numpy()

    mj = mujoco.MjModel.from_xml_path(str(MJCF)); dj = mujoco.MjData(mj)
    spec = mujoco.MjSpec.from_file(str(URDF))  # URDF has no inertials (as KITE's): bound mass/inertia for compiling only
    spec.compiler.boundmass, spec.compiler.boundinertia = 1e-4, 1e-8
    mu = spec.compile(); du = mujoco.MjData(mu)
    res = {"n_coords": nc, "n_physical": int(bank.joint_valid[0].sum()), "J_MAX": J_MAX, "coords": coords}

    # 4 limits
    lim_err = max(max(abs(lo[i] - mj.jnt_range[mj.joint(c).id][0]), abs(hi[i] - mj.jnt_range[mj.joint(c).id][1])) for i, c in enumerate(coords))
    res["limits_max_abs_err"] = float(lim_err)
    res["coords_independent_expected"] = sorted(set(j for j in [mj.joint(k).name for k in range(mj.njnt)]
                                                    if not j.startswith("rh_WR") and j not in COUPLED)) == sorted(coords)

    rng = np.random.default_rng(0)
    Q = [np.zeros(nc), lo, hi] + [lo + rng.random(nc) * (hi - lo) for _ in range(256)]
    e_ours, e_urdf, e_mimic, ten = 0.0, 0.0, 0.0, []
    tend = {mj.tendon(t).name: t for t in range(mj.ntendon)}
    for q in Q:
        qc = torch.zeros(1, J_MAX, dtype=torch.float64); qc[0, :nc] = torch.tensor(q)
        qp = bank.expand(r, qc)
        Rs, ts = bank.fk(r, qp, torch.eye(3, dtype=torch.float64)[None], torch.zeros(1, 3, dtype=torch.float64))
        P, _ = bank.points(r, Rs, ts, torch.arange(len(links) * len(OFF))[None])
        P = P[0].numpy()
        qmap = dict(zip(coords, q)) | {k: q[coords.index(v)] for k, v in COUPLED.items()}
        ref = mj_points(mj, dj, qmap | {"rh_WRJ1": 0.0, "rh_WRJ2": 0.0}, links, "rh_palm")
        e_ours = max(e_ours, float(np.abs(P - ref).max()))
        uq = mj_points(mu, du, qmap, links, "rh_palm")
        e_urdf = max(e_urdf, float(np.abs(uq - ref).max()))
        # 3 coupling: our physical J1 equals J2; tendon length in actuator range
        names = {n: i for i, n in enumerate(coords)}
        for j1, j2 in COUPLED.items():
            # physical value of J1 from our expansion = mimic(J2)
            jp = [k for k in range(J_MAX) if int(bank.mimic_src[0, k]) >= 0 and int(bank.coord_joint[0, names[j2]]) == int(bank.mimic_src[0, k])]
            e_mimic = max(e_mimic, max(abs(float(qp[0, k]) - q[names[j2]]) for k in jp))
            ten.append(float(dj.ten_length[tend[j1.replace("J1", "J0")]]) - 2 * q[names[j2]])
    act = {mj.actuator(a).name: mj.actuator_ctrlrange[a].tolist() for a in range(mj.nu) if mj.actuator(a).name.endswith("J0")}
    res |= {"fk_ours_vs_mjcf_max_abs_m": e_ours, "fk_urdf_in_mujoco_vs_mjcf_max_abs_m": e_urdf, "mimic_J1_eq_J2_max_abs": e_mimic,
            "tendon_len_minus_2xJ2_max_abs": float(np.abs(ten).max()), "J0_actuator_ctrlrange": act,
            # MJCF: J1, J2 in [0, 1.5708] each, tendon actuator ctrlrange [0, 3.1415] (rounded pi): 2*1.5708 exceeds it by
            # ~9e-5 rad. Reported, not hidden; tolerance 1e-3 rad for the pass flag.
            "J2_range_x2_minus_actuator_max": {f: round(2 * hi[coords.index(f"rh_{f}J2")] - act[f"rh_A_{f}J0"][1], 6) for f in ("FF", "MF", "RF", "LF")},
            "n_configs": len(Q)}
    res["pass"] = bool(res["fk_ours_vs_mjcf_max_abs_m"] < 1e-6 and res["fk_urdf_in_mujoco_vs_mjcf_max_abs_m"] < 1e-6
                       and res["mimic_J1_eq_J2_max_abs"] < 1e-9 and res["tendon_len_minus_2xJ2_max_abs"] < 1e-6
                       and res["limits_max_abs_err"] < 1e-6 and res["coords_independent_expected"]
                       and res["n_physical"] <= J_MAX and max(res["J2_range_x2_minus_actuator_max"].values()) < 1e-3)
    res["pending"] = ["4096-point KITE surface template", "collision primitives", "surface coverage (every finger, covering radius < 45 mm)",
                      "contact reproduction from KITE-recipe records", "zero-pose self-collision", "500 eval-only goals + cross-body example bank"]
    (AD / "validation.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
