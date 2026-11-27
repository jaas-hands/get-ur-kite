"""Shadow Hand E3M5: surface template and 500 evaluation-only records (2026-11-19). Shadow is the reserved external target.
These records are used for evaluation only, never for training, selection or tuning.

Shadow is not in KITE's robot registry, so KITE's own sampler cannot be used. The records follow kite_data's kite_code
regime instead. That regime is KITE's released recipe, the same one used for the LEAP and primitive bodies:
  * contact pose: wrist translation U[-0.4, 0.4]^3, rotation uniform on SO(3);
  * joints: KITE's mixture distribution over each coordinate's range, with J1 = J2 (the adapter's mimic coupling);
  * self-colliding poses are rejected (MuJoCo, Menagerie's collision geoms, wrist fixed at 0);
  * contacts: K ~ U{2..6}, each a template point on a uniformly chosen region (palm, ff, mf, rf, lf, th),
    with no repeated points;
  * joint_active: KITE's contact-relevant mask over the 18 coordinates.

Template: 4096 area-weighted points on the adapter URDF's link meshes (Menagerie visual meshes), with outward normals.
Visibility-filtered per part as in kite_data.template: a point is kept if >= 8 of 32 hemisphere rays escape its own
mesh. FPS-32 indices are taken at the zero pose.

Outputs, next to the adapter:
  * shadow_surface_template_n4096.pt (KITE template keys, read by robots.urdf_desc);
  * shadow_eval.npz (KITE record schema, read by multifamily.load_kite_records);
  * shadow_records_check.json (zero-pose collision, rejection rate, contact reproduction via the decoder's FK,
    FK vs MuJoCo, coverage).
    PYTHONPATH=src python scripts/decoder/shadow/make_shadow_records.py
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
import os
from pathlib import Path

import mujoco
import numpy as np
import torch
import trimesh

from getkite.decoder.multifamily import J_MAX, load_kite_records
from getkite.decoder.robots import RobotBank, robot_arrays, urdf_desc
from getkite.decoder.rotations import euler_xyz_to_matrix, matrix_to_euler_xyz
from kite_data.sample import kite_mixture_unit
from kite_data.template import farthest_point_sampling
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
EXT_ROOT = Path(os.environ.get("GETKITE_EXTERNAL", _REPO.parent / "external_assets"))  # third-party hand assets (docs/external_hands.md)

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests/decoder"))
from test_surface_sampling import _finger_of_slot, _hand_points  # noqa: E402

AD = EXT_ROOT / "shadow_adapter"
MJCF = EXT_ROOT / "mujoco_menagerie/shadow_hand/right_hand.xml"
URDF = AD / "shadow_hand_right_extended.urdf"
TPL = AD / "shadow_surface_template_n4096.pt"
OUT = AD / "shadow_eval.npz"
N, SEED, K_SLOTS, PEN_TOL = 500, 20261120, 6, 1e-4
REGION_OF = lambda link: next((r for r in ("ff", "mf", "rf", "lf", "th") if link.startswith(f"rh_{r}")), "palm")  # noqa: E731
REGIONS = ("palm", "ff", "mf", "rf", "lf", "th")


def build_template(seed=0):
    root = ET.parse(URDF).getroot()
    pts, nrm, wts, lnk, names = [], [], [], [], []
    rng = np.random.default_rng(seed)
    dirs = None
    for L in root.findall("link"):
        v = L.find("visual")
        if v is None:
            continue
        names.append(L.get("name"))
        o = v.find("origin"); xyz = [float(x) for x in o.get("xyz").split()]; rpy = [float(x) for x in o.get("rpy").split()]
        assert rpy == [0, 0, 0]
        mesh = trimesh.load(AD / v.find("geometry/mesh").get("filename"), force="mesh", process=False)
        mesh.apply_scale(0.001); mesh.apply_translation(xyz)
        n = 20000
        P, fi = trimesh.sample.sample_surface(mesh, n, seed=int(rng.integers(1 << 31)))
        Nn = mesh.face_normals[fi]
        if dirs is None:
            i = np.arange(32) + 0.5; phi = np.arccos(1 - 2 * i / 32); th = np.pi * (1 + 5 ** 0.5) * i
            dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
        flip = np.where(Nn @ dirs.T < 0, -1.0, 1.0)
        hit = mesh.ray.intersects_any(np.repeat(P + 1e-5 * Nn, 32, 0), (flip[:, :, None] * dirs[None]).reshape(-1, 3)).reshape(n, 32)
        keep = (~hit).sum(1) >= 8
        pts.append(P[keep]); nrm.append(Nn[keep]); wts.append(np.full(keep.sum(), mesh.area / n)); lnk.append(np.full(keep.sum(), len(names) - 1))
    P, Nn, W, Lk = map(np.concatenate, (pts, nrm, wts, lnk))
    pick = np.sort(rng.choice(len(P), size=4096, replace=False, p=W / W.sum()))
    return names, Lk[pick], P[pick], Nn[pick]


class ShadowCollider:
    """Menagerie MJCF, wrist joints fixed at 0; pairs touching at the zero pose (palm/knuckle bases) are ignored."""

    def __init__(self, desc):
        self.m = mujoco.MjModel.from_xml_path(str(MJCF)); self.d = mujoco.MjData(self.m)
        self.coords = desc.coords
        self.mimic = {j.name: j.mimic for j in desc.joints if j.mimic is not None}
        self.adr = {j.name: self.m.jnt_qposadr[self.m.joint(j.name).id] for j in desc.joints if j.type != "fixed"}
        self.body = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) for b in self.m.geom_bodyid]
        self.ignore = set()
        self.ignore = self.pairs(np.zeros(len(self.coords)))

    def set(self, qc):
        self.d.qpos[:] = 0
        v = dict(zip(self.coords, qc))
        for n, (src, mult, off) in self.mimic.items():
            v[n] = v[src] * mult + off
        for n, x in v.items():
            self.d.qpos[self.adr[n]] = x

    def pairs(self, qc):
        self.set(qc); mujoco.mj_kinematics(self.m, self.d); mujoco.mj_collision(self.m, self.d)
        out = set()
        for c in self.d.contact[: self.d.ncon]:
            a, b = self.body[c.geom1], self.body[c.geom2]
            if c.dist < -PEN_TOL and "forearm" not in a + b and "wrist" not in a + b:
                p = tuple(sorted((a, b)))
                if p not in self.ignore:
                    out.add(p)
        return out


def main():
    names, Lk, P, Nn = build_template()
    torch.save({"link_names": names, "surface_template_link_indices": torch.as_tensor(Lk),
                "surface_template_points_local": torch.as_tensor(P, dtype=torch.float32),
                "surface_template_normals_local": torch.as_tensor(Nn, dtype=torch.float32),
                "meta": {"note": "4096-point visibility-filtered template on the adapter URDF meshes (make_shadow_records.py)"}}, TPL)
    desc = urdf_desc(URDF, TPL, "shadow_hand_right", "shadow")
    bank = RobotBank([desc])
    col = ShadowCollider(desc)
    rng = np.random.default_rng(SEED)
    lo, hi = bank.coords_of(torch.zeros(1, dtype=torch.long), bank.lower[[0]])[0, :len(desc.coords)].numpy(), \
        bank.coords_of(torch.zeros(1, dtype=torch.long), bank.upper[[0]])[0, :len(desc.coords)].numpy()
    qs, tried = [], 0
    while len(qs) < N:
        t = kite_mixture_unit(64, len(desc.coords), rng)
        for q in lo + t * (hi - lo):
            tried += 1
            if not col.pairs(q):
                qs.append(q)
                if len(qs) == N:
                    break
    q2 = np.array(qs)
    t2 = rng.uniform(-0.4, 0.4, size=(N, 3))
    qq = rng.normal(size=(N, 4)); qq /= np.linalg.norm(qq, axis=1, keepdims=True)
    from scipy.spatial.transform import Rotation
    R2 = Rotation.from_quat(qq).as_matrix()
    e2 = matrix_to_euler_xyz(torch.as_tensor(R2)).numpy()
    R2 = euler_xyz_to_matrix(torch.as_tensor(e2)).numpy()  # exactly what the loader reconstructs
    # contacts: region uniformly among the non-empty regions, then a point; no repeats
    reg = np.array([REGIONS.index(REGION_OF(names[l])) for l in Lk])
    by_reg = [np.flatnonzero(reg == r) for r in range(len(REGIONS))]
    k = rng.integers(2, K_SLOTS + 1, size=N)
    tidx = np.full((N, K_SLOTS), -1)
    for i in range(N):
        seen = set()
        for s in range(k[i]):
            while True:
                pts = by_reg[int(rng.integers(len(REGIONS)))]
                c = int(pts[int(rng.random() * len(pts))])
                if c not in seen:
                    seen.add(c); tidx[i, s] = c; break
    r = torch.zeros(N, dtype=torch.long)
    qc = torch.zeros(N, J_MAX, dtype=torch.float32); qc[:, :len(desc.coords)] = torch.as_tensor(q2, dtype=torch.float32)
    Rs, ts = bank.fk(r, bank.expand(r, qc), torch.as_tensor(R2, dtype=torch.float32), torch.as_tensor(t2, dtype=torch.float32))
    Pw, Nw = bank.points(r, Rs, ts, torch.as_tensor(np.maximum(tidx, 0)))
    valid = tidx >= 0
    # contact-relevant mask over coordinates: a coordinate drives a physical joint upstream of a contacted link
    jparent = {j.child: j for j in desc.joints}
    coord_of = {j.name: (j.mimic[0] if j.mimic else j.name) for j in desc.joints if j.type != "fixed"}
    def up_coords(link):
        out = set()
        while link in jparent:
            j = jparent[link]
            if j.type != "fixed":
                out.add(desc.coords.index(coord_of[j.name]))
            link = j.parent
        return out
    link_up = [up_coords(n) for n in names]
    active = np.zeros((N, len(desc.coords)), bool)
    for i in range(N):
        for c in tidx[i][valid[i]]:
            active[i, list(link_up[Lk[c]])] = True
    rec = {"q2_t": t2, "q2_euler": e2, "q2_joints": q2, "joint_active": active,
           "c_pos": np.where(valid[..., None], Pw.double().numpy(), 0.0), "c_normal": np.where(valid[..., None], Nw.double().numpy(), 0.0),
           "c_valid": valid, "c_tpl": tidx}
    meta = {"robot": "shadow_hand_right", "seed": SEED, "n": N, "coords": desc.coords,
            "sampler": "kite_data kite_code regime re-implemented for the Shadow adapter (make_shadow_records.py); not KITE's own sampler",
            "rotation_convention": "intrinsic_xyz_euler", "status": "reserved external target: evaluation only"}
    np.savez_compressed(OUT, **rec, meta=json.dumps(meta))
    # checks
    bank2 = RobotBank([desc]); bank2.index = {"shadow_hand_right": 0} if not hasattr(bank2, "index") else bank2.index
    R = load_kite_records(OUT, bank2, "shadow_hand_right")
    Rs2, ts2 = bank2.fk(R.r, bank2.expand(R.r, R.qc2), R.R2, R.t2)
    P2, _ = bank2.points(R.r, Rs2, ts2, R.c_tpl.clamp_min(0))
    rep = float(((P2 - R.c_pos).norm(dim=-1) * R.c_valid).max())
    # FK vs MuJoCo on template points' links (palm frame), 5 stored poses
    ferr = 0.0
    for i in range(5):
        col.set(q2[i]); mujoco.mj_kinematics(col.m, col.d)
        bp = col.m.body("rh_palm").id; Rb, tb = col.d.xmat[bp].reshape(3, 3), col.d.xpos[bp]
        qci = qc[[i]]; Ri, ti = bank.fk(r[[i]], bank.expand(r[[i]], qci), torch.eye(3)[None], torch.zeros(1, 3))
        Pi, _ = bank.points(r[[i]], Ri, ti, torch.arange(4096)[None])
        for li, n in enumerate(names):
            sel = np.flatnonzero(Lk == li)[:20]
            b = col.m.body(n).id
            Pm = ((P[sel] @ col.d.xmat[b].reshape(3, 3).T + col.d.xpos[b]) - tb) @ Rb
            ferr = max(ferr, float(np.abs(Pm - Pi[0, sel].double().numpy()).max()))
    a = robot_arrays(desc, "hand_frame")
    Ph = _hand_points(a); finger = _finger_of_slot(a.parent_slot[: int(a.n_joint)])
    f_of_pt = np.array([finger[int(s)] for s in a.tpl_anchor]); covered = set(f_of_pt[a.surf_idx].tolist())
    chk = {"n": N, "accept_rate": round(N / tried, 4), "zero_pose_ignored_pairs": sorted(map(list, col.ignore)),
           "contact_reproduction_max_m": rep, "fk_decoder_vs_mujoco_max_abs_m": ferr,
           "goal_out_of_limits": int(((q2 < lo - 1e-6) | (q2 > hi + 1e-6)).any(1).sum()),
           "template_points_per_region": {g: int((np.array([REGION_OF(names[l]) for l in Lk]) == g).sum()) for g in REGIONS},
           "fingers_uncovered": [int(f) for f, c in zip(*np.unique(f_of_pt, return_counts=True)) if f != 0 and c >= 0.05 * len(Ph) and f not in covered],
           "cover_radius_mm": float(np.sqrt(((Ph[:, None] - Ph[a.surf_idx][None]) ** 2).sum(-1)).min(1).max() * 1000),
           "mean_k": float(k.mean()), "active_frac": float(active.mean())}
    chk["pass"] = bool(rep < 1e-5 and ferr < 1e-5 and chk["goal_out_of_limits"] == 0 and not chk["fingers_uncovered"])
    (AD / "shadow_records_check.json").write_text(json.dumps(chk, indent=1))
    print(json.dumps(chk, indent=1))


if __name__ == "__main__":
    main()
