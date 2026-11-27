"""External evaluation hands (2026-11-26): MJCF -> KITE-style *_extended URDF, surface template and 500 evaluation-only records.

Generalises scripts/decoder/shadow/ (make_shadow_urdf.py + make_shadow_records.py) to other public hand models, so the decoder
can be scored on more hand families that neither KITE's encoder (psi: Wuji, Allegro, Barrett, Robotiq gripper, MANO) nor any
decoder has seen. Evaluation only: never used for training, selection or tuning.

Sources (downloaded outside the KITE checkout, ~/code/uni-kite/external_assets):
  * SPIDER (facebookresearch/spider, CC-BY-NC): ability, inspire, xhand, schunk, metahand (right.xml);
  * MuJoCo Menagerie (Apache-2.0): sharpa_wave, tetheria_aero_hand_open (shadow_dexee dropped: its fingers are not under a palm body).

Adapter rules (no tuning on performance; same as the Shadow adapter where it applies):
  * floating base = the hand's root body (HANDS[name]["root"]); every body above it (virtual forearm / mount) is dropped and its
    joints removed; joints ON the root body (some files put the 6 virtual joints there) are removed too. Six virtual_joint_* ahead of
    the root, as KITE's URDFs.
  * kinematics from the compiled MJCF (body pos/quat, hinge axis/range). A hinge not at its body origin moves the link frame to the
    hinge (child geometry and child bodies shifted accordingly), so URDF joints sit on their axes.
  * couplings: MJCF joint equality constraints q1 = c0 + c1 q2 become <mimic joint=q2 multiplier=c1 offset=c0> on q1. Tendons are
    actuation only and ignored. Hands whose files model coupled joints as independent (SPIDER) stay independent, as KITE does for Barrett.
  * geometry per link: mesh geoms of the visual group (2) if present, else every mesh geom, else primitives; written as one OBJ per
    link in the link frame (compiled, scaled vertices).
Records (kite_code regime, as Shadow): wrist translation U[-0.4, 0.4]^3, rotation uniform on SO(3); joints = KITE's mixture
distribution over each coordinate's range; self-colliding poses rejected with MuJoCo using, for the SPIDER hands, KITE's own fitted
collision primitives (KITE/assets/collision_primitives/spider__<hand>__right.json, fitted on the same SPIDER files; read only) and, for
the Menagerie hands, their collision geoms (group 3); pairs touching at the zero pose ignored; K ~ U{2..6} contacts, each a template point on a uniformly chosen region
(palm = root + rigidly attached links; one region per finger subtree), no repeats; contact-relevant joint mask.

    PYTHONPATH=src python scripts/decoder/external_hands/make_hand_adapter.py --hand xhand [--hand ...] [--n 500]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as Rot

from getkite.decoder.multifamily import J_MAX, load_kite_records
from getkite.decoder.robots import RobotBank, robot_arrays, urdf_desc
from getkite.decoder.rotations import euler_xyz_to_matrix, matrix_to_euler_xyz
from kite_data.sample import kite_mixture_unit

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests/decoder"))
from test_surface_sampling import _finger_of_slot, _hand_points  # noqa: E402

EXT = Path.home() / "code/uni-kite/external_assets"
SP = EXT / "spider_repo/spider/assets/robots"
MN = EXT / "mujoco_menagerie"
KP = Path.home() / "code/uni-kite/KITE/assets/collision_primitives"  # KITE's fitted self-collision primitives (read only)
HANDS = {
    "ability": {"xml": SP / "ability/right.xml", "root": "right_base", "license": "CC-BY-NC (SPIDER)", "collision": "kite"},
    "inspire": {"xml": SP / "inspire/right.xml", "root": "right_hand_base", "license": "CC-BY-NC (SPIDER)", "collision": "kite"},
    "xhand": {"xml": SP / "xhand/right.xml", "root": "right_hand_link", "license": "CC-BY-NC (SPIDER)", "collision": "kite"},
    "schunk": {"xml": SP / "schunk/right.xml", "root": "right_hand_base_link", "license": "CC-BY-NC (SPIDER)", "collision": "kite"},
    "metahand": {"xml": SP / "metahand/right.xml", "root": "right_palm", "license": "CC-BY-NC (SPIDER)", "collision": "kite"},
    "sharpa": {"xml": MN / "sharpa_wave/right_hand.xml", "root": "right_hand_C_MC", "license": "Apache-2.0 (Menagerie)", "collision": "group3"},
    "aero": {"xml": MN / "tetheria_aero_hand_open/right_hand.xml", "root": "palm", "license": "Apache-2.0 (Menagerie)", "collision": "group3"},
}
SEED0, K_SLOTS, PEN_TOL = 20261126, 6, 1e-4
VIRTUAL = [("x", "prismatic", "1 0 0", -2, 2), ("y", "prismatic", "0 1 0", -2, 2), ("z", "prismatic", "0 0 1", -2, 2),
           ("roll", "revolute", "1 0 0", -3.14159265, 3.14159265), ("pitch", "revolute", "0 1 0", -3.14159265, 3.14159265),
           ("yaw", "revolute", "0 0 1", -3.14159265, 3.14159265)]


def f(x):
    return " ".join(f"{v:.10g}" for v in x)


def qmat(q):
    return Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def rpy_of(q):
    return Rot.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")


def geom_mesh(m, g):
    t = m.geom_type[g]; s = m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        i = m.geom_dataid[g]; va, vn = m.mesh_vertadr[i], m.mesh_vertnum[i]; fa, fn = m.mesh_faceadr[i], m.mesh_facenum[i]
        return trimesh.Trimesh(m.mesh_vert[va:va + vn].copy(), m.mesh_face[fa:fa + fn].copy(), process=False)
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return trimesh.creation.box(extents=2 * s)
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=3, radius=s[0])
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return trimesh.creation.capsule(height=2 * s[1], radius=s[0])
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(radius=s[0], height=2 * s[1])
    if t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        e = trimesh.creation.icosphere(subdivisions=3, radius=1.0); e.apply_scale(s); return e
    return None


def surface_geoms(m, b):
    gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
    mesh = [g for g in gs if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    vis = [g for g in mesh if m.geom_group[g] == 2]
    return vis or mesh or [g for g in gs if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE]


def collision_geoms(m, sub):
    """Collision group 3 if it covers every link that has surface geometry; otherwise (SPIDER files keep group-3 geoms on the
    fingertips only, DEX-EE on two links) the surface geoms themselves (MuJoCo collides their convex hulls: conservative)."""
    gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] in sub]
    c3 = [g for g in gs if m.geom_group[g] == 3]
    surf = [g for b in sub for g in surface_geoms(m, b)]
    if c3 and {m.geom_bodyid[g] for g in surf} <= {m.geom_bodyid[g] for g in c3}:
        return c3
    return surf


def build_adapter(name, out):
    H = HANDS[name]
    m = mujoco.MjModel.from_xml_path(str(H["xml"]))
    root = m.body(H["root"]).id
    anc = lambda b: (lambda s: s)(_anc(m, b))  # noqa: E731
    sub = [b for b in range(m.nbody) if b == root or root in anc(b)]
    eq = {}
    for i in range(m.neq):
        if m.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT and m.eq_active0[i]:
            c = m.eq_data[i][:5]
            assert np.allclose(c[2:], 0), f"{name}: nonlinear joint coupling"
            eq[m.joint(m.eq_obj1id[i]).name] = (m.joint(m.eq_obj2id[i]).name, float(c[1]), float(c[0]))
    shift, jinfo = {}, {}
    for b in sub:
        js = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b]
        if b == root:
            shift[b] = np.zeros(3); jinfo[b] = None; continue
        assert len(js) <= 1, f"{name}: body {m.body(b).name} has {len(js)} joints"
        if js:
            j = js[0]
            assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE, f"{name}: {m.joint(j).name} not a hinge"
            assert m.jnt_limited[j], f"{name}: {m.joint(j).name} unlimited"
            shift[b] = m.jnt_pos[j].copy(); jinfo[b] = j
        else:
            shift[b] = np.zeros(3); jinfo[b] = None
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    L, J = [], []
    lname = lambda b: m.body(b).name.replace("/", "_") or f"body{b}"  # noqa: E731
    for b in sub:
        parts = []
        for g in surface_geoms(m, b):
            tm = geom_mesh(m, g)
            if tm is None:
                continue
            T = np.eye(4); T[:3, :3] = qmat(m.geom_quat[g]); T[:3, 3] = m.geom_pos[g] - shift[b]
            tm.apply_transform(T); parts.append(tm)
        if parts:
            mesh = trimesh.util.concatenate(parts)
            fn = f"meshes/{lname(b)}.obj"; mesh.export(out / fn)
            g = f'<origin xyz="0 0 0" rpy="0 0 0" /><geometry><mesh filename="{fn}" /></geometry>'
            L.append(f'  <link name="{lname(b)}">\n    <visual>{g}</visual>\n    <collision>{g}</collision>\n  </link>')
        else:
            L.append(f'  <link name="{lname(b)}" />')
        if b == root:
            continue
        p = m.body_parentid[b]
        xyz = m.body_pos[b] + qmat(m.body_quat[b]) @ shift[b] - shift[p]
        org = f'<origin xyz="{f(xyz)}" rpy="{f(rpy_of(m.body_quat[b]))}" />'
        j = jinfo[b]
        if j is None:
            J.append(f'  <joint name="{lname(b)}_fixed" type="fixed">{org}<parent link="{lname(p)}" /><child link="{lname(b)}" /></joint>')
            continue
        jn = m.joint(j).name
        mim = f'<mimic joint="{eq[jn][0]}" multiplier="{eq[jn][1]:.10g}" offset="{eq[jn][2]:.10g}" />' if jn in eq else ""
        J.append(f'  <joint name="{jn}" type="revolute">{org}<parent link="{lname(p)}" /><child link="{lname(b)}" />'
                 f'<axis xyz="{f(m.jnt_axis[j])}" /><limit lower="{m.jnt_range[j][0]:.10g}" upper="{m.jnt_range[j][1]:.10g}" '
                 f'effort="10" velocity="10" />{mim}</joint>')
    V = ['  <link name="virtual_base_link" />'] + [f'  <link name="virtual_joint_{n}_link" />' for n, *_ in VIRTUAL[:-1]]
    prev = "virtual_base_link"
    for n, typ, ax, lo, hi in VIRTUAL:
        child = lname(root) if n == "yaw" else f"virtual_joint_{n}_link"
        V.append(f'  <joint name="virtual_joint_{n}" type="{typ}"><origin xyz="0 0 0" rpy="0 0 0" /><parent link="{prev}" />'
                 f'<child link="{child}" /><axis xyz="{ax}" /><limit lower="{lo}" upper="{hi}" effort="1000" velocity="1000" /></joint>')
        prev = child
    urdf = out / f"{name}_right_extended.urdf"
    urdf.write_text(f'<?xml version="1.0" encoding="utf-8"?>\n<robot name="{name}_right">\n' + "\n".join(V + L + J) + "\n</robot>\n")
    return m, root, sub, shift, eq, urdf, lname


def _anc(m, b):
    out = []
    while b != 0:
        b = m.body_parentid[b]; out.append(b)
    return out


def build_template(urdf, out, seed):
    import xml.etree.ElementTree as ET
    rootx = ET.parse(urdf).getroot()
    pts, nrm, wts, lnk, names = [], [], [], [], []
    rng = np.random.default_rng(seed)
    i = np.arange(32) + 0.5; phi = np.arccos(1 - 2 * i / 32); th = np.pi * (1 + 5 ** 0.5) * i
    dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    for Lx in rootx.findall("link"):
        v = Lx.find("visual")
        if v is None:
            continue
        names.append(Lx.get("name"))
        mesh = trimesh.load(out / v.find("geometry/mesh").get("filename"), force="mesh", process=False)
        n = 20000
        P, fi = trimesh.sample.sample_surface(mesh, n, seed=int(rng.integers(1 << 31)))
        Nn = mesh.face_normals[fi]
        flip = np.where(Nn @ dirs.T < 0, -1.0, 1.0)
        hit = mesh.ray.intersects_any(np.repeat(P + 1e-5 * Nn, 32, 0), (flip[:, :, None] * dirs[None]).reshape(-1, 3)).reshape(n, 32)
        keep = (~hit).sum(1) >= 8
        pts.append(P[keep]); nrm.append(Nn[keep]); wts.append(np.full(keep.sum(), mesh.area / n)); lnk.append(np.full(keep.sum(), len(names) - 1))
    P, Nn, W, Lk = map(np.concatenate, (pts, nrm, wts, lnk))
    pick = np.sort(rng.choice(len(P), size=4096, replace=False, p=W / W.sum()))
    return names, Lk[pick], P[pick], Nn[pick]


class Collider:
    """The source MJCF with only the hand subtree's collision geoms enabled (all against all; MuJoCo skips parent-child pairs);
    the root's / forearm's joints stay at 0; pairs touching at the zero pose are ignored."""

    def __init__(self, m, sub, desc, lname, xml, mode, hand):
        spec = mujoco.MjSpec.from_file(str(xml))
        assert len(spec.geoms) == m.ngeom
        if mode == "group3":
            cg = {g for g in range(m.ngeom) if m.geom_bodyid[g] in set(sub) and m.geom_group[g] == 3}
            for g, sg in enumerate(spec.geoms):
                sg.contype = 1 if g in cg else 0; sg.conaffinity = 1 if g in cg else 0
            self.source = "Menagerie collision geoms (group 3)"
        else:  # KITE's primitives: rounded boxes (box, half extents + radius) and capsules (axis z), pose = link_from_primitive
            for sg in spec.geoms:
                sg.contype = 0; sg.conaffinity = 0
            P = json.load(open(KP / f"spider__{hand}__right.json"))["primitives"]
            for pr in P:
                T = np.array(pr["link_from_primitive"]); q = Rot.from_matrix(T[:3, :3]).as_quat()
                if pr["type"] == "rounded_box":
                    gt, size = mujoco.mjtGeom.mjGEOM_BOX, np.array(pr["params"]["half_extents"]) + pr["params"]["radius"]
                else:
                    assert pr["type"] == "capsule", pr["type"]
                    gt, size = mujoco.mjtGeom.mjGEOM_CAPSULE, np.array([pr["params"]["radius"], pr["params"]["half_length"], 0.0])
                spec.body(pr["link_name"]).add_geom(type=gt, size=size, pos=T[:3, 3], quat=[q[3], q[0], q[1], q[2]],
                                                    contype=1, conaffinity=1, group=4)
            self.source = f"KITE collision primitives spider__{hand}__right.json ({len(P)})"
        m = spec.compile()
        self.m = m; self.d = mujoco.MjData(m)
        self.coords = desc.coords
        self.mimic = {j.name: j.mimic for j in desc.joints if j.mimic is not None}
        self.adr = {j.name: m.jnt_qposadr[m.joint(j.name).id] for j in desc.joints if j.type != "fixed"}
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
            if c.dist < -PEN_TOL:
                p = tuple(sorted((int(self.m.geom_bodyid[c.geom1]), int(self.m.geom_bodyid[c.geom2]))))
                if p not in self.ignore:
                    out.add(p)
        return out


def make(name, n_rec):
    out = EXT / f"{name}_adapter"; out.mkdir(parents=True, exist_ok=True)
    m, root, sub, shift, eq, urdf, lname = build_adapter(name, out)
    seed = SEED0 + list(HANDS).index(name)
    names, Lk, P, Nn = build_template(urdf, out, seed)
    tpl = out / f"{name}_surface_template_n4096.pt"
    torch.save({"link_names": names, "surface_template_link_indices": torch.as_tensor(Lk),
                "surface_template_points_local": torch.as_tensor(P, dtype=torch.float32),
                "surface_template_normals_local": torch.as_tensor(Nn, dtype=torch.float32),
                "meta": {"note": "4096-point visibility-filtered template on the adapter meshes (external_hands/make_hand_adapter.py)"}}, tpl)
    robot = f"{name}_right"
    desc = urdf_desc(urdf, tpl, robot, name)
    bank = RobotBank([desc])
    col = Collider(m, sub, desc, lname, HANDS[name]["xml"], HANDS[name]["collision"], name)
    rng = np.random.default_rng(seed)
    r0 = torch.zeros(1, dtype=torch.long)
    nc = len(desc.coords)
    lo = bank.coords_of(r0, bank.lower[[0]])[0, :nc].numpy(); hi = bank.coords_of(r0, bank.upper[[0]])[0, :nc].numpy()
    qs, tried = [], 0
    while len(qs) < n_rec:
        t = kite_mixture_unit(64, nc, rng)
        for q in lo + t * (hi - lo):
            tried += 1
            if not col.pairs(q):
                qs.append(q)
                if len(qs) == n_rec:
                    break
        if tried > 200 * n_rec:
            raise RuntimeError(f"{name}: acceptance too low ({len(qs)}/{tried})")
    q2 = np.array(qs)
    t2 = rng.uniform(-0.4, 0.4, size=(n_rec, 3))
    qq = rng.normal(size=(n_rec, 4)); qq /= np.linalg.norm(qq, axis=1, keepdims=True)
    e2 = matrix_to_euler_xyz(torch.as_tensor(Rot.from_quat(qq).as_matrix())).numpy()
    R2 = euler_xyz_to_matrix(torch.as_tensor(e2)).numpy()
    # regions: palm = root + links reached from it through fixed joints only; each finger = a subtree hanging off the palm
    parent = {j.child: j for j in desc.joints}
    def region(link):
        chain = [link]
        while chain[-1] in parent:
            chain.append(parent[chain[-1]].parent)
        chain = chain[::-1]  # root ... link
        for a, b in zip(chain, chain[1:]):
            if parent[b].type != "fixed":
                return b  # the first link below a moving joint names the finger
        return "palm"
    regs = sorted({region(names[l]) for l in Lk}, key=lambda x: (x != "palm", x))
    reg = np.array([regs.index(region(names[l])) for l in Lk])
    by_reg = [np.flatnonzero(reg == r) for r in range(len(regs))]
    k = rng.integers(2, K_SLOTS + 1, size=n_rec)
    tidx = np.full((n_rec, K_SLOTS), -1)
    for i in range(n_rec):
        seen = set()
        for s in range(k[i]):
            while True:
                pts = by_reg[int(rng.integers(len(regs)))]
                c = int(pts[int(rng.random() * len(pts))])
                if c not in seen:
                    seen.add(c); tidx[i, s] = c; break
    r = torch.zeros(n_rec, dtype=torch.long)
    qc = torch.zeros(n_rec, J_MAX, dtype=torch.float32); qc[:, :nc] = torch.as_tensor(q2, dtype=torch.float32)
    Rs, ts = bank.fk(r, bank.expand(r, qc), torch.as_tensor(R2, dtype=torch.float32), torch.as_tensor(t2, dtype=torch.float32))
    Pw, Nw = bank.points(r, Rs, ts, torch.as_tensor(np.maximum(tidx, 0)))
    valid = tidx >= 0
    coord_of = {j.name: (j.mimic[0] if j.mimic else j.name) for j in desc.joints if j.type != "fixed"}
    def up_coords(link):
        s = set()
        while link in parent:
            j = parent[link]
            if j.type != "fixed":
                s.add(desc.coords.index(coord_of[j.name]))
            link = j.parent
        return s
    link_up = [up_coords(x) for x in names]
    active = np.zeros((n_rec, nc), bool)
    for i in range(n_rec):
        for c in tidx[i][valid[i]]:
            active[i, list(link_up[Lk[c]])] = True
    rec = {"q2_t": t2, "q2_euler": e2, "q2_joints": q2, "joint_active": active,
           "c_pos": np.where(valid[..., None], Pw.double().numpy(), 0.0), "c_normal": np.where(valid[..., None], Nw.double().numpy(), 0.0),
           "c_valid": valid, "c_tpl": tidx}
    npz = out / f"{name}_eval.npz"
    meta = {"robot": robot, "seed": seed, "n": n_rec, "coords": desc.coords, "rotation_convention": "intrinsic_xyz_euler",
            "sampler": "kite_data kite_code regime re-implemented (external_hands/make_hand_adapter.py); not KITE's own sampler",
            "status": "external evaluation hand: evaluation only"}
    np.savez_compressed(npz, **rec, meta=json.dumps(meta))
    # checks: record round trip; decoder FK vs MuJoCo on template points (root frame); coverage
    R = load_kite_records(npz, bank, robot)
    Rs2, ts2 = bank.fk(R.r, bank.expand(R.r, R.qc2), R.R2, R.t2)
    P2, _ = bank.points(R.r, Rs2, ts2, R.c_tpl.clamp_min(0))
    rep = float(((P2 - R.c_pos).norm(dim=-1) * R.c_valid).max())
    body_of = {lname(b): b for b in sub}
    ferr = 0.0
    for i in range(5):
        col.set(q2[i]); mujoco.mj_kinematics(col.m, col.d)
        Rb, tb = col.d.xmat[root].reshape(3, 3), col.d.xpos[root]
        Ri, ti = bank.fk(r[[i]], bank.expand(r[[i]], qc[[i]]), torch.eye(3)[None], torch.zeros(1, 3))
        Pi, _ = bank.points(r[[i]], Ri, ti, torch.arange(4096)[None])
        for li, nm in enumerate(names):
            sel = np.flatnonzero(Lk == li)[:20]
            if not len(sel):
                continue
            b = body_of[nm]
            Pm = (((P[sel] + shift[b]) @ col.d.xmat[b].reshape(3, 3).T + col.d.xpos[b]) - tb) @ Rb
            ferr = max(ferr, float(np.abs(Pm - Pi[0, sel].double().numpy()).max()))
    encl = None
    if HANDS[name]["collision"] == "kite":  # do KITE's primitives sit on this file's links? fraction of template points within 3 mm
        prim = json.load(open(KP / f"spider__{name}__right.json"))["primitives"]
        ok = np.zeros(len(P), bool)
        for pr in prim:
            if pr["link_name"] not in names:
                continue
            sel = np.flatnonzero(Lk == names.index(pr["link_name"]))
            T = np.array(pr["link_from_primitive"]); pb = (P[sel] + shift[body_of[pr["link_name"]]] - T[:3, 3]) @ T[:3, :3]
            if pr["type"] == "rounded_box":
                d_ = np.linalg.norm(np.maximum(np.abs(pb) - np.array(pr["params"]["half_extents"]), 0), axis=1) - pr["params"]["radius"]
            else:
                z = np.clip(pb[:, 2], -pr["params"]["half_length"], pr["params"]["half_length"])
                d_ = np.linalg.norm(pb - np.stack([0 * z, 0 * z, z], 1), axis=1) - pr["params"]["radius"]
            ok[sel] |= d_ <= 3e-3
        encl = float(ok.mean())
    a = robot_arrays(desc, "hand_frame")
    Ph = _hand_points(a); finger = _finger_of_slot(a.parent_slot[: int(a.n_joint)])
    f_of_pt = np.array([finger[int(s)] for s in a.tpl_anchor]); covered = set(f_of_pt[a.surf_idx].tolist())
    ext = P.max(0) - P.min(0)
    chk = {"hand": name, "robot": robot, "source": str(HANDS[name]["xml"]), "license": HANDS[name]["license"], "root": HANDS[name]["root"],
           "n_links": len(sub), "n_coords": nc, "n_physical_joints": sum(j.type != "fixed" for j in desc.joints),
           "couplings": {k_: {"mimic_of": v[0], "multiplier": v[1], "offset": v[2]} for k_, v in eq.items()},
           "regions": regs, "template_points_per_region": {g: int((reg == i_).sum()) for i_, g in enumerate(regs)},
           "collision_source": col.source, "n": n_rec, "accept_rate": round(n_rec / tried, 4), "zero_pose_ignored_pairs": len(col.ignore), "primitive_enclosure_frac_3mm": encl,
           "contact_reproduction_max_m": rep, "fk_decoder_vs_mujoco_max_abs_m": ferr,
           "goal_out_of_limits": int(((q2 < lo - 1e-6) | (q2 > hi + 1e-6)).any(1).sum()),
           "fingers_uncovered": [int(f_) for f_, c in zip(*np.unique(f_of_pt, return_counts=True)) if f_ != 0 and c >= 0.05 * len(Ph) and f_ not in covered],
           "cover_radius_mm": float(np.sqrt(((Ph[:, None] - Ph[a.surf_idx][None]) ** 2).sum(-1)).min(1).max() * 1000),
           "hand_extent_zero_pose_m": [round(float(x), 4) for x in (Ph.max(0) - Ph.min(0))], "mean_k": float(k.mean()),
           "active_frac": float(active.mean())}
    chk["pass"] = bool(rep < 1e-5 and ferr < 1e-5 and chk["goal_out_of_limits"] == 0 and not chk["fingers_uncovered"]
                       and nc <= J_MAX)
    (out / f"{name}_records_check.json").write_text(json.dumps(chk, indent=1))
    (out / "adapter_info.json").write_text(json.dumps({k_: chk[k_] for k_ in ("hand", "robot", "source", "license", "root", "n_coords",
                                                                               "n_physical_joints", "couplings", "regions")}
                                                      | {"status": "external evaluation hand: never used for training, selection or tuning"}, indent=1))
    print(json.dumps(chk), flush=True)
    return chk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", action="append", required=True, choices=list(HANDS) + ["all"])
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    hands = list(HANDS) if "all" in a.hand else a.hand
    res = {}
    for h in hands:
        try:
            res[h] = make(h, a.n)["pass"]
        except Exception as e:  # report and continue with the other hands
            res[h] = f"FAILED: {type(e).__name__}: {e}"
            print(h, res[h], flush=True)
    print("summary", json.dumps(res))


if __name__ == "__main__":
    main()
