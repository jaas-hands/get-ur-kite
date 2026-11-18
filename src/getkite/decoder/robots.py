"""Robot-independent description, padding and batched FK for the multi-family scratch decoder.

One RobotDesc per robot (morphology). Adapters build it from a family's native files:
    urdf_desc(...)   KITE's URDF assets (Allegro, Barrett, ...): the six virtual floating-base joints are
                     stripped; the base is the child link of the last virtual joint.
    leap_desc(...)   our LEAP variants (kite_data hand.json + template.npz); locked joints become fixed.

Conventions (shared with the data records):
  * Root pose (R, t) is the floating base link in KITE's world frame (metres).
  * "Physical joints" are all non-fixed joints after the base, in the adapter's order. Mimic joints are
    physical joints whose value is q[src] * mult + offset; they are graph nodes and FK joints but get no
    output. "Coordinates" are the independently controllable joints: the order of the native action
    vector (minus the base), which is how predictions are unpadded.
  * Fixed joints are folded into the constant transform A of the next physical joint (as in HandBank),
    and every link is rigidly attached to one frame ("slot"): slot 0 is the base, slot j+1 is physical
    joint j's child frame after its motion.
  * Template points are stored in their anchor slot's frame, so contact provenance is one gather.

Graph nodes for attention are [root, joint 0..J_MAX-1, surface 0..N_SURF-1]; a surface token is a leaf
child of its anchor slot. Joint ordering inside the graph is the physical order; nothing here carries a
learned or canonical identity (no family or joint ids reach the model).
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import torch

from .rotations import axis_angle_matrix

J_MAX = 24  # physical joints per robot (Shadow: 24 incl. coupled; LEAP/Allegro 16; Barrett 8)
N_SURF = 32
N_NBR = 32  # template points pooled into each surface token
N_BODY = 1 + J_MAX + N_SURF
REVOLUTE, PRISMATIC = 0, 1


def rpy_matrix(rpy) -> np.ndarray:
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx  # URDF: fixed-axis roll, pitch, yaw


def quat_wxyz_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def make_T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


@dataclass
class Joint:
    name: str
    type: str  # "revolute" | "prismatic" | "fixed"
    parent: str
    child: str
    T: np.ndarray  # (4, 4) parent link frame -> joint frame
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    lower: float = 0.0
    upper: float = 0.0
    mimic: tuple | None = None  # (source joint name, multiplier, offset)


@dataclass
class RobotDesc:
    name: str  # unique morphology id
    family: str  # for sampling/reporting only; never a model input
    base_link: str
    joints: list  # list[Joint], any order; fixed ones included
    coords: list  # independent coordinate names in native action order
    tpl_link_names: list
    tpl_link: np.ndarray  # (P,) index into tpl_link_names
    tpl_pos: np.ndarray  # (P, 3) link-local
    tpl_normal: np.ndarray  # (P, 3) link-local
    surf_idx: np.ndarray | None = None  # (N_SURF,) token centres; None = fallback FPS (see robot_arrays surface_sampling)
    source_hash: str = ""

    def fingerprint(self) -> str:
        """Hash of the kinematics and template (identity check for splits/leakage, not a model input)."""
        h = hashlib.sha256()
        for j in sorted(self.joints, key=lambda j: j.name):
            h.update(f"{j.name}|{j.type}|{j.parent}|{j.child}|{j.lower:.6f}|{j.upper:.6f}|{j.mimic}".encode())
            h.update(np.round(j.T, 6).tobytes() + np.round(j.axis, 6).tobytes())
        h.update(np.round(self.tpl_pos, 6).tobytes())
        return h.hexdigest()[:16]


# ---------------------------------------------------------------------------------------------- adapters
VIRTUAL_PREFIX = "virtual_joint_"


def urdf_desc(urdf: str | Path, template: str | Path, name: str, family: str, coords: list | None = None,
              ignore_mimic: bool = False) -> RobotDesc:
    """KITE URDF + KITE surface template (.pt). coords: native independent-joint order (default: every
    non-fixed, non-mimic, non-virtual joint in file order). ignore_mimic=True follows KITE's released
    convention (its robot model never parses <mimic>: Barrett's 8 joints are 8 independent DOFs, and its
    synthetic data samples them independently)."""
    root = ET.parse(urdf).getroot()
    joints, base = [], None
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        a = j.find("axis")
        axis = np.array([float(v) for v in a.get("xyz").split()]) if a is not None else np.array([1.0, 0, 0])
        lim = j.find("limit")
        m = j.find("mimic")
        jt = j.get("type")
        jt = "revolute" if jt == "continuous" else jt
        jj = Joint(j.get("name"), jt, j.find("parent").get("link"), j.find("child").get("link"),
                   make_T(rpy_matrix(rpy), xyz), axis / np.linalg.norm(axis),
                   float(lim.get("lower", 0)) if lim is not None else 0.0,
                   float(lim.get("upper", 0)) if lim is not None else 0.0,
                   (m.get("joint"), float(m.get("multiplier", 1)), float(m.get("offset", 0)))
                   if m is not None and not ignore_mimic else None)
        if jj.name.startswith(VIRTUAL_PREFIX):
            base = jj.child if jj.name.endswith("yaw") else base
            continue
        joints.append(jj)
    if base is None:
        raise ValueError(f"{urdf}: no virtual floating-base joints; unsupported (expected KITE's *_extended URDF)")
    for jt in {j.type for j in joints} - {"revolute", "prismatic", "fixed"}:
        raise ValueError(f"{urdf}: unsupported joint type {jt}")
    if coords is None:
        coords = [j.name for j in joints if j.type != "fixed" and j.mimic is None]
    import torch as _t
    t = _t.load(template, map_location="cpu", weights_only=False)
    return RobotDesc(name=name, family=family, base_link=base, joints=joints, coords=list(coords),
                     tpl_link_names=list(t["link_names"]), tpl_link=t["surface_template_link_indices"].numpy(),
                     tpl_pos=t["surface_template_points_local"].numpy().astype(np.float64),
                     tpl_normal=t["surface_template_normals_local"].numpy().astype(np.float64),
                     source_hash=hashlib.sha256(Path(urdf).read_bytes() + Path(template).read_bytes()).hexdigest()[:16])


def leap_desc(hand_dir: str | Path) -> RobotDesc:
    """A kite_data LEAP variant. Locked joints are fixed joints at their lock angle (0 rad)."""
    hand_dir = Path(hand_dir)
    d = json.loads((hand_dir / "hand.json").read_text())
    tpl = np.load(hand_dir / "template.npz")
    lock = d.get("generator", {}).get("lock_angle", 0.0)
    joints, coords = [], [None] * d["n_dof"]
    for j in d["joints"]:
        T = make_T(quat_wxyz_matrix(j["origin_quat_wxyz"]), j["origin_xyz"])
        if j["locked"]:  # rigid at the lock angle: fold the rotation into the fixed transform
            Rl = axis_angle_matrix(torch.tensor(j["axis"], dtype=torch.float64), torch.tensor(lock, dtype=torch.float64)).numpy()
            joints.append(Joint(j["name"], "fixed", j["parent_link"], j["child_link"], T @ make_T(Rl, np.zeros(3))))
            continue
        joints.append(Joint(j["name"], "revolute", j["parent_link"], j["child_link"], T, np.asarray(j["axis"], float),
                            j["lower"], j["upper"]))
        coords[j["dof"]] = j["name"]
    assert None not in coords
    return RobotDesc(name=d["id"], family="leap", base_link="palm_lower", joints=joints, coords=coords,
                     tpl_link_names=[str(s) for s in tpl["link_names"]], tpl_link=tpl["link"].astype(np.int64),
                     tpl_pos=tpl["pos"].astype(np.float64), tpl_normal=tpl["normal"].astype(np.float64),
                     surf_idx=tpl["fps"].astype(np.int64),
                     source_hash=hashlib.sha256((hand_dir / "hand.json").read_bytes()).hexdigest()[:16])


# ---------------------------------------------------------------------------------------------- arrays
def _tree_matrices(parents, valid):
    n = len(parents)
    chain = []
    for i in range(n):
        c, d, p = {i: 0}, 1, parents[i]
        while p >= 0:
            c[p] = d
            d, p = d + 1, parents[p]
        chain.append(c)
    spd = np.full((n, n), -1, np.int16)
    anc = np.zeros((n, n), np.int16)
    for i in range(n):
        if not valid[i]:
            continue
        for j in range(n):
            if not valid[j]:
                continue
            common = set(chain[i]) & set(chain[j])
            if common:
                spd[i, j] = min(chain[i][c] + chain[j][c] for c in common)
            if j != i and j in chain[i]:
                anc[i, j] = chain[i][j]
    return spd, anc


def _fps(P: np.ndarray, k: int) -> np.ndarray:
    idx, dist, last = np.zeros(k, np.int64), np.full(len(P), np.inf), 0
    for i in range(1, k):
        dist = np.minimum(dist, ((P - P[last]) ** 2).sum(-1))
        last = int(np.argmax(dist))
        idx[i] = last
    return idx


@dataclass
class RobotArrays:
    n_joint: np.ndarray  # () physical joints
    n_coord: np.ndarray  # () independent coordinates
    parent_slot: np.ndarray  # (J,)
    A: np.ndarray  # (J, 4, 4)
    axis: np.ndarray  # (J, 3)
    jtype: np.ndarray  # (J,) REVOLUTE / PRISMATIC
    lower: np.ndarray  # (J,)
    upper: np.ndarray  # (J,)
    joint_valid: np.ndarray  # (J,)
    mimic_src: np.ndarray  # (J,) physical index of the source joint, -1 if independent
    mimic_mult: np.ndarray  # (J,)
    mimic_off: np.ndarray  # (J,)
    coord_joint: np.ndarray  # (J,) physical joint driven by coordinate c (padded -1)
    joint_coord: np.ndarray  # (J,) coordinate of an independent physical joint, -1 otherwise
    tpl_anchor: np.ndarray  # (P,)
    tpl_pos: np.ndarray  # (P, 3) anchor frame
    tpl_normal: np.ndarray  # (P, 3)
    surf_idx: np.ndarray  # (N_SURF,)
    surf_nbr: np.ndarray  # (N_SURF, N_NBR) template points pooled into each token (same anchor)
    spd: np.ndarray  # (N_BODY, N_BODY)
    anc: np.ndarray
    node_valid: np.ndarray  # (N_BODY,)


SURFACE_SAMPLING = ("link_local_legacy", "hand_frame")


def _hand_frame_template(parent_slot, A, axis, jtype, lower, upper, n_phys, tpl_anchor, pos):
    """Template points in ONE hand (base) frame at the zero pose (joints clamped to their limits)."""
    T = [np.eye(4)]
    for i in range(n_phys):
        q = float(np.clip(0.0, lower[i], upper[i]))
        Tq = np.eye(4)
        if jtype[i] == PRISMATIC:
            Tq[:3, 3] = axis[i] * q
        else:
            k = axis[i] / np.linalg.norm(axis[i])
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            Tq[:3, :3] = np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K
        T.append(T[parent_slot[i]] @ A[i] @ Tq)
    T = np.stack(T)
    Ta = T[tpl_anchor]
    return np.einsum("pij,pj->pi", Ta[:, :3, :3], pos) + Ta[:, :3, 3]


def robot_arrays(desc: RobotDesc, surface_sampling: str = "link_local_legacy") -> RobotArrays:
    by_child = {j.child: j for j in desc.joints}
    phys = [j for j in desc.joints if j.type != "fixed"]

    def depth(j):  # movable ancestors; a stable sort by depth lists every parent before its children
        d, link = 0, j.parent
        while link != desc.base_link:
            p = by_child[link]
            d, link = d + (p.type != "fixed"), p.parent
        return d
    phys = sorted(phys, key=depth)
    if len(phys) > J_MAX:
        raise ValueError(f"{desc.name}: {len(phys)} physical joints > J_MAX={J_MAX}")
    pidx = {j.name: i for i, j in enumerate(phys)}

    def anchor_of(link):  # (slot, link frame in slot frame)
        T = np.eye(4)
        while link != desc.base_link:
            j = by_child.get(link)
            if j is None:
                raise ValueError(f"{desc.name}: link {link} not connected to base {desc.base_link}")
            if j.type != "fixed":
                return pidx[j.name] + 1, T
            T = j.T @ T
            link = j.parent
        return 0, T

    J = J_MAX
    A = np.tile(np.eye(4), (J, 1, 1))
    axis = np.tile([0.0, 0.0, 1.0], (J, 1))
    lower, upper = np.zeros(J), np.zeros(J)
    parent_slot, jtype = np.zeros(J, np.int64), np.zeros(J, np.int64)
    msrc, mmul, moff = np.full(J, -1, np.int64), np.zeros(J), np.zeros(J)
    for i, j in enumerate(phys):
        s, Tp = anchor_of(j.parent)
        if s > i:
            raise ValueError(f"{desc.name}: joints must be listed parent-first ({j.name})")
        parent_slot[i], A[i] = s, Tp @ j.T
        axis[i], lower[i], upper[i] = j.axis, j.lower, j.upper
        jtype[i] = PRISMATIC if j.type == "prismatic" else REVOLUTE
        if j.mimic is not None:
            if j.mimic[0] not in pidx or phys[pidx[j.mimic[0]]].mimic is not None:
                raise ValueError(f"{desc.name}: unsupported mimic chain at {j.name}")
            msrc[i], mmul[i], moff[i] = pidx[j.mimic[0]], j.mimic[1], j.mimic[2]
    coord_joint, joint_coord = np.full(J, -1, np.int64), np.full(J, -1, np.int64)
    for c, n in enumerate(desc.coords):
        i = pidx[n]
        if msrc[i] >= 0:
            raise ValueError(f"{desc.name}: coordinate {n} is a mimic joint")
        coord_joint[c], joint_coord[i] = i, c
    for i in range(len(phys)):
        if msrc[i] < 0 and joint_coord[i] < 0:
            raise ValueError(f"{desc.name}: independent joint {phys[i].name} is not a coordinate")

    anchors = [anchor_of(l) for l in desc.tpl_link_names]
    slot_of_link = np.array([a[0] for a in anchors])
    C = np.stack([a[1] for a in anchors])
    li = desc.tpl_link
    tpl_anchor = slot_of_link[li]
    pos = np.einsum("pij,pj->pi", C[li, :3, :3], desc.tpl_pos) + C[li, :3, 3]
    nrm = np.einsum("pij,pj->pi", C[li, :3, :3], desc.tpl_normal)

    # Surface-token centres. Dataset-provided FPS (LEAP) wins. Fallback for URDF robots:
    #   link_local_legacy: FPS on LINK-LOCAL template points (pre-2026-11-17 behaviour, kept so existing checkpoints
    #                      see their original inputs; it concentrates tokens on the palm -- docs/decoder_audit.md section 37A)
    #   hand_frame:        FPS on template points in one hand frame at the zero pose (corrected policy)
    assert surface_sampling in SURFACE_SAMPLING, surface_sampling
    if desc.surf_idx is not None:
        surf_idx = desc.surf_idx
    elif surface_sampling == "hand_frame":
        surf_idx = _fps(_hand_frame_template(parent_slot, A, axis, jtype, lower, upper, len(phys), tpl_anchor, pos), N_SURF)
    else:
        surf_idx = _fps(desc.tpl_pos, N_SURF)
    nbr = np.zeros((N_SURF, N_NBR), np.int64)
    for s, c in enumerate(surf_idx):  # nearest points on the same rigid body: pose-independent pooling
        same = np.flatnonzero(tpl_anchor == tpl_anchor[c])
        order = same[np.argsort(((pos[same] - pos[c]) ** 2).sum(-1))]
        nbr[s] = np.resize(order[:N_NBR], N_NBR)  # repeat if the body has fewer points

    valid = np.arange(J) < len(phys)
    parents = [-1] + [int(parent_slot[i]) if valid[i] else -1 for i in range(J)] + [int(tpl_anchor[i]) for i in surf_idx]
    node_valid = np.concatenate([[True], valid, np.ones(N_SURF, bool)])
    spd, anc = _tree_matrices(parents, node_valid)
    return RobotArrays(np.array(len(phys)), np.array(len(desc.coords)), parent_slot, A, axis, jtype, lower, upper,
                       valid, msrc, mmul, moff, coord_joint, joint_coord, tpl_anchor, pos, nrm, surf_idx, nbr,
                       spd, anc, node_valid)


class RobotBank:
    """All robots of an experiment as stacked padded tensors (leading robot axis) + batched FK."""

    def __init__(self, descs: list, device="cpu", dtype=torch.float32, surface_sampling: str = "link_local_legacy"):
        self.descs = list(descs)
        self.surface_sampling = surface_sampling
        self.names = [d.name for d in descs]
        self.family = [d.family for d in descs]
        P = max(len(d.tpl_pos) for d in descs)
        arrs = [robot_arrays(d, surface_sampling) for d in descs]
        self.n_points = torch.tensor([len(d.tpl_pos) for d in descs], device=device)
        for f in fields(RobotArrays):
            xs = [getattr(a, f.name) for a in arrs]
            if f.name in ("tpl_anchor", "tpl_pos", "tpl_normal"):  # pad templates (MANO has 778 points)
                xs = [np.concatenate([x, np.repeat(x[-1:], P - len(x), 0)]) for x in xs]
            t = torch.from_numpy(np.stack(xs))
            t = t.to(dtype) if t.is_floating_point() else (t.long() if t.dtype == torch.int16 else t)
            setattr(self, f.name, t.to(device))
        self.index = {n: i for i, n in enumerate(self.names)}

    def to(self, device):
        for f in fields(RobotArrays):
            setattr(self, f.name, getattr(self, f.name).to(device))
        self.n_points = self.n_points.to(device)
        return self

    @property
    def range(self):
        return (self.upper - self.lower).clamp_min(1e-6)

    @property
    def coord_valid(self):
        return self.coord_joint >= 0

    def expand(self, r, qc):
        """Independent coordinates (B, J_MAX, padded) -> physical joint values (B, J_MAX) incl. mimics."""
        jc = self.joint_coord[r]
        q = torch.where(jc >= 0, torch.gather(qc, 1, jc.clamp_min(0)), 0.0)
        ms = self.mimic_src[r]
        qm = torch.gather(q, 1, ms.clamp_min(0)) * self.mimic_mult[r] + self.mimic_off[r]
        q = torch.where(ms >= 0, qm, q)
        return torch.where(self.joint_valid[r], q, 0.0)

    def coords_of(self, r, qp):
        """Physical joint values -> independent coordinates (padded with 0)."""
        cj = self.coord_joint[r]
        return torch.where(cj >= 0, torch.gather(qp, 1, cj.clamp_min(0)), 0.0)

    def fk(self, r, qp, R0, t0):
        """Slot frames for robots r (B,), physical joints qp (B, J), base pose. -> R (B,1+J,3,3), t (B,1+J,3)."""
        A, par = self.A[r], self.parent_slot[r]
        ax = self.axis[r]
        rev = (self.jtype[r] == REVOLUTE)
        Rq = axis_angle_matrix(ax, torch.where(rev, qp, 0.0))
        tq = ax * torch.where(rev, 0.0, qp)[..., None]
        RA = A[..., :3, :3] @ Rq
        tA = A[..., :3, 3] + (A[..., :3, :3] @ tq[..., None]).squeeze(-1)
        Rs, ts = [R0], [t0]
        ar = torch.arange(len(r), device=qp.device)
        for j in range(J_MAX):
            Rall, tall = torch.stack(Rs, 1), torch.stack(ts, 1)
            Rp, tp = Rall[ar, par[:, j]], tall[ar, par[:, j]]
            Rs.append(Rp @ RA[:, j])
            ts.append((Rp @ tA[:, j, :, None]).squeeze(-1) + tp)
        return torch.stack(Rs, 1), torch.stack(ts, 1)

    def points(self, r, R, t, idx):
        """World positions/normals of template points idx (B, M); entries < 0 return 0."""
        safe = idx.clamp_min(0)
        anc = torch.gather(self.tpl_anchor[r], 1, safe)
        p = torch.gather(self.tpl_pos[r], 1, safe[..., None].expand(-1, -1, 3))
        n = torch.gather(self.tpl_normal[r], 1, safe[..., None].expand(-1, -1, 3))
        ar = torch.arange(len(r), device=idx.device)[:, None]
        Ra, ta = R[ar, anc], t[ar, anc]
        keep = (idx >= 0)[..., None]
        return ((Ra @ p[..., None]).squeeze(-1) + ta) * keep, (Ra @ n[..., None]).squeeze(-1) * keep
