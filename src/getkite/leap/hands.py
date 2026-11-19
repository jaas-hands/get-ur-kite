"""LEAP hand descriptors from get-ur-kite-dataset, padded into tensors, with batched forward kinematics.

A ``HandBank`` holds a set of hands (a split) side by side, so a batch can mix hands:

    bank = HandBank.load(hand_ids, device="cuda")
    T = bank.fk(h, q, wrist)                  # (B, MAX_LINKS, 4, 4) link frames
    pts, nrm = bank.hand_points(h, T)         # (B, 32, 3) KITE-style hand tokens at that pose

Conventions follow the dataset (``kite_data/hand.py``): links are ``[palm] + [child link of
joint i]`` in the descriptor's joint order (parents before children); a joint's frame is
``parent @ origin @ Rot(axis, q)``; locked joints are rigid (their origin already includes the
lock angle). The wrist pose is the palm's pose. Graph matrices are over ``[wrist] + movable
joints`` (wrist = node 0), exactly as ``Hand.graph()`` computes them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from getkite.leap.paths import hands_dir

MAX_DOF = 16  # movable joints (the dataset loader pads to the same size)
MAX_JOINTS = 16  # joints including locked ones (full LEAP)
MAX_LINKS = MAX_JOINTS + 1  # palm + one link per joint
MAX_NODES = MAX_DOF + 1  # wrist + movable joints
N_HAND_POINTS = 32  # the template's farthest-point samples (KITE's hand tokens)

# Per-movable-joint features, in this order (``HandBank.joint_features``). All come from the URDF.
JOINT_FEATURES = (
    "origin_x", "origin_y", "origin_z",  # offset from the parent link, in its frame (m)
    "quat_w", "quat_x", "quat_y", "quat_z",  # orientation relative to the parent link
    "axis_x", "axis_y", "axis_z",  # joint axis in its own frame
    "lower", "upper",  # limits (rad)
    "depth", "leaf",  # depth among movable joints (finger root = 1), and whether it has no movable child
)


def quat_wxyz_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) unit quaternions (w, x, y, z) -> (..., 3, 3) rotation matrices."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def pose_matrix(pos: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    """(..., 3) position and (..., 4) quaternion -> (..., 4, 4) transform."""
    T = torch.zeros(pos.shape[:-1] + (4, 4), dtype=pos.dtype, device=pos.device)
    T[..., :3, :3] = quat_wxyz_to_matrix(quat_wxyz)
    T[..., :3, 3] = pos
    T[..., 3, 3] = 1.0
    return T


def axis_angle_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues: (..., 3) unit axes and (...,) angles -> (..., 3, 3)."""
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).reshape(axis.shape[:-1] + (3, 3))
    s, c = torch.sin(angle)[..., None, None], torch.cos(angle)[..., None, None]
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device).expand_as(K)
    return eye + s * K + (1 - c) * (K @ K)


def graph_matrices(parent_dof: list[int]) -> dict[str, np.ndarray]:
    """Shortest-path, ancestor and descendant distances over [wrist] + movable joints (``Hand.graph``)."""
    parents = [-1] + [p + 1 for p in parent_dof]
    n = len(parents)
    adj = np.zeros((n, n), bool)
    for i, p in enumerate(parents):
        if p >= 0:
            adj[i, p] = adj[p, i] = True
    spd = np.full((n, n), -1, int)
    for s in range(n):
        spd[s, s], frontier = 0, [s]
        while frontier:
            nxt = []
            for u in frontier:
                for v in np.flatnonzero(adj[u]):
                    if spd[s, v] < 0:
                        spd[s, v] = spd[s, u] + 1
                        nxt.append(v)
            frontier = nxt
    anc = np.zeros((n, n), int)
    for i in range(n):
        d, p = 1, parents[i]
        while p >= 0:
            anc[i, p] = d
            d, p = d + 1, parents[p]
    return {"spd": spd, "anc": anc, "desc": anc.T.copy()}


@dataclass
class HandBank:
    """Padded descriptors of H hands. Index a hand by its position in ``hand_ids``."""

    hand_ids: list[str]
    n_dof: torch.Tensor  # (H,)
    # kinematics, over all joints in descriptor order (locked included)
    joint_valid: torch.Tensor  # (H, MAX_JOINTS) bool
    joint_locked: torch.Tensor  # (H, MAX_JOINTS) bool
    joint_dof: torch.Tensor  # (H, MAX_JOINTS) long, -1 if locked or padding
    joint_parent_link: torch.Tensor  # (H, MAX_JOINTS) long, index into links
    joint_origin: torch.Tensor  # (H, MAX_JOINTS, 4, 4)
    joint_axis: torch.Tensor  # (H, MAX_JOINTS, 3)
    # per movable joint (the decoder's joint tokens)
    dof_valid: torch.Tensor  # (H, MAX_DOF) bool
    dof_link: torch.Tensor  # (H, MAX_DOF) long: the link each movable joint moves (its frame)
    dof_axis: torch.Tensor  # (H, MAX_DOF, 3): each movable joint's axis in its own frame
    joint_features: torch.Tensor  # (H, MAX_DOF, len(JOINT_FEATURES))
    lower: torch.Tensor  # (H, MAX_DOF)
    upper: torch.Tensor  # (H, MAX_DOF)
    # graph over [wrist] + movable joints, padded with -1 (spd) / 0 (anc, desc)
    spd: torch.Tensor  # (H, MAX_NODES, MAX_NODES) long
    anc: torch.Tensor  # (H, MAX_NODES, MAX_NODES) long
    desc: torch.Tensor  # (H, MAX_NODES, MAX_NODES) long
    # hand tokens: the template's 32 farthest-point samples, in their link frames
    point_link: torch.Tensor  # (H, 32) long
    point_pos: torch.Tensor  # (H, 32, 3)
    point_normal: torch.Tensor  # (H, 32, 3)
    # the full 4,096-point template (for contact losses); None unless loaded
    template_link: torch.Tensor | None = None  # (H, 4096) long
    template_pos: torch.Tensor | None = None  # (H, 4096, 3)
    template_normal: torch.Tensor | None = None  # (H, 4096, 3)

    @classmethod
    def load(cls, hand_ids, root: Path | None = None, device="cpu", full_template: bool = False) -> "HandBank":
        root = Path(root) if root is not None else hands_dir()
        H = len(hand_ids)
        f32 = np.float32
        n_dof = np.zeros(H, np.int64)
        jv = np.zeros((H, MAX_JOINTS), bool); jl = np.zeros((H, MAX_JOINTS), bool)
        jd = np.full((H, MAX_JOINTS), -1, np.int64); jp = np.zeros((H, MAX_JOINTS), np.int64)
        jo = np.tile(np.eye(4, dtype=f32), (H, MAX_JOINTS, 1, 1)); ja = np.zeros((H, MAX_JOINTS, 3), f32)
        dv = np.zeros((H, MAX_DOF), bool); dl = np.zeros((H, MAX_DOF), np.int64); dax = np.zeros((H, MAX_DOF, 3), f32)
        feat = np.zeros((H, MAX_DOF, len(JOINT_FEATURES)), f32)
        lo = np.zeros((H, MAX_DOF), f32); hi = np.zeros((H, MAX_DOF), f32)
        spd = np.full((H, MAX_NODES, MAX_NODES), -1, np.int64)
        anc = np.zeros((H, MAX_NODES, MAX_NODES), np.int64); desc = np.zeros_like(anc)
        pl = np.zeros((H, N_HAND_POINTS), np.int64); pp = np.zeros((H, N_HAND_POINTS, 3), f32); pn = np.zeros_like(pp)
        tl = tp = tn = None
        for h, hid in enumerate(hand_ids):
            d = json.loads((root / hid / "hand.json").read_text())
            links = [lk["name"] for lk in d["links"]]
            joints = d["joints"]
            assert len(joints) <= MAX_JOINTS and links == [links[0]] + [j["child_link"] for j in joints], hid
            n_dof[h] = d["n_dof"]
            parent_dof = [0] * d["n_dof"]
            for i, j in enumerate(joints):
                jv[h, i], jl[h, i] = True, bool(j["locked"])
                jp[h, i] = links.index(j["parent_link"])
                R = quat_wxyz_to_matrix(torch.tensor(j["origin_quat_wxyz"], dtype=torch.float64)).numpy()
                jo[h, i, :3, :3], jo[h, i, :3, 3] = R, j["origin_xyz"]
                ja[h, i] = j["axis"]
                if not j["locked"]:
                    k = j["dof"]
                    jd[h, i], dv[h, k], dl[h, k], dax[h, k] = k, True, i + 1, j["axis"]
                    parent_dof[k] = j["parent_dof"]
                    feat[h, k] = [*j["origin_xyz"], *j["origin_quat_wxyz"], *j["axis"], j["lower"], j["upper"],
                                  j["depth"], float(j["leaf"])]
                    lo[h, k], hi[h, k] = j["lower"], j["upper"]
            g = graph_matrices(parent_dof)
            n = d["n_dof"] + 1
            spd[h, :n, :n], anc[h, :n, :n], desc[h, :n, :n] = g["spd"], g["anc"], g["desc"]
            t = np.load(root / hid / "template.npz")
            assert list(t["link_names"]) == links, hid
            fps = t["fps"].astype(np.int64)
            pl[h], pp[h], pn[h] = t["link"][fps], t["pos"][fps], t["normal"][fps]
            if full_template:
                if tl is None:
                    n_t = len(t["link"])
                    tl = np.zeros((H, n_t), np.int64); tp = np.zeros((H, n_t, 3), f32); tn = np.zeros_like(tp)
                tl[h], tp[h], tn[h] = t["link"], t["pos"], t["normal"]
        T = lambda a: None if a is None else torch.as_tensor(a, device=device)  # noqa: E731
        return cls(list(hand_ids), T(n_dof), T(jv), T(jl), T(jd), T(jp), T(jo), T(ja), T(dv), T(dl), T(dax), T(feat), T(lo), T(hi),
                   T(spd), T(anc), T(desc), T(pl), T(pp), T(pn), T(tl), T(tp), T(tn))

    @property
    def device(self) -> torch.device:
        return self.joint_origin.device

    def index(self, hand_id: str) -> int:
        return self.hand_ids.index(hand_id)

    def fk(self, h: torch.Tensor, q: torch.Tensor, wrist: torch.Tensor | None = None) -> torch.Tensor:
        """Link frames, (B, MAX_LINKS, 4, 4). h: (B,) hand indices; q: (B, MAX_DOF) joint angles
        (padding ignored); wrist: (B, 4, 4) palm pose (identity if None). Padded links are identity."""
        B = h.shape[0]
        dtype = q.dtype
        T = torch.eye(4, dtype=dtype, device=q.device).repeat(B, MAX_LINKS, 1, 1)
        if wrist is not None:
            T[:, 0] = wrist.to(dtype)
        origin, axis = self.joint_origin[h].to(dtype), self.joint_axis[h].to(dtype)
        dof, parent, valid = self.joint_dof[h], self.joint_parent_link[h], self.joint_valid[h]
        rows = torch.arange(B, device=q.device)
        for i in range(MAX_JOINTS):
            if not bool(valid[:, i].any()):
                break
            angle = torch.where(dof[:, i] >= 0, q.gather(1, dof[:, i].clamp_min(0)[:, None])[:, 0], torch.zeros_like(q[:, 0]))
            Rq = torch.eye(4, dtype=dtype, device=q.device).repeat(B, 1, 1)
            Rq[:, :3, :3] = axis_angle_matrix(axis[:, i], angle)
            M = T[rows, parent[:, i]] @ origin[:, i] @ Rq
            T[:, i + 1] = torch.where(valid[:, i, None, None], M, T[:, i + 1])
        return T

    def transform_points(self, T: torch.Tensor, link: torch.Tensor, pos: torch.Tensor, normal: torch.Tensor):
        """Link-frame points to the world: T (B, L, 4, 4), link (B, N), pos/normal (B, N, 3)."""
        Tl = torch.gather(T, 1, link[:, :, None, None].expand(-1, -1, 4, 4))
        R, t = Tl[..., :3, :3], Tl[..., :3, 3]
        return (R @ pos[..., None])[..., 0] + t, (R @ normal[..., None])[..., 0]

    def hand_points(self, h: torch.Tensor, T: torch.Tensor):
        """The 32 hand tokens (world positions and normals) at link frames T."""
        return self.transform_points(T, self.point_link[h], self.point_pos[h].to(T.dtype), self.point_normal[h].to(T.dtype))

    def template_points(self, h: torch.Tensor, T: torch.Tensor, idx: torch.Tensor):
        """Template points ``idx`` (B, K) in the world (needs ``full_template=True``)."""
        if self.template_link is None:
            raise RuntimeError("load the HandBank with full_template=True")
        link = torch.gather(self.template_link[h], 1, idx)
        pos = torch.gather(self.template_pos[h], 1, idx[..., None].expand(-1, -1, 3)).to(T.dtype)
        nrm = torch.gather(self.template_normal[h], 1, idx[..., None].expand(-1, -1, 3)).to(T.dtype)
        return self.transform_points(T, link, pos, nrm)
