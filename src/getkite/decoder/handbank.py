"""HandBank: every hand of a split as padded tensors, plus batched differentiable FK.

Built only from the dataset's outputs (data/hands/<hand_id>/hand.json and template.npz), so training
never needs the third-party LEAP assets.

Frames ("slots"): slot 0 is the wrist (palm) frame; slot j+1 is movable joint j's child-link frame
after its rotation. Locked joints are folded into fixed transforms, so FK is one rotation per movable
joint:
    T[slot j+1] = T[parent_slot[j]] @ A[j] @ Rot(axis[j], q[j])
Every link is rigidly attached to one slot (its anchor): link frame = T[anchor] @ C_link. Template
points are stored in anchor frames, so a contact point is one gather + one transform.

Graph nodes (for attention biases) are [wrist, joint 0..MAX_DOF-1, surface 0..N_SURF-1]. A surface
token (one of the template's 32 farthest-point samples) is treated as a leaf child of its anchor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from .rotations import axis_angle_matrix

MAX_DOF = 16
N_SLOTS = MAX_DOF + 1
N_SURF = 32
N_BODY = 1 + MAX_DOF + N_SURF
JOINT_FEAT_DIM = 20
FINGERS = ("index", "middle", "ring", "thumb")


def _transform(xyz, quat_wxyz) -> np.ndarray:
    w, x, y, z = quat_wxyz
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, xyz
    return T


def _tree_matrices(parents: list[int], valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shortest-path and ancestor-distance matrices for a forest given by parent indices (-1 = root)."""
    n = len(parents)
    chain = []  # ancestors of each node with distance, self included at 0
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


@dataclass
class HandArrays:
    """One hand, padded. Shapes are given for a single hand; HandBank stacks them on a leading axis."""
    n_dof: np.ndarray  # ()
    parent_slot: np.ndarray  # (D,) int: slot of the frame each joint hangs off
    A: np.ndarray  # (D, 4, 4) fixed transform from the parent slot frame to the joint frame
    axis: np.ndarray  # (D, 3)
    lower: np.ndarray  # (D,)
    upper: np.ndarray  # (D,)
    dof_valid: np.ndarray  # (D,) bool
    joint_feat: np.ndarray  # (D, JOINT_FEAT_DIM) static URDF features rho_j
    finger: np.ndarray  # (D,) int finger index, -1 for padding
    tpl_anchor: np.ndarray  # (P,) slot of each template point
    tpl_pos: np.ndarray  # (P, 3) anchor frame
    tpl_normal: np.ndarray  # (P, 3) anchor frame
    tpl_region: np.ndarray  # (P,) 0 palm, 1..4 index/middle/ring/thumb
    surf_idx: np.ndarray  # (N_SURF,) template indices of the surface tokens
    spd: np.ndarray  # (N_BODY, N_BODY) int16, -1 = unrelated/padding
    anc: np.ndarray  # (N_BODY, N_BODY) int16, anc[i, j] = d > 0 if j is an ancestor of i
    node_valid: np.ndarray  # (N_BODY,) bool
    slot: np.ndarray  # (D,) canonical LEAP joint id (URDF joint name '0'..'15'), -1 for padding
    kite_hand_idx: np.ndarray  # (N_SURF,) template indices KITE would use as its 32 hand tokens


def kite_local_fps(points_link_local: np.ndarray, k: int) -> np.ndarray:
    """KITE's hand-token selection, reproduced exactly (robot_model.py _farthest_point_sample_indices_local):
    greedy FPS starting at index 0 over the template points *in their own link frames* (not the
    assembled hand). The pretrained scene encoder only ever saw hand tokens chosen this way."""
    P = np.asarray(points_link_local, np.float32)
    idx = np.zeros(k, np.int64)
    dist = np.full(len(P), np.inf, np.float32)
    last = 0
    for i in range(1, k):
        dist = np.minimum(dist, ((P - P[last]) ** 2).sum(-1))
        last = int(np.argmax(dist))
        idx[i] = last
    return idx


def load_hand_arrays(hand_dir: Path) -> HandArrays:
    desc = json.loads((hand_dir / "hand.json").read_text())
    tpl = np.load(hand_dir / "template.npz")
    nd = desc["n_dof"]
    assert nd <= MAX_DOF

    anchor = {"palm_lower": (0, np.eye(4))}  # link -> (slot, C_link)
    A = np.tile(np.eye(4), (MAX_DOF, 1, 1))
    axis = np.tile([0.0, 0.0, 1.0], (MAX_DOF, 1))
    lower, upper = np.zeros(MAX_DOF), np.zeros(MAX_DOF)
    parent_slot = np.zeros(MAX_DOF, np.int64)
    finger = np.full(MAX_DOF, -1, np.int64)
    slot_id = np.full(MAX_DOF, -1, np.int64)
    depth, leaf, n_locked = np.zeros(MAX_DOF), np.zeros(MAX_DOF), np.zeros(MAX_DOF)
    for j in desc["joints"]:
        slot, Cp = anchor[j["parent_link"]]
        O = Cp @ _transform(j["origin_xyz"], j["origin_quat_wxyz"])
        if j["locked"]:
            anchor[j["child_link"]] = (slot, O)
            if slot > 0:
                n_locked[slot - 1] += 1
            continue
        d = j["dof"]
        assert slot == j["parent_dof"] + 1, "movable joints must hang off their movable parent"
        A[d], parent_slot[d] = O, slot
        axis[d], lower[d], upper[d] = j["axis"], j["lower"], j["upper"]
        finger[d] = FINGERS.index(j["finger"])
        slot_id[d] = int(j["name"]) if str(j["name"]).isdigit() else 4 * finger[d] + j.get("k", 0)
        depth[d], leaf[d] = j["depth"], j["leaf"]
        anchor[j["child_link"]] = (d + 1, np.eye(4))

    link_names = [str(s) for s in tpl["link_names"]]
    C = np.stack([anchor[l][1] for l in link_names])
    slot_of_link = np.array([anchor[l][0] for l in link_names])
    li = tpl["link"].astype(np.int64)
    tpl_anchor = slot_of_link[li]
    pos = np.einsum("pij,pj->pi", C[li, :3, :3], tpl["pos"]) + C[li, :3, 3]
    nrm = np.einsum("pij,pj->pi", C[li, :3, :3], tpl["normal"])

    # reach: the farthest template point rigidly attached to each joint, in its frame (captures locked
    # distal links and link stretch, which the joint origins alone do not)
    reach = np.zeros((MAX_DOF, 3))
    for d in range(nd):
        sel = tpl_anchor == d + 1
        if sel.any():
            reach[d] = pos[sel][np.argmax(np.linalg.norm(pos[sel], axis=1))]

    valid = np.arange(MAX_DOF) < nd
    feat = np.concatenate([
        A[:, :3, 3] * 10.0,  # decimetres: offsets are a few cm
        A[:, :3, 0], A[:, :3, 1],  # 6D rotation of the joint frame in its parent
        axis, lower[:, None], upper[:, None],
        depth[:, None] / 4.0, leaf[:, None], n_locked[:, None] / 3.0, reach * 10.0,
    ], axis=1)
    assert feat.shape[1] == JOINT_FEAT_DIM
    feat[~valid] = 0.0

    surf_idx = tpl["fps"].astype(np.int64)
    parents = [-1] + [int(parent_slot[d]) if valid[d] else -1 for d in range(MAX_DOF)]
    parents += [int(tpl_anchor[i]) for i in surf_idx]
    node_valid = np.concatenate([[True], valid, np.ones(N_SURF, bool)])
    spd, anc = _tree_matrices(parents, node_valid)

    region_names = [str(s) for s in tpl["region_names"]]
    region_map = np.array([(("palm",) + FINGERS).index(r) for r in region_names])
    return HandArrays(
        n_dof=np.array(nd), parent_slot=parent_slot, A=A, axis=axis, lower=lower, upper=upper,
        dof_valid=valid, joint_feat=feat, finger=finger, tpl_anchor=tpl_anchor, tpl_pos=pos,
        tpl_normal=nrm, tpl_region=region_map[tpl["region"].astype(np.int64)], surf_idx=surf_idx,
        spd=spd, anc=anc, node_valid=node_valid, slot=slot_id,
        kite_hand_idx=kite_local_fps(tpl["pos"], N_SURF),
    )


class HandBank:
    """All hands of one or more splits, stacked: attribute tensors have a leading hand axis."""

    def __init__(self, hand_ids: list[str], hands_dir: Path, device="cpu", dtype=torch.float32):
        self.hand_ids = list(hand_ids)
        arrays = [load_hand_arrays(Path(hands_dir) / h) for h in self.hand_ids]
        for f in fields(HandArrays):
            a = np.stack([getattr(x, f.name) for x in arrays])
            t = torch.from_numpy(a)
            if t.is_floating_point():
                t = t.to(dtype)
            elif t.dtype == torch.int16:
                t = t.long()
            setattr(self, f.name, t.to(device))
        self.index = {h: i for i, h in enumerate(self.hand_ids)}

    def to(self, device):
        for f in fields(HandArrays):
            setattr(self, f.name, getattr(self, f.name).to(device))
        return self

    @property
    def range(self) -> torch.Tensor:
        return (self.upper - self.lower).clamp_min(1e-6)

    # ------------------------------------------------------------------ FK
    def fk(self, h: torch.Tensor, q: torch.Tensor, wrist_R: torch.Tensor, wrist_t: torch.Tensor):
        """Slot frames for hands h (B,), joints q (B, MAX_DOF), wrist pose (B,3,3), (B,3).

        Returns R (B, N_SLOTS, 3, 3), t (B, N_SLOTS, 3). Padded joints give identity offsets (unused).
        """
        A, par = self.A[h], self.parent_slot[h]
        Rq = axis_angle_matrix(self.axis[h], q)  # (B, D, 3, 3)
        RA, tA = A[..., :3, :3] @ Rq, A[..., :3, 3]
        Rs, ts = [wrist_R], [wrist_t]
        ar = torch.arange(len(h), device=q.device)
        for j in range(MAX_DOF):
            Rall, tall = torch.stack(Rs, 1), torch.stack(ts, 1)
            Rp, tp = Rall[ar, par[:, j]], tall[ar, par[:, j]]
            Rs.append(Rp @ RA[:, j])
            ts.append((Rp @ tA[:, j, :, None]).squeeze(-1) + tp)
        return torch.stack(Rs, 1), torch.stack(ts, 1)

    def points(self, h, R, t, idx):
        """World positions and normals of template points idx (B, M) (any value < 0 is returned as 0)."""
        safe = idx.clamp_min(0)
        anc = torch.gather(self.tpl_anchor[h], 1, safe)
        p = torch.gather(self.tpl_pos[h], 1, safe[..., None].expand(-1, -1, 3))
        n = torch.gather(self.tpl_normal[h], 1, safe[..., None].expand(-1, -1, 3))
        ar = torch.arange(len(h), device=idx.device)[:, None]
        Ra, ta = R[ar, anc], t[ar, anc]
        x = (Ra @ p[..., None]).squeeze(-1) + ta
        nw = (Ra @ n[..., None]).squeeze(-1)
        keep = (idx >= 0)[..., None]
        return x * keep, nw * keep

    def all_points(self, h, R, t):
        """All template points of each hand in the world: (B, P, 3) positions and normals."""
        anc = self.tpl_anchor[h]
        ar = torch.arange(len(h), device=R.device)[:, None]
        Ra, ta = R[ar, anc], t[ar, anc]
        return (Ra @ self.tpl_pos[h][..., None]).squeeze(-1) + ta, (Ra @ self.tpl_normal[h][..., None]).squeeze(-1)
