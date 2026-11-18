"""GPU-resident split data. The whole split lives on the device; a batch is an index gather.

The 1M-example training split is ~0.5 GB as tensors, so there is no DataLoader and no CPU bottleneck.
batch() returns everything in the model frame:
    world       as stored (KITE's floating-base frame). Required for KITE's psi.
    wrist_init  everything expressed in q_init's wrist frame, so q_init's wrist is the identity. The
                problem becomes invariant to where the hand is. The world-frame contacts and start
                pose are kept as c_pos_world / c_normal_world / R_init_world / t_init_world, so a
                world-frame intent backend (KITE's psi) still sees exactly what KITE's policy would
                give it, and the decoder can be told where the hand is in that frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .handbank import MAX_DOF, HandBank
from .rotations import axis_angle_matrix, euler_xyz_to_matrix, matrix_to_euler_xyz, quat_to_matrix


def jitter_normals(n: torch.Tensor, sigma: float, gen: torch.Generator | None = None) -> torch.Tensor:
    """KITE's train-time normal noise (contact_policy_probe.py): rotate each normal by a rotation
    vector ~ N(0, sigma^2 I3). Positions are not perturbed (pos_scale = 0)."""
    rv = torch.randn(n.shape, generator=gen).to(n) * sigma
    ang = rv.norm(dim=-1).clamp_min(1e-12)
    return (axis_angle_matrix(rv / ang[..., None], ang) @ n[..., None]).squeeze(-1)

def _quat_from_matrix(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrices -> unit quaternions (w, x, y, z), w >= 0 (Shepperd's method, batched)."""
    m = R
    tr = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    cand = torch.stack([tr, m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]], -1)
    i = cand.argmax(-1)
    w = torch.stack([1 + tr, m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]], -1)
    x = torch.stack([m[..., 2, 1] - m[..., 1, 2], 1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
                     m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0]], -1)
    y = torch.stack([m[..., 0, 2] - m[..., 2, 0], m[..., 0, 1] + m[..., 1, 0],
                     1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2], m[..., 1, 2] + m[..., 2, 1]], -1)
    z = torch.stack([m[..., 1, 0] - m[..., 0, 1], m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1],
                     1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]], -1)
    q = torch.stack([w, x, y, z], -2).gather(-1, i[..., None, None].expand(*i.shape, 4, 1)).squeeze(-1)
    q = torch.nn.functional.normalize(q, dim=-1)
    return q * torch.where(q[..., :1] < 0, -1.0, 1.0)


KEYS = {
    "k": torch.long, "template_idx": torch.long, "contact_pos": torch.float32, "contact_normal": torch.float32,
    "joints_init": torch.float32, "joints_goal": torch.float32, "mask": torch.bool, "dof_valid": torch.bool,
    "wrist_pos_init": torch.float32, "wrist_quat_init": torch.float32,
    "wrist_pos_goal": torch.float32, "wrist_quat_goal": torch.float32, "hand": torch.long,
    "joints_contact": torch.float32, "wrist_pos_contact": torch.float32, "wrist_quat_contact": torch.float32,
}


def data_root(cfg) -> Path:
    if cfg.data_dir:
        return Path(cfg.data_dir)
    from . import paths
    return Path(paths.get("GETKITE_DATASET"))


def load_split_arrays(root: Path, split: str, regime: str) -> dict:
    """One split, or several joined with '+' (hand indices offset, hand_ids concatenated)."""
    from kite_data.dataset import load_split
    parts = [load_split(s, regime=regime, root=root / "examples" / regime) for s in split.split("+")]
    if len(parts) == 1:
        return parts[0]
    off = np.cumsum([0] + [len(d["hand_ids"]) for d in parts[:-1]])
    out = {k: np.concatenate([d[k] + o if k == "hand" else d[k] for d, o in zip(parts, off)]) for k in parts[0]}
    return out


def part_mask(hand: np.ndarray, frac: float, part: str) -> np.ndarray:
    """Examples of each hand in `part`: 'holdout' = the last ceil(frac * n_hand) examples of every hand
    (examples within a hand are i.i.d., so this is a random split), 'fit' = the rest, 'all' = everything."""
    if part == "all" or frac <= 0:
        if part == "holdout":
            raise ValueError("part='holdout' needs holdout_frac > 0")
        return np.ones(len(hand), bool)
    if part not in ("fit", "holdout"):
        raise ValueError(part)
    order = np.argsort(hand, kind="stable")  # files are per hand, so this is the identity, but be safe
    h = hand[order]
    start = np.searchsorted(h, h, side="left")
    count = np.bincount(h)[h]
    rank = np.arange(len(h)) - start
    hold = np.empty(len(hand), bool)
    hold[order] = rank >= count - np.ceil(frac * count)
    return hold if part == "holdout" else ~hold


def parse_split(name: str) -> tuple[str, str]:
    """'train@holdout' -> ('train', 'holdout'); 'val' -> ('val', 'all')."""
    split, _, part = name.partition("@")
    return split, part or "all"


class SplitData:
    def __init__(self, cfg, split: str, device="cpu", max_examples: int = 0, seed: int = 0, part: str | None = None):
        """split may carry a part ('train@holdout'); an explicit part argument overrides it."""
        root = data_root(cfg)
        split, p = parse_split(split)
        part = part or p
        d = load_split_arrays(root, split, cfg.regime)
        sel = np.flatnonzero(part_mask(d["hand"], cfg.holdout_frac, part))
        if max_examples and len(sel) > max_examples:  # stratified by construction: files are per hand
            sel = np.sort(np.random.default_rng(seed).choice(sel, max_examples, replace=False))
        self.split, self.part, self.frame, self.n, self.cfg = split, part, cfg.frame, len(sel), cfg
        self.t = {k: torch.as_tensor(d[k][sel]).to(dt).to(device) for k, dt in KEYS.items()}
        assert self.t["joints_init"].shape[1] == MAX_DOF
        self.bank = HandBank(list(d["hand_ids"]), root / "hands", device=device)
        self.device = device

    def batch(self, idx: torch.Tensor, gen: torch.Generator | None = None) -> dict:
        """Examples idx in the model frame. With gen given and cfg.resample_q_init, the start pose is
        drawn afresh from q_contact (KITE re-draws it every epoch); otherwise the stored one is used."""
        t = {k: v[idx] for k, v in self.t.items()}
        if gen is not None and self.cfg.resample_q_init:
            t |= self.resample_start(t, gen)
        R0, t0 = quat_to_matrix(t["wrist_quat_init"]), t["wrist_pos_init"]
        Rg, tg = quat_to_matrix(t["wrist_quat_goal"]), t["wrist_pos_goal"]
        c_pos, c_nrm = t["contact_pos"], t["contact_normal"]
        world = {}
        if self.frame == "wrist_init":
            world = {"c_pos_world": c_pos, "c_normal_world": c_nrm, "R_init_world": R0, "t_init_world": t0}
            Rt = R0.transpose(1, 2)
            c_pos = (c_pos - t0[:, None]) @ R0  # row-vector form of R0^T (x - t0)
            c_nrm = c_nrm @ R0
            Rg, tg = Rt @ Rg, (Rt @ (tg - t0)[..., None]).squeeze(-1)
            R0, t0 = torch.eye(3, device=R0.device).expand_as(R0).contiguous(), torch.zeros_like(t0)
        c_valid = t["template_idx"] >= 0
        return {
            "h": t["hand"], "q_init": t["joints_init"], "q_goal": t["joints_goal"], "mask": t["mask"],
            "dof_valid": t["dof_valid"], "R_init": R0, "t_init": t0, "R_goal": Rg, "t_goal": tg,
            "c_pos": c_pos, "c_normal": c_nrm, "c_valid": c_valid, "tidx": t["template_idx"], "k": t["k"],
        } | world

    def resample_start(self, t: dict, gen: torch.Generator) -> dict:
        """KITE's start-pose noise (contact_policy_probe_dataset.py): joints q_c + N(0, (0.05 range)^2),
        clamped; wrist translation + N(0, 5 mm^2) per axis; rotation N(0, (5 deg)^2) added to each
        intrinsic-XYZ Euler angle ('euler_xyz', KITE) or as a rotation vector ('rotvec', our stored data).
        The label is recomputed: q_goal = mask ? q_contact : q_init; the wrist goal is the contact pose."""
        cfg, h, valid = self.cfg, t["hand"], t["dof_valid"]
        lower, upper = self.bank.lower[h], self.bank.upper[h]
        randn = lambda *s: torch.randn(*s, generator=gen).to(self.device)  # noqa: E731
        qc = t["joints_contact"]
        q0 = torch.minimum(torch.maximum(qc + randn(qc.shape) * cfg.joint_noise_frac * (upper - lower), lower), upper)
        q0 = torch.where(valid, q0, torch.zeros_like(q0))
        B = len(qc)
        p0 = t["wrist_pos_contact"] + randn(B, 3) * cfg.wrist_pos_noise
        Rc = quat_to_matrix(t["wrist_quat_contact"])
        sig = cfg.wrist_rot_noise_deg * torch.pi / 180
        if cfg.wrist_rot_noise == "euler_xyz":
            e = matrix_to_euler_xyz(Rc) + randn(B, 3) * sig
            R0 = euler_xyz_to_matrix(torch.remainder(e + torch.pi, 2 * torch.pi) - torch.pi)
        else:
            rv = randn(B, 3) * sig
            ang = rv.norm(dim=-1).clamp_min(1e-12)
            R0 = Rc @ axis_angle_matrix(rv / ang[:, None], ang)
        return {"joints_init": q0, "joints_goal": torch.where(t["mask"], qc, q0), "wrist_pos_init": p0,
                "wrist_quat_init": _quat_from_matrix(R0)}

    def iterate(self, batch_size: int):
        for s in range(0, self.n, batch_size):
            yield self.batch(torch.arange(s, min(s + batch_size, self.n), device=self.device))

    def sample(self, batch_size: int, gen: torch.Generator, normal_noise: float = 0.0) -> dict:
        """A training batch. With normal_noise > 0, the model's intent input gets KITE's normal
        noise (c_normal_in); losses and metrics keep the clean c_normal."""
        b = self.batch(torch.randint(0, self.n, (batch_size,), generator=gen, device="cpu").to(self.device), gen)
        if normal_noise > 0:
            b["c_normal_in"] = jitter_normals(b["c_normal"], normal_noise, gen) * b["c_valid"][..., None]
        return b
