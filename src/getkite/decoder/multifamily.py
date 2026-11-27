"""Common record schema, KITE's start/goal recipe and balanced multi-family sampling.

A record is one KITE-style synthetic sample on one robot (RobotBank index r):
    t2, R2       contact (generating) base pose q2, world frame (metres)
    qc2          (J_MAX,) contact coordinates, padded with 0
    active       (J_MAX,) KITE's contact-relevant mask (joint has a contacting descendant link)
    c_pos, c_normal, c_valid, c_tpl   (K,) world contacts at q2 and their template indices
Start/goal (KITE released code, data_process/contact_policy_probe_dataset.py, q1 'gaussian' mode):
    q1 = q2 + noise: base xyz + N(0, 5 mm^2); each intrinsic-XYZ Euler angle + N(0, (5 deg)^2) (wrapped);
         joints + N(0, (0.05 range)^2), clamped to limits
    goal: base = q2; joint = q2 where active else q1 (every coordinate supervised; inactive target = stay)
Training redraws q1 every batch; evaluation pools fix q1 with a seed so every model sees the same records.

Sources: KITE embodiments via get-ur-kite-dataset scripts/kite_family/gen_kite_family.py (KITE's own sampler) and LEAP variants via
kite_data's kite_code regime (same recipe; its `mask` is the same contact-relevant mask).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from .robots import J_MAX, RobotBank
from .rotations import euler_xyz_to_matrix, matrix_to_euler_xyz, quat_to_matrix

K_MAX = 6


@dataclass
class Records:
    r: torch.Tensor  # (N,)
    t2: torch.Tensor  # (N, 3)
    R2: torch.Tensor  # (N, 3, 3)
    e2: torch.Tensor  # (N, 3) intrinsic-XYZ Euler of R2 as stored by the source (KITE perturbs these angles)
    qc2: torch.Tensor  # (N, J_MAX)
    active: torch.Tensor  # (N, J_MAX) bool
    c_pos: torch.Tensor  # (N, K, 3)
    c_normal: torch.Tensor
    c_valid: torch.Tensor  # (N, K) bool
    c_tpl: torch.Tensor  # (N, K) long, -1 = padding

    def __len__(self):
        return len(self.r)

    def index(self, i) -> "Records":
        return Records(**{f.name: getattr(self, f.name)[i] for f in fields(self)})

    def to(self, device) -> "Records":
        return Records(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    @staticmethod
    def cat(rs: list) -> "Records":
        return Records(**{f.name: torch.cat([getattr(x, f.name) for x in rs]) for f in fields(Records)})


def _pad(a: np.ndarray, n: int, axis: int, value=0):
    pad = [(0, 0)] * a.ndim
    pad[axis] = (0, n - a.shape[axis])
    return np.pad(a, pad, constant_values=value)


def _records(r_index, t2, e2, qc2, active, c_pos, c_nrm, c_valid, c_tpl) -> Records:
    N, D = qc2.shape
    K = c_pos.shape[1]
    assert D <= J_MAX and K <= K_MAX
    f = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)  # noqa: E731
    e2 = torch.as_tensor(np.asarray(e2), dtype=torch.float64)
    return Records(
        r=torch.full((N,), r_index, dtype=torch.long), t2=f(t2), R2=euler_xyz_to_matrix(e2).float(), e2=e2.float(),
        qc2=f(_pad(qc2, J_MAX, 1)), active=torch.as_tensor(_pad(active, J_MAX, 1, False)),
        c_pos=f(_pad(c_pos, K_MAX, 1)), c_normal=f(_pad(c_nrm, K_MAX, 1)),
        c_valid=torch.as_tensor(_pad(c_valid, K_MAX, 1, False)),
        c_tpl=torch.as_tensor(_pad(c_tpl.astype(np.int64), K_MAX, 1, -1)))


def load_kite_records(npz: str | Path, bank: RobotBank, robot: str, rows=None) -> Records:
    z = np.load(npz)
    meta = json.loads(str(z["meta"]))
    desc = bank.descs[bank.index[robot]]
    assert meta["coords"] == desc.coords, f"{npz}: coordinate order differs from the robot description"
    sel = slice(None) if rows is None else rows
    return _records(bank.index[robot], z["q2_t"][sel], z["q2_euler"][sel], z["q2_joints"][sel], z["joint_active"][sel],
                    z["c_pos"][sel], z["c_normal"][sel], z["c_valid"][sel], z["c_tpl"][sel])


def load_leap_records(npz: str | Path, bank: RobotBank, rows=None) -> Records:
    z = np.load(npz)
    hid = str(z["hand_id"])
    sel = slice(None) if rows is None else rows
    k = z["k"][sel]
    tpl = z["template_idx"][sel].astype(np.int64)
    valid = np.arange(tpl.shape[1])[None] < k[:, None]
    e2 = matrix_to_euler_xyz(quat_to_matrix(torch.as_tensor(z["wrist_quat_contact"][sel], dtype=torch.float64))).numpy()
    return _records(bank.index[hid], z["wrist_pos_contact"][sel], e2, z["joints_contact"][sel], z["mask"][sel],
                    z["contact_pos"][sel], z["contact_normal"][sel], valid, np.where(valid, tpl, -1))


# ---------------------------------------------------------------------------------------------- recipe
@dataclass
class StartGoal:
    qc1: torch.Tensor
    t1: torch.Tensor
    R1: torch.Tensor
    qcg: torch.Tensor
    tg: torch.Tensor
    Rg: torch.Tensor


def draw_start(bank: RobotBank, rec: Records, gen: torch.Generator | None = None, xyz_std=0.005, rot_std_deg=5.0,
               joint_frac=0.05) -> StartGoal:
    """KITE's gaussian q1 and the contact-relevant goal mix."""
    dev, N = rec.t2.device, len(rec)
    rnd = lambda *s: torch.randn(*s, generator=gen, device="cpu").to(dev)  # noqa: E731  (CPU generator: reproducible)
    t1 = rec.t2 + rnd(N, 3) * xyz_std
    e = rec.e2 + rnd(N, 3) * math.radians(rot_std_deg)
    e = torch.remainder(e + math.pi, 2 * math.pi) - math.pi
    R1 = euler_xyz_to_matrix(e)
    r = rec.r
    cl, cu = bank.coords_of(r, bank.lower[r]), bank.coords_of(r, bank.upper[r])
    cv = bank.coord_valid[r]
    q1 = rec.qc2 + rnd(N, J_MAX) * joint_frac * (cu - cl)
    q1 = torch.where(cv, torch.minimum(torch.maximum(q1, cl), cu), 0.0)
    qg = torch.where(rec.active & cv, rec.qc2, q1)
    return StartGoal(q1, t1, R1, qg, rec.t2, rec.R2)


def fixed_start(bank, rec: Records, seed: int, **noise) -> StartGoal:
    return draw_start(bank, rec, torch.Generator().manual_seed(seed), **noise)


# ---------------------------------------------------------------------------------------------- sampling
class BalancedSampler:
    """Family uniformly, then morphology uniformly within the family, then an example of that robot."""

    def __init__(self, rec: Records, family_of_robot: list, seed=0):
        self.gen = torch.Generator().manual_seed(seed)
        r = rec.r.cpu().numpy()
        self.by_robot = {int(i): torch.as_tensor(np.flatnonzero(r == i)) for i in np.unique(r)}
        fams = {}
        for i in self.by_robot:
            fams.setdefault(family_of_robot[i], []).append(i)
        self.families = sorted(fams)
        self.robots = [fams[f] for f in self.families]

    def sample(self, n: int) -> torch.Tensor:
        fi = torch.randint(len(self.families), (n,), generator=self.gen)
        out = torch.empty(n, dtype=torch.long)
        for f in range(len(self.families)):
            sel = (fi == f).nonzero().squeeze(1)
            if len(sel) == 0:
                continue
            robots = self.robots[f]
            ri = torch.randint(len(robots), (len(sel),), generator=self.gen)
            for k, rob in enumerate(robots):
                s2 = sel[ri == k]
                pool = self.by_robot[rob]
                out[s2] = pool[torch.randint(len(pool), (len(s2),), generator=self.gen)]
        return out
