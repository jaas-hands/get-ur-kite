"""KITE's decoder heads on the team's LEAP data (the baselines of the evaluation matrix).

`KiteHead` loads a KITE decoder checkpoint (the LEAP head from `scripts/train_leap_head.py`, or
KITE's `release.pt`) and runs one head on `getkite.leap.data.LeapSplit` batches:

    head = KiteHead.load("outputs/leap_head/b0/last.pt")
    wrist_pred, joints_pred = head.predict(batch)          # dataset frames, (B, 4, 4) and (B, 16)

Frames. KITE's LEAP (DRO-Grasp's URDF) and the dataset's nominal hand are the same robot with the
same joint names and order; they differ only in where the wrist frame sits. KITE's configuration is
`[x, y, z, roll, pitch, yaw, q_0 .. q_15]` for the virtual chain (R = Rx(roll) Ry(pitch) Rz(yaw)), and
the palm hangs below it by a fixed offset `T_palm` (DRO's `base_joint`). The dataset's wrist pose is
the palm's pose, so `palm = virtual @ T_palm`. Checked to 0.11 mm / 0.06 deg over all 17 links
(`tests/test_kite_head.py`).

Hand tokens. KITE's head sees 32 points of KITE's own surface template, moved by FK. That is right for
the nominal hand. For lesioned or stretched hands (B1, B2) the tokens must come from that hand: pass
`hand_points` / `hand_normals` to `predict` (e.g. the dataset's 32 points of that hand, `batch["hand_points"]`).
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import torch

from getkite.leap.hands import MAX_DOF

REPO = Path(__file__).resolve().parents[2]
KITE_ROOT = Path(os.environ.get("KITE_ROOT", REPO / "third_party" / "KITE")).resolve()
LEAP_NAME = "leaphand_right"
LEAP_URDF = KITE_ROOT / "assets" / "dro_urdf_robot" / "leaphand" / "leap_hand_right_extended.urdf"
N_WRIST = 6


@contextlib.contextmanager
def _in_kite():
    """KITE resolves some paths relative to its root and imports its modules top-level."""
    old = os.getcwd()
    if str(KITE_ROOT) not in sys.path:
        sys.path.insert(0, str(KITE_ROOT))
    os.chdir(KITE_ROOT)
    try:
        yield
    finally:
        os.chdir(old)


# ---------------------------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------------------------

def _rpy_matrix(r: float, p: float, y: float) -> torch.Tensor:
    """URDF rpy: R = Rz(y) Ry(p) Rx(r)."""
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return torch.tensor([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=torch.float64)


def palm_offset(urdf: Path = LEAP_URDF) -> torch.Tensor:
    """T_palm (4, 4): the palm's pose in the frame of KITE's virtual wrist chain (DRO's `base_joint`)."""
    root = ET.parse(urdf).getroot()
    T = torch.eye(4, dtype=torch.float64)
    # Fixed joints from the end of the virtual chain down to palm_lower.
    child_of = {j.find("parent").get("link"): j for j in root.findall("joint")}
    link = "virtual_link_yaw"
    while link != "palm_lower":
        j = child_of[link]
        if j.get("type") != "fixed":
            raise ValueError(f"non-fixed joint {j.get('name')} between the virtual chain and the palm")
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        M = torch.eye(4, dtype=torch.float64)
        M[:3, :3], M[:3, 3] = _rpy_matrix(*rpy), torch.tensor(xyz, dtype=torch.float64)
        T = T @ M
        link = j.find("child").get("link")
    return T


def euler_xyz_to_matrix(e: torch.Tensor) -> torch.Tensor:
    """(..., 3) intrinsic x-y-z angles -> R = Rx(a) Ry(b) Rz(c)."""
    a, b, c = e.unbind(-1)
    ca, sa, cb, sb, cc, sc = a.cos(), a.sin(), b.cos(), b.sin(), c.cos(), c.sin()
    return torch.stack([
        torch.stack([cb * cc, -cb * sc, sb], -1),
        torch.stack([sa * sb * cc + ca * sc, -sa * sb * sc + ca * cc, -sa * cb], -1),
        torch.stack([-ca * sb * cc + sa * sc, ca * sb * sc + sa * cc, ca * cb], -1),
    ], -2)


def matrix_to_euler_xyz(R: torch.Tensor) -> torch.Tensor:
    """Inverse of `euler_xyz_to_matrix` (b in [-pi/2, pi/2])."""
    b = torch.asin(R[..., 0, 2].clamp(-1.0, 1.0))
    a = torch.atan2(-R[..., 1, 2], R[..., 2, 2])
    c = torch.atan2(-R[..., 0, 1], R[..., 0, 0])
    return torch.stack([a, b, c], -1)


def to_kite_q(wrist: torch.Tensor, joints: torch.Tensor, T_palm: torch.Tensor) -> torch.Tensor:
    """Dataset palm pose (B, 4, 4) + joints (B, 16) -> KITE configuration (B, 22)."""
    V = wrist @ torch.linalg.inv(T_palm.to(wrist))
    return torch.cat([V[:, :3, 3], matrix_to_euler_xyz(V[:, :3, :3]), joints[:, :MAX_DOF]], dim=1)


def from_kite_q(q: torch.Tensor, T_palm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """KITE configuration (B, 22) -> dataset palm pose (B, 4, 4) and joints (B, 16)."""
    V = torch.eye(4, dtype=q.dtype, device=q.device).repeat(q.shape[0], 1, 1)
    V[:, :3, :3], V[:, :3, 3] = euler_xyz_to_matrix(q[:, 3:6]), q[:, :3]
    return V @ T_palm.to(q), q[:, N_WRIST:N_WRIST + MAX_DOF]


# ---------------------------------------------------------------------------------------------
# The head
# ---------------------------------------------------------------------------------------------

@dataclass
class KiteHead:
    bank: object            # KITE's ContactProbePolicyPairedCoTrain
    name: str               # head name, e.g. "leaphand_right"
    robot: object           # KITE's RobotModel of that hand (for its hand tokens)
    T_palm: torch.Tensor
    device: torch.device
    checkpoint: Path | None = None

    @classmethod
    def load(cls, checkpoint, name: str = LEAP_NAME, device="cuda", prefer_ema: bool = True) -> "KiteHead":
        checkpoint = Path(checkpoint).resolve()
        device = torch.device(device)
        with _in_kite():
            from experiment.high_level_policy.contact_encoding import _load_cotrain_probe_policy
            from omegaconf import OmegaConf
            from paper_config import runtime_config
            from robot_model.robot_model import create_robot_model

            meta_path = checkpoint.with_name("dataset_meta.pt")
            meta = torch.load(meta_path, map_location="cpu", weights_only=False)
            cfg = runtime_config(OmegaConf.load(KITE_ROOT / "conf" / "kite.yaml"))
            bank, _, saved_cfg = _load_cotrain_probe_policy(
                cfg, checkpoint_path=str(checkpoint), device=device, prefer_ema=prefer_ema, cotrain_meta=meta,
            )
            bank.eval()
            n_points = int(meta["per_manipulator"][name]["surface_num_points"])
            robot = create_robot_model(name, device=device, num_points=n_points)
            robot.build_hand_fps_template(int(saved_cfg.model.hand_points))
        return cls(bank=bank, name=name, robot=robot, T_palm=palm_offset(), device=device, checkpoint=checkpoint)

    @property
    def head(self):
        return self.bank.head_for(self.name)

    @torch.no_grad()
    def latent(self, contact_pos, contact_normal, valid) -> torch.Tensor:
        """KITE's ψ on world-frame contacts (B, K, 3) x2 with a validity mask (B, K)."""
        cloud = torch.cat([contact_pos, contact_normal], -1).float().to(self.device)
        valid = valid.bool().to(self.device)
        return self.head.encode_contact_latent({"contact_cloud": cloud * valid[..., None], "contact_valid_mask": valid})

    def hand_tokens(self, q_kite: torch.Tensor):
        """KITE's own 32 hand points / normals at KITE configurations (B, 22)."""
        return self.robot.transform_hand_fps_template_batch(q_kite.float().to(self.robot.device))

    @torch.no_grad()
    def predict_kite(self, z, q_init_kite, hand_points=None, hand_normals=None) -> torch.Tensor:
        """One decoder call in KITE's layout: latent (B, D), start (B, 22) -> goal (B, 22)."""
        q = q_init_kite.float().to(self.device)
        if hand_points is None:
            hand_points, hand_normals = self.hand_tokens(q)
        B, dof = q.shape
        batch = {
            "q1_padded": q,
            "hand_points_world": hand_points.float().to(self.device),
            "hand_normals_world": hand_normals.float().to(self.device),
            "action_valid_mask": torch.ones((B, dof), dtype=torch.bool, device=self.device),
            "robot_index": torch.zeros(B, dtype=torch.long, device=self.device),
            "local_dof": torch.full((B,), dof, device=self.device),
        }
        return self.head.predict_action_from_contact_latent(z.float().to(self.device), batch)["action_pred"][:, :dof]

    @torch.no_grad()
    def predict(self, batch: dict, *, use_batch_z: bool = False, dataset_hand_tokens: bool = False):
        """A `LeapSplit` batch -> predicted palm pose (B, 4, 4) and joints (B, 16), dataset frames.

        `use_batch_z` uses `batch["z"]` (our ψ wrapper) instead of running KITE's ψ again; the two agree.
        `dataset_hand_tokens` feeds the dataset's 32 hand points of the batch's hand instead of KITE's
        template (needed for lesioned and stretched hands).
        """
        T_palm = self.T_palm.to(self.device)
        wrist0 = batch["wrist_init"].double().to(self.device)
        q0 = to_kite_q(wrist0, batch["joints_init"].double().to(self.device), T_palm)
        z = batch["z"] if use_batch_z else self.latent(batch["contact_pos"], batch["contact_normal"], batch["contact_valid"])
        points = normals = None
        if dataset_hand_tokens:
            points, normals = batch["hand_points"], batch["hand_normals"]
        q = self.predict_kite(z, q0, points, normals).double()
        return from_kite_q(q, T_palm)
