"""Rotation helpers (torch). Quaternions are (w, x, y, z), matching kite_data. 6D = first two columns of R (KITE)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))


def matrix_to_6d(R: torch.Tensor) -> torch.Tensor:
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def sixd_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt (Zhou et al. 2019)."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def axis_angle_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues: axis (..., 3) unit, angle (...) -> (..., 3, 3)."""
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).reshape(axis.shape[:-1] + (3, 3))
    s = torch.sin(angle)[..., None, None]
    c = torch.cos(angle)[..., None, None]
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device).expand_as(K)
    return eye + s * K + (1 - c) * (K @ K)


def geodesic_angle(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """Angle (rad) of R1^T R2, numerically safe."""
    tr = (R1.transpose(-1, -2) @ R2).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(((tr - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7))


def matrix_to_euler_xyz(R: torch.Tensor) -> torch.Tensor:
    """Intrinsic XYZ Euler angles (R = Rx(a) Ry(b) Rz(c), scipy 'XYZ'), KITE's base-rotation format.
    Used only on data (no gradients needed); at gimbal lock c is set to 0."""
    sy = R[..., 0, 2].clamp(-1.0, 1.0)
    cy = torch.sqrt(R[..., 0, 0] ** 2 + R[..., 0, 1] ** 2)
    b = torch.atan2(sy, cy)  # better conditioned than asin near +-90 deg
    singular = cy < 1e-6
    a = torch.where(singular, torch.atan2(R[..., 2, 1], R[..., 1, 1]), torch.atan2(-R[..., 1, 2], R[..., 2, 2]))
    c = torch.where(singular, torch.zeros_like(b), torch.atan2(-R[..., 0, 1], R[..., 0, 0]))
    return torch.stack([a, b, c], -1)


def euler_xyz_to_matrix(e: torch.Tensor) -> torch.Tensor:
    one, zero = torch.ones_like(e[..., 0]), torch.zeros_like(e[..., 0])
    ax = axis_angle_matrix(torch.stack([one, zero, zero], -1), e[..., 0])
    ay = axis_angle_matrix(torch.stack([zero, one, zero], -1), e[..., 1])
    az = axis_angle_matrix(torch.stack([zero, zero, one], -1), e[..., 2])
    return ax @ ay @ az


def sixd_to_matrix_symmetric(d6: torch.Tensor) -> torch.Tensor:
    """KITE's 6D -> rotation (residual_action_utils.rot6d_to_matrix): symmetric orthogonalisation
    of the two column vectors (not Gram-Schmidt). Matches Gram-Schmidt only on orthonormal input."""
    n = lambda v: F.normalize(v, dim=-1, eps=1e-8)  # noqa: E731
    x, y = n(d6[..., :3]), n(d6[..., 3:])
    a, b = n(x + y), n(x - y)
    x, y = n(a + b), n(a - b)
    return torch.stack([x, y, n(torch.cross(x, y, dim=-1))], dim=-1)
