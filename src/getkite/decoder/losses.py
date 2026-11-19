"""Losses (PROJECT_OVERVIEW 'Loss') and evaluation metrics. Everything here runs in fp32."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .handbank import HandBank
from .rotations import geodesic_angle, matrix_to_6d

XYZ_STD, ROT_STD, JOINT_STD_FRAC = 0.005, math.radians(5.0), 0.05  # KITE's start-pose noise model
_I6 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def kite_residual(t, R, q, b, bank: HandBank):
    """KITE's normalised residual (residual_action_utils + closed-form normaliser), from absolute values:
    [(t - t_init)/5mm, (6D(R R_init^T) - I6)/5deg, (q - q_init)/(0.05 range)], world frame."""
    i6 = torch.tensor(_I6, device=t.device, dtype=t.dtype)
    xyz = (t - b["t_init"]) / XYZ_STD
    rot = (matrix_to_6d(R @ b["R_init"].transpose(1, 2)) - i6) / ROT_STD
    j = (q - b["q_init"]) / (JOINT_STD_FRAC * bank.range[b["h"]])
    return torch.cat([xyz, rot, j], -1)


def _masked_mean(x, m):
    m = m.float()
    return (x * m).sum(-1) / m.sum(-1).clamp_min(1)


def compute_losses(out: dict, b: dict, bank: HandBank, cfg) -> dict:
    h, valid = b["h"], b["dof_valid"]
    rng, lower, upper = bank.range[h], bank.lower[h], bank.upper[h]
    sp, sr = cfg.sigma_pos, math.radians(cfg.sigma_rot_deg)

    # action: wrist and fingers averaged separately so many joints don't drown out the wrist
    l_wrist = ((out["t"] - b["t_goal"]) ** 2).sum(-1) / sp**2 + \
        ((out["R"] - b["R_goal"]) ** 2).sum((-1, -2)) / (2 * sr**2)  # chordal ~ angle^2 for small angles
    dq = out["q"] - b["q_goal"]
    if cfg.joint_loss_units == "range":
        dq = dq / rng
    l_joint = _masked_mean(dq**2, valid)

    # contact: FK of the prediction must put the labelled template points on the targets
    R, t = bank.fk(h, out["q"], out["R"], out["t"])
    x, n = bank.points(h, R, t, b["tidx"])
    cv = b["c_valid"]
    l_cpos = _masked_mean(((x - b["c_pos"]) ** 2).sum(-1) / sp**2, cv)
    l_cnrm = _masked_mean(1 - (n * b["c_normal"]).sum(-1), cv)

    # self-model (get arch): where each joint's frame lands at q_goal
    l_fk = None
    if "p_fk" in out:
        with torch.no_grad():
            _, tg = bank.fk(h, b["q_goal"], b["R_goal"], b["t_goal"])
        l_fk = _masked_mean(((out["p_fk"] - tg[:, 1:]) ** 2).sum(-1) / sp**2, valid)

    l_limit = (_masked_mean(F.relu((out["q"] - upper) / rng) ** 2 + F.relu((lower - out["q"]) / rng) ** 2, valid)
               * valid.float().sum(-1))
    l_mask = None
    if "mask_logit" in out:
        l_mask = _masked_mean(F.binary_cross_entropy_with_logits(out["mask_logit"], b["mask"].float(),
                                                                 reduction="none"), valid)

    # KITE's loss: every residual channel in units of its start-pose noise std, masked mean over
    # 9 pose channels + the hand's joints (non-moving joints have target residual 0)
    rv = torch.cat([torch.ones_like(valid[:, :9]), valid], -1).float()
    r_pred = kite_residual(out["t"], out["R"], out["q"], b, bank)
    r_tgt = kite_residual(b["t_goal"], b["R_goal"], b["q_goal"], b, bank)
    l_res = ((r_pred - r_tgt) ** 2 * rv).sum(-1) / rv.sum(-1)

    parts = {"residual": l_res, "wrist": l_wrist, "joint": l_joint, "contact": l_cpos + cfg.w_normal * l_cnrm,
             "limit": l_limit}
    w = {"residual": cfg.w_residual, "wrist": cfg.w_wrist, "joint": cfg.w_joint, "contact": cfg.w_contact,
         "limit": cfg.w_limit}
    if l_fk is not None:
        parts["fk"], w["fk"] = l_fk, cfg.w_fk
    if l_mask is not None:
        parts["mask"], w["mask"] = l_mask, cfg.w_mask
    losses = {k: v.mean() for k, v in parts.items()}
    losses["total"] = sum(w[k] * losses[k] for k in parts)
    return losses


@torch.no_grad()
def contact_metrics(bank: HandBank, b: dict, q, R, t, cfg, clamp: bool = True, nearest: bool = True) -> dict:
    """Per-example metrics for a configuration (q, R, t). Returns tensors of shape (B,)."""
    h, valid, cv = b["h"], b["dof_valid"], b["c_valid"]
    lower, upper = bank.lower[h], bank.upper[h]
    viol = (((q > upper + 1e-3) | (q < lower - 1e-3)) & valid).any(-1).float()
    if clamp:  # deployment clamps to limits; score what would be executed
        q = torch.minimum(torch.maximum(q, lower), upper)
    Rs, ts = bank.fk(h, q, R, t)
    x, n = bank.points(h, Rs, ts, b["tidx"])
    pe = (x - b["c_pos"]).norm(dim=-1) * 1000
    ne = torch.rad2deg(torch.acos((n * b["c_normal"]).sum(-1).clamp(-1, 1)))
    ok = ((pe <= cfg.success_pos_mm) & (ne <= cfg.success_normal_deg)) | ~cv
    m = {
        "contact_mm": _masked_mean(pe, cv), "normal_deg": _masked_mean(ne, cv),
        "success": ok.all(-1).float(), "limit_violation": viol,
        "wrist_mm": (t - b["t_goal"]).norm(dim=-1) * 1000,
        "wrist_deg": torch.rad2deg(geodesic_angle(R, b["R_goal"])),
        "joint_deg_moving": _masked_mean(torch.rad2deg((q - b["q_goal"]).abs()), b["mask"] & valid),
    }
    if nearest:  # assignment-free: any template point may realise a target (decoder chose another finger)
        P, _ = bank.all_points(h, Rs, ts)
        d = torch.cdist(b["c_pos"], P).amin(-1) * 1000
        m["contact_mm_nearest"] = _masked_mean(d, cv)
    return m
