"""KITE's frozen intent encoder psi: contacts -> latent intent z (256-d).

psi is KITE's ``_ContactOnlyEncoder`` (a 2-layer set transformer over world-frame
(position, outward normal) pairs with a CLS read-out). All heads of the released decoder bank
share it; its weights are read from ``checkpoints/decoder/release.pt`` (EMA weights, as KITE's
scripts use them) and frozen.

    psi = IntentEncoder.load(device="cuda")
    z = psi(contact_pos, contact_normal, k)        # (B, 6, 3), (B, 6, 3), (B,) -> (B, 256)

KITE perturbs psi's input normals while training its decoders (``contact_noise``:
``normal_scale=0.1``, ``pos_scale=0``): pass ``normal_noise=0.1`` during decoder training to
match.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from getkite.kite_bridge import add_kite_to_path
from getkite.leap.paths import kite_checkpoint

MAX_CONTACTS = 6
LATENT_DIM = 256
# psi's architecture in the released checkpoint (conf/kite.yaml: d_model, num_heads, mlp_ratio,
# latent_intent_interface.num_encoder_layers)
PSI_KWARGS = dict(dim=256, num_heads=8, num_layers=2, mlp_ratio=4.0)


def rotate_normals(normals: torch.Tensor, scale: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """KITE's contact-normal augmentation: rotate each normal by a Gaussian rotation vector of std ``scale`` (rad)."""
    rvec = torch.randn(normals.shape, dtype=normals.dtype, device=normals.device, generator=generator) * float(scale)
    theta = rvec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    axis = rvec / theta
    cos, sin = torch.cos(theta), torch.sin(theta)
    return normals * cos + torch.cross(axis, normals, dim=-1) * sin + axis * (axis * normals).sum(-1, keepdim=True) * (1 - cos)


class IntentEncoder(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def load(cls, checkpoint: Path | None = None, device="cpu", head: str = "allegro_right", prefer_ema: bool = True) -> "IntentEncoder":
        add_kite_to_path()
        from networks.contact_probe_policy_paired import _ContactOnlyEncoder

        ck = torch.load(Path(checkpoint or kite_checkpoint()), map_location="cpu", weights_only=False)
        sd = ck["ema_model"] if prefer_ema and ck.get("ema_model") else ck["model"]
        prefix = f"heads.{head}.contact_only_encoder."
        weights = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if not weights:
            raise KeyError(f"no {prefix}* weights in the checkpoint")
        encoder = _ContactOnlyEncoder(**PSI_KWARGS)
        encoder.load_state_dict(weights)
        return cls(encoder).to(device)

    def train(self, mode: bool = True):  # always frozen
        return super().train(False)

    @torch.no_grad()
    def forward(self, contact_pos: torch.Tensor, contact_normal: torch.Tensor, k: torch.Tensor | None = None,
                valid: torch.Tensor | None = None, normal_noise: float = 0.0,
                generator: torch.Generator | None = None) -> torch.Tensor:
        """z for contact sets: pos/normal (B, N, 3); either k (B,) contact counts or valid (B, N) masks."""
        if valid is None:
            if k is None:
                raise ValueError("pass k or valid")
            valid = torch.arange(contact_pos.shape[1], device=contact_pos.device)[None] < k[:, None].to(contact_pos.device)
        pos = contact_pos.float()
        nrm = contact_normal.float()
        if normal_noise > 0:
            nrm = torch.where(valid[..., None], rotate_normals(nrm, normal_noise, generator), nrm)
        cloud = torch.cat([pos, nrm], dim=-1) * valid[..., None]
        return self.encoder(contact_feat_in=cloud, contact_xyz=pos * valid[..., None], contact_mask=valid.bool())
