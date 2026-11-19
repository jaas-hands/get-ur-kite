"""Intent backends: turn a contact set c = {(x_i, n_i)} into tokens the decoder attends to.

Every backend takes contacts already expressed in the model frame (metres) and returns IntentOut:
    tokens (B, M, d_model), token_valid (B, M), z (B, z_dim) - a global vector also fed to every token.

  raw       No z bottleneck: one token per contact. The information upper bound, and the fallback if
            KITE's psi never ships (PROJECT_OVERVIEW open question 1).
  own_psi   Our re-implementation of KITE's psi (set transformer + read-out token -> z in R^256),
            trained jointly with the decoder; z is unpacked into M intent tokens like KITE's decoder.
            Same interface as KITE's, so swapping in the real psi is a backend change.
  kite_psi  KITE's released psi (checkpoints/decoder/release.pt), frozen, with KITE's pretrained
            unpacker (fine-tuned). See kite_release.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class IntentOut:
    tokens: torch.Tensor
    token_valid: torch.Tensor
    z: torch.Tensor


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.GELU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class RawContactIntent(nn.Module):
    def __init__(self, d_model: int, z_dim: int, pos_scale: float):
        super().__init__()
        self.pos_scale = pos_scale
        self.embed = mlp(6, d_model, d_model)
        self.pool = mlp(d_model, d_model, z_dim)
        self.z_dim = z_dim

    def forward(self, pos, normal, valid) -> IntentOut:
        tok = self.embed(torch.cat([pos * self.pos_scale, normal], -1))
        w = valid[..., None].float()
        z = self.pool((tok * w).sum(1) / w.sum(1).clamp_min(1))
        return IntentOut(tok, valid, z)


class SetPsi(nn.Module):
    """KITE-style psi: permutation-invariant transformer over contacts with a read-out token."""

    def __init__(self, z_dim=256, width=256, layers=4, heads=8, pos_scale=2.5):
        super().__init__()
        self.pos_scale = pos_scale
        self.embed = mlp(6, width, width)
        self.readout = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, z_dim))

    def forward(self, pos, normal, valid) -> torch.Tensor:
        tok = self.embed(torch.cat([pos * self.pos_scale, normal], -1))
        x = torch.cat([self.readout.expand(len(tok), -1, -1), tok], 1)
        pad = torch.cat([torch.zeros_like(valid[:, :1]), ~valid], 1)
        return self.out(self.encoder(x, src_key_padding_mask=pad)[:, 0])


class ZIntent(nn.Module):
    """psi(c) -> z, then unpack z into n_tokens intent tokens (KITE's 'unpacker')."""

    def __init__(self, psi: nn.Module, z_dim: int, d_model: int, n_tokens: int = 4, freeze_psi: bool = False):
        super().__init__()
        self.psi, self.z_dim, self.n_tokens, self.freeze_psi = psi, z_dim, n_tokens, freeze_psi
        if freeze_psi:
            self.psi.requires_grad_(False)
        # z standardisation: identity until fit_z_stats() is called (do it for a frozen external psi,
        # whose z scale we don't control)
        self.register_buffer("z_mean", torch.zeros(z_dim))
        self.register_buffer("z_std", torch.ones(z_dim))
        self.unpack = nn.Sequential(nn.LayerNorm(z_dim), nn.Linear(z_dim, n_tokens * d_model))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_psi:
            self.psi.eval()
        return self

    def encode(self, pos, normal, valid) -> torch.Tensor:
        if self.freeze_psi:
            with torch.no_grad():
                z = self.psi(pos, normal, valid)
        else:
            z = self.psi(pos, normal, valid)
        return (z - self.z_mean) / self.z_std

    @torch.no_grad()
    def fit_z_stats(self, batches):
        zs = torch.cat([self.psi(*b) for b in batches])
        self.z_mean.copy_(zs.mean(0))
        self.z_std.copy_(zs.std(0).clamp_min(1e-4))

    def forward(self, pos, normal, valid) -> IntentOut:
        z = self.encode(pos, normal, valid)
        tok = self.unpack(z).view(len(z), self.n_tokens, -1)
        return IntentOut(tok, torch.ones(tok.shape[:2], dtype=torch.bool, device=z.device), z)


class KiteZIntent(nn.Module):
    needs_world = True  # psi must see world-frame contacts, whatever the decoder's frame
    """KITE's released psi (frozen) + KITE's pretrained unpacker (fine-tuned) -> intent tokens.

    The unpacker sees raw z, exactly as in KITE. z itself is returned standardised (identity until
    fit_z_stats is called) for the optional z_in_tokens concatenation.
    """

    def __init__(self, psi: nn.Module, unpacker: nn.Module, z_dim: int, d_model: int, train_unpacker: bool = True):
        super().__init__()
        self.psi, self.unpacker, self.z_dim = psi, unpacker, z_dim
        self.psi.requires_grad_(False)
        self.unpacker.requires_grad_(train_unpacker)
        self.proj = nn.Identity() if z_dim == d_model else nn.Linear(z_dim, d_model)
        self.register_buffer("z_mean", torch.zeros(z_dim))
        self.register_buffer("z_std", torch.ones(z_dim))

    def train(self, mode=True):
        super().train(mode)
        self.psi.eval()
        return self

    @torch.no_grad()
    def raw_z(self, pos, normal, valid):
        return self.psi(pos, normal, valid).float()

    @torch.no_grad()
    def fit_z_stats(self, batches):
        zs = torch.cat([self.raw_z(*b) for b in batches])
        self.z_mean.copy_(zs.mean(0))
        self.z_std.copy_(zs.std(0).clamp_min(1e-4))

    def forward(self, pos, normal, valid) -> IntentOut:
        z = self.raw_z(pos, normal, valid)
        tok = self.proj(self.unpacker(z))
        return IntentOut(tok, torch.ones(tok.shape[:2], dtype=torch.bool, device=z.device),
                         (z - self.z_mean) / self.z_std)


def build_intent(cfg, release=None) -> nn.Module:
    if cfg.intent == "raw":
        return RawContactIntent(cfg.d_model, cfg.z_dim, cfg.resolved_pos_scale())
    if cfg.intent == "own_psi":
        psi = SetPsi(cfg.z_dim, cfg.psi_width, cfg.psi_layers, cfg.n_heads, cfg.resolved_pos_scale())
        if cfg.psi_checkpoint:
            psi.load_state_dict(torch.load(cfg.psi_checkpoint, map_location="cpu"))
        return ZIntent(psi, cfg.z_dim, cfg.d_model, cfg.n_intent_tokens, freeze_psi=cfg.freeze_psi)
    if cfg.intent == "kite_psi":
        # KITE's z is computed from world-frame contacts. With frame=wrist_init the model is fed
        # c_pos_world / c_normal_world (see GETDecoder.intent_inputs); psi's input never changes.
        from .kite_release import KitePsi, load_release
        rel = release or load_release(cfg.kite_ckpt, cfg.kite_root or None)
        assert cfg.z_dim == rel.meta["d_model"], f"set z_dim={rel.meta['d_model']} (KITE's latent size)"
        return KiteZIntent(KitePsi(rel.modules["contact_only_encoder"]), rel.modules["cls_unpacker"],
                           rel.meta["d_model"], cfg.d_model, train_unpacker=not cfg.freeze_unpacker)
    raise ValueError(cfg.intent)
