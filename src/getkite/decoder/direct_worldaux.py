"""direct_worldaux_v1 (2026-11-18): a direct GET decoder with input-level world-contact supervision.

    frozen psi z --(standardise)--> shared intent trunk (MLP, z ONLY) --> h
        h --> M intent tokens (+ start-wrist WORLD pose feature added to every token) --> GET (robot graph, state) --> motion
        h --> training-only world-frame contact-set head (positions, normals, existence)   [auxiliary output only]

Inputs in scope are the same as J1: frozen z, start wrist world pose, current configuration, deterministic robot
description/geometry (corrected hand-frame surface sampling). No learned robot / joint identities.

Differences from J1 (intentional): the movement path NEVER consumes the predicted contacts (positions, normals,
existence logits, masks); there is no reconstructed-contact bottleneck and no exact intent-coordinate transform. The
contact head reads the trunk features BEFORE any robot-transformer mixing; at deployment it is dropped.
Differences from the earlier direct model (scratch_c3_aux50k_10x_wrist): the aux target is WORLD-frame contacts read
from the z-only trunk (the arrangement that worked standalone, audit section 30), not wrist-frame contacts read from the
final mixed tokens; the intent tokens also receive the start-wrist world pose (the earlier model gave it to the root
token only).

Trunk/head hyper-parameters come from the reference world-frame translator config (set_bridge_zworld_c3_20k_b):
hidden 512, 4 linear layers incl. output (3 x [Linear, LayerNorm, GELU] + output), slots 6, out_scale 0.1, and its
set-matching loss (spatial_bridge.ContactSetBridge.set_loss, historical version, unchanged).
"""
from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .get_scratch import GETScratch, ScratchConfig, fourier, mlp
from .rotations import matrix_to_6d
from .spatial_bridge import ContactSetBridge, SetBridgeConfig


class IntentTrunk(nn.Module):
    """z only -> h (B, hidden). Same layer design as SpatialBridge.trunk, without the pose inputs."""

    def __init__(self, c: SetBridgeConfig):
        super().__init__()
        self.c = c
        self.register_buffer("z_mu", torch.zeros(c.z_dim))
        self.register_buffer("z_sd", torch.ones(c.z_dim))
        mods, d = [], c.z_dim
        for _ in range(c.layers - 1):
            mods += [nn.Linear(d, c.hidden), nn.LayerNorm(c.hidden), nn.GELU()]
            d = c.hidden
        self.trunk = nn.Sequential(*mods)
        self.hidden = d

    @torch.no_grad()
    def set_z_stats(self, z):
        self.z_mu.copy_(z.mean(0))
        self.z_sd.copy_(z.std(0).clamp_min(1e-4))

    def forward(self, z):
        return self.trunk((z - self.z_mu) / self.z_sd)


class ContactAuxHead(nn.Module):
    """Training-only: h -> world-frame contact set (S slots: position m, unit normal, existence logit); the set loss is
    the reference translator's exact-matching loss (unchanged)."""

    def __init__(self, c: SetBridgeConfig, hidden: int):
        super().__init__()
        self.c = c
        self.head = nn.Linear(hidden, c.slots * 7)
        self.register_buffer("perms", torch.tensor(list(itertools.permutations(range(c.slots))), dtype=torch.long),
                             persistent=False)

    pair_costs = ContactSetBridge.pair_costs
    match = ContactSetBridge.match
    set_loss = ContactSetBridge.set_loss

    def forward(self, h):
        o = self.head(h).view(len(h), self.c.slots, 7)
        return o[..., :3] * self.c.out_scale, F.normalize(o[..., 3:6], dim=-1), o[..., 6]


class DirectWorldAux(nn.Module):
    def __init__(self, decoder_cfg: ScratchConfig, set_cfg: SetBridgeConfig):
        super().__init__()
        c = decoder_cfg
        assert c.intent_source == "external" and c.frame == "wrist" and c.world_pose_in, \
            "direct_worldaux: external intent tokens, wrist frame, start-wrist world pose input"
        self.trunk = IntentTrunk(set_cfg)
        self.contacts = ContactAuxHead(set_cfg, self.trunk.hidden)  # auxiliary only
        d, nf, nr = c.d_model, 3 * 2 * c.pos_fourier, 6 * 2 * c.rot_fourier
        self.intent_tok = nn.Linear(self.trunk.hidden, c.n_intent_tokens * d)
        self.intent_pose = mlp(9 + nf + nr, d, d)  # start-wrist world pose feature, added to every intent token
        self.decoder = GETScratch(c)
        self.c = self.decoder.c  # evaluate()/target_residuals read the decoder config

    def pose_feature(self, R0w, t0w):
        c = self.c
        wp = torch.cat([t0w * c.pos_scale, matrix_to_6d(R0w)], -1)
        return torch.cat([wp, fourier(wp[:, :3], c.pos_fourier, c.fourier_base), fourier(wp[:, 3:], c.rot_fourier, c.fourier_base)], -1)

    def intent(self, z, R0w, t0w):
        h = self.trunk(z)
        it = self.intent_tok(h).view(len(z), self.c.n_intent_tokens, self.c.d_model)
        return it + self.intent_pose(self.pose_feature(R0w, t0w))[:, None], h

    def forward(self, bank, r, qc0, R0, t0, z, aux=True):
        it, h = self.intent(z, R0, t0)
        out = self.decoder(bank, r, qc0, R0, t0, None, intent_tokens=it)
        if aux:  # computed from h; nothing above reads it
            out["contact_set"] = self.contacts(h)
        return out

    def predict_contacts(self, z):
        """Auxiliary world-frame contact set from z (scoring only; never used for motion)."""
        return self.contacts(self.trunk(z))

    def deploy_parameters(self):
        """Parameters used at inference (the auxiliary contact head is dropped)."""
        return [p for n, p in self.named_parameters() if not n.startswith("contacts.")]
