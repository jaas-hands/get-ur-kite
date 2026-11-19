"""J1: one jointly trained decoder with an explicit internal contact stage (2026-11-18).

    frozen psi z --> world-frame contact-set predictor (ContactSetBridge, inputs: z only, pose held at identity/origin)
                 --> pack (pos*v, normal*v, v) per slot, slots ordered by existence logit, hard existence mask
                 --> raw-contact GET motion decoder (its own exact transform into the start-wrist frame) --> motion

Both learned components are randomly initialised and trained together (contact set loss + motion loss). One forward
interface forward(bank, r, qc0, R0, t0, z) and one checkpoint. Motion-loss gradients reach the contact predictor
through the predicted positions and normals (the analytic wrist transform and the packing are differentiable);
existence is trained by the contact loss only (the hard mask used in the forward pass is the inference mask).
psi stays frozen outside this module. The internal contact representation is kept explicit, not removed.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .get_scratch import GETScratch, ScratchConfig
from .multifamily import K_MAX
from .spatial_bridge import ContactSetBridge, SetBridgeConfig


def pack_contacts(pos, nrm, logit):
    """Differentiable in pos/nrm. Slots sorted by existence logit (desc); valid = sigmoid > 0.5, first slot forced
    valid (as in the two-stage evaluation). -> z (B, 7*S) in RawContacts layout, valid (B, S) bool, order (B, S)."""
    o = logit.argsort(1, descending=True)
    p = pos.gather(1, o[..., None].expand_as(pos))
    n = nrm.gather(1, o[..., None].expand_as(nrm))
    v = logit.gather(1, o).sigmoid() > 0.5
    v = torch.cat([torch.ones_like(v[:, :1]), v[:, 1:]], 1)
    vf = v[..., None].to(p.dtype)
    return torch.cat([p * vf, n * vf, vf], -1).flatten(1), v, o


class JointContactGET(nn.Module):
    def __init__(self, decoder_cfg: ScratchConfig, set_cfg: SetBridgeConfig):
        super().__init__()
        assert decoder_cfg.intent_source == "raw_contacts" and set_cfg.slots == K_MAX
        decoder_cfg.n_contacts, decoder_cfg.z_dim = K_MAX, 7 * K_MAX
        self.contacts = ContactSetBridge(set_cfg)
        self.decoder = GETScratch(decoder_cfg)
        self.c = self.decoder.c  # evaluate()/target_residuals read the decoder config

    def predict_contacts(self, z):
        B = len(z)
        eye = torch.eye(3, device=z.device, dtype=z.dtype).expand(B, 3, 3)
        return self.contacts(z, eye, torch.zeros(B, 3, device=z.device, dtype=z.dtype))  # world frame

    def forward(self, bank, r, qc0, R0, t0, z):
        pos, nrm, logit = self.predict_contacts(z)
        zc, v, _ = pack_contacts(pos, nrm, logit)
        out = self.decoder(bank, r, qc0, R0, t0, zc)
        out["contact_set"] = (pos, nrm, logit)
        out["contact_valid"] = v
        return out
