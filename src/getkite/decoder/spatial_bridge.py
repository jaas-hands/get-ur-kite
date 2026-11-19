"""Spatial bridge (2026-11-17): a randomly initialised MLP that maps ONLY frozen psi's z and the starting wrist's world
pose to the target contact centroid expressed in the starting-wrist frame,

    c_wrist = R0^T (mean_k valid c_k - t0)

Inputs at prediction time: z (KITE psi of the deployment contact request), R0, t0 (start wrist world pose). Target
contacts, goal configuration and correspondences are used only to build the training label (synthetic source data)
and never enter forward(). z is standardised with training statistics and gets no Fourier features (it is a latent
code, not coordinates); only the pose gets the decoder's usual scaled + Fourier encoding.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from .get_scratch import fourier
from .rotations import matrix_to_6d


@dataclass
class BridgeConfig:
    z_dim: int = 256
    hidden: int = 512
    layers: int = 4  # Linear layers incl. output
    pos_scale: float = 2.5  # same world scaling as the decoder (+-0.4 m -> +-1)
    pos_fourier: int = 8
    rot_fourier: int = 8
    out_scale: float = 0.1  # output unit = 10 cm

    def to_dict(self):
        return asdict(self)


def centroid_wrist_label(c_pos, c_valid, R0, t0):
    """Supervision only: R0^T (valid-contact centroid - t0), metres, (N, 3)."""
    v = c_valid.float()[..., None]
    c = (c_pos * v).sum(1) / v.sum(1).clamp_min(1)
    return ((c - t0)[:, None] @ R0).squeeze(1)


class SpatialBridge(nn.Module):
    def __init__(self, c: BridgeConfig = BridgeConfig()):
        super().__init__()
        self.c = c
        self.register_buffer("z_mu", torch.zeros(c.z_dim))
        self.register_buffer("z_sd", torch.ones(c.z_dim))
        n_in = c.z_dim + 9 + 3 * 2 * c.pos_fourier + 6 * 2 * c.rot_fourier
        mods, d = [], n_in
        for _ in range(c.layers - 1):
            mods += [nn.Linear(d, c.hidden), nn.LayerNorm(c.hidden), nn.GELU()]
            d = c.hidden
        self.trunk = nn.Sequential(*mods)
        self.head = nn.Linear(d, 3)

    @torch.no_grad()
    def set_z_stats(self, z):
        self.z_mu.copy_(z.mean(0))
        self.z_sd.copy_(z.std(0).clamp_min(1e-4))

    def features(self, z, R0, t0):
        c = self.c
        tp = t0 * c.pos_scale
        r6 = matrix_to_6d(R0)
        x = torch.cat([(z - self.z_mu) / self.z_sd, tp, r6, fourier(tp, c.pos_fourier), fourier(r6, c.rot_fourier)], -1)
        return self.trunk(x)

    def forward(self, z, R0, t0):
        """(B, z_dim), (B, 3, 3), (B, 3) -> predicted c_wrist (B, 3) in metres."""
        return self.head(self.features(z, R0, t0)) * self.c.out_scale


# ---------------------------------------------------------------------------------------------------------------------
# Contact-set bridge (2026-11-17): same inputs (frozen z, start wrist world pose), predicts the target contact SET in the
# starting-wrist frame: S unordered slots, each (position, unit normal, existence logit). Count = number of slots with
# existence > 0.5; the ground-truth count is never supplied at inference. Trained with an exact set matching (all S!
# slot->contact assignments; S = K_MAX = 6 -> 720) whose cost and loss share the same terms, so a slot's position and
# normal are always supervised by the same target contact.
# ---------------------------------------------------------------------------------------------------------------------
import itertools  # noqa: E402

import torch.nn.functional as F  # noqa: E402


@dataclass
class SetBridgeConfig(BridgeConfig):
    slots: int = 6
    pos_unit: float = 0.01  # position cost/loss: squared error in cm, summed over xyz
    w_normal: float = 10.0  # x (1 - cos)
    w_exist: float = 1.0  # x BCE on existence
    raw_frame: str = "world"  # raw control only: "wrist" = request mapped analytically into the start-wrist frame first
    raw_fourier: int = 8  # raw control only: Fourier octaves on the request's contact positions (pose encoding unaffected)


def contact_set_label(c_pos, c_normal, c_valid, R0, t0):
    """Supervision only: wrist-frame contacts R0^T (c - t0), normals R0^T n; invalid slots zeroed. (N,K,3) x2, (N,K)."""
    v = c_valid[..., None].to(c_pos.dtype)
    return ((c_pos - t0[:, None]) @ R0) * v, (c_normal @ R0) * v, c_valid.bool()


class ContactSetBridge(SpatialBridge):
    def __init__(self, c: SetBridgeConfig = SetBridgeConfig()):
        super().__init__(c)
        self.head = nn.Linear(self.head.in_features, c.slots * 7)
        self.register_buffer("perms", torch.tensor(list(itertools.permutations(range(c.slots))), dtype=torch.long),
                             persistent=False)

    def forward(self, z, R0, t0):
        """-> pos (B,S,3) metres wrist frame, normal (B,S,3) unit, exist_logit (B,S)."""
        o = self.head(self.features(z, R0, t0)).view(len(z), self.c.slots, 7)
        return o[..., :3] * self.c.out_scale, F.normalize(o[..., 3:6], dim=-1), o[..., 6]

    def pair_costs(self, pos, nrm, logit, gp, gn, gv):
        """Per (slot i, target j) cost, (B,S,K). Valid target: position + normal + BCE(exist,1); padded target:
        BCE(exist,0). Pads the target set to S slots."""
        c, S = self.c, self.c.slots
        K = gp.shape[1]
        if K < S:
            pad = lambda x, v=0: F.pad(x, (0, 0) * (x.dim() - 2) + (0, S - K), value=v)  # noqa: E731
            gp, gn, gv = pad(gp), pad(gn), F.pad(gv, (0, S - K), value=False)
        dp = (((pos[:, :, None] - gp[:, None]) / c.pos_unit) ** 2).sum(-1)
        dn = c.w_normal * (1 - (nrm[:, :, None] * gn[:, None]).sum(-1))
        l1 = F.softplus(-logit)[:, :, None]  # -log sigmoid
        l0 = F.softplus(logit)[:, :, None]  # -log(1 - sigmoid)
        m = gv[:, None].expand(-1, S, -1).to(pos.dtype)
        return m * (dp + dn + c.w_exist * l1) + (1 - m) * c.w_exist * l0, dp, dn, m

    def match(self, cost):
        """Exact min-cost assignment slot i -> target perm[i], (B,S) long."""
        P = self.perms
        tot = cost[:, torch.arange(self.c.slots, device=cost.device)[None], P].sum(-1)  # (B, S!)
        return P[tot.argmin(-1)]

    def set_loss(self, pos, nrm, logit, gp, gn, gv):
        cost, dp, dn, m = self.pair_costs(pos, nrm, logit, gp, gn, gv)
        with torch.no_grad():
            a = self.match(cost)
        g = lambda x: x.gather(2, a[..., None]).squeeze(-1)  # noqa: E731
        mv = g(m)
        n_valid = mv.sum(1).clamp_min(1)
        out = {"pos": (g(dp) * mv).sum(1) / n_valid, "normal": (g(dn) * mv).sum(1) / n_valid,
               "exist": F.binary_cross_entropy_with_logits(logit, mv, reduction="none").mean(1) * self.c.w_exist}
        out = {k: v.mean() for k, v in out.items()}
        out["total"] = out["pos"] + out["normal"] + out["exist"]
        return out, a


# ---------------------------------------------------------------------------------------------------------------------
# Raw-contact control (2026-11-17): the identical set bridge, but z is replaced by a learned 256-d encoding of the raw
# contact REQUEST (world-frame positions + normals of its contacts, the same tensors psi consumes). Permutation-
# invariant, padded slots excluded by masking (their contents never matter). Validates the set-output machinery only.
# ---------------------------------------------------------------------------------------------------------------------
class RawSetEncoder(nn.Module):
    def __init__(self, c: SetBridgeConfig, d: int = 256, layers: int = 2, heads: int = 4):
        super().__init__()
        self.c = c
        self.inp = nn.Sequential(nn.Linear(6 + 3 * 2 * c.raw_fourier, d), nn.GELU(), nn.Linear(d, d))
        self.att = nn.ModuleList([nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
                                  for _ in range(layers)])
        self.out = nn.Sequential(nn.Linear(2 * d, c.z_dim), nn.LayerNorm(c.z_dim))

    def forward(self, c_pos, c_normal, c_valid):
        v = c_valid.bool()
        p = c_pos * self.c.pos_scale
        x = self.inp(torch.cat([p, c_normal, fourier(p, self.c.raw_fourier)], -1))
        x = torch.where(v[..., None], x, torch.zeros_like(x))  # padded contents never enter
        for layer in self.att:
            x = layer(x, src_key_padding_mask=~v)
        m = v[..., None].to(x.dtype)
        mean = (x * m).sum(1) / m.sum(1).clamp_min(1)
        mx = torch.where(v[..., None], x, torch.full_like(x, -1e4)).amax(1)
        return self.out(torch.cat([mean, mx], -1))


class RawContactSetBridge(ContactSetBridge):
    """Inputs: the contact request (c_pos, c_normal, c_valid: world frame, as fed to psi) + start wrist pose."""

    def __init__(self, c: SetBridgeConfig = SetBridgeConfig()):
        super().__init__(c)
        self.enc = RawSetEncoder(c)

    def forward(self, c_pos, c_normal, c_valid, R0, t0):
        if self.c.raw_frame == "wrist":
            c_pos, c_normal = request_to_wrist(c_pos, c_normal, c_valid, R0, t0)
        return super().forward(self.enc(c_pos, c_normal, c_valid), R0, t0)


def request_to_wrist(c_pos, c_normal, c_valid, R0, t0):
    """Analytic map of the contact REQUEST into the start-wrist frame (deployment-visible: request + start pose)."""
    v = c_valid[..., None].to(c_pos.dtype)
    return ((c_pos - t0[:, None]) @ R0) * v, (c_normal @ R0) * v


class SetMatcher(nn.Module):
    """The contact-set bridge's exact matching + loss, standalone (used for the direct decoder's auxiliary head)."""

    def __init__(self, c: SetBridgeConfig = SetBridgeConfig()):
        super().__init__()
        self.c = c
        self.register_buffer("perms", torch.tensor(list(itertools.permutations(range(c.slots))), dtype=torch.long),
                             persistent=False)

    pair_costs = ContactSetBridge.pair_costs
    match = ContactSetBridge.match
    set_loss = ContactSetBridge.set_loss
