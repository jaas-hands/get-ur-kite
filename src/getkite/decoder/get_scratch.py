"""get_universal_scratch: a randomly initialised GET decoder over any robot in a RobotBank.

    G(z, q_init, robot description) -> q_goal       (z = frozen KITE psi(contacts), computed outside)

Tokens: [root] [physical joint 0..J_MAX-1] [surface 0..N_SURF-1] [intent 0..M-1]
  root     start floating-base pose in KITE's world frame (t * pos_scale, 6D rotation)
  joint    normalised value, sin/cos (revolute), type (revolute / prismatic / mimic), limits, mimic
           multiplier, axis (joint frame and world), parent->joint transform, world FK pose at q_init.
           One shared projection for every robot: no learned joint, family or embodiment ids.
  surface  a shared point encoder pools N_NBR template points (same rigid body) around each of N_SURF
           centres, posed at q_init: max-pool of MLP([(p - c) * nbr_scale, n]) + MLP([c * pos_scale, n_c]).
  intent   standardised z -> M tokens by a new shared projector (no pretrained KITE unpacker).
Attention: per-layer, per-head learned biases on clipped shortest-path distance, ancestor and descendant
distance and the token-type pair (GET-Zero). Intent tokens and unrelated pairs use the 'none' bucket;
padded joints are masked as keys and excluded from every loss.

Outputs are KITE-normalised residuals (verified against KITE's residual_action_utils, see MODEL_PLAN):
  root   9 = dxyz / 5 mm (world)  |  (6D(R_goal R_init^T) - I6) / 5 deg   -> R = R6d_sym(I6 + .) R_init
  joint  dq / (5 % of range) for every independent coordinate (mimic joints follow their source)
Heads are zero-initialised, so an untrained model predicts q_init exactly.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

from .robots import J_MAX, N_NBR, N_SURF, PRISMATIC, REVOLUTE, RobotBank
from .rotations import matrix_to_6d, sixd_to_matrix_symmetric

T_ROOT, T_JOINT, T_SURF, T_INTENT = range(4)
I6 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
JOINT_IN = 1 + 2 + 3 + 3 + 1 + 3 + 3 + 3 + 6 + 3 + 6  # see _joint_features


@dataclass
class ScratchConfig:
    # architecture
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    mlp_ratio: int = 4
    n_intent_tokens: int = 4
    z_dim: int = 256
    z_in_tokens: bool = False  # also concatenate z into every token input (off: intent only via tokens)
    surface_tokens: bool = True
    tree_bias: bool = True
    max_spd: int = 12
    max_anc: int = 6
    pos_scale: float = 2.5  # KITE world box +-0.4 m -> [-1, 1]
    nbr_scale: float = 20.0  # neighbourhood offsets: 5 cm -> 1
    pos_fourier: int = 0  # extra sin/cos features of world positions (0 = off; diagnostic only)
    rot_fourier: int = 0  # same for the root's and joints' world 6D rotations (diagnostic only)
    fourier_base: float = 1.0  # lowest frequency (cycles per unit of the scaled feature)
    # intent source: "psi" = frozen KITE psi(contacts) -> M intent tokens (the experiment);
    # "raw_contacts" = diagnostic reference that bypasses psi: one token per raw world contact
    # [xyz * pos_scale, normal, Fourier(xyz)], invalid contacts masked; z is then the flattened
    # (K_MAX, 7) [xyz, normal, valid] cloud and z_mu/z_sd are unused.
    intent_source: str = "psi"
    n_contacts: int = 6
    # KITE residual normaliser (closed form)
    res_xyz_m: float = 0.005
    res_rot_deg: float = 5.0
    res_joint_frac: float = 0.05

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.GELU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


def zero_last(m: nn.Sequential):
    nn.init.zeros_(m[-1].weight)
    nn.init.zeros_(m[-1].bias)
    return m


def fourier(x, n, base=1.0):
    if n == 0:
        return x[..., :0]
    f = base * (2.0 ** torch.arange(n, device=x.device, dtype=x.dtype)) * math.pi
    a = x[..., None] * f
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


class GraphBias(nn.Module):
    def __init__(self, heads, max_spd, max_anc):
        super().__init__()
        self.spd = nn.Embedding(max_spd + 2, heads)  # last bucket = no relation
        self.anc = nn.Embedding(max_anc + 1, heads)
        self.desc = nn.Embedding(max_anc + 1, heads)
        self.pair = nn.Embedding(16, heads)
        for e in (self.spd, self.anc, self.desc, self.pair):
            nn.init.zeros_(e.weight)

    def forward(self, g):
        b = self.spd(g["spd"]) + self.anc(g["anc"]) + self.desc(g["desc"]) + self.pair(g["pair"])
        return b.permute(0, 3, 1, 2) + g["key_mask"]


class Block(nn.Module):
    def __init__(self, c: ScratchConfig):
        super().__init__()
        d, self.h = c.d_model, c.n_heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, c.mlp_ratio * d), nn.GELU(), nn.Linear(c.mlp_ratio * d, d))
        self.bias = GraphBias(c.n_heads, c.max_spd, c.max_anc)

    def forward(self, x, g):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=self.bias(g).to(q.dtype))
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))
        return x + self.ff(self.ln2(x))


class GETScratch(nn.Module):
    def __init__(self, c: ScratchConfig):
        super().__init__()
        self.c = c
        d, nf = c.d_model, 3 * 2 * c.pos_fourier
        nr = 6 * 2 * c.rot_fourier
        zd = c.z_dim if c.z_in_tokens else 0
        self.register_buffer("z_mu", torch.zeros(c.z_dim))
        self.register_buffer("z_sd", torch.ones(c.z_dim))
        self.root_in = mlp(9 + nf + nr + zd, d, d)
        self.joint_in = mlp(JOINT_IN + nf + nr + zd, d, d)
        self.nbr_enc = mlp(6, d, d)  # shared point encoder
        self.surf_in = mlp(6 + nf + d + zd, d, d)
        self.type_emb = nn.Embedding(4, d)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.ln_out = nn.LayerNorm(d)
        self.root_head = zero_last(mlp(d, d, 9))
        self.joint_head = zero_last(mlp(d, d, 1))
        # intent encoder built last, so every backbone parameter gets the same initial values whichever
        # intent source is used (the psi and raw_contacts variants differ only in this module)
        assert c.intent_source in ("psi", "raw_contacts"), c.intent_source
        if c.intent_source == "psi":
            self.intent_proj = mlp(c.z_dim, 2 * d, c.n_intent_tokens * d)
        else:
            assert not c.z_in_tokens
            self.contact_in = mlp(6 + nf, d, d)

    @torch.no_grad()
    def set_z_stats(self, mu, sd):
        self.z_mu.copy_(mu)
        self.z_sd.copy_(sd.clamp_min(1e-3))

    # ------------------------------------------------------------------ inputs
    def _joint_features(self, bank: RobotBank, r, qp, Rs, ts):
        lo, up, rng = bank.lower[r], bank.upper[r], bank.range[r]
        rev = (bank.jtype[r] == REVOLUTE)
        mim = bank.mimic_src[r] >= 0
        qn = 2 * (qp - lo) / rng - 1
        sc = torch.stack([torch.where(rev, qp.sin(), 0.0), torch.where(rev, qp.cos(), 0.0)], -1)
        typ = torch.stack([rev & ~mim, ~rev & ~mim, mim], -1).float()
        ax_local = bank.axis[r]
        Rj = Rs[:, 1:]  # joint child frame after motion; the axis is fixed in it
        ax_world = (Rj @ ax_local[..., None]).squeeze(-1)
        A = bank.A[r]
        f = [qn[..., None], sc, typ, torch.stack([lo, up, rng], -1), bank.mimic_mult[r][..., None],
             ax_local, ax_world, A[..., :3, 3] * 10.0, matrix_to_6d(A[..., :3, :3]),
             ts[:, 1:] * self.c.pos_scale, matrix_to_6d(Rj)]
        x = torch.cat(f, -1)
        return x, ts[:, 1:]

    def _graph(self, bank: RobotBank, r, n_body, n_intent, intent_valid=None):
        c, B, dev = self.c, len(r), r.device
        T = n_body + n_intent
        none = c.max_spd + 1
        spd = torch.full((B, T, T), none, dtype=torch.long, device=dev)
        anc = torch.zeros((B, T, T), dtype=torch.long, device=dev)
        s = bank.spd[r][:, :n_body, :n_body]
        if c.tree_bias:
            spd[:, :n_body, :n_body] = torch.where(s < 0, none, s.clamp(max=c.max_spd))
            anc[:, :n_body, :n_body] = bank.anc[r][:, :n_body, :n_body].clamp(max=c.max_anc)
        types = torch.tensor([T_ROOT] + [T_JOINT] * J_MAX + ([T_SURF] * N_SURF if c.surface_tokens else [])
                             + [T_INTENT] * n_intent, device=dev)
        if intent_valid is None:
            intent_valid = torch.ones(B, n_intent, dtype=torch.bool, device=dev)
        valid = torch.cat([bank.node_valid[r][:, :n_body], intent_valid], 1)
        key_mask = torch.zeros((B, 1, 1, T), device=dev).masked_fill(~valid[:, None, None, :], float("-inf"))
        return {"spd": spd, "anc": anc, "desc": anc.transpose(1, 2),
                "pair": (types[:, None] * 4 + types[None, :]).expand(B, T, T), "key_mask": key_mask}

    # ------------------------------------------------------------------ forward
    def forward(self, bank: RobotBank, r, qc0, R0, t0, z):
        """r (B,) robot index, qc0 (B, J_MAX) start coordinates (padded), R0/t0 start base pose (world),
        z (B, z_dim) frozen psi output. Returns normalised residuals and the decoded goal."""
        c, B = self.c, len(r)
        raw = c.intent_source == "raw_contacts"
        zs = z if raw else (z - self.z_mu) / self.z_sd
        zc = [zs] if c.z_in_tokens else []

        def wz(x):
            return torch.cat([x] + [zz[:, None].expand(-1, x.shape[1], -1) for zz in zc], -1)

        qp = bank.expand(r, qc0)
        Rs, ts = bank.fk(r, qp, R0, t0)  # fp32 FK at the start pose
        root = torch.cat([t0 * c.pos_scale, matrix_to_6d(R0), fourier(t0 * c.pos_scale, c.pos_fourier, c.fourier_base),
                          fourier(matrix_to_6d(R0), c.rot_fourier, c.fourier_base)], -1)[:, None]
        jx, jp = self._joint_features(bank, r, qp, Rs, ts)
        jx = torch.cat([jx, fourier(jp * c.pos_scale, c.pos_fourier, c.fourier_base),
                        fourier(matrix_to_6d(Rs[:, 1:]), c.rot_fourier, c.fourier_base)], -1)
        toks = [self.root_in(wz(root)) + self.type_emb.weight[T_ROOT],
                self.joint_in(wz(jx)) + self.type_emb.weight[T_JOINT]]
        if c.surface_tokens:
            cp, cn = bank.points(r, Rs, ts, bank.surf_idx[r])  # (B, S, 3)
            nb = bank.surf_nbr[r].flatten(1)
            npos, nnrm = bank.points(r, Rs, ts, nb)
            npos, nnrm = npos.view(B, N_SURF, N_NBR, 3), nnrm.view(B, N_SURF, N_NBR, 3)
            loc = torch.cat([(npos - cp[:, :, None]) * c.nbr_scale, nnrm], -1)
            pooled = self.nbr_enc(loc).amax(2)
            sx = torch.cat([cp * c.pos_scale, cn, fourier(cp * c.pos_scale, c.pos_fourier, c.fourier_base), pooled], -1)
            toks.append(self.surf_in(wz(sx)) + self.type_emb.weight[T_SURF])
        if raw:
            cz = z.view(B, c.n_contacts, 7)
            cpos, iv = cz[..., :3] * c.pos_scale, cz[..., 6] > 0.5
            it = self.contact_in(torch.cat([cpos, cz[..., 3:6], fourier(cpos, c.pos_fourier, c.fourier_base)], -1))
            n_int = c.n_contacts
        else:
            it = self.intent_proj(zs).view(B, c.n_intent_tokens, c.d_model)
            iv, n_int = None, c.n_intent_tokens
        toks.append(it + self.type_emb.weight[T_INTENT])
        x = torch.cat(toks, 1)
        n_body = 1 + J_MAX + (N_SURF if c.surface_tokens else 0)
        g = self._graph(bank, r, n_body, n_int, iv)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda and getattr(self, "amp", True)):
            for blk in self.blocks:
                x = blk(x, g)
        x = self.ln_out(x.float())

        res_root = self.root_head(x[:, 0])  # (B, 9)
        res_phys = self.joint_head(x[:, 1:1 + J_MAX]).squeeze(-1)  # (B, J) one per physical joint
        cj = bank.coord_joint[r]
        res_coord = torch.where(cj >= 0, torch.gather(res_phys, 1, cj.clamp_min(0)), 0.0)  # mimic outputs unused
        return {"res_root": res_root, "res_coord": res_coord} | self.decode(bank, r, qc0, R0, t0, res_root, res_coord)

    def decode(self, bank, r, qc0, R0, t0, res_root, res_coord):
        c = self.c
        t = t0 + res_root[:, :3] * c.res_xyz_m
        d6 = torch.tensor(I6, device=R0.device, dtype=R0.dtype) + res_root[:, 3:] * math.radians(c.res_rot_deg)
        R = sixd_to_matrix_symmetric(d6) @ R0
        crng = bank.coords_of(r, bank.range[r])
        qc = qc0 + res_coord * c.res_joint_frac * crng
        return {"t": t, "R": R, "qc": torch.where(bank.coord_valid[r], qc, 0.0)}


def target_residuals(bank: RobotBank, r, qc0, R0, t0, qcg, Rg, tg, c: ScratchConfig):
    """KITE's normalised residual target for (start -> goal); inverse of GETScratch.decode."""
    rt = (tg - t0) / c.res_xyz_m
    rr = (matrix_to_6d(Rg @ R0.transpose(1, 2)) - torch.tensor(I6, device=R0.device, dtype=R0.dtype)) / math.radians(c.res_rot_deg)
    crng = bank.coords_of(r, bank.range[r]).clamp_min(1e-6)
    rj = torch.where(bank.coord_valid[r], (qcg - qc0) / (c.res_joint_frac * crng), 0.0)
    return torch.cat([rt, rr], -1), rj
