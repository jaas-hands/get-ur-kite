"""KITE-trunk decoder: KITE's pretrained shared decoder + morphology-conditioned joint tokens.

KITE's released decoder bank already shares almost everything across embodiments (see
kite_release.py). Per embodiment it learns only a joint identity embedding per joint (plus small
I/O layers and normalizers), which is why it cannot decode for a hand it was not trained on. This
module keeps KITE's shared pipeline and replaces exactly that per-embodiment part:

    contacts --psi (KITE, frozen)--> z --unpacker--> 4 z tokens
    32 hand points at q_init (FK) --hand_proj--> hand tokens
    hand tokens <-> z tokens: KITE's bottleneck scene encoder (3 layers)          -> memory (32 tokens)
    action tokens = [wrist xyz, wrist rot] + one token per joint
        joint token j = joint_encoder(j) + qref_joint_proj(q_init_j normalised)
            joint_encoder = "get":  MLP(URDF features rho_j, FK pose of joint j at q_init in world)
                          = "slot": learned embedding per canonical LEAP joint (KITE's recipe, with
                                    one shared table for the whole family) - the baseline
    KITE's action decoder (2 x [self-attn, cross-attn to memory]); with "get", self-attention gets
    learned kinematic-tree biases (GET-Zero), and padded joints are masked
    heads (shared across hands): xyz(3), rot6d(6), one joint MLP applied to every joint token

Outputs follow KITE's residual convention exactly (residual_action_utils.py), so KITE's head
weights are a meaningful initialisation:
    t = t_init + r_xyz / 200                         (world-frame translation, std 5 mm)
    R = R6d(r_rot / 11.459 + I6) @ R_init            (world-frame left delta, std 5 deg)
    q_j = q_init_j + r_j * 0.05 * range_j            (std 5 % of the joint range)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .handbank import JOINT_FEAT_DIM, MAX_DOF, HandBank
from .kite_release import KitePsi, KiteRelease, load_release
from .rotations import matrix_to_6d, matrix_to_euler_xyz, sixd_to_matrix_symmetric

XYZ_STD = 0.005
ROT_STD = math.radians(5.0)
JOINT_STD_FRAC = 0.05
IDENTITY_6D = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
N_POSE = 2
N_SLOTS_CANON = 16  # LEAP joint ids 0..15


def _load(module: nn.Module, state: dict | None):
    if state:
        module.load_state_dict(state)


class TreeBias(nn.Module):
    """Learned per-head attention biases from the kinematic tree, for the action self-attention.
    Token i in [xyz, rot, joint 0..15] maps to tree node [0, 0, 1..16] (both pose tokens = wrist)."""

    def __init__(self, n_heads, max_spd=12, max_anc=6):
        super().__init__()
        self.max_spd, self.max_anc = max_spd, max_anc
        self.spd = nn.Embedding(max_spd + 2, n_heads)
        self.anc = nn.Embedding(max_anc + 1, n_heads)
        self.desc = nn.Embedding(max_anc + 1, n_heads)
        self.pair = nn.Embedding(9, n_heads)
        for e in (self.spd, self.anc, self.desc, self.pair):
            nn.init.zeros_(e.weight)  # starts as KITE's unbiased attention

    def forward(self, g):
        b = self.spd(g["spd"]) + self.anc(g["anc"]) + self.desc(g["desc"]) + self.pair(g["pair"])
        return b.permute(0, 3, 1, 2)


def tree_graph(bank: HandBank, h, max_spd, max_anc):
    node = torch.tensor([0, 0] + list(range(1, MAX_DOF + 1)), device=h.device)
    spd = bank.spd[h][:, node][:, :, node]
    anc = bank.anc[h][:, node][:, :, node]
    none = max_spd + 1
    typ = torch.tensor([0, 1] + [2] * MAX_DOF, device=h.device)
    return {"spd": torch.where(spd < 0, none, spd.clamp(max=max_spd)), "anc": anc.clamp(max=max_anc),
            "desc": anc.transpose(1, 2).clamp(max=max_anc),
            "pair": (typ[:, None] * 3 + typ[None, :]).expand(len(h), -1, -1)}


class KiteTrunkDecoder(nn.Module):
    def __init__(self, cfg: Config, release: KiteRelease | None = None):
        super().__init__()
        self.cfg = cfg
        rel = release or load_release(cfg.kite_ckpt, cfg.kite_root or None)
        m, d = rel.modules, rel.meta["d_model"]
        self.d, self.n_heads = d, rel.meta["num_heads"]
        self.psi = KitePsi(m["contact_only_encoder"]).requires_grad_(False)
        self.unpacker, self.hand_proj = m["cls_unpacker"], m["hand_proj"]
        self.scene, self.action = m["bottleneck_scene_encoder"], m["action_decoder"]
        for name in cfg.kite_freeze:
            getattr(self, name).requires_grad_(False)
        self.register_buffer("z_mean", torch.zeros(d))
        self.register_buffer("z_std", torch.ones(d))

        # per-embodiment I/O, now shared across hands; initialised from one of KITE's heads
        src = rel.per_head.get(cfg.kite_init_head) if cfg.kite_init_head else None
        if cfg.kite_init_head and src is None:
            raise KeyError(f"kite_init_head={cfg.kite_init_head!r} not in {rel.heads}")
        g = lambda k: None if src is None else src.get(k)  # noqa: E731
        self.pose_queries = nn.Parameter(g("action_queries").clone() if src is not None
                                         else torch.randn(N_POSE, d) * 0.02)
        self.qref_xyz, self.qref_rot = nn.Linear(3, d), nn.Linear(3, d)
        self.qref_joint = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        self.token_norm = nn.LayerNorm(d)
        self.head_xyz, self.head_rot = nn.Linear(d, 3), nn.Linear(d, 6)
        self.head_joint = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
        for mod, key in ((self.qref_xyz, "qref_xyz_proj"), (self.qref_rot, "qref_rot_proj"),
                         (self.qref_joint, "qref_joint_proj"), (self.token_norm, "action_token_norm"),
                         (self.head_xyz, "head_xyz"), (self.head_rot, "head_rot"), (self.head_joint, "head_joints")):
            _load(mod, g(key))

        if cfg.joint_encoder == "slot":
            self.slot_embed = nn.Parameter(torch.randn(N_SLOTS_CANON, d) * 0.02)
            self.joint_in, self.tree_bias = None, None
        elif cfg.joint_encoder == "get":
            n_in = 3 + JOINT_FEAT_DIM + 3 + 6 + 3 + (4 if cfg.finger_id else 0)
            self.joint_in = nn.Sequential(nn.Linear(n_in, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
            nn.init.normal_(self.joint_in[-1].weight, std=0.02 / math.sqrt(d))  # start at the scale of KITE's
            nn.init.zeros_(self.joint_in[-1].bias)  # joint identity embeddings (std 0.02)
            self.slot_embed = None
            self.tree_bias = nn.ModuleList(TreeBias(self.n_heads, cfg.max_spd, cfg.max_anc)
                                           for _ in self.action.self_layers) if cfg.tree_bias else None
        else:
            raise ValueError(cfg.joint_encoder)

    def train(self, mode=True):
        super().train(mode)
        self.psi.eval()
        return self

    @torch.no_grad()
    def fit_z_stats(self, batches):
        zs = torch.cat([self.psi(*b).float() for b in batches])
        self.z_mean.copy_(zs.mean(0))
        self.z_std.copy_(zs.std(0).clamp_min(1e-4))

    def hand_points(self, bank, h, Rs, ts):
        idx = bank.kite_hand_idx[h] if self.cfg.hand_points == "kite" else bank.surf_idx[h]
        return bank.points(h, Rs, ts, idx)

    def joint_tokens(self, bank, h, q0, Rs, ts):
        lower, rng = bank.lower[h], bank.range[h]
        qn = 2 * (q0 - lower) / rng - 1  # KITE's absolute normaliser: [-1, 1] over the limits
        valid = bank.dof_valid[h]
        qref = self.qref_joint((qn * valid)[..., None])
        if self.slot_embed is not None:
            return self.slot_embed[bank.slot[h].clamp_min(0)] + qref
        Rj, tj = Rs[:, 1:], ts[:, 1:]  # joint frames at q_init, world
        axis_w = (Rj @ bank.axis[h][..., None]).squeeze(-1)
        f = [qn[..., None], torch.sin(q0)[..., None], torch.cos(q0)[..., None], bank.joint_feat[h],
             tj * 2.5, matrix_to_6d(Rj), axis_w]
        if self.cfg.finger_id:
            f.append(F.one_hot(bank.finger[h].clamp_min(0), 4).float())
        return self.joint_in(torch.cat(f, -1)) + qref

    def forward(self, batch: dict, bank: HandBank) -> dict:
        h, q0, R0, t0 = batch["h"], batch["q_init"], batch["R_init"], batch["t_init"]
        B, valid = len(h), bank.dof_valid[h]
        Rs, ts = bank.fk(h, q0, R0, t0)
        hx, hn = self.hand_points(bank, h, Rs, ts)
        cn = batch.get("c_normal_in", batch["c_normal"])  # train-time normal noise, if enabled

        with torch.no_grad():
            z = self.psi(batch["c_pos"], cn, batch["c_valid"]).float()
        use_amp = self.cfg.amp == "bf16" and q0.is_cuda
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            hand, zt = self.scene(hand_feat=self.hand_proj(torch.cat([hx, hn], -1)), hand_xyz=hx,
                                  z_tokens=self.unpacker(z))
            e = matrix_to_euler_xyz(R0) / math.pi
            pose = self.pose_queries[None].expand(B, -1, -1) + torch.stack(
                [self.qref_xyz(t0 * 2.5), self.qref_rot(e)], 1)  # translation box +-0.4 m -> [-1, 1]
            tok = self.token_norm(torch.cat([pose, self.joint_tokens(bank, h, q0, Rs, ts)], 1))
            key_ok = torch.cat([torch.ones(B, N_POSE, dtype=torch.bool, device=h.device), valid], 1)
            pad = torch.zeros(B, 1, 1, tok.shape[1], device=h.device).masked_fill(~key_ok[:, None, None], float("-inf"))
            g = tree_graph(bank, h, self.cfg.max_spd, self.cfg.max_anc) if self.tree_bias is not None else None
            for i, (sl, cl) in enumerate(zip(self.action.self_layers, self.action.cross_layers)):
                bias = pad if g is None else pad + self.tree_bias[i](g)
                tok = sl(src=tok, src_xyz=None, attn_mask=bias.to(tok.dtype))
                tok = cl(src=tok, src_xyz=None, context=hand, context_xyz=hx, attn_mask=None)
        tok = tok.float()

        r_xyz, r_rot = self.head_xyz(tok[:, 0]), self.head_rot(tok[:, 1])
        r_j = self.head_joint(tok[:, N_POSE:]).squeeze(-1) * valid
        t_hat = t0 + r_xyz * XYZ_STD
        R_hat = sixd_to_matrix_symmetric(r_rot * ROT_STD + IDENTITY_6D.to(tok)) @ R0
        q_hat = torch.where(valid, q0 + r_j * JOINT_STD_FRAC * bank.range[h], q0)
        return {"t": t_hat, "R": R_hat, "q": q_hat, "z": (z - self.z_mean) / self.z_std,
                "r_norm": torch.cat([r_xyz, r_rot, r_j], -1)}
