"""GET decoder: G_phi(intent, q_init, rho_e, G_e) -> q_goal for any LEAP variant, one set of weights.

Token sequence (length T = 1 + MAX_DOF + N_SURF + M):
    [wrist] [joint 0..15] [surface 0..31] [intent 0..M-1]
Padded joints are masked out as keys. Attention logits get learned per-head, per-layer biases:
    b_spd[shortest path] + b_anc[ancestor dist] + b_desc[descendant dist] + b_type[type_i, type_j]
Body tokens (wrist, joints, surface) use the kinematic tree; intent tokens use a 'no relation' bucket,
so they interact with the body only through content and the type-pair bias.

Outputs are residuals from q_init and are zero-initialised, so an untrained model predicts q_init:
    wrist:  t = t_init + R_init (wrist_step * dt);  R = R_init * R6d(identity + d6)
    joints: q_j = q_init_j + range_j * out_j       (shared head: any number of joints)
    self-model: p_j = p_j(q_init) + fk_step * out_j (where joint j's frame lands at q_goal)
    mask logits: which joints should move (auxiliary)
FK runs in fp32; only the transformer runs under autocast (bf16 FK error would be ~mm).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .handbank import JOINT_FEAT_DIM, MAX_DOF, N_SURF, HandBank
from .intent import build_intent, mlp

N_SLOTS_CANON = 16  # LEAP joint ids 0..15 (as in kite_trunk)
from .rotations import matrix_to_6d, sixd_to_matrix

T_WRIST, T_JOINT, T_SURF, T_INTENT = range(4)
IDENTITY_6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


class GraphBias(nn.Module):
    def __init__(self, n_heads, max_spd, max_anc):
        super().__init__()
        self.spd = nn.Embedding(max_spd + 2, n_heads)  # last index = no relation
        self.anc = nn.Embedding(max_anc + 1, n_heads)
        self.desc = nn.Embedding(max_anc + 1, n_heads)
        self.pair = nn.Embedding(16, n_heads)
        for e in (self.spd, self.anc, self.desc, self.pair):
            nn.init.zeros_(e.weight)

    def forward(self, g) -> torch.Tensor:
        b = self.spd(g["spd"]) + self.anc(g["anc"]) + self.desc(g["desc"]) + self.pair(g["type_pair"])
        return b.permute(0, 3, 1, 2) + g["key_mask"]  # (B, H, T, T)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d, h = cfg.d_model, cfg.n_heads
        self.h, self.dropout = h, cfg.dropout
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, cfg.ff_mult * d), nn.GELU(), nn.Linear(cfg.ff_mult * d, d))
        self.bias = GraphBias(h, cfg.max_spd, cfg.max_anc)

    def forward(self, x, g):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        bias = self.bias(g).to(q.dtype)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))
        return x + self.ff(self.ln2(x))


def zero_last(m: nn.Sequential, bias=None):
    nn.init.zeros_(m[-1].weight)
    with torch.no_grad():
        m[-1].bias.copy_(torch.zeros_like(m[-1].bias) if bias is None else torch.tensor(bias))
    return m


class GETDecoder(nn.Module):
    def __init__(self, cfg: Config, release=None):
        super().__init__()
        self.cfg = cfg
        d, zd = cfg.d_model, (cfg.z_dim if cfg.z_in_tokens else 0)
        self.ps = cfg.resolved_pos_scale()
        self.intent = build_intent(cfg, release)
        # A world-frame intent (KITE's z) under a wrist_init decoder frame is only decodable if the decoder
        # knows where the start wrist is in that world frame: the wrist token then also gets the world pose.
        self.world_pose_in = cfg.frame == "wrist_init" and getattr(self.intent, "needs_world", False)
        self.wrist_in = mlp(9 + (9 if self.world_pose_in else 0) + zd, d, d)
        if cfg.joint_encoder == "slot":  # KITE's recipe: one embedding per canonical LEAP joint id + q_init only
            self.slot_embed = nn.Parameter(torch.randn(N_SLOTS_CANON, d) * 0.02)
            self.joint_in = mlp(3 + zd, d, d)
        elif cfg.joint_encoder == "get":  # from the URDF: joint features + FK pose at q_init
            self.slot_embed = None
            self.joint_in = mlp(3 + JOINT_FEAT_DIM + 9 + (4 if cfg.finger_id else 0) + zd, d, d)
        else:
            raise ValueError(cfg.joint_encoder)
        self.surf_in = mlp(3 + 3 + 5 + zd, d, d)
        self.type_emb = nn.Embedding(4, d)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_out = nn.LayerNorm(d)
        self.wrist_head = zero_last(mlp(d, d, 9), (0, 0, 0) + IDENTITY_6D)
        self.joint_head = zero_last(mlp(d, d, 1))
        self.fk_head = zero_last(mlp(d, d, 3))
        self.mask_head = mlp(d, d, 1)

    # ------------------------------------------------------------------ graph
    def _graph(self, bank: HandBank, h, n_intent, intent_valid):
        n_body = 1 + MAX_DOF + (N_SURF if self.cfg.surface_tokens else 0)
        T = n_body + n_intent
        B, dev = len(h), h.device
        none = self.cfg.max_spd + 1
        spd = torch.full((B, T, T), none, dtype=torch.long, device=dev)
        anc = torch.zeros((B, T, T), dtype=torch.long, device=dev)
        s = bank.spd[h][:, :n_body, :n_body]
        spd[:, :n_body, :n_body] = torch.where(s < 0, none, s.clamp(max=self.cfg.max_spd))
        anc[:, :n_body, :n_body] = bank.anc[h][:, :n_body, :n_body].clamp(max=self.cfg.max_anc)
        if not self.cfg.tree_bias:  # graph-off ablation: every pair looks unrelated; only token types remain
            spd.fill_(none)
            anc.zero_()
        types = torch.tensor([T_WRIST] + [T_JOINT] * MAX_DOF + ([T_SURF] * N_SURF if self.cfg.surface_tokens else [])
                             + [T_INTENT] * n_intent, device=dev)
        valid = torch.cat([bank.node_valid[h][:, :n_body], intent_valid], 1)
        key_mask = torch.zeros((B, 1, 1, T), device=dev).masked_fill(~valid[:, None, None, :], float("-inf"))
        return {"spd": spd, "anc": anc, "desc": anc.transpose(1, 2),
                "type_pair": (types[:, None] * 4 + types[None, :]).expand(B, T, T), "key_mask": key_mask}

    def intent_inputs(self, batch: dict):
        """(pos, normal, valid) for the intent backend. KITE's psi needs world-frame contacts: when the
        decoder runs in wrist_init, rotate the (possibly noise-jittered) normals back with R_init_world."""
        n = batch.get("c_normal_in", batch["c_normal"])
        if getattr(self.intent, "needs_world", False):
            if "c_pos_world" not in batch:
                assert self.cfg.frame == "world", "wrist_init batch lacks c_pos_world"
                return batch["c_pos"], n, batch["c_valid"]
            return batch["c_pos_world"], n @ batch["R_init_world"].transpose(1, 2), batch["c_valid"]
        return batch["c_pos"], n, batch["c_valid"]

    # ------------------------------------------------------------------ forward
    def forward(self, batch: dict, bank: HandBank) -> dict:
        cfg, ps = self.cfg, self.ps
        h, q0, R0, t0 = batch["h"], batch["q_init"], batch["R_init"], batch["t_init"]
        B = len(h)
        lower, rng = bank.lower[h], bank.range[h]
        Rs, ts = bank.fk(h, q0, R0, t0)  # fp32 FK at the current state: p_j(q_t)

        it = self.intent(*self.intent_inputs(batch))
        zcat = [it.z] if cfg.z_in_tokens else []

        def with_z(x, n):
            return torch.cat([x] + [z[:, None].expand(-1, n, -1) for z in zcat], -1)

        wf = [t0 * ps, matrix_to_6d(R0)]
        if self.world_pose_in:  # world box is +-0.4 m -> [-1, 1], as KITE's qref_xyz
            wf += [batch["t_init_world"] * 2.5, matrix_to_6d(batch["R_init_world"])]
        wrist = self.wrist_in(with_z(torch.cat(wf, -1)[:, None], 1))
        qn = 2 * (q0 - lower) / rng - 1
        jf = [qn[..., None], torch.sin(q0)[..., None], torch.cos(q0)[..., None]]
        if self.slot_embed is None:
            jf += [bank.joint_feat[h], ts[:, 1:] * ps, matrix_to_6d(Rs[:, 1:])]
            if cfg.finger_id:
                jf.append(F.one_hot(bank.finger[h].clamp_min(0), 4).float())
        joints = self.joint_in(with_z(torch.cat(jf, -1), MAX_DOF))
        if self.slot_embed is not None:
            joints = joints + self.slot_embed[bank.slot[h].clamp_min(0)]
        toks = [wrist + self.type_emb.weight[T_WRIST], joints + self.type_emb.weight[T_JOINT]]
        if cfg.surface_tokens:
            sidx = bank.surf_idx[h]
            sp, sn = bank.points(h, Rs, ts, sidx)
            reg = F.one_hot(torch.gather(bank.tpl_region[h], 1, sidx), 5).float()
            toks.append(self.surf_in(with_z(torch.cat([sp * ps, sn, reg], -1), N_SURF)) + self.type_emb.weight[T_SURF])
        toks.append(it.tokens + self.type_emb.weight[T_INTENT])
        x = torch.cat(toks, 1)
        g = self._graph(bank, h, it.tokens.shape[1], it.token_valid)

        use_amp = cfg.amp == "bf16" and x.is_cuda
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            for blk in self.blocks:
                x = blk(x, g)
        x = self.ln_out(x.float())

        hw, hj = x[:, 0], x[:, 1:1 + MAX_DOF]
        w = self.wrist_head(hw)
        t_hat = t0 + (R0 @ (w[:, :3] * cfg.wrist_step)[..., None]).squeeze(-1)
        R_hat = R0 @ sixd_to_matrix(w[:, 3:])
        valid = bank.dof_valid[h]
        q_hat = torch.where(valid, q0 + rng * self.joint_head(hj).squeeze(-1), q0)
        p_hat = ts[:, 1:] + cfg.fk_step * self.fk_head(hj)
        return {"t": t_hat, "R": R_hat, "q": q_hat, "p_fk": p_hat,
                "mask_logit": self.mask_head(hj).squeeze(-1), "z": it.z}


def build_model(cfg: Config, release=None) -> nn.Module:
    """The decoder named by cfg.arch. release: a preloaded KiteRelease (avoids re-reading the file)."""
    if cfg.arch == "kite_trunk":
        from .kite_trunk import KiteTrunkDecoder
        return KiteTrunkDecoder(cfg, release)
    if cfg.arch == "get":
        return GETDecoder(cfg, release)
    raise ValueError(cfg.arch)
