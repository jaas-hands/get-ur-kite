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
    intent_source: str = "psi"  # also "external": intent tokens are built OUTSIDE (direct_worldaux) and passed to forward()
    n_contacts: int = 6
    contact_fourier: int = -1  # Fourier octaves on raw contact xyz only (-1 = same as pos_fourier)
    # geometry frame: "world" (KITE's) or "wrist" = the start wrist frame (R0, t0). In "wrist" every geometric
    # input (contacts, normals, hand FK poses, joint axes, surface points) is expressed as R0^T (x - t0), the root
    # token becomes the identity pose, and the wrist update is predicted in that frame:
    #   t = t0 + R0 (dxyz * res_xyz_m),  R = R0 R6d_sym(I6 + d6 * res_rot)     (world: t0 + dxyz, R6d(.) R0)
    # Joint residuals are unchanged. With raw_contacts the contacts are transformed into the wrist frame. With psi,
    # psi still sees the original world-frame contacts (KITE-compatible); z is not geometric and is never transformed
    # or Fourier-encoded, so the decoder must also be given the start wrist's world pose (world_pose_in=True).
    frame: str = "world"
    # add the start wrist's world pose [t0 * pos_scale, 6D(R0), Fourier(t0 * pos_scale; pos_fourier),
    # Fourier(6D(R0); rot_fourier)] to the root token through its own projection (wrist frame only; required for psi)
    world_pose_in: bool = False
    # intent_source="psi_scene": z (raw, unstandardised) goes through KITE's released, frozen hand_proj ->
    # cls_unpacker -> bottleneck_scene_encoder together with KITE's 32 hand tokens (greedy link-local FPS, posed at
    # q_init in the WORLD frame, exactly as KITE feeds them); the resulting 32 hand + 4 z tokens become the intent
    # tokens through a new projection. Hand tokens additionally carry their wrist-frame position/normal.
    kite_ckpt: str = ""
    kite_root: str = ""
    kite_hand_points: int = 32
    # spatial bridge (2026-11-17; psi + wrist + world_pose_in only): a random-init SpatialBridge(z, R0w, t0w) predicts
    # the target contact centroid in the start-wrist frame (out["bridge_c"], metres; auxiliary-supervised by the
    # trainer, labels from training data only). Its prediction (wrist-frame position, same encoding as hand surface
    # points) and hidden features form ONE extra intent token. Gradients from the decoder loss also reach the bridge.
    bridge: bool = False
    bridge_hidden: int = 512
    bridge_layers: int = 4
    # direct decoder auxiliary contact-set head (2026-11-17): TRAINING-ONLY. Reads the final-layer intent tokens (mean)
    # + root token and predicts the request's contact set in the start-wrist frame (out["aux_set"]: positions m,
    # unit normals, existence logits), supervised by the trainer with the set-matching loss. Nothing in the motion
    # path (root/joint heads, decode) reads it; drop it at deployment.
    aux_set: bool = False
    aux_set_slots: int = 6
    aux_set_hidden: int = 256
    amp: bool = True  # bf16 autocast in the transformer blocks on CUDA (False = fp32 throughout)
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
        assert c.intent_source in ("psi", "raw_contacts", "psi_scene", "external"), c.intent_source
        assert c.frame in ("world", "wrist"), c.frame
        assert not c.world_pose_in or c.frame == "wrist", "world_pose_in only adds information in the wrist frame"
        assert c.frame == "world" or c.intent_source == "raw_contacts" or c.world_pose_in, \
            "psi's z is world-frame: a wrist-frame psi decoder needs world_pose_in=True"
        if c.intent_source in ("psi", "psi_scene", "external"):
            assert c.contact_fourier <= 0, "contact Fourier features apply to raw contact xyz only, never to z"
        if c.intent_source == "psi":
            self.intent_proj = mlp(c.z_dim, 2 * d, c.n_intent_tokens * d)
        elif c.intent_source == "psi_scene":
            from .kite_release import load_release
            rel = load_release(c.kite_ckpt, c.kite_root or None)
            kd = rel.meta["d_model"]
            assert c.z_dim == kd, (c.z_dim, kd)
            self.kite = nn.ModuleDict({k: rel.modules[k] for k in ("hand_proj", "cls_unpacker", "bottleneck_scene_encoder")})
            self.kite.eval().requires_grad_(False)
            self.scene_in = mlp(kd + 1 + 6 + nf, d, d)
        elif c.intent_source == "external":  # no intent encoder here: the caller supplies (B, M, d_model) intent tokens
            assert not c.z_in_tokens and not c.bridge and not c.aux_set
        else:
            assert not c.z_in_tokens
            self.contact_in = mlp(6 + 3 * 2 * self._cf(), d, d)
        if c.world_pose_in:  # built after everything else: backbone initial values unchanged
            self.world_pose_proj = mlp(9 + nf + nr, d, d)
        if c.bridge:  # built last of all: backbone, intent and world-pose initial values unchanged
            assert c.intent_source == "psi" and c.frame == "wrist" and c.world_pose_in, "bridge: psi + wrist + world_pose_in"
            from .spatial_bridge import BridgeConfig, SpatialBridge
            self.bridge = SpatialBridge(BridgeConfig(z_dim=c.z_dim, hidden=c.bridge_hidden, layers=c.bridge_layers,
                                                     pos_scale=c.pos_scale))
            self.bridge_tok = mlp(3 + nf + c.bridge_hidden, d, d)
        if c.aux_set:  # built last: backbone initial values unchanged
            assert c.intent_source == "psi", "aux_set: the direct psi decoder"
            h = c.aux_set_hidden
            self.aux_head = nn.Sequential(nn.Linear(2 * d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 7 * c.aux_set_slots))

    def train(self, mode=True):  # KITE's frozen modules never leave eval mode
        super().train(mode)
        if hasattr(self, "kite"):
            self.kite.eval()
        return self

    def _kite_fps(self, bank):
        idx = getattr(bank, "_kite_fps_idx", None)
        if idx is None or idx.shape[1] != self.c.kite_hand_points:
            from .handbank import kite_local_fps
            idx = torch.stack([torch.as_tensor(kite_local_fps(d.tpl_pos, self.c.kite_hand_points)) for d in bank.descs])
            bank._kite_fps_idx = idx = idx.to(bank.lower.device)
        return idx

    @torch.no_grad()
    def kite_scene(self, bank, r, Rs, ts, R0w, t0w, z):
        """KITE's frozen scene encoding. Rs/ts: FK in the model's geometry frame; returns hand_feat (B,N,kd),
        z_tokens (B,4,kd), hand points/normals in the model frame (B,N,3)."""
        hp, hn = bank.points(r, Rs, ts, self._kite_fps(bank)[r])
        if self.c.frame == "wrist":  # back to the world frame for KITE
            hpw, hnw = hp @ R0w.transpose(1, 2) + t0w[:, None], hn @ R0w.transpose(1, 2)
        else:
            hpw, hnw = hp, hn
        hpw, hnw, z = hpw.float(), hnw.float(), z.float()
        hf = self.kite["hand_proj"](torch.cat([hpw, hnw], -1))
        zt = self.kite["cls_unpacker"](z)
        hf, zt = self.kite["bottleneck_scene_encoder"](hand_feat=hf, hand_xyz=hpw, z_tokens=zt)
        return hf, zt, hp, hn

    def _cf(self):
        return self.c.pos_fourier if self.c.contact_fourier < 0 else self.c.contact_fourier

    @torch.no_grad()
    def set_z_stats(self, mu, sd):
        self.z_mu.copy_(mu)
        self.z_sd.copy_(sd.clamp_min(1e-3))
        if self.c.bridge:
            self.bridge.z_mu.copy_(mu)
            self.bridge.z_sd.copy_(sd.clamp_min(1e-3))

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
    def forward(self, bank: RobotBank, r, qc0, R0, t0, z, intent_tokens=None):
        """r (B,) robot index, qc0 (B, J_MAX) start coordinates (padded), R0/t0 start base pose (world),
        z (B, z_dim) frozen psi output. Returns normalised residuals and the decoded goal.
        intent_source="external": z is unused and intent_tokens (B, M, d_model) are the intent tokens."""
        c, B = self.c, len(r)
        assert (intent_tokens is not None) == (c.intent_source == "external")
        raw = c.intent_source == "raw_contacts"
        zs = z if c.intent_source != "psi" else (z - self.z_mu) / self.z_sd
        zc = [zs] if c.z_in_tokens else []

        def wz(x):
            return torch.cat([x] + [zz[:, None].expand(-1, x.shape[1], -1) for zz in zc], -1)

        qp = bank.expand(r, qc0)
        R0w, t0w = R0, t0  # the start wrist pose in the world: kept for the contact transform and for decode
        if c.frame == "wrist":  # every geometric input in the start wrist frame; the root is then the identity
            R0 = torch.eye(3, device=R0.device, dtype=R0.dtype).expand(B, 3, 3)
            t0 = torch.zeros_like(t0)
        Rs, ts = bank.fk(r, qp, R0, t0)  # fp32 FK at the start pose
        root = torch.cat([t0 * c.pos_scale, matrix_to_6d(R0), fourier(t0 * c.pos_scale, c.pos_fourier, c.fourier_base),
                          fourier(matrix_to_6d(R0), c.rot_fourier, c.fourier_base)], -1)[:, None]
        jx, jp = self._joint_features(bank, r, qp, Rs, ts)
        jx = torch.cat([jx, fourier(jp * c.pos_scale, c.pos_fourier, c.fourier_base),
                        fourier(matrix_to_6d(Rs[:, 1:]), c.rot_fourier, c.fourier_base)], -1)
        root_tok = self.root_in(wz(root)) + self.type_emb.weight[T_ROOT]
        if c.world_pose_in:
            wp = torch.cat([t0w * c.pos_scale, matrix_to_6d(R0w)], -1)
            root_tok = root_tok + self.world_pose_proj(torch.cat(
                [wp, fourier(wp[:, :3], c.pos_fourier, c.fourier_base), fourier(wp[:, 3:], c.rot_fourier, c.fourier_base)], -1))[:, None]
        toks = [root_tok,
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
            iv = cz[..., 6] > 0.5
            if c.frame == "wrist":
                m = iv[..., None].to(cz.dtype)
                cz = torch.cat([((cz[..., :3] - t0w[:, None]) @ R0w) * m, (cz[..., 3:6] @ R0w) * m, cz[..., 6:]], -1)
            cpos = cz[..., :3] * c.pos_scale
            it = self.contact_in(torch.cat([cpos, cz[..., 3:6], fourier(cpos, self._cf(), c.fourier_base)], -1))
            n_int = c.n_contacts
        elif c.intent_source == "external":
            it, iv, n_int = intent_tokens, None, intent_tokens.shape[1]
        elif c.intent_source == "psi_scene":
            hf, zt, hp, hn = self.kite_scene(bank, r, Rs, ts, R0w, t0w, z)
            hpos = hp * c.pos_scale
            hx = torch.cat([hf, torch.zeros_like(hf[..., :1]), hpos, hn, fourier(hpos, c.pos_fourier, c.fourier_base)], -1)
            zx = torch.cat([zt, torch.ones_like(zt[..., :1]), torch.zeros(B, zt.shape[1], hx.shape[-1] - zt.shape[-1] - 1,
                                                                          device=zt.device, dtype=zt.dtype)], -1)
            it = self.scene_in(torch.cat([hx, zx], 1))
            iv, n_int = None, hx.shape[1] + zx.shape[1]
        else:
            it = self.intent_proj(zs).view(B, c.n_intent_tokens, c.d_model)
            iv, n_int = None, c.n_intent_tokens
            if c.bridge:  # inputs: z and the start wrist WORLD pose only
                bf = self.bridge.features(z, R0w, t0w)
                bc = self.bridge.head(bf) * self.bridge.c.out_scale  # (B, 3) wrist frame, metres
                bpos = bc * c.pos_scale
                bt = self.bridge_tok(torch.cat([bpos, fourier(bpos, c.pos_fourier, c.fourier_base), bf], -1))
                it = torch.cat([it, bt[:, None]], 1)
                n_int += 1
        toks.append(it + self.type_emb.weight[T_INTENT])
        x = torch.cat(toks, 1)
        n_body = 1 + J_MAX + (N_SURF if c.surface_tokens else 0)
        g = self._graph(bank, r, n_body, n_int, iv)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda and c.amp and getattr(self, "amp", True)):
            for blk in self.blocks:
                x = blk(x, g)
        x = self.ln_out(x.float())

        res_root = self.root_head(x[:, 0])  # (B, 9)
        res_phys = self.joint_head(x[:, 1:1 + J_MAX]).squeeze(-1)  # (B, J) one per physical joint
        cj = bank.coord_joint[r]
        res_coord = torch.where(cj >= 0, torch.gather(res_phys, 1, cj.clamp_min(0)), 0.0)  # mimic outputs unused
        extra = {"bridge_c": bc} if c.bridge else {}
        if c.aux_set:  # training-only readout of the final tokens; never consumed by the heads above or decode
            a = self.aux_head(torch.cat([x[:, n_body:n_body + n_int].mean(1), x[:, 0]], -1)).view(B, c.aux_set_slots, 7)
            extra["aux_set"] = (a[..., :3] * 0.1, torch.nn.functional.normalize(a[..., 3:6], dim=-1), a[..., 6])
        return {"res_root": res_root, "res_coord": res_coord} | extra | self.decode(bank, r, qc0, R0w, t0w, res_root, res_coord)

    def decode(self, bank, r, qc0, R0, t0, res_root, res_coord):
        c = self.c
        d6 = torch.tensor(I6, device=R0.device, dtype=R0.dtype) + res_root[:, 3:] * math.radians(c.res_rot_deg)
        if c.frame == "wrist":  # update expressed in the start wrist frame
            t = t0 + (R0 @ (res_root[:, :3] * c.res_xyz_m)[..., None]).squeeze(-1)
            R = R0 @ sixd_to_matrix_symmetric(d6)
        else:
            t = t0 + res_root[:, :3] * c.res_xyz_m
            R = sixd_to_matrix_symmetric(d6) @ R0
        crng = bank.coords_of(r, bank.range[r])
        qc = qc0 + res_coord * c.res_joint_frac * crng
        return {"t": t, "R": R, "qc": torch.where(bank.coord_valid[r], qc, 0.0)}


def target_residuals(bank: RobotBank, r, qc0, R0, t0, qcg, Rg, tg, c: ScratchConfig):
    """KITE's normalised residual target for (start -> goal); inverse of GETScratch.decode."""
    i6 = torch.tensor(I6, device=R0.device, dtype=R0.dtype)
    if getattr(c, "frame", "world") == "wrist":  # wrist update in the start wrist frame
        rt = ((tg - t0)[:, None] @ R0).squeeze(1) / c.res_xyz_m
        rr = (matrix_to_6d(R0.transpose(1, 2) @ Rg) - i6) / math.radians(c.res_rot_deg)
    else:
        rt = (tg - t0) / c.res_xyz_m
        rr = (matrix_to_6d(Rg @ R0.transpose(1, 2)) - i6) / math.radians(c.res_rot_deg)
    crng = bank.coords_of(r, bank.range[r]).clamp_min(1e-6)
    rj = torch.where(bank.coord_valid[r], (qcg - qc0) / (c.res_joint_frac * crng), 0.0)
    return torch.cat([rt, rr], -1), rj
