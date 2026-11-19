"""Load the parts of KITE's released decoder bank (checkpoints/decoder/release.pt) that we reuse.

KITE's checkpoint is a ContactProbePolicyPairedCoTrain: one head per embodiment whose
shape-agnostic submodules are aliased (shared) across heads. With the released config
(conf/kite.yaml) the shared and per-embodiment parts are:

  shared (we reuse)                          per embodiment (we replace with GET)
  hand_proj            Linear(6 -> d)        action_queries        (2, d) wrist xyz/rot queries
  contact_only_encoder psi: contacts -> z    joint_identity_embed  (n_joints, d)  <- the part GET replaces
  cls_unpacker         z -> 4 intent tokens  qref_{xyz,rot,joint}_proj, action_token_norm
  bottleneck_scene_encoder  hand pts <-> z   head_xyz, head_rot, head_joints
  action_decoder       queries -> memory     action / residual normalizers

The modules are built from KITE's own classes (import from a KITE checkout: set KITE_ROOT or
pass kite_root), and their sizes are read from the weights, so nothing here depends on KITE's
config-name translation. num_heads is not recoverable from weights (QK-norm has no affine
parameters): it comes from the checkpoint's config, default 8 as in conf/kite.yaml.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

SHARED = ("hand_proj", "contact_only_encoder", "cls_unpacker", "bottleneck_scene_encoder", "action_decoder")
PER_HEAD_PARAMS = ("action_queries", "joint_identity_embed")
PER_HEAD_MODULES = ("qref_xyz_proj", "qref_rot_proj", "qref_joint_proj", "action_token_norm",
                    "head_xyz", "head_rot", "head_joints")


def kite_networks(kite_root: str | None = None):
    """Import KITE's networks.contact_probe_policy_paired from a checkout (pure PyTorch)."""
    from . import paths
    root = kite_root or paths.get("KITE_ROOT")  # env KITE_ROOT, else the repo's third_party/KITE submodule
    if root:
        root = str(Path(root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
    try:
        return importlib.import_module("networks.contact_probe_policy_paired")
    except ImportError as e:
        raise ImportError("KITE's code is needed: clone https://github.com/helloqxwang/KITE and set KITE_ROOT "
                          "(or the kite_root config key) to the checkout.") from e


def _sub(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def _n_layers(sd: dict, prefix: str) -> int:
    idx = {int(m.group(1)) for k in sd if (m := re.match(re.escape(prefix) + r"(\d+)\.", k))}
    return max(idx) + 1 if idx else 0


def _cfg_get(cfg, *path, default=None):
    for p in path:
        if not isinstance(cfg, dict) or p not in cfg:
            return default
        cfg = cfg[p]
    return cfg


@dataclass
class KiteRelease:
    modules: nn.ModuleDict  # SHARED, built from KITE's classes, weights loaded
    heads: list[str]  # embodiment names in the bank
    per_head: dict = field(default_factory=dict)  # name -> {param/module name: state dict or tensor}
    meta: dict = field(default_factory=dict)  # d_model, num_heads, unpack_tokens, ...
    config: dict | None = None

    def head_state(self, head: str, name: str):
        return self.per_head[head].get(name)


def build_shared(P, *, d_model, num_heads, mlp_ratio, enc_layers, scene_layers, action_layers,
                 unpack_tokens, unpack_ratio) -> nn.ModuleDict:
    return nn.ModuleDict({
        "hand_proj": nn.Linear(6, d_model),
        "contact_only_encoder": P._ContactOnlyEncoder(dim=d_model, num_heads=num_heads, num_layers=enc_layers,
                                                      mlp_ratio=mlp_ratio),
        "cls_unpacker": P._ClsUnpacker(dim=d_model, num_tokens=unpack_tokens, hidden_ratio=unpack_ratio),
        "bottleneck_scene_encoder": P._BottleneckSceneEncoder(dim=d_model, num_heads=num_heads,
                                                              num_layers=scene_layers, mlp_ratio=mlp_ratio,
                                                              num_z_tokens=unpack_tokens),
        "action_decoder": P._PairedActionDecoder(dim=d_model, num_heads=num_heads, num_layers=action_layers,
                                                 mlp_ratio=mlp_ratio),
    })


def load_release(ckpt_path: str, kite_root: str | None = None, prefer_ema: bool = True,
                 num_heads: int | None = None) -> KiteRelease:
    P = kite_networks(kite_root)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["ema_model"] if prefer_ema and "ema_model" in ck else ck["model"]
    cfg = ck.get("config") if isinstance(ck.get("config"), dict) else None
    heads = sorted({k.split(".")[1] for k in sd if k.startswith("heads.")})
    if not heads:
        raise ValueError(f"{ckpt_path}: no 'heads.*' keys; not a KITE co-training checkpoint")
    h0 = f"heads.{heads[0]}."
    if f"{h0}contact_only_encoder.input_proj.weight" not in sd:
        raise ValueError("checkpoint has no contact_only_encoder: trained without the latent-intent bottleneck")
    if f"{h0}cls_unpacker.proj.0.weight" not in sd:
        raise NotImplementedError("only the cls_unpack bottleneck with unpack_tokens > 1 is supported "
                                  "(the released config uses 4 tokens)")

    d = sd[f"{h0}hand_proj.weight"].shape[0]
    ff_hidden = sd[f"{h0}action_decoder.self_layers.0.ffn.net.0.weight"].shape[0]
    unpack_hidden = sd[f"{h0}cls_unpacker.proj.0.weight"].shape[0]
    meta = {
        "d_model": d,
        "num_heads": int(num_heads or _cfg_get(cfg, "model", "num_heads", default=8)),
        "mlp_ratio": ff_hidden / d,
        "enc_layers": _n_layers(sd, f"{h0}contact_only_encoder.self_layers."),
        "scene_layers": _n_layers(sd, f"{h0}bottleneck_scene_encoder.h_self_layers."),
        "action_layers": _n_layers(sd, f"{h0}action_decoder.self_layers."),
        "unpack_tokens": sd[f"{h0}cls_unpacker.proj.2.weight"].shape[0] // d,
        "unpack_ratio": unpack_hidden / d,
    }
    modules = build_shared(P, **{k: meta[k] for k in ("d_model", "num_heads", "mlp_ratio", "enc_layers",
                                                     "scene_layers", "action_layers", "unpack_tokens",
                                                     "unpack_ratio")})
    for name, m in modules.items():
        m.load_state_dict(_sub(sd, f"{h0}{name}."), strict=True)
        for h in heads[1:]:  # shared means identical in every head; check, don't assume
            other = _sub(sd, f"heads.{h}.{name}.")
            for k, v in m.state_dict().items():
                if k in other and not torch.equal(other[k], v):
                    raise ValueError(f"{name}.{k} differs between heads {heads[0]} and {h}: not shared")
    per_head = {}
    for h in heads:
        hp = f"heads.{h}."
        entry = {n: sd[hp + n] for n in PER_HEAD_PARAMS if hp + n in sd}
        entry.update({n: _sub(sd, f"{hp}{n}.") for n in PER_HEAD_MODULES if any(k.startswith(f"{hp}{n}.") for k in sd)})
        entry.update({n: sd[hp + n] for n in ("residual_normalizer_scale", "residual_normalizer_offset",
                                              "action_normalizer_scale", "action_normalizer_offset") if hp + n in sd})
        per_head[h] = entry
    return KiteRelease(modules=modules, heads=heads, per_head=per_head, meta=meta, config=cfg)


class KitePsi(nn.Module):
    """KITE's intent encoder psi with our call signature: (pos, normal, valid) -> z (B, d).

    Contacts are world-frame metres and outward unit normals, exactly what KITE's
    encode_contact_latent consumes (contact_cloud[..., :6] = xyz, normal; xyz also feeds the
    xyz-aware attention). Padded slots are masked; their values are zeroed for safety.
    """

    def __init__(self, contact_only_encoder: nn.Module):
        super().__init__()
        self.enc = contact_only_encoder

    def forward(self, pos, normal, valid):
        w = valid[..., None].to(pos.dtype)
        pos, normal = pos * w, normal * w
        return self.enc(contact_feat_in=torch.cat([pos, normal], -1), contact_xyz=pos, contact_mask=valid.bool())
