"""z-shuffling sensitivity probe for the direct decoders (no training). Version-controlled copy of the ad hoc probe
quoted in runs/direct_frozentrunk_d1/report.md (2026-11-19). 384 source-val goals (randperm seed 0); starts
fixed_start(val_start_seed, config start noise) by default (the original ad hoc probe used seed 0: --start_seed 0); z permuted across the batch
(randperm seed 1). For each checkpoint: mean |output|, mean |output(z) - output(z permuted)|, mean across-batch std of
the wrist (res_root) and joint (res_coord) residuals, and the intent-token change.
Token statistics (added 2026-11-19): at the input of the first GET block (after type embeddings), the mean L2 norm per
token type (root / valid joint / surface / intent) and the mean L2 norm of d(motion loss)/d(token) per type, with the
motion residual loss of training (targets from the same starts); plus parameter-gradient norms per input encoder.

    PYTHONPATH=src python scripts/decoder/diagnostics/direct_zperm_probe.py [--start_seed 0 --out ${GETKITE_RUNS}/direct_zperm_probe/probe.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from getkite.decoder import paths
from getkite.decoder.direct_worldaux import DirectWorldAux
from getkite.decoder.get_scratch import ScratchConfig, target_residuals
from getkite.decoder.robots import J_MAX, N_SURF
from getkite.decoder.multifamily import fixed_start
from getkite.decoder.spatial_bridge import SetBridgeConfig
from getkite.decoder.train_scratch import FrozenPsi, build_bank, load_config, load_pool, residual_loss, start_noise

def gn(ps):
    g = [p.grad.flatten() for p in ps if p.grad is not None]
    return float(torch.cat(g).norm()) if g else 0.0


def token_stats(m, bank, rb, sg, z, families):
    """Per-token-type input norms and motion-loss gradients at the first GET block."""
    cap = {}

    def hook(mod, args):
        x = args[0]
        x.retain_grad()
        cap["x"] = x
    hd = m.decoder.blocks[0].register_forward_pre_hook(hook)
    m.zero_grad(set_to_none=True)
    out = m(bank, rb.r, sg.qc1, sg.R1, sg.t1, z, aux=False)
    hd.remove()
    tr_root, tr_j = target_residuals(bank, rb.r, sg.qc1, sg.R1, sg.t1, sg.qcg, sg.Rg, sg.tg, m.c)
    fi = torch.tensor([families.index(f) for f in bank.family])[rb.r]
    lm, _, _ = residual_loss(out, tr_root, tr_j, bank.coord_valid[rb.r], fi, len(families))
    lm.backward()
    x, g = cap["x"].detach(), cap["x"].grad
    nb = 1 + J_MAX + N_SURF
    valid = bank.node_valid[rb.r]
    spans = {"root": slice(0, 1), "joint": slice(1, 1 + J_MAX), "surface": slice(1 + J_MAX, nb), "intent": slice(nb, x.shape[1])}
    r = {"motion_loss": float(lm)}
    for k, sl in spans.items():
        msk = valid[:, sl] if k != "intent" else torch.ones(x.shape[0], x.shape[1] - nb, dtype=torch.bool)
        xn, gn_ = x[:, sl].norm(dim=-1)[msk], g[:, sl].norm(dim=-1)[msk]
        r[f"tok_norm/{k}"] = float(xn.mean())
        r[f"tok_grad/{k}"] = float(gn_.mean())
    d = m.decoder
    r["param_grad/intent_tok+pose"] = gn(list(m.intent_tok.parameters()) + list(m.intent_pose.parameters()))
    r["param_grad/root_in(+world_pose)"] = gn(list(d.root_in.parameters()) + list(d.world_pose_proj.parameters()))
    r["param_grad/joint_in"] = gn(d.joint_in.parameters())
    r["param_grad/surf_in(+nbr_enc)"] = gn(list(d.surf_in.parameters()) + list(d.nbr_enc.parameters()))
    r["param_grad/blocks"] = gn(d.blocks.parameters())
    r["ratio_tok_grad_intent_over_surface"] = r["tok_grad/intent"] / max(r["tok_grad/surface"], 1e-12)
    r["ratio_tok_norm_intent_over_surface"] = r["tok_norm/intent"] / max(r["tok_norm/surface"], 1e-12)
    m.zero_grad(set_to_none=True)
    return r


LAYER_CKPTS = {"direct_frozentrunk_d1/last.pt": "direct", "direct_worldaux_v1/best.pt": "direct",
               "direct_anchor_a1/best.pt": "anchor", "j1_joint50k_hf/best.pt": "j1"}


def load_any(path, kind):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if kind == "j1":
        from getkite.decoder.joint_contact_get import JointContactGET
        m = JointContactGET(ScratchConfig.from_dict(ck["model_config"]), SetBridgeConfig(**ck["set_config"]))
    elif kind == "anchor":
        from getkite.decoder.train_direct_anchor import AnchorDirect
        A = ck["anchor"]
        m = AnchorDirect(ScratchConfig.from_dict(ck["model_config"]), SetBridgeConfig(**ck["set_config"]), A["n_anchors"], A["f_dim"], A["out_scale"])
    else:
        m = DirectWorldAux(ScratchConfig.from_dict(ck["model_config"]), SetBridgeConfig(**ck["set_config"]))
    m.load_state_dict(ck["model"]); m.decoder.amp = False
    return m.eval(), ck.get("step")


@torch.no_grad()
def root_attention(m, bank, r, n_body, store):
    """Pre-hooks recomputing each block's attention for the ROOT query (index 0): mean (over goals and heads) attention
    mass on root / joint / surface / intent keys. Exactly Block.forward's q, k, graph bias and key mask."""
    hooks = []
    for i, b in enumerate(m.decoder.blocks):
        def pre(mod, args, i=i):
            x, g = args
            B, T, d = x.shape
            q, k, _ = mod.qkv(mod.ln1(x.float())).view(B, T, 3, mod.h, d // mod.h).permute(2, 0, 3, 1, 4)
            a = (q[:, :, :1] @ k.transpose(-1, -2)) / math.sqrt(d // mod.h) + mod.bias(g)[:, :, :1].float()
            a = a.softmax(-1)[:, :, 0]  # (B, H, T)
            sp = {"root": slice(0, 1), "joint": slice(1, 1 + J_MAX), "surface": slice(1 + J_MAX, n_body), "intent": slice(n_body, T)}
            store[f"block{i}"] = {k_: float(a[..., sl].sum(-1).mean()) for k_, sl in sp.items()}
        hooks.append(b.register_forward_pre_hook(pre))
    return hooks


@torch.no_grad()
def layer_sensitivity(m, kind, bank, rb, sg, z, perm):
    """Relative z-sensitivity at every stage of the movement path: mean |x(z) - x(z permuted)| divided by the mean
    across-goal SD of x(z), per token type (root / valid joints / surface / intent). 0 = ignores z; ~1.4 = output as
    different as two random goals' outputs (for independent permuted pairs, E|x-y| ~ 1.13 sd)."""
    cap = {}
    hooks = [m.decoder.blocks[0].register_forward_pre_hook(lambda mod, a: cap.__setitem__("in", a[0].float().clone()))]
    for i, b in enumerate(m.decoder.blocks):
        hooks.append(b.register_forward_hook(lambda mod, a, o, i=i: cap.__setitem__(f"block{i}", o.float().clone())))
    hooks.append(m.decoder.ln_out.register_forward_hook(lambda mod, a, o: cap.__setitem__("ln_out", o.float().clone())))
    kw = {} if kind == "j1" else {"aux": False}
    att = {}
    hooks += root_attention(m, bank, rb.r, 1 + J_MAX + N_SURF, att)
    runs = []
    for zz in (z, z[perm]):
        cap.clear()
        o = m(bank, rb.r, sg.qc1, sg.R1, sg.t1, zz, **kw)
        st = dict(cap)
        st["res_root"], st["res_coord"] = o["res_root"][:, None], o["res_coord"][:, None]
        if kind != "j1":
            st["trunk_h"] = m.trunk(zz)[:, None]
        else:
            pos, nrm, lg = m.predict_contacts(zz)
            st["contact_stage"] = torch.cat([pos, nrm], -1)
        runs.append(st)
    a, b = runs
    nb = 1 + J_MAX + N_SURF
    valid = bank.node_valid[rb.r]
    res = {}
    for k in a:
        x, y = a[k], b[k]
        if x.dim() == 3 and x.shape[1] > nb - 1:
            spans = {"root": slice(0, 1), "joint": slice(1, 1 + J_MAX), "surface": slice(1 + J_MAX, nb), "intent": slice(nb, x.shape[1])}
            for t, sl in spans.items():
                xs, ys = x[:, sl], y[:, sl]
                msk = valid[:, sl] if t != "intent" else torch.ones(xs.shape[:2], dtype=torch.bool)
                sd = xs.std(0)  # across goals, per token slot and dim
                w = msk.float()[..., None]
                d = ((xs - ys).abs() * w).sum() / (w.sum() * xs.shape[-1])
                s_ = (sd * (msk.float().mean(0) > 0.5)[..., None]).sum() / ((msk.float().mean(0) > 0.5).sum() * xs.shape[-1]).clamp_min(1)
                res[f"{k}/{t}"] = {"delta": float(d), "sd_across_goals": float(s_), "rel": float(d / s_.clamp_min(1e-12))}
        else:
            x2, y2 = x.flatten(1), y.flatten(1)
            sd = x2.std(0).mean()
            d = (x2 - y2).abs().mean()
            res[k] = {"delta": float(d), "sd_across_goals": float(sd), "rel": float(d / sd.clamp_min(1e-12))}
    for h in hooks:
        h.remove()
    res["_root_attention_mass"] = att  # from the second (z-permuted) pass; mass sums are permutation-robust in mean
    return res


CKPTS = ["direct_frozentrunk_d1/last.pt", "direct_worldaux_v1/best.pt", "direct_worldaux_v1/last.pt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/decoder/scratch/direct_frozentrunk_d1.json")
    ap.add_argument("--start_seed", type=int, default=None, help="default: the benchmark's val_start_seed; 0 = the original ad hoc probe")
    ap.add_argument("--out", default="${GETKITE_RUNS}/direct_zperm_probe/probe_val_start_seed.json")
    ap.add_argument("--layers", action="store_true", help="layer-by-layer relative z-sensitivity (adds A1 and J1 as positive control); writes --layers_out")
    ap.add_argument("--layers_out", default="${GETKITE_RUNS}/direct_zperm_probe/probe_layers_attn_val_start_seed.json")
    a = ap.parse_args()
    R = Path(paths.get("GETKITE_RUNS"))
    cfg = load_config(a.config, [])
    bank = build_bank(cfg, "cpu")
    val = load_pool(cfg["val"], bank)
    idx = torch.randperm(len(val), generator=torch.Generator().manual_seed(0))[:384]
    rb = val.index(idx)
    if a.start_seed is None:
        a.start_seed = cfg["val_start_seed"]
    sg = fixed_start(bank, rb, a.start_seed, **start_noise(cfg))
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).eval()
    with torch.no_grad():
        z = psi(rb.c_pos, rb.c_normal, rb.c_valid)
    perm = torch.randperm(len(z), generator=torch.Generator().manual_seed(1))
    if a.layers:
        L = {"_settings": {"config": a.config, "n_goals": len(idx), "goal_seed": 0, "perm_seed": 1, "start_seed": a.start_seed,
                           "doc": layer_sensitivity.__doc__}}
        for name, kind in LAYER_CKPTS.items():
            m, step = load_any(R / name, kind)
            L[name] = {"kind": kind, "step": step, "stages": layer_sensitivity(m, kind, bank, rb, sg, z, perm)}
            print(name, {k: round(v["rel"], 3) for k, v in L[name]["stages"].items() if not k.startswith("_") and any(t in k for t in ("root", "intent", "res_", "trunk", "contact"))}, flush=True)
            print("  root attention mass", L[name]["stages"]["_root_attention_mass"], flush=True)
        out = Path(paths.expand(a.layers_out)); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(L, indent=1))
        print("wrote", out)
        return
    res = {"_settings": {"config": a.config, "n_goals": len(idx), "goal_seed": 0, "perm_seed": 1, "start_seed": a.start_seed,
                         "noise": start_noise(cfg)}}
    for name in CKPTS:
        ck = torch.load(R / name, map_location="cpu", weights_only=False)
        m = DirectWorldAux(ScratchConfig.from_dict(ck["model_config"]), SetBridgeConfig(**ck["set_config"]))
        m.load_state_dict(ck["model"]); m.decoder.amp = False; m.eval()
        with torch.no_grad():
            o = m(bank, rb.r, sg.qc1, sg.R1, sg.t1, z, aux=False)
            op = m(bank, rb.r, sg.qc1, sg.R1, sg.t1, z[perm], aux=False)
            it, _ = m.intent(z, sg.R1, sg.t1)
            itp, _ = m.intent(z[perm], sg.R1, sg.t1)
        r = {"step": ck.get("step")}
        for k in ("res_root", "res_coord"):
            x, y = o[k], op[k]
            r[k] = {"mean_abs_out": float(x.abs().mean()), "mean_abs_delta_zperm": float((x - y).abs().mean()),
                    "across_batch_std": float(x.std(0).mean())}
        r["intent_tokens"] = {"mean_abs": float(it.abs().mean()), "mean_abs_delta_zperm": float((it - itp).abs().mean())}
        r["token_stats"] = token_stats(m, bank, rb, sg, z, sorted(set(bank.family)))
        res[name] = r
        print(name, json.dumps(r), flush=True)
    out = Path(paths.expand(a.out)); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
