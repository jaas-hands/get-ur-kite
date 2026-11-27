"""Layer probe on the T1 direct-GET checkpoint (2026-11-19; evaluation only). Direct track, separate from W1.

On T1's own validation targets (fresh rigid translations) and fixed starts:
  1 layer-wise relative z-sensitivity + root-token attention mass per block (direct_zperm_probe.layer_sensitivity),
    384 targets, z permuted across targets.
  2 linear decodability of the REQUIRED translation (start-wrist frame, mm) from features at each stage, 2-fold
    cross-fitted ridge on all 2,000 validation targets, R^2 per axis: standardised z, intent tokens entering the GET,
    intent tokens / root token after every block and after ln_out, and the root head output res_root[:3].
    Distinguishes "the information never reaches the root token" from "it reaches it but the head does not read it".

    PYTHONPATH=src python scripts/decoder/diagnostics/t1_layer_probe.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from getkite.decoder import paths
from getkite.decoder.direct_worldaux import DirectWorldAux
from getkite.decoder.get_scratch import ScratchConfig
from getkite.decoder.multifamily import fixed_start
from getkite.decoder.robots import J_MAX, N_SURF
from getkite.decoder.spatial_bridge import SetBridgeConfig
from getkite.decoder.train_scratch import FrozenPsi, build_bank, load_config, load_pool
from getkite.decoder.translation_diag import make_targets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from direct_zperm_probe import layer_sensitivity  # noqa: E402


def ridge_cv(X, Y, lam=1e-2, seed=0):
    """2-fold cross-fitted ridge; X standardised on the fit fold. -> R^2 per output column (on held-out folds)."""
    X, Y = X.double(), Y.double()
    n = len(X)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    P = torch.empty_like(Y)
    for a, b in ((idx[: n // 2], idx[n // 2:]), (idx[n // 2:], idx[: n // 2])):
        mu, sd = X[a].mean(0), X[a].std(0).clamp_min(1e-8)
        Xa = torch.cat([(X[a] - mu) / sd, torch.ones(len(a), 1, dtype=X.dtype)], 1)
        Xb = torch.cat([(X[b] - mu) / sd, torch.ones(len(b), 1, dtype=X.dtype)], 1)
        W = torch.linalg.solve(Xa.T @ Xa + lam * len(a) * torch.eye(Xa.shape[1], dtype=X.dtype), Xa.T @ Y[a])
        P[b] = Xb @ W
    return [float(1 - ((P[:, j] - Y[:, j]) ** 2).sum() / ((Y[:, j] - Y[:, j].mean()) ** 2).sum()) for j in range(Y.shape[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/decoder/scratch/t1_translation_allegro.json")
    ap.add_argument("--ckpt", default="${GETKITE_RUNS}/t1_translation_allegro/get/best.pt")
    ap.add_argument("--out", default="${GETKITE_RUNS}/t1_translation_allegro/layer_probe.json")
    a = ap.parse_args()
    cfg = load_config(a.config, [])
    T = cfg["t1"]
    bank = build_bank(cfg, "cpu")
    base = load_pool([T["base"]], bank).index(torch.tensor([T["base_row"]]))
    val, _ = make_targets(base, T["n_val"], T["box_m"], T["val_seed"])
    noise = {"xyz_std": T["start_xyz_std"], "rot_std_deg": 0.0, "joint_frac": 0.0}
    sg = fixed_start(bank, val, cfg["val_start_seed"], **noise)  # = the training script's validation starts
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).eval()
    with torch.no_grad():
        z = psi(val.c_pos, val.c_normal, val.c_valid)
    ck = torch.load(paths.expand(a.ckpt), map_location="cpu", weights_only=False)
    m = DirectWorldAux(ScratchConfig.from_dict(ck["model_config"]), SetBridgeConfig(**ck["set_config"]))
    m.load_state_dict(ck["model"]); m.decoder.amp = False; m.eval()
    res = {"_settings": {"ckpt": paths.expand(a.ckpt), "step": ck.get("step"), "n_val": len(val), "doc": __doc__}}

    # 1 layer sensitivity + root attention on 384 targets
    sel = torch.randperm(len(val), generator=torch.Generator().manual_seed(0))[:384]
    rb = val.index(sel)
    g = type(sg)(**{k: getattr(sg, k)[sel] for k in sg.__dataclass_fields__})
    perm = torch.randperm(384, generator=torch.Generator().manual_seed(1))
    res["layer_sensitivity"] = layer_sensitivity(m, "direct", bank, rb, g, z[sel], perm)

    # 2 linear decodability of the required translation (start-wrist frame, mm) along the path, all 2,000 targets
    need = torch.einsum("nji,nj->ni", sg.R1, sg.tg - sg.t1) * 1000
    cap = {}
    keep = lambda k, x: cap.setdefault(k, []).append(x.float()[:, list(range(0, 1)) + list(range(1 + J_MAX + N_SURF, x.shape[1]))].clone())  # noqa: E731  root + intent only
    hooks = [m.decoder.blocks[0].register_forward_pre_hook(lambda mod, x: keep("in", x[0]))]
    for i, b in enumerate(m.decoder.blocks):
        hooks.append(b.register_forward_hook(lambda mod, x, o, i=i: keep(f"block{i}", o)))
    hooks.append(m.decoder.ln_out.register_forward_hook(lambda mod, x, o: keep("ln_out", o)))
    outs = {"res_root": [], "t": []}
    with torch.no_grad():
        for s0 in range(0, len(val), 250):
            i = slice(s0, s0 + 250)
            oo = m(bank, val.r[i], sg.qc1[i], sg.R1[i], sg.t1[i], z[i], aux=False)
            outs["res_root"].append(oo["res_root"]); outs["t"].append(oo["t"])
    for h in hooks:
        h.remove()
    cap = {k: torch.cat(v) for k, v in cap.items()}
    o = {k: torch.cat(v) for k, v in outs.items()}
    nb = 1  # captured tensors hold [root, intent...] only
    D = {"z (standardised)": [ridge_cv(z, need)]}
    # the decoder input root token carries the start pose only (world pose feature), not intent
    for k in ["in"] + [f"block{i}" for i in range(len(m.decoder.blocks))] + ["ln_out"]:
        x = cap[k]
        D[f"{k}/intent tokens"] = [ridge_cv(x[:, nb:].flatten(1), need)]
        D[f"{k}/root token"] = [ridge_cv(x[:, 0], need)]
    D["res_root[:3] (head output)"] = [ridge_cv(o["res_root"][:, :3], need)]
    got = torch.einsum("nji,nj->ni", sg.R1, o["t"] - sg.t1) * 1000
    D["predicted correction (identity map R^2)"] = [[float(1 - ((got[:, j] - need[:, j]) ** 2).sum() / ((need[:, j] - need[:, j].mean()) ** 2).sum()) for j in range(3)]]
    res["required_translation_R2_xyz"] = {k: [round(x, 4) for x in v[0]] for k, v in D.items()}
    res["required_translation_sd_mm"] = [round(float(x), 3) for x in need.std(0)]
    res["predicted_translation_sd_mm"] = [round(float(x), 3) for x in got.std(0)]
    out = Path(paths.expand(a.out)); out.write_text(json.dumps(res, indent=1))
    L = res["layer_sensitivity"]
    print("rel z-sensitivity:", {k: round(v["rel"], 3) for k, v in L.items() if not k.startswith("_") and any(t in k for t in ("root", "intent", "res_", "trunk"))})
    print("root attention mass:", L["_root_attention_mass"])
    print("required-translation R2 (x, y, z):")
    for k, v in res["required_translation_R2_xyz"].items():
        print(f"  {k:45s} {v}")
    print("sd mm need / got:", res["required_translation_sd_mm"], res["predicted_translation_sd_mm"])
    print("wrote", out)


if __name__ == "__main__":
    main()
