"""Where does contact information get lost in the direct psi decoder? (2026-11-17, frozen models, probes only)

    python scripts/decoder/diagnostics/aux_probe.py --out $GETKITE_RUNS/aux_probe

For each frozen decoder checkpoint, features are taken at four stages for the C3 training pool (one fixed start per
record) and the fresh-goal validation pool (fixed evaluation start):
  z          standardised psi z (the decoder's own z_mu/z_sd)
  intent_in  the intent tokens entering the transformer (intent_proj output, 4 x d, flattened)
  aux_in     the exact aux-head input: mean of final intent tokens ++ final root token
  final_all  all final intent tokens (flattened) ++ final root token
A fresh 2-layer MLP probe (random init, trained on the training pool only) predicts, from each feature set:
  count (2..6, classification), world-frame contact centroid, start-wrist-frame contact centroid (mm).
The model's own aux head (if present) is scored on the same records for reference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from getkite.decoder import paths
from getkite.decoder.get_scratch import GETScratch, ScratchConfig
from getkite.decoder.multifamily import fixed_start
from getkite.decoder.robots import J_MAX, N_SURF
from getkite.decoder.spatial_bridge import centroid_wrist_label
from getkite.decoder.train_scratch import FrozenPsi, build_bank, load_config, load_pool, start_noise

RUNS, REPO = paths.get("GETKITE_RUNS"), str(paths.REPO)


def features(model, psi, bank, rec, sg, bs=1024):
    c = model.c
    n_body = 1 + J_MAX + (N_SURF if c.surface_tokens else 0)
    grab = {}
    h1 = model.intent_proj.register_forward_hook(lambda m, i, o: grab.__setitem__("intent_in", o))
    h2 = model.ln_out.register_forward_hook(lambda m, i, o: grab.__setitem__("final", o))
    out = {k: [] for k in ("z", "intent_in", "aux_in", "final_all", "aux_count")}
    with torch.no_grad():
        for s in range(0, len(rec), bs):
            i = slice(s, s + bs)
            z = psi(rec.c_pos[i], rec.c_normal[i], rec.c_valid[i])
            o = model(bank, rec.r[i], sg.qc1[i], sg.R1[i], sg.t1[i], z)
            x = grab["final"]
            it = x[:, n_body:n_body + c.n_intent_tokens]
            out["z"].append((z - model.z_mu) / model.z_sd)
            out["intent_in"].append(grab["intent_in"].flatten(1))
            out["aux_in"].append(torch.cat([it.mean(1), x[:, 0]], -1))
            out["final_all"].append(torch.cat([it.flatten(1), x[:, 0]], -1))
            if "aux_set" in o:
                out["aux_count"].append((o["aux_set"][2].sigmoid() > 0.5).sum(1))
    h1.remove(); h2.remove()
    return {k: torch.cat(v).float() if v else None for k, v in out.items()}


def targets(rec, sg):
    v = rec.c_valid
    eye = torch.eye(3, device=v.device).expand(len(v), 3, 3)
    return {"count": v.sum(1).long(), "c_world": centroid_wrist_label(rec.c_pos, v, eye, torch.zeros_like(sg.t1)),
            "c_wrist": centroid_wrist_label(rec.c_pos, v, sg.R1, sg.t1)}


def probe(Xtr, Ytr, Xev_list, kind, steps=4000, seed=0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-4)
    nout = 7 if kind == "count" else 3
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, nout)).to(Xtr.device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    ysc = 0.01  # regression targets in cm
    for _ in range(steps):
        j = torch.randint(0, len(Xtr), (512,), device=Xtr.device)
        p = net((Xtr[j] - mu) / sd)
        loss = nn.functional.cross_entropy(p, Ytr[j]) if kind == "count" else (((p - Ytr[j] / ysc)) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    res = []
    with torch.no_grad():
        for X, Y in Xev_list:
            p = net((X - mu) / sd)
            if kind == "count":
                res.append(round(float((p.argmax(-1) == Y).float().mean()), 3))
            else:
                res.append(round(float((p * ysc - Y).norm(dim=-1).mean() * 1000), 2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=f"{REPO}/configs/decoder/scratch/c3_psi50k_10x_wrist.json")
    ap.add_argument("--ckpts", nargs="+", default=[f"{RUNS}/scratch_c3_aux50k_10x_wrist/best.pt",
                                                     f"{RUNS}/scratch_c3_aux50k_10x_wrist/last.pt",
                                                     f"{RUNS}/scratch_c3_psi50k_10x_wrist/best.pt"])
    ap.add_argument("--n_train", type=int, default=30000)
    ap.add_argument("--out", default=f"{RUNS}/aux_probe")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    cfg = load_config(a.config, [])
    bank = build_bank(cfg, dev)
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    noise = start_noise(cfg)
    train = load_pool(cfg["train"], bank).to(dev)
    train = train.index(torch.randperm(len(train), generator=torch.Generator().manual_seed(3))[:a.n_train].to(dev))
    val = load_pool(cfg["val"], bank).to(dev)
    sgt = fixed_start(bank, train, cfg.get("train_start_seed", 0) + 17, **noise)
    sgt = type(sgt)(**{k: getattr(sgt, k).to(dev) for k in sgt.__dataclass_fields__})
    sgv = fixed_start(bank, val, cfg["val_start_seed"], **noise)
    sgv = type(sgv)(**{k: getattr(sgv, k).to(dev) for k in sgv.__dataclass_fields__})
    Tt, Tv = targets(train, sgt), targets(val, sgv)
    ntr_eval = min(5000, len(train))
    R = {"n_train": len(train), "n_val": len(val),
         "count_majority_baseline_val": round(float((Tv["count"] == Tv["count"].mode().values).float().mean()), 3)}
    for ck_path in a.ckpts:
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        m = GETScratch(ScratchConfig.from_dict(ck["model_config"])).to(dev)
        m.load_state_dict(ck["model"]); m.amp = False; m.eval()
        Ft, Fv = features(m, psi, bank, train, sgt), features(m, psi, bank, val, sgv)
        name = f"{Path(ck_path).parent.name}/{Path(ck_path).stem}@{ck.get('step')}"
        row = {"z_sd_mean": round(float(m.z_sd.mean()), 4)}
        if Fv["aux_count"] is not None:
            row["own_aux_head_count_acc"] = {"train": round(float((Ft["aux_count"][:ntr_eval] == Tt["count"][:ntr_eval]).float().mean()), 3),
                                             "val": round(float((Fv["aux_count"] == Tv["count"]).float().mean()), 3)}
        for feat in ("z", "intent_in", "aux_in", "final_all"):
            row[feat] = {}
            for kind in ("count", "c_world", "c_wrist"):
                tr, va = probe(Ft[feat], Tt[kind], [(Ft[feat][:ntr_eval], Tt[kind][:ntr_eval]), (Fv[feat], Tv[kind])], kind)
                row[feat][kind] = {"train": tr, "val": va}
            print(name, feat, json.dumps(row[feat]), flush=True)
        R[name] = row
        print(name, "own aux", row.get("own_aux_head_count_acc"), flush=True)
    (out / "results.json").write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
