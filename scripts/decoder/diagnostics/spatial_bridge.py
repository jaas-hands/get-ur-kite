"""Train + evaluate the spatial bridge (src/getkite/decoder/spatial_bridge.py) on the C3 data (2026-11-17).

    python scripts/decoder/diagnostics/spatial_bridge.py --config configs/decoder/scratch/c3_psi50k_10x_wrist.json \
        --out $GETKITE_RUNS/bridge_c3 [--steps 20000]

Training: the C3 training pool, family-balanced batches, start pose redrawn every batch with KITE's start noise (the
decoder's deployment start distribution), normal jitter as in decoder training. Label = target contact centroid in the
starting-wrist frame (built from the source synthetic contacts; supervision only). Bridge inputs = psi(z), R0, t0.

Evaluation (fixed evaluation start, cfg val_start_seed): fresh goals = cfg["val"] (all three families) and the held-out
LEAP hands (cfg extra_eval leap_heldout, never seen in training); training goals reported for the memorisation gap.
Per family: mean / median L2 error (mm) and R^2, against
  zero        predict the start wrist origin (c_wrist = 0)
  mean        training-label mean (one global vector)
  fam_mean    per-family training-label mean (family is known at deployment)
  ridge       linear map from the bridge's exact inputs (standardised z, pose + Fourier) -- a linear bridge
  structured  reference only: ridge z -> world centroid, then the exact transform R0^T(c - t0)
Checkpoints: last.pt and best.pt (selected on cfg["val"] macro mm; leap_heldout never used for selection).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from getkite.decoder.get_scratch import fourier
from getkite.decoder.multifamily import BalancedSampler, draw_start, fixed_start
from getkite.decoder.rotations import matrix_to_6d
from getkite.decoder.spatial_bridge import BridgeConfig, SpatialBridge, centroid_wrist_label
from getkite.decoder.train_scratch import FrozenPsi, build_bank, jitter_normals, load_config, load_pool, start_noise


def to_dev(sg, dev):
    return type(sg)(**{k: getattr(sg, k).to(dev) for k in sg.__dataclass_fields__})


def predict_inputs(psi, rec, sg, gen=None, normal_noise=0.0):
    """The ONLY quantities the bridge sees: z of the contact request, and the start wrist pose (R1, t1)."""
    n = jitter_normals(rec.c_normal, normal_noise, gen) if normal_noise > 0 else rec.c_normal
    return psi(rec.c_pos, n, rec.c_valid), sg.R1, sg.t1


def batched(fn, n, bs=2048):
    return torch.cat([fn(slice(s, s + bs)) for s in range(0, n, bs)])


def eval_set(rec, bank, psi, seed, cfg, dev):
    sg = to_dev(fixed_start(bank, rec, seed, **start_noise(cfg)), dev)
    z = batched(lambda i: psi(rec.c_pos[i], rec.c_normal[i], rec.c_valid[i]), len(rec))
    y = centroid_wrist_label(rec.c_pos, rec.c_valid, sg.R1, sg.t1)
    cw = centroid_wrist_label(rec.c_pos, rec.c_valid, torch.eye(3, device=dev).expand(len(rec), 3, 3), torch.zeros_like(sg.t1))
    return {"z": z, "R0": sg.R1, "t0": sg.t1, "y": y, "c_world": cw, "r": rec.r}


def lin_feats(bridge, z, R0, t0):
    c = bridge.c
    tp, r6 = t0 * c.pos_scale, matrix_to_6d(R0)
    X = torch.cat([(z - bridge.z_mu) / bridge.z_sd, tp, r6, fourier(tp, c.pos_fourier), fourier(r6, c.rot_fourier)], -1)
    return torch.cat([X, torch.ones_like(X[:, :1])], -1).double()


def ridge(X, Y, lam=1e-1):
    return torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1], device=X.device, dtype=X.dtype), X.T @ Y)


def metrics(pred, y, fam_of, families, prefix=""):
    e = (pred - y).norm(dim=-1) * 1000
    out = {}
    for f in families + ["all"]:
        sel = torch.ones_like(e, dtype=torch.bool) if f == "all" else torch.as_tensor(fam_of == f, device=e.device)
        if not sel.any():
            continue
        ys, ps = y[sel], pred[sel]
        r2 = 1 - float(((ps - ys) ** 2).sum() / ((ys - ys.mean(0)) ** 2).sum())
        out[f] = {"mean_mm": round(float(e[sel].mean()), 2), "median_mm": round(float(e[sel].median()), 2),
                  "r2": round(r2, 3), "n": int(sel.sum())}
    return out


def rec_hashes(rec):
    cp = (rec.c_pos * rec.c_valid[..., None]).cpu().numpy().round(5)
    return {hashlib.sha1(np.concatenate([[int(rec.r[i])], cp[i].ravel()]).tobytes()).hexdigest() for i in range(len(rec))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(a.config, [])
    torch.manual_seed(a.seed)
    bank = build_bank(cfg, dev)
    families = sorted(set(bank.family))
    fam = np.array([str(x) for x in bank.family], dtype=object)
    train = load_pool(cfg["train"], bank).to(dev)
    # cfg["val"] is the selection pool (C3: fresh goals; C2: the training goals at a new start); extra_eval pools are
    # reported only (C3: leap_heldout; C2: fresh_goals + leap_heldout)
    pools = {"val": load_pool(cfg["val"], bank).to(dev) if cfg.get("val") else train}  # as train_scratch: no val = train goals
    pools |= {k: load_pool(v, bank).to(dev) for k, v in cfg.get("extra_eval", {}).items() if not k.startswith("train_goals")}
    g = torch.Generator().manual_seed(1)
    pools["train_goals"] = train.index(torch.randperm(len(train), generator=g)[:6000].to(dev))
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev)

    # ---- separation checks: evaluation goals vs training goals
    htr = rec_hashes(train)
    checks = {k: {"n": len(p), "overlap_with_train": len(rec_hashes(p) & htr)} for k, p in pools.items() if k != "train_goals"}
    checks["leap_heldout_robots_in_train"] = len(set(pools["leap_heldout"].r.tolist()) & set(train.r.tolist()))
    print("separation:", checks)

    bridge = SpatialBridge(BridgeConfig()).to(dev)
    # training statistics: one fixed start draw over the whole training pool (labels/z only from training records)
    trs = eval_set(train, bank, psi, cfg.get("train_start_seed", 0) + 17, cfg, dev)
    bridge.set_z_stats(trs["z"])
    fam_tr = fam[trs["r"].cpu().numpy()]
    mu = trs["y"].mean(0)
    fam_mu = {f: trs["y"][torch.as_tensor(fam_tr == f, device=dev)].mean(0) for f in families}
    Xtr = lin_feats(bridge, trs["z"], trs["R0"], trs["t0"])
    W_lin = ridge(Xtr, trs["y"].double())
    Zs = torch.cat([(trs["z"] - bridge.z_mu) / bridge.z_sd, torch.ones_like(trs["z"][:, :1])], -1).double()
    W_cw = ridge(Zs, trs["c_world"].double())

    E = {k: eval_set(p, bank, psi, cfg["val_start_seed"], cfg, dev) for k, p in pools.items()}
    baselines = {}
    for k, d in E.items():
        fo = fam[d["r"].cpu().numpy()]
        cw_hat = (torch.cat([(d["z"] - bridge.z_mu) / bridge.z_sd, torch.ones_like(d["z"][:, :1])], -1).double() @ W_cw).float()
        preds = {"zero": torch.zeros_like(d["y"]), "mean": mu.expand_as(d["y"]),
                 "fam_mean": torch.stack([fam_mu[f] for f in fo]),
                 "ridge": (lin_feats(bridge, d["z"], d["R0"], d["t0"]) @ W_lin).float(),
                 "structured": ((cw_hat - d["t0"])[:, None] @ d["R0"]).squeeze(1)}
        baselines[k] = {n: metrics(p, d["y"], fo, families) for n, p in preds.items()}
        baselines[k]["label_spread_mm"] = round(float((d["y"] - d["y"].mean(0)).norm(dim=-1).mean() * 1000), 2)

    opt = torch.optim.AdamW(bridge.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    sampler = BalancedSampler(train.to("cpu"), bank.family, seed=a.seed)
    gen = torch.Generator().manual_seed(a.seed + 7)
    log = open(out / "log.jsonl", "w")
    best, hist = (1e9, -1), []

    def evaluate():
        bridge.eval()
        res = {}
        with torch.no_grad():
            for k, d in E.items():
                p = batched(lambda i: bridge(d["z"][i], d["R0"][i], d["t0"][i]), len(d["y"]))
                res[k] = metrics(p, d["y"], fam[d["r"].cpu().numpy()], families)
        bridge.train()
        return res

    t_start = time.time()
    for step in range(1, a.steps + 1):
        idx = sampler.sample(a.batch).to(dev)
        rb = train.index(idx)
        sg = to_dev(draw_start(bank, rb, gen, **start_noise(cfg)), dev)
        z, R0, t0 = predict_inputs(psi, rb, sg, gen, cfg["normal_noise_rad"])
        y = centroid_wrist_label(rb.c_pos, rb.c_valid, sg.R1, sg.t1)  # supervision only
        loss = (((bridge(z, R0, t0) - y) / 0.01) ** 2).sum(-1).mean()  # cm^2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0:
            log.write(json.dumps({"step": step, "loss_cm2": round(float(loss.detach()), 4), "grad_norm": round(float(gn), 3)}) + "\n")
        if step % a.eval_every == 0 or step == a.steps:
            res = evaluate()
            macro = float(np.mean([res["val"][f]["mean_mm"] for f in families]))
            hist.append({"step": step, "val_macro_mm": round(macro, 2), **{k: {f: v[f]["mean_mm"] for f in v} for k, v in res.items()}})
            log.write(json.dumps({"step": step, "eval": res}) + "\n")
            log.flush()
            print(f"step {step} loss {float(loss):.3f} cm2  val {[res['val'][f]['mean_mm'] for f in families]} "
                  f"{ {k: res[k]['all']['mean_mm'] for k in res if k != 'val'} } train {res['train_goals']['all']['mean_mm']} "
                  f"({time.time() - t_start:.0f}s)", flush=True)
            state = {"model": bridge.state_dict(), "config": bridge.c.to_dict(), "step": step, "args": vars(a)}
            torch.save(state, out / "last.pt")
            if macro < best[0]:
                best = (macro, step)
                torch.save(state, out / "best.pt")
                best_res = res
    results = {"separation": checks, "baselines": baselines, "last": res, "best_step": best[1], "best": best_res,
               "history": hist, "args": vars(a), "config_file": a.config, "bridge_config": bridge.c.to_dict(),
               "inputs": "forward(z, R0, t0): z = frozen KITE psi(contact request), R0/t0 = start wrist world pose"}
    (out / "results.json").write_text(json.dumps(results, indent=1))
    fmt = lambda m: " ".join(f"{f}={m[f]['mean_mm']:.2f}({m[f]['r2']:+.2f})" for f in m)  # noqa: E731
    for k in E:
        print(f"\n[{k}] label spread {baselines[k]['label_spread_mm']} mm")
        for n in ("zero", "mean", "fam_mean", "ridge", "structured"):
            print(f"  {n:10s} {fmt(baselines[k][n])}")
        print(f"  {'bridge':10s} {fmt(res[k])}   (last, step {a.steps})")
        print(f"  {'bridge*':10s} {fmt(best_res[k])}   (val-selected, step {best[1]})")


if __name__ == "__main__":
    main()
