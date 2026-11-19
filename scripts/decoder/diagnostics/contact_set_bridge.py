"""Train + evaluate the contact-set bridge (src/getkite/decoder/spatial_bridge.py: ContactSetBridge), 2026-11-17.

    python scripts/decoder/diagnostics/contact_set_bridge.py --config configs/decoder/scratch/c3_psi50k_10x_wrist.json \
        --out $GETKITE_RUNS/set_bridge_c3 [--steps 20000]

Inputs at prediction time: frozen psi(z) of the contact request + the start wrist world pose (R0, t0). Output: S=6
unordered slots (wrist-frame position, unit normal, existence). Predicted count = #slots with existence > 0.5 (at least
one); the ground-truth count is never given at inference. Training: C3 training pool, family-balanced batches, start
redrawn every batch with KITE's start noise, normal jitter on the psi input only; exact set matching (720 assignments)
with shared position / normal / existence cost (so position and normal of a slot are supervised by the same contact).

Evaluation at the fixed evaluation start (cfg val_start_seed): val = fresh source goals (selection pool, macro Chamfer
mm), leap_heldout = unseen LEAP hands (reported only), train_goals = 6000 training goals (memorisation gap).
Evaluation matching: predicted set (existing slots) vs true set, min-cost assignment on position maximising the
number of pairs. Reported per family:
  count accuracy / bias; matched position error (mm) and normal angle (deg) over all matched pairs;
  for tau in (5, 10) mm: recall (true contacts with a prediction within tau), missing = 1 - recall,
  precision, duplicates (unmatched predictions within tau of an already covered true contact), exact-set rate;
  symmetric Chamfer (mm).
Baselines (same evaluation): chance = the contact set of a random same-family training goal in its own start-wrist
frame (typical grasp geometry, no information about this request); z-NN = the training goal nearest in standardised z,
its world contacts mapped exactly into the evaluation start-wrist frame (reference only: analytic transform).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from getkite.decoder.multifamily import BalancedSampler, draw_start, fixed_start
from getkite.decoder.spatial_bridge import ContactSetBridge, RawContactSetBridge, SetBridgeConfig, contact_set_label, request_to_wrist
from getkite.decoder.train_scratch import FrozenPsi, build_bank, load_config, load_pool, start_noise

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("_sb", Path(__file__).with_name("spatial_bridge.py"))
_sb = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_sb)
predict_inputs, batched, rec_hashes, to_dev = _sb.predict_inputs, _sb.batched, _sb.rec_hashes, _sb.to_dev

TAUS = (5.0, 10.0)
BIG = 1e4


def eval_set(rec, bank, psi, seed, cfg, dev):
    sg = to_dev(fixed_start(bank, rec, seed, **start_noise(cfg)), dev)
    z = batched(lambda i: psi(rec.c_pos[i], rec.c_normal[i], rec.c_valid[i]), len(rec))
    gp, gn, gv = contact_set_label(rec.c_pos, rec.c_normal, rec.c_valid, sg.R1, sg.t1)
    return {"z": z, "R0": sg.R1, "t0": sg.t1, "gp": gp, "gn": gn, "gv": gv, "r": rec.r,
            "wp": rec.c_pos, "wn": rec.c_normal, "req_pos": rec.c_pos, "req_nrm": rec.c_normal, "req_valid": rec.c_valid}


def set_eval(perms, pp, pn, pe, gp, gn, gv):
    """Per-record matching of predicted (existing) vs true contacts. Returns per-record tensors."""
    S = pp.shape[1]
    pe = pe.clone()
    none = ~pe.any(1)
    pe[none] = False  # handled by caller (forces >= 1 slot)
    D = (pp[:, :, None] - gp[:, None]).norm(dim=-1) * 1000  # mm (B,S,K)
    real = pe[:, :, None] & gv[:, None]
    one = pe[:, :, None] ^ gv[:, None]
    cost = torch.where(real, D, torch.where(one, torch.full_like(D, BIG), torch.zeros_like(D)))
    tot = cost[:, torch.arange(S, device=cost.device)[None], perms].sum(-1)
    a = perms[tot.argmin(-1)]  # slot i -> target a[i]
    Dm = D.gather(2, a[..., None]).squeeze(-1)
    rr = real.gather(2, a[..., None]).squeeze(-1)  # matched real-real pairs (B,S)
    gnm = gn.gather(1, a[..., None].expand_as(gn))
    ang = torch.rad2deg(torch.acos((pn * gnm).sum(-1).clamp(-1, 1)))
    npred, ngt = pe.sum(1), gv.sum(1)
    out = {"npred": npred, "ngt": ngt, "Dm": Dm, "ang": ang, "rr": rr}
    Dmin_pred = torch.where(gv[:, None], D, torch.full_like(D, BIG)).amin(2)  # pred -> nearest true
    Dmin_gt = torch.where(pe[:, :, None], D, torch.full_like(D, BIG)).amin(1)  # true -> nearest pred
    out["chamfer"] = 0.5 * ((Dmin_gt * gv).sum(1) / ngt.clamp_min(1) + (Dmin_pred * pe).sum(1) / npred.clamp_min(1))
    for t in TAUS:
        hit = rr & (Dm < t)
        nh = hit.sum(1)
        dup = pe & ~hit & (Dmin_pred < t)
        out[f"hits@{t:g}"], out[f"dup@{t:g}"] = nh, dup.sum(1)
        out[f"exact@{t:g}"] = (npred == ngt) & (nh == ngt)
    return out


def summarise(o, sel):
    s = lambda k: o[k][sel]  # noqa: E731
    rr = s("rr")
    dm, ang = s("Dm")[rr], s("ang")[rr]
    npred, ngt = s("npred").float(), s("ngt").float()
    r = {"n": int(sel.sum()), "count_acc": round(float((npred == ngt).float().mean()), 3),
         "count_bias": round(float((npred - ngt).mean()), 3), "count_mae": round(float((npred - ngt).abs().mean()), 3),
         "pos_mm": round(float(dm.mean()), 2), "pos_med_mm": round(float(dm.median()), 2),
         "normal_deg": round(float(ang.mean()), 2), "normal_med_deg": round(float(ang.median()), 2),
         "chamfer_mm": round(float(s("chamfer").mean()), 2)}
    for t in TAUS:
        h = s(f"hits@{t:g}").float().sum()
        r[f"missing@{t:g}"] = round(float(1 - h / ngt.sum()), 3)
        r[f"precision@{t:g}"] = round(float(h / npred.sum().clamp_min(1)), 3)
        r[f"dup@{t:g}"] = round(float(s(f"dup@{t:g}").float().sum() / npred.sum().clamp_min(1)), 3)
        r[f"exact@{t:g}"] = round(float(s(f"exact@{t:g}").float().mean()), 3)
    return r


def metrics(perms, pp, pn, pe, d, fam_of, families):
    pe = pe.clone()
    none = ~pe.any(1)
    if none.any():  # predicted set is never empty: keep the single most likely slot
        pe[none, :] = False
        pe[none, 0] = True  # caller passes slots sorted by existence
    o = set_eval(perms, pp, pn, pe, d["gp"], d["gn"], d["gv"])
    res = {}
    for f in families + ["all"]:
        sel = torch.ones(len(pp), dtype=torch.bool, device=pp.device) if f == "all" else torch.as_tensor(fam_of == f, device=pp.device)
        if sel.any():
            res[f] = summarise(o, sel)
    return res


def bridge_args(bridge, d, i):
    """The prediction inputs: z (or, for the raw control, the contact REQUEST tensors psi itself consumes) + start pose.
    Labels gp/gn/gv are never passed."""
    if isinstance(bridge, RawContactSetBridge):
        return d["req_pos"][i], d["req_nrm"][i], d["req_valid"][i], d["R0"][i], d["t0"][i]
    if getattr(bridge, "world_target", False):  # z only: the pose inputs are held at a constant (identity, origin)
        z = d["z"][i]
        return z, torch.eye(3, device=z.device).expand(len(z), 3, 3), torch.zeros(len(z), 3, device=z.device)
    return d["z"][i], d["R0"][i], d["t0"][i]


def to_wrist_all(pp, pn, R0, t0):
    """World-frame prediction -> evaluation start-wrist frame (exact, analytic; applied to every slot)."""
    return (pp - t0[:, None]) @ R0, pn @ R0


def predict(bridge, d, frame="wrist"):
    pp, pn, lg = (torch.cat(x) for x in zip(*[bridge(*bridge_args(bridge, d, i))
                                               for i in (slice(s, s + 2048) for s in range(0, len(d["z"]), 2048))]))
    if getattr(bridge, "world_target", False) and frame == "wrist":
        pp, pn = to_wrist_all(pp, pn, d["R0"], d["t0"])
    order = lg.argsort(1, descending=True)  # sort slots by existence (for the >=1 rule)
    tk = lambda x: x.gather(1, order[..., None].expand_as(x)) if x.dim() == 3 else x.gather(1, order)  # noqa: E731
    return tk(pp), tk(pn), tk(lg).sigmoid() > 0.5


def retrieved(idx, trw, d):
    """Training goals idx (world contacts) mapped exactly into d's evaluation start-wrist frame."""
    p, n, v = contact_set_label(trw["wp"][idx], trw["wn"][idx], trw["gv"][idx], d["R0"], d["t0"])
    return p, n, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--input", choices=["z", "raw"], default="z")
    ap.add_argument("--raw_fourier", type=int, default=8, help="raw control: Fourier octaves on contact positions")
    ap.add_argument("--raw_frame", choices=["world", "wrist"], default="world")
    ap.add_argument("--target", choices=["wrist", "world"], default="wrist",
                    help="world: z bridge predicts the WORLD-frame set from z alone; evaluation maps it exactly to the start-wrist frame")
    ap.add_argument("--overfit", type=int, default=0, help="train on N fixed training goals at a fixed start, no noise")
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
    pools = {"val": load_pool(cfg["val"], bank).to(dev) if cfg.get("val") else train}
    pools |= {k: load_pool(v, bank).to(dev) for k, v in cfg.get("extra_eval", {}).items() if not k.startswith("train_goals")}
    g = torch.Generator().manual_seed(1)
    pools["train_goals"] = train.index(torch.randperm(len(train), generator=g)[:6000].to(dev))
    full_train = train
    if a.overfit:  # fixed examples: same records, same start (val_start_seed), clean normals -> train == "overfit" pool
        train = train.index(torch.randperm(len(train), generator=torch.Generator().manual_seed(5))[:a.overfit].to(dev))
        pools["overfit"] = train
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev)

    htr = rec_hashes(full_train)
    checks = {k: {"n": len(p), "overlap_with_train": len(rec_hashes(p) & htr)} for k, p in pools.items() if k not in ("train_goals", "overfit")}
    checks["leap_heldout_robots_in_train"] = len(set(pools["leap_heldout"].r.tolist()) & set(full_train.r.tolist()))
    print("separation:", checks, flush=True)
    if a.input == "raw":
        print("NOTE raw control: the request's own validity mask is an input (it defines the request set, so its count is "
              "visible to the encoder); the z bridge sees the request only through psi's z.", flush=True)

    bridge = (RawContactSetBridge if a.input == "raw" else ContactSetBridge)(SetBridgeConfig(raw_fourier=a.raw_fourier, raw_frame=a.raw_frame)).to(dev)
    bridge.world_target = a.target == "world"
    assert not (bridge.world_target and a.input == "raw")
    trs = eval_set(full_train, bank, psi, cfg.get("train_start_seed", 0) + 17, cfg, dev)
    bridge.set_z_stats(trs["z"])  # z bridge only (the raw encoder ignores these buffers' use: it has its own LayerNorm)
    perms = bridge.perms
    fam_tr = fam[trs["r"].cpu().numpy()]
    ztr = (trs["z"] - bridge.z_mu) / bridge.z_sd

    E = {k: eval_set(p, bank, psi, cfg["val_start_seed"], cfg, dev) for k, p in pools.items()}
    baselines = {}
    rng = np.random.default_rng(0)
    for k, d in E.items():
        fo = fam[d["r"].cpu().numpy()]
        # chance: random same-family training goal
        idx_c = torch.as_tensor([rng.choice(np.flatnonzero(fam_tr == f)) for f in fo], device=dev)
        # z-NN (exclude exact self-match for the train_goals subset)
        zq = (d["z"] - bridge.z_mu) / bridge.z_sd
        nn_idx = []
        for s in range(0, len(zq), 1024):
            D = torch.cdist(zq[s:s + 1024], ztr)
            if k == "train_goals":
                D[D < 1e-4] = float("inf")
            nn_idx.append(D.argmin(1))
        idx_nn = torch.cat(nn_idx)
        baselines[k] = {}
        # chance: that goal's contact set in ITS OWN start-wrist frame (typical wrist-relative grasp geometry)
        baselines[k]["chance"] = metrics(perms, trs["gp"][idx_c], trs["gn"][idx_c], trs["gv"][idx_c], d, fo, families)
        baselines[k]["zNN"] = metrics(perms, *retrieved(idx_nn, trs, d), d, fo, families)
        # transform-and-copy reference: the request mapped analytically into the start-wrist frame, existence = request
        # mask (must be ~0 error: checks the label frame and the evaluation, and is the raw control's ceiling)
        cp, cn = request_to_wrist(d["req_pos"], d["req_nrm"], d["req_valid"], d["R0"], d["t0"])
        baselines[k]["copy"] = metrics(perms, cp, cn, d["req_valid"].bool(), d, fo, families)

    opt = torch.optim.AdamW(bridge.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    sampler = BalancedSampler(train.to("cpu"), bank.family, seed=a.seed)
    gen = torch.Generator().manual_seed(a.seed + 7)
    log = open(out / "log.jsonl", "w")
    best, hist, best_res = (1e9, -1), [], None

    def evaluate():
        bridge.eval()
        res = {}
        with torch.no_grad():
            for k, d in E.items():
                res[k] = metrics(perms, *predict(bridge, d), d, fam[d["r"].cpu().numpy()], families)
                if bridge.world_target:  # same predictions scored in the world frame against world labels
                    dw = {**d, "gp": d["wp"] * d["gv"][..., None], "gn": d["wn"] * d["gv"][..., None]}
                    res[k + "/world"] = metrics(perms, *predict(bridge, d, frame="world"), dw, fam[d["r"].cpu().numpy()], families)
        bridge.train()
        return res

    if a.overfit:
        Ofx = E["overfit"]
    t_start, acc = time.time(), {}
    nn_rad = 0.0 if a.overfit else cfg["normal_noise_rad"]
    for step in range(1, a.steps + 1):
        if a.overfit:  # fixed examples at their fixed start
            j = torch.randint(0, a.overfit, (min(a.batch, a.overfit),), generator=gen).to(dev)
            d = {k: v[j] for k, v in Ofx.items()}
            L, _ = bridge.set_loss(*bridge(*bridge_args(bridge, d, slice(None))), d["gp"], d["gn"], d["gv"])
        else:
            rb = train.index(sampler.sample(a.batch).to(dev))
            sg = to_dev(draw_start(bank, rb, gen, **start_noise(cfg)), dev)
            gp, gn, gv = contact_set_label(rb.c_pos, rb.c_normal, rb.c_valid, sg.R1, sg.t1)  # supervision only
            if a.input == "raw":
                nrm = _sb.jitter_normals(rb.c_normal, nn_rad, gen) if nn_rad > 0 else rb.c_normal
                out_b = bridge(rb.c_pos, nrm, rb.c_valid, sg.R1, sg.t1)  # the request itself, as psi would see it
            elif bridge.world_target:
                z = predict_inputs(psi, rb, sg, gen, nn_rad)[0]
                gp, gn, gv = contact_set_label(rb.c_pos, rb.c_normal, rb.c_valid, torch.eye(3, device=dev).expand(len(z), 3, 3),
                                               torch.zeros(len(z), 3, device=dev))  # world-frame labels
                out_b = bridge(*bridge_args(bridge, {"z": z, "R0": None, "t0": None}, slice(None)))
            else:
                out_b = bridge(*predict_inputs(psi, rb, sg, gen, nn_rad))
            L, _ = bridge.set_loss(*out_b, gp, gn, gv)
        opt.zero_grad(set_to_none=True)
        L["total"].backward()
        gnorm = torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        opt.step()
        sched.step()
        for kk, v in L.items():
            acc[kk] = acc.get(kk, 0.0) + float(v.detach())
        if step % 200 == 0:
            log.write(json.dumps({"step": step, **{kk: round(v / 200, 5) for kk, v in acc.items()}, "grad_norm": round(float(gnorm), 3)}) + "\n")
            last_loss, acc = {kk: round(v / 200, 4) for kk, v in acc.items()}, {}
        if step % a.eval_every == 0 or step == a.steps:
            res = evaluate()
            macro = float(np.mean([res["val"][f]["chamfer_mm"] for f in families]))
            hist.append({"step": step, "val_macro_chamfer": round(macro, 2), "loss": last_loss,
                         **{k: {f: {m: v[f][m] for m in ("chamfer_mm", "pos_mm", "count_acc", "missing@5")} for f in v} for k, v in res.items()}})
            log.write(json.dumps({"step": step, "eval": res}) + "\n")
            log.flush()
            print(f"step {step} loss {last_loss} | val chamfer {[res['val'][f]['chamfer_mm'] for f in families]} "
                  f"pos {[res['val'][f]['pos_mm'] for f in families]} cnt {[res['val'][f]['count_acc'] for f in families]} "
                  f"| heldout ch {res['leap_heldout']['all']['chamfer_mm']} pos {res['leap_heldout']['all']['pos_mm']} "
                  f"| train ch {res['train_goals']['all']['chamfer_mm']} "
                  + (f"| overfit ch {res['overfit']['all']['chamfer_mm']} pos {res['overfit']['all']['pos_mm']} cnt {res['overfit']['all']['count_acc']} " if a.overfit else "")
                  + f"({time.time() - t_start:.0f}s)", flush=True)
            state = {"model": bridge.state_dict(), "config": bridge.c.to_dict(), "step": step, "args": vars(a)}
            torch.save(state, out / "last.pt")
            if macro < best[0]:
                best, best_res = (macro, step), res
                torch.save(state, out / "best.pt")
    results = {"separation": checks, "baselines": baselines, "last": res, "best_step": best[1], "best": best_res,
               "history": hist, "args": vars(a), "config_file": a.config, "bridge_config": bridge.c.to_dict(),
               "inputs": "forward(z, R0, t0): z = frozen KITE psi(contact request), R0/t0 = start wrist world pose"}
    (out / "results.json").write_text(json.dumps(results, indent=1))
    cols = ["count_acc", "count_bias", "pos_mm", "normal_deg", "missing@5", "missing@10", "dup@5", "precision@5", "exact@5", "chamfer_mm"]
    for k in E:
        if (k + "/world") in res:
            print(f"\n[{k}/world frame, bridge only]  " + "  ".join(cols))
            for n, m in (("bridge_last", res[k + "/world"]), (f"bridge*@{best[1]}", best_res[k + "/world"])):
                for f in m:
                    print(f"  {n:14s} {f:8s} " + "  ".join(f"{m[f][c]}" for c in cols))
        print(f"\n[{k}]  " + "  ".join(cols))
        rows = [(n, baselines[k][n]) for n in ("chance", "zNN", "copy")] + [("bridge_last", res[k]), (f"bridge*@{best[1]}", best_res[k])]
        for n, m in rows:
            for f in m:
                print(f"  {n:14s} {f:8s} " + "  ".join(f"{m[f][c]}" for c in cols))


if __name__ == "__main__":
    main()
