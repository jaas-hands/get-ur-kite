"""T1: translation-only diagnostic for the direct decoder (2026-11-19). Separate from W1 and from the multi-family runs.

Question: can the direct path (frozen psi z -> intent tokens -> GET -> wrist correction) learn the simplest intent-to-
motion map, a pure wrist translation, and how does it compare with a small network that sees z and the start pose?

Task (one Allegro morphology; nothing but translation varies):
  * one base goal record (allegro_right_train.npz, row `base_row`): fixed finger configuration qc2, fixed wrist
    orientation R2, and its contacts (2-6) on the hand.
  * targets: the whole goal rigidly translated by Delta ~ U[-box, box]^3 (world). Goal wrist t2 + Delta; contacts
    c_pos + Delta; normals, orientation, fingers unchanged. Train targets and validation targets are disjoint draws
    (different seeds); a fixed subset of the train targets is scored as "training goals".
  * starts: t1 = goal + N(0, xyz_std) per axis (KITE's 5 mm), R1 = R2 and fingers = qc2 exactly (no rotation/joint noise).
    Resampled every batch in training; fixed (val_start_seed) for evaluation.
  * inputs at inference: z = psi(request contacts) (frozen), the start state (wrist pose, fingers), the robot
    description. No reconstructed contact coordinates enter the movement path.
  * target: KITE's normalised root residual (target_residuals: wrist-frame translation / res_xyz_m, 6D rotation / res_rot_deg;
    the rotation part is 0 here), identical for both models.

Models (`--model`):
  get  direct GET, the direct_worldaux_v1 architecture (z-only trunk -> 4 intent tokens + start-wrist world pose -> GET),
       fresh init, motion residual loss only (root + joint; the joint target is 0 here). The auxiliary contact head is
       constructed (same module layout) but never trained or used. v1's optimiser groups: GET lr 3e-4, trunk lr 1e-3.
  mlp  small network: [standardised z, start pose feature (t1 * pos_scale, 6D R1, Fourier 8 / 8 of both, exactly the GET's
       world-pose feature)] -> 3 x [Linear 512, LayerNorm, GELU] -> 9 root residuals; same root loss (mean over 9).
       AdamW lr 1e-3, wd 1e-4, clip 1.0.
Both: same records, starts, z standardisation (from the training targets), batch, steps, schedule (warmup 500 + cosine).

Reported (train subset and validation, fixed starts): translation error mm, rotation error deg, no-movement translation
error, how far the wrist moved, and correction correlation (start-wrist frame: slope, R2_lin, R2_id from
transfer_benchmark.wrist_fit, and per-axis Pearson r between predicted and required translation). Also a closed-form
ridge readout of the target translation from z (evaluation-only reference: is the target linearly available in z?).

    python -m getkite.decoder.translation_diag --config configs/decoder/scratch/t1_translation_allegro.json --model get|mlp \
        [--checks_only] [--steps N] [--set k=v ...]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from .direct_worldaux import DirectWorldAux
from .get_scratch import GETScratch, ScratchConfig, fourier, target_residuals
from .multifamily import Records, draw_start, fixed_start
from .rotations import geodesic_angle, matrix_to_6d
from .spatial_bridge import SetBridgeConfig
from .train_direct_worldaux import gnorm
from .train_scratch import FrozenPsi, build_bank, git_state, jitter_normals, load_config, load_pool, residual_loss, sha16


def make_targets(base: Records, n: int, box: float, seed: int) -> Records:
    g = torch.Generator().manual_seed(seed)
    d = ((torch.rand(n, 3, generator=g) * 2 - 1) * box).to(base.t2)  # CPU generator: identical draws on any device
    rep = lambda x: x.expand(n, *x.shape[1:]).clone()  # noqa: E731
    v = rep(base.c_valid)
    return Records(r=rep(base.r), t2=rep(base.t2) + d, R2=rep(base.R2), e2=rep(base.e2), qc2=rep(base.qc2),
                   active=rep(base.active), c_pos=rep(base.c_pos) + d[:, None] * v[..., None], c_normal=rep(base.c_normal),
                   c_valid=v, c_tpl=rep(base.c_tpl)), d


class PoseZMLP(nn.Module):
    def __init__(self, c: ScratchConfig, z_dim=256, hidden=512, layers=3):
        super().__init__()
        self.c = c
        self.register_buffer("z_mu", torch.zeros(z_dim)); self.register_buffer("z_sd", torch.ones(z_dim))
        d_in = z_dim + 9 + 3 * 2 * c.pos_fourier + 6 * 2 * c.rot_fourier
        mods = []
        for _ in range(layers):
            mods += [nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU()]
            d_in = hidden
        self.net = nn.Sequential(*mods, nn.Linear(hidden, 9))

    @torch.no_grad()
    def set_z_stats(self, z):
        self.z_mu.copy_(z.mean(0)); self.z_sd.copy_(z.std(0).clamp_min(1e-4))

    def pose_feature(self, R0w, t0w):  # = DirectWorldAux.pose_feature
        c = self.c
        wp = torch.cat([t0w * c.pos_scale, matrix_to_6d(R0w)], -1)
        return torch.cat([wp, fourier(wp[:, :3], c.pos_fourier, c.fourier_base), fourier(wp[:, 3:], c.rot_fourier, c.fourier_base)], -1)

    def forward(self, bank, r, qc0, R0, t0, z, aux=False):
        res_root = self.net(torch.cat([(z - self.z_mu) / self.z_sd, self.pose_feature(R0, t0)], -1))
        # decode exactly as GETScratch.decode (wrist frame); fingers unchanged
        dec = decode(self.c, bank, r, qc0, R0, t0, res_root, torch.zeros_like(qc0))
        return {"res_root": res_root, "res_coord": torch.zeros_like(qc0)} | dec


def decode(c, bank, r, qc0, R0, t0, res_root, res_coord):
    """GETScratch.decode with config c (wrist-frame update; fingers from res_coord)."""
    return GETScratch.decode(SimpleNamespace(c=c), bank, r, qc0, R0, t0, res_root, res_coord)


def corr_stats(sg, out):
    """Start-wrist-frame translation correction: required vs predicted."""
    R1, t1, tg, t = (x.detach().double().cpu().numpy() for x in (sg.R1, sg.t1, sg.tg, out["t"]))
    need = np.einsum("nji,nj->ni", R1, tg - t1); got = np.einsum("nji,nj->ni", R1, t - t1)
    A, *_ = np.linalg.lstsq(need, got, rcond=None)
    r = {"slope": float((got * need).sum() / (need * need).sum()),
         "R2_lin": float(1 - ((got - need @ A) ** 2).sum() / max(((got - got.mean(0)) ** 2).sum(), 1e-12)),
         "R2_id": float(1 - ((got - need) ** 2).sum() / ((need - need.mean(0)) ** 2).sum())}
    for i, ax in enumerate("xyz"):
        g, n = got[:, i], need[:, i]
        r[f"pearson_{ax}"] = float(np.corrcoef(g, n)[0, 1]) if g.std() > 1e-12 else 0.0
    return r


@torch.no_grad()
def evaluate_t(model, psi, bank, rec, sg, bs=1024):
    model.eval()
    o = {"t": [], "R": [], "qc": []}
    for s in range(0, len(rec), bs):
        i = torch.arange(s, min(s + bs, len(rec)), device=rec.r.device)
        rb = rec.index(i)
        g = type(sg)(**{k: getattr(sg, k)[i] for k in sg.__dataclass_fields__})
        p = model(bank, rb.r, g.qc1, g.R1, g.t1, psi(rb.c_pos, rb.c_normal, rb.c_valid), aux=False)
        for k in o:
            o[k].append(p[k])
    o = {k: torch.cat(v) for k, v in o.items()}
    res = {"trans_mm": float(((o["t"] - sg.tg).norm(dim=-1) * 1000).mean()),
           "trans_mm_median": float(((o["t"] - sg.tg).norm(dim=-1) * 1000).median()),
           "noop_trans_mm": float(((sg.t1 - sg.tg).norm(dim=-1) * 1000).mean()),
           "rot_deg": float(torch.rad2deg(geodesic_angle(o["R"], sg.Rg)).mean()),
           "moved_mm": float(((o["t"] - sg.t1).norm(dim=-1) * 1000).mean()),
           "finger_move_rad": float((o["qc"] - sg.qc1).abs().max())}
    res |= corr_stats(sg, o)
    model.train()
    return res


def ridge_reference(psi, train, val, sub, lam=1e-2):
    """Closed-form ridge from standardised z to the target wrist position (world, metres); evaluation-only reference."""
    def zz(x):
        return torch.cat([psi(x.c_pos[s:s + 4096], x.c_normal[s:s + 4096], x.c_valid[s:s + 4096]) for s in range(0, len(x), 4096)]).double()
    with torch.no_grad():
        zt, zv, zs = zz(train), zz(val), zz(sub)
    mu, sd = zt.mean(0), zt.std(0).clamp_min(1e-4)
    X = torch.cat([(zt - mu) / sd, torch.ones(len(zt), 1, dtype=zt.dtype, device=zt.device)], 1)
    W = torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1], dtype=X.dtype, device=X.device), X.T @ train.t2.double())
    out = {}
    for k, z, rec in (("train_subset", zs, sub), ("val", zv, val)):
        P = torch.cat([(z - mu) / sd, torch.ones(len(z), 1, dtype=z.dtype, device=z.device)], 1) @ W
        out[k] = {"target_position_mm": float(((P - rec.t2.double()).norm(dim=-1) * 1000).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", choices=("get", "mlp"), required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--checks_only", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    T = cfg["t1"]
    stop_at = a.steps or cfg["total_steps"]
    out = Path(cfg["out_dir"]) / a.model; out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert dev.type == "cuda" or a.checks_only, "T1 training needs a GPU (a node with an old driver falls back to CPU)"
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    bank = build_bank(cfg, dev)
    assert len(bank.names) == 1 and bank.names[0] == "allegro_right" and bank.surface_sampling == "hand_frame"
    base = load_pool([T["base"]], bank).index(torch.tensor([T["base_row"]])).to(dev)
    train, dtr = make_targets(base, T["n_train"], T["box_m"], T["train_seed"])
    val, dva = make_targets(base, T["n_val"], T["box_m"], T["val_seed"])
    sub = train.index(torch.arange(T["n_train_eval"]))
    noise = {"xyz_std": T["start_xyz_std"], "rot_std_deg": 0.0, "joint_frac": 0.0}
    mc = ScratchConfig.from_dict(cfg["model"])
    ref = torch.load(cfg["set_config_from"], map_location="cpu", weights_only=False)
    scfg = SetBridgeConfig(**ref["config"])
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    with torch.no_grad():
        zs = torch.cat([psi(train.c_pos[s:s + 4096], train.c_normal[s:s + 4096], train.c_valid[s:s + 4096]) for s in range(0, len(train), 4096)])
    if a.model == "get":
        model = DirectWorldAux(mc, scfg).to(dev)
        model.trunk.set_z_stats(zs)
        groups = [{"params": list(model.decoder.parameters()) + list(model.intent_tok.parameters()) + list(model.intent_pose.parameters()),
                   "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]},
                  {"params": list(model.trunk.parameters()), "lr": cfg["trunk_lr"], "weight_decay": cfg["trunk_weight_decay"]}]
        clip_groups = [groups[0]["params"], groups[1]["params"]]
        clips = [cfg["grad_clip"], cfg["trunk_grad_clip"]]
    else:
        model = PoseZMLP(mc, hidden=T["mlp_hidden"], layers=T["mlp_layers"]).to(dev)
        model.set_z_stats(zs)
        groups = [{"params": list(model.parameters()), "lr": T["mlp_lr"], "weight_decay": cfg["weight_decay"]}]
        clip_groups, clips = [groups[0]["params"]], [cfg["grad_clip"]]
    params = [p for g in groups for p in g["params"]]

    # ---------------- checks
    C = {"model": a.model}
    sg0 = fixed_start(bank, val, cfg["val_start_seed"], **noise)
    C["fingers_fixed_max_abs"] = float((sg0.qc1 - sg0.qcg).abs().max())
    C["orientation_fixed_max_abs"] = float((sg0.R1 - sg0.Rg).abs().max())
    C["goal_orientation_equals_base"] = float((sg0.Rg - base.R2).abs().max())
    C["start_xyz_std_mm"] = float(((sg0.t1 - sg0.tg) * 1000).std())
    Rs, ts = bank.fk(base.r, bank.expand(base.r, base.qc2), base.R2, base.t2)
    p0, _ = bank.points(base.r, Rs, ts, base.c_tpl)
    Rs2, ts2 = bank.fk(val.r[:64], bank.expand(val.r[:64], val.qc2[:64]), val.R2[:64], val.t2[:64])
    p2, _ = bank.points(val.r[:64], Rs2, ts2, val.c_tpl[:64])
    v = val.c_valid[:64]
    C["base_contact_reproduction_mm"] = float(((p0 - base.c_pos).norm(dim=-1)[base.c_valid]).mean() * 1000)
    C["translated_contact_reproduction_mm"] = float(((p2 - val.c_pos[:64]).norm(dim=-1)[v]).mean() * 1000)
    C["n_contacts"] = int(base.c_valid.sum())
    C["train_val_min_target_distance_mm"] = float(torch.cdist(dva[:500].double().cpu(), dtr.double().cpu()).min() * 1000)
    with torch.no_grad():
        zv = psi(val.c_pos[:512], val.c_normal[:512], val.c_valid[:512])
        C["z_sd_across_targets_mean"] = float(zv.std(0).mean())
    m = copy.deepcopy(model).eval()
    if a.model == "get":
        m.decoder.amp = False
        g0 = torch.Generator().manual_seed(7)
        with torch.no_grad():
            for h in (m.decoder.root_head, m.decoder.joint_head):
                h[-1].weight.copy_(torch.randn(h[-1].weight.shape, generator=g0) * 1e-2)
    rb = val.index(torch.arange(64, device=dev))
    g = type(sg0)(**{k: getattr(sg0, k)[:64].to(dev) for k in sg0.__dataclass_fields__})
    z = psi(rb.c_pos, rb.c_normal, rb.c_valid)
    with torch.no_grad():
        o1 = m(bank, rb.r, g.qc1, g.R1, g.t1, z, aux=False)
        rb2 = copy.copy(rb)
        for f in ("t2", "e2", "R2", "qc2", "c_tpl"):
            x = getattr(rb2, f)
            setattr(rb2, f, x + (3 if x.dtype == torch.long else 0.37))
        o2 = m(bank, rb2.r, g.qc1, g.R1, g.t1, z, aux=False)
        C["labels_not_inputs_max_abs"] = max(float((o1[k] - o2[k]).abs().max()) for k in ("t", "R", "qc"))
        if a.model == "get":
            m2 = copy.deepcopy(m); m2.contacts.head.weight.normal_()
            o3 = m2(bank, rb.r, g.qc1, g.R1, g.t1, z, aux=False)
            C["aux_head_no_effect_max_abs"] = max(float((o1[k] - o3[k]).abs().max()) for k in ("t", "R", "qc"))
        tr_root, tr_j = target_residuals(bank, rb.r, g.qc1, g.R1, g.t1, g.qcg, g.Rg, g.tg, mc)
        d = decode(mc, bank, rb.r, g.qc1, g.R1, g.t1, tr_root, tr_j)
        C["decode_roundtrip_max_abs"] = max(float((d["t"] - g.tg).abs().max()), float((d["R"] - g.Rg).abs().max()))
        C["target_rot_residual_max_abs"] = float(tr_root[:, 3:].abs().max())
    o = m(bank, rb.r, g.qc1, g.R1, g.t1, z, aux=False)
    L = ((o["res_root"] - tr_root) ** 2).mean()
    L.backward()
    C["grad_to_all_trainable"] = gnorm([p.grad for p in m.parameters() if p.requires_grad])
    if a.model == "get":
        C["grad_to_trunk"] = gnorm([p.grad for p in m.trunk.parameters()])
        C["grad_to_aux_head_none"] = all(p.grad is None for p in m.contacts.parameters())
    C["n_params_trained"] = sum(p.numel() for p in params)
    C["ridge_reference"] = ridge_reference(psi, train, val, sub)
    C["pass"] = bool(C["fingers_fixed_max_abs"] == 0.0 and C["orientation_fixed_max_abs"] < 1e-6 and C["labels_not_inputs_max_abs"] == 0.0
                     and C["decode_roundtrip_max_abs"] < 1e-4 and C["target_rot_residual_max_abs"] < 1e-3 and C["grad_to_all_trainable"] > 0
                     and C["train_val_min_target_distance_mm"] > 0 and abs(C["translated_contact_reproduction_mm"] - C["base_contact_reproduction_mm"]) < 1e-3
                     and (a.model == "mlp" or (C["aux_head_no_effect_max_abs"] == 0.0 and C["grad_to_aux_head_none"])))
    (out / "checks.json").write_text(json.dumps(C, indent=1))
    print("checks", json.dumps(C), flush=True)
    assert C["pass"], "T1 checks failed"
    if a.checks_only:
        return

    # ---------------- training
    opt = torch.optim.AdamW(groups)
    W, Tt = cfg["warmup_steps"], cfg["total_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / W) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / Tt))))
    val_sg = fixed_start(bank, val, cfg["val_start_seed"], **noise)
    sub_sg = fixed_start(bank, sub, cfg["val_start_seed"] + 1, **noise)
    val_sg, sub_sg = (type(x)(**{k: getattr(x, k).to(dev) for k in x.__dataclass_fields__}) for x in (val_sg, sub_sg))
    step, best = 0, None
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step, best = ck["step"], ck.get("best")
    (out / "config.json").write_text(json.dumps(cfg | {"_model": a.model}, indent=1))
    (out / "provenance.json").write_text(json.dumps({"kind": f"t1_translation_{a.model}", "n_params_trained": C["n_params_trained"],
                                                     "psi_ckpt_sha16": psi.ckpt_sha, "base_npz_sha16": sha16(T["base"]["npz"]),
                                                     "model_repo": git_state(Path(__file__).parents[3]),
                                                     "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else None}, indent=1))
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    gen = torch.Generator().manual_seed(cfg["seed"] + 1000 + step)
    log = open(out / "log.jsonl", "a")

    def save(path, extra=None):
        tmp = Path(str(path) + ".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "best": best,
                    "config": cfg, "model_config": mc.to_dict(), "set_config": scfg.to_dict(), "kind": f"t1_{a.model}"} | (extra or {}), tmp)
        os.replace(tmp, path)

    def do_eval():
        return {"val": evaluate_t(model, psi, bank, val, val_sg), "train_subset": evaluate_t(model, psi, bank, sub, sub_sg)}

    if step == 0:
        e = do_eval(); log.write(json.dumps({"step": 0, "eval": e}) + "\n"); log.flush()
        print(json.dumps({"step": 0, "eval": e}), flush=True)
    t_last, n_last, t0 = time.time(), 0, time.time()
    model.train()
    if a.model == "get":
        model.trunk.train(); model.contacts.eval()
    fam0 = torch.zeros(cfg["batch_size"], dtype=torch.long, device=dev)
    while step < stop_at and not stop["flag"]:
        idx = torch.randint(len(train), (cfg["batch_size"],), generator=gen).to(dev)
        rb = train.index(idx)
        sg = draw_start(bank, rb, gen, **noise)
        z = psi(rb.c_pos, jitter_normals(rb.c_normal, cfg["normal_noise_rad"], gen), rb.c_valid)
        o = model(bank, rb.r, sg.qc1, sg.R1, sg.t1, z, aux=False)
        tr_root, tr_j = target_residuals(bank, rb.r, sg.qc1, sg.R1, sg.t1, sg.qcg, sg.Rg, sg.tg, mc)
        if a.model == "get":
            loss, lr_, lj_ = residual_loss(o, tr_root, tr_j, bank.coord_valid[rb.r], fam0, 1)
        else:
            lr_ = ((o["res_root"] - tr_root) ** 2).mean(-1); loss = lr_.mean(); lj_ = torch.zeros_like(lr_)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gns = [float(torch.nn.utils.clip_grad_norm_(ps, c)) for ps, c in zip(clip_groups, clips)]
        opt.step(); sched.step()
        step += 1; n_last += len(idx)
        if step % cfg["log_every"] == 0:
            with torch.no_grad():
                tl = float(((o["res_root"][:, :3] - tr_root[:, :3]) ** 2).mean())
            row = {"step": step, "lr": sched.get_last_lr()[0], "train/loss": loss.item(), "train/root": lr_.mean().item(),
                   "train/trans_resid_mse": tl, "train/joint": lj_.mean().item(), "grad": gns,
                   "examples_per_s": n_last / (time.time() - t_last), "wall_s": time.time() - t0,
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0.0}
            log.write(json.dumps(row) + "\n"); log.flush()
            print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in row.items()}), flush=True)
            t_last, n_last = time.time(), 0
        if step % cfg["eval_every"] == 0 or step == stop_at:
            e = do_eval()
            log.write(json.dumps({"step": step, "eval": e}) + "\n"); log.flush()
            print(json.dumps({"step": step, "eval": e}), flush=True)
            key = -e["val"]["trans_mm"]
            if best is None or key > best["key"]:
                best = {"key": key, "step": step}
                save(out / "best.pt", {"selected_on": "val translation error (validation targets are fresh positions)"})
        if step % cfg["ckpt_every"] == 0:
            save(out / "last.pt")
    save(out / "last.pt")
    if step >= stop_at:
        (out / "DONE").write_text(json.dumps({"best": best, "wall_end": time.time()}))


if __name__ == "__main__":
    main()
