"""Train/evaluate get_universal_scratch (frozen KITE psi + randomly initialised GET decoder).

    python -m getkite.decoder.train_scratch --config configs/scratch/c1_overfit.json [--set key=value ...]

Config (JSON):
  out_dir, seed, kite_root, kite_ckpt
  robots: list of {kind: "kite", robot, urdf, template, ignore_mimic} | {kind: "leap", hand_dir}
  train / val: list of {kind: "kite"|"leap", npz, robot (kite only), rows: [start, stop] (optional)}
  start: "fixed" (one q1 per record, seed train_start_seed; C1) | "resample" (fresh KITE q1 each batch)
  val_start_seed: q1 seed of the fixed evaluation records
  normal_noise_rad: train-time normal noise before psi (KITE 0.1; 0 for C1)
  model: ScratchConfig fields; batch_size, lr, weight_decay, warmup_steps, total_steps, grad_clip,
  eval_every, log_every, ckpt_every
Loss: KITE's normalised residual. Per example: mean over the 9 root dims + mean over the robot's valid
coordinates (all supervised, as in KITE's released full_mix); averaged within each family, then across
families. Checkpoint selection: val macro success over families, then macro contact error (tie-break).
Writes out_dir/{config.json, provenance.json, log.jsonl, last.pt, best.pt, DONE}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .get_scratch import GETScratch, ScratchConfig, target_residuals
from .multifamily import K_MAX, BalancedSampler, Records, draw_start, fixed_start, load_kite_records, load_leap_records
from .robots import RobotBank, leap_desc, urdf_desc
from .rotations import axis_angle_matrix, geodesic_angle

DEFAULTS = dict(seed=0, start="resample", train_start_seed=1, val_start_seed=12345, normal_noise_rad=0.1,
                batch_size=256, lr=3e-4, weight_decay=1e-4, warmup_steps=500, total_steps=5000, grad_clip=1.0,
                eval_every=500, log_every=100, ckpt_every=500, model={}, success_pos_mm=10.0,
                success_normal_deg=15.0, eval_batch=512)


def sha16(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def load_config(path, overrides):
    cfg = dict(DEFAULTS) | json.loads(Path(path).read_text())
    for item in overrides or []:
        k, v = item.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        if k.startswith("model."):
            cfg["model"] = dict(cfg["model"]) | {k[6:]: v}
        else:
            cfg[k] = v
    from . import paths
    return paths.expand(cfg)


def build_bank(cfg, device):
    descs = []
    for r in cfg["robots"]:
        if r["kind"] == "kite":
            coords = r.get("coords")
            if coords is None and r.get("coords_npz"):  # KITE's joint order, as exported with the records
                coords = json.loads(str(np.load(r["coords_npz"])["meta"]))["coords"]
            descs.append(urdf_desc(r["urdf"], r["template"], r["robot"], r.get("family", r["robot"]),
                                   coords=coords, ignore_mimic=r.get("ignore_mimic", True)))
        else:
            descs.append(leap_desc(r["hand_dir"]))
    return RobotBank(descs, device=device)


def load_pool(spec_list, bank) -> Records:
    out = []
    for s in spec_list:
        rows = slice(*s["rows"]) if "rows" in s else None
        if s["kind"] == "kite":
            out.append(load_kite_records(s["npz"], bank, s["robot"], rows))
        else:
            out.append(load_leap_records(s["npz"], bank, rows))
    return Records.cat(out)


def jitter_normals(n, sigma, gen=None):
    if sigma <= 0:
        return n
    rv = torch.randn(n.shape, generator=gen, device="cpu").to(n) * sigma
    ang = rv.norm(dim=-1).clamp_min(1e-12)
    return (axis_angle_matrix(rv / ang[..., None], ang) @ n[..., None]).squeeze(-1)


class FrozenPsi(torch.nn.Module):
    """KITE's released psi, frozen and in eval mode (asserted on every call)."""

    def __init__(self, kite_ckpt, kite_root):
        super().__init__()
        from .kite_release import KitePsi, load_release
        rel = load_release(kite_ckpt, kite_root)
        self.psi = KitePsi(rel.modules["contact_only_encoder"]).eval().requires_grad_(False)
        self.ckpt_sha = sha16(kite_ckpt)

    def train(self, mode=True):  # never leaves eval mode
        return super().train(False)

    @torch.no_grad()
    def forward(self, pos, normal, valid):
        assert not self.psi.training and not any(p.requires_grad for p in self.psi.parameters())
        return self.psi(pos.float(), normal.float(), valid)


class RawContacts(torch.nn.Module):
    """Diagnostic reference intent source: the raw contact cloud itself (no psi), flattened (K, 7)."""

    ckpt_sha = "none (raw_contacts diagnostic: psi bypassed)"

    @torch.no_grad()
    def forward(self, pos, normal, valid):
        return torch.cat([pos.float() * valid[..., None], normal.float() * valid[..., None],
                          valid[..., None].float()], -1).flatten(1)


def residual_loss(out, tr_root, tr_j, coord_valid, family_idx, n_fam):
    """Per example: mean_9(root^2) + mean_valid(joint^2); then mean per family, then mean over families."""
    lr = ((out["res_root"] - tr_root) ** 2).mean(-1)
    m = coord_valid.float()
    lj = (((out["res_coord"] - tr_j) ** 2) * m).sum(-1) / m.sum(-1).clamp_min(1)
    per = lr + lj
    fams = [per[family_idx == f].mean() for f in range(n_fam) if (family_idx == f).any()]
    return torch.stack(fams).mean(), lr.detach(), lj.detach()


@torch.no_grad()
def pose_metrics(bank, rec: Records, r, qc, t, R, sg, cfg):
    """Contact correspondence and wrist errors for a predicted pose (joints clamped to limits)."""
    cl, cu = bank.coords_of(r, bank.lower[r]), bank.coords_of(r, bank.upper[r])
    cv = bank.coord_valid[r]
    viol = ((qc < cl - 1e-6) | (qc > cu + 1e-6)) & cv
    qcc = torch.where(cv, torch.minimum(torch.maximum(qc, cl), cu), 0.0)
    Rs, ts = bank.fk(r, bank.expand(r, qcc), R, t)
    p, n = bank.points(r, Rs, ts, rec.c_tpl)
    v = rec.c_valid
    d_mm = (p - rec.c_pos).norm(dim=-1) * 1000
    ang = torch.rad2deg(torch.acos((n * rec.c_normal).sum(-1).clamp(-1, 1)))
    vf = v.float()
    ok = ((d_mm <= cfg["success_pos_mm"]) & (ang <= cfg["success_normal_deg"])) | ~v
    return {"contact_mm": (d_mm * vf).sum(1) / vf.sum(1), "normal_deg": (ang * vf).sum(1) / vf.sum(1),
            "success": ok.all(1).float(), "wrist_mm": (t - sg.tg).norm(dim=-1) * 1000,
            "wrist_deg": torch.rad2deg(geodesic_angle(R, sg.Rg)),
            "limit_violation": viol.any(1).float()}


@torch.no_grad()
def evaluate(model, psi, bank, rec: Records, sg, cfg, families):
    model.eval()
    acc = {}
    B = cfg["eval_batch"]
    for s in range(0, len(rec), B):
        idx = torch.arange(s, min(s + B, len(rec)), device=rec.r.device)
        rb = rec.index(idx)
        g = type(sg)(**{k: getattr(sg, k)[idx] for k in sg.__dataclass_fields__})
        r = rb.r
        z = psi(rb.c_pos, rb.c_normal, rb.c_valid)
        out = model(bank, r, g.qc1, g.R1, g.t1, z)
        tr_root, tr_j = target_residuals(bank, r, g.qc1, g.R1, g.t1, g.qcg, g.Rg, g.tg, model.c)
        cv = bank.coord_valid[r].float()
        rows = {
            "model": pose_metrics(bank, rb, r, out["qc"], out["t"], out["R"], g, cfg),
            "noop": pose_metrics(bank, rb, r, g.qc1, g.t1, g.R1, g, cfg),
            "wrist_only": pose_metrics(bank, rb, r, g.qc1, g.tg, g.Rg, g, cfg),
        }
        rows["model"]["res_loss"] = ((out["res_root"] - tr_root) ** 2).mean(-1) + \
            (((out["res_coord"] - tr_j) ** 2) * cv).sum(-1) / cv.sum(-1)
        rows["noop"]["res_loss"] = (tr_root ** 2).mean(-1) + ((tr_j ** 2) * cv).sum(-1) / cv.sum(-1)
        for who, mets in rows.items():
            for k, v in mets.items():
                acc.setdefault((who, k), []).append(v.cpu())
        acc.setdefault(("_", "r"), []).append(r.cpu())
    r_all = torch.cat(acc.pop(("_", "r")))
    fam_of_r = np.array(bank.family)[r_all.numpy()]
    res = {}
    for (who, k), vs in acc.items():
        v = torch.cat(vs).numpy()
        for f in families:
            sel = fam_of_r == f
            if sel.any():
                res[f"{f}/{who}/{k}"] = float(v[sel].mean())
        per_robot = [v[r_all.numpy() == i].mean() for i in np.unique(r_all.numpy())]
        res[f"all/{who}/{k}"] = float(np.mean([res[f"{f}/{who}/{k}"] for f in families if f"{f}/{who}/{k}" in res]))
        res[f"robot_macro/{who}/{k}"] = float(np.mean(per_robot))
    model.train()
    return res


def git_state(path):
    try:
        h = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        d = subprocess.run(["git", "-C", str(path), "status", "--porcelain"], capture_output=True, text=True).stdout
        return {"head": h, "dirty_files": len(d.splitlines())}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--eval-only", default="")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    bank = build_bank(cfg, dev)
    families = sorted(set(bank.family))
    fam_index = torch.tensor([families.index(f) for f in bank.family], device=dev)
    train = load_pool(cfg["train"], bank).to(dev)
    val = load_pool(cfg["val"], bank).to(dev) if cfg.get("val") else train
    extra = {k: load_pool(v, bank).to(dev) for k, v in cfg.get("extra_eval", {}).items()}  # reported, never selected on
    mc = ScratchConfig.from_dict(cfg["model"])
    if mc.intent_source == "raw_contacts":
        mc.n_contacts, mc.z_dim = K_MAX, 7 * K_MAX
        psi = RawContacts().to(dev)
    else:
        psi = FrozenPsi(cfg["kite_ckpt"], cfg.get("kite_root")).to(dev)
    model = GETScratch(mc).to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    val_sg = fixed_start(bank, val, cfg["val_start_seed"])
    val_sg = type(val_sg)(**{k: getattr(val_sg, k).to(dev) for k in val_sg.__dataclass_fields__})
    extra_sg = {}
    for k, v in extra.items():
        g = fixed_start(bank, v, cfg["val_start_seed"])
        extra_sg[k] = type(g)(**{f: getattr(g, f).to(dev) for f in g.__dataclass_fields__})
    train_fixed = None
    if cfg["start"] == "fixed":
        train_fixed = fixed_start(bank, train, cfg["train_start_seed"])
        train_fixed = type(train_fixed)(**{k: getattr(train_fixed, k).to(dev) for k in train_fixed.__dataclass_fields__})

    # z standardisation from the training pool (noisy normals as in training)
    gen = torch.Generator().manual_seed(cfg["seed"] + 7)
    with torch.no_grad():
        zs = []
        for s in range(0, len(train), 4096):
            rb = train.index(torch.arange(s, min(s + 4096, len(train)), device=dev))
            for _ in range(4 if cfg["normal_noise_rad"] > 0 else 1):
                zs.append(psi(rb.c_pos, jitter_normals(rb.c_normal, cfg["normal_noise_rad"], gen), rb.c_valid))
        zs = torch.cat(zs)
        model.set_z_stats(zs.mean(0), zs.std(0) if len(zs) > 1 else torch.ones_like(zs[0]))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    W, Tt = cfg["warmup_steps"], cfg["total_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / W) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / Tt))))
    step, best = 0, None
    if a.eval_only:
        ck = torch.load(a.eval_only, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        res = evaluate(model, psi, bank, val, val_sg, cfg, families)
        for k, v in extra.items():
            res |= {f"{k}:{m}": x for m, x in evaluate(model, psi, bank, v, extra_sg[k], cfg, families).items()}
        print(json.dumps(res, indent=1))
        return
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step, best = ck["step"], ck.get("best")
        print(f"resumed from step {step}")
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    data_files = sorted({s["npz"] for s in cfg["train"] + cfg.get("val", []) + sum(cfg.get("extra_eval", {}).values(), [])})
    prov = {"n_trainable_params": n_params, "psi_ckpt_sha16": psi.ckpt_sha, "model_repo": git_state(Path(__file__).parents[3]),
            "kite_repo": git_state(cfg.get("kite_root") or ""), "data_sha16": {f: sha16(f) for f in data_files},
            "robots": {d.name: {"family": d.family, "fingerprint": d.fingerprint(), "source_sha16": d.source_hash,
                                "n_coord": len(d.coords)} for d in bank.descs},
            "n_train": len(train), "n_val": len(val), "device": str(dev),
            "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else None}
    (out / "provenance.json").write_text(json.dumps(prov, indent=1))
    print(f"{n_params / 1e6:.3f}M trainable params on {dev}; train {len(train)} / val {len(val)} records; families {families}")

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    sampler = BalancedSampler(train.to("cpu"), bank.family, seed=cfg["seed"])
    log = open(out / "log.jsonl", "a")
    sgen = torch.Generator().manual_seed(cfg["seed"] + 1000 + step)
    t_last, n_last = time.time(), 0
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def save(path, extra=None):
        tmp = Path(str(path) + ".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                    "best": best, "config": cfg, "model_config": mc.to_dict()} | (extra or {}), tmp)
        os.replace(tmp, path)

    model.train()
    while step < Tt and not stop["flag"]:
        idx = sampler.sample(cfg["batch_size"]).to(dev)
        rb = train.index(idx)
        if train_fixed is not None:
            sg = type(train_fixed)(**{k: getattr(train_fixed, k)[idx] for k in train_fixed.__dataclass_fields__})
        else:
            sg = draw_start(bank, rb, sgen)
        z = psi(rb.c_pos, jitter_normals(rb.c_normal, cfg["normal_noise_rad"], sgen), rb.c_valid)
        out_m = model(bank, rb.r, sg.qc1, sg.R1, sg.t1, z)
        tr_root, tr_j = target_residuals(bank, rb.r, sg.qc1, sg.R1, sg.t1, sg.qcg, sg.Rg, sg.tg, mc)
        loss, lr_, lj_ = residual_loss(out_m, tr_root, tr_j, bank.coord_valid[rb.r], fam_index[rb.r], len(families))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step(); sched.step()
        step += 1
        n_last += len(idx)
        if step % cfg["log_every"] == 0:
            dt = time.time() - t_last
            row = {"step": step, "lr": sched.get_last_lr()[0], "train/loss": loss.item(), "train/root": lr_.mean().item(),
                   "train/joint": lj_.mean().item(), "grad_norm": float(gn), "examples_per_s": n_last / dt,
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0.0}
            for f in range(len(families)):
                sel = fam_index[rb.r] == f
                if sel.any():
                    row[f"train/{families[f]}/res"] = (lr_[sel] + lj_[sel]).mean().item()
            log.write(json.dumps(row) + "\n"); log.flush()
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
            t_last, n_last = time.time(), 0
        if step % cfg["eval_every"] == 0 or step == Tt:
            res = evaluate(model, psi, bank, val, val_sg, cfg, families)
            key = (res["all/model/success"], -res["all/model/contact_mm"])
            for k, v in extra.items():
                res |= {f"{k}:{m}": x for m, x in evaluate(model, psi, bank, v, extra_sg[k], cfg, families).items()}
            row = {"step": step, "eval": res}
            log.write(json.dumps(row) + "\n"); log.flush()
            brief = {k: round(v, 3) for k, v in res.items() if k.split("/")[-1] in ("contact_mm", "success", "res_loss", "wrist_mm")
                     and "robot_macro" not in k and "/wrist_only/" not in k}
            print(json.dumps({"step": step} | brief), flush=True)
            if best is None or key > tuple(best["key"]):
                best = {"key": list(key), "step": step}
                save(out / "best.pt", {"selected_on": "val macro success, then macro contact_mm"})
        if step % cfg["ckpt_every"] == 0:
            save(out / "last.pt")
    save(out / "last.pt")
    if step >= Tt:
        (out / "DONE").write_text(json.dumps({"best": best, "wall_end": time.time()}))


if __name__ == "__main__":
    main()
