"""Train the GET decoder.

    python -m getkite.decoder.train --config configs/own_psi.json --set out_dir=runs/own_psi total_steps=50000
    python -m getkite.decoder.train --config configs/raw.json --eval-only runs/raw/best.pt

Writes <out_dir>/config.json, log.jsonl (train + eval rows), last.pt and best.pt (by val success).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path

import torch

from .config import Config
from .data import SplitData
from .losses import compute_losses, contact_metrics
from .model import build_model


def lr_at(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    p = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


@torch.no_grad()
def evaluate(model, data: SplitData, cfg, batch_size=1024) -> dict:
    """Mean metrics for the model and two reference points: no-op (predict q_init) and wrist-only (the
    model's wrist with fingers at q_init). Model vs wrist-only isolates what the finger outputs add."""
    model.eval()
    sums, n = {}, 0
    for b in data.iterate(batch_size):
        out = model(b, data.bank)
        rows = {f"model/{k}": v for k, v in contact_metrics(data.bank, b, out["q"], out["R"], out["t"], cfg).items()}
        rows.update({f"noop/{k}": v for k, v in contact_metrics(data.bank, b, b["q_init"], b["R_init"], b["t_init"], cfg,
                                                               nearest=False).items()})
        # wrist-only: the model's wrist, fingers left at q_init - shows whether finger corrections help
        rows.update({f"wrist_only/{k}": v for k, v in contact_metrics(data.bank, b, b["q_init"], out["R"], out["t"], cfg,
                                                                     nearest=False).items()})
        if "mask_logit" in out:
            mask_pred = (out["mask_logit"] > 0) == b["mask"]
            rows["model/mask_acc"] = (mask_pred | ~b["dof_valid"]).float().mean(-1)
        for k, v in rows.items():
            sums[k] = sums.get(k, 0.0) + float(v.sum())
        n += len(b["h"])
    model.train()
    return {k: v / n for k, v in sums.items()} | {"n": n}


def print_param_groups(model):
    """Trainable / total parameters per top-level module (shows what kite_freeze left trainable)."""
    rows = {}
    for n, p in model.named_parameters():
        k = n.split(".")[0]
        tr, tot = rows.get(k, (0, 0))
        rows[k] = (tr + (p.numel() if p.requires_grad else 0), tot + p.numel())
    for k, (tr, tot) in rows.items():
        print(f"  {k:16s} {tr / 1e6:8.3f}M trainable / {tot / 1e6:8.3f}M")


KITE_PRETRAINED = ("unpacker.", "hand_proj.", "scene.", "action.", "intent.unpacker.")


def param_groups(model, cfg):
    """AdamW groups: weight decay only on matrices; KITE's pretrained shared modules get lr * kite_trunk_lr_mult."""
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        mult = cfg.kite_trunk_lr_mult if n.startswith(KITE_PRETRAINED) else 1.0
        wd = cfg.weight_decay if (p.ndim >= 2 and "bias" not in n.split(".")[-1]) else 0.0
        groups.setdefault((mult, wd), []).append(p)
    return [{"params": ps, "lr_mult": m, "weight_decay": wd} for (m, wd), ps in groups.items()]


def save(path, model, opt, step, cfg, best):
    """Atomic: write a temp file, then rename, so a kill mid-save never leaves a corrupt checkpoint."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "config": cfg.to_dict(),
                "best": best}, tmp)
    os.replace(tmp, path)


def truncate_log(path: Path, step: int):
    """On resume, drop log rows written after the checkpoint (they will be re-done)."""
    if not path.exists():
        return
    import json as _json
    keep = [ln for ln in path.read_text().splitlines() if ln.strip() and _json.loads(ln).get("step", 0) <= step]
    path.write_text("".join(ln + "\n" for ln in keep))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--resume", help="checkpoint to resume from")
    ap.add_argument("--eval-only", metavar="CKPT")
    ap.add_argument("--eval-out", metavar="JSON", help="with --eval-only: also write the results here (atomically)")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config, args.set)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg).to(dev)
    print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M trainable params on {dev}")
    print_param_groups(model)
    evals = {s: SplitData(cfg, s, dev, cfg.eval_max_examples) for s in cfg.eval_splits}

    if args.eval_only:
        ck = torch.load(args.eval_only, map_location=dev)
        model.load_state_dict(ck["model"])
        res = {s: evaluate(model, d, cfg) for s, d in evals.items()}
        res["_checkpoint"] = {"path": args.eval_only, "step": ck.get("step"), "selected_on_val_success": ck.get("best")}
        print(json.dumps(res, indent=1))
        if args.eval_out:
            tmp = Path(args.eval_out + ".tmp")
            tmp.write_text(json.dumps(res, indent=1))
            os.replace(tmp, args.eval_out)
        return res

    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=1))
    train = SplitData(cfg, cfg.train_split, dev, cfg.train_max_examples, part=cfg.train_part)
    print(f"train: {train.n:,} examples ({train.part}), {len(train.bank.hand_ids)} hands")
    if hasattr(model, "intent_inputs") and hasattr(model.intent, "fit_z_stats"):
        g = torch.Generator().manual_seed(cfg.seed)
        model.intent.fit_z_stats([model.intent_inputs(train.batch(
            torch.randint(0, train.n, (min(2048, train.n),), generator=g).to(dev))) for _ in range(8)])
        print(f"z stats: mean |mu| {model.intent.z_mean.abs().mean():.3g}, mean sd {model.intent.z_std.mean():.3g}")
    opt = torch.optim.AdamW(param_groups(model, cfg), lr=cfg.lr, betas=(0.9, 0.98))
    step, best = 0, -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck["best"]
        truncate_log(out / "log.jsonl", step)
        print(f"resumed at step {step} (best val success {best:.4g})", flush=True)
    if step >= cfg.total_steps:
        (out / "DONE").write_text(f"{step}\n")
        print("already finished")
        return
    stop = {"sig": None}  # preemption / time limit: Slurm sends SIGTERM, then SIGKILL ~30 s later

    def on_signal(signum, _frame):
        stop["sig"] = signum
    for sg in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sg, on_signal)
    if cfg.wandb:
        import wandb
        wandb.init(project="get-ur-kite", config=cfg.to_dict(), dir=str(out))

    gen = torch.Generator().manual_seed(cfg.seed + step)
    log = open(out / "log.jsonl", "a")
    t0, acc = time.time(), {}
    while step < cfg.total_steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg) * g["lr_mult"]
        b = train.sample(cfg.batch_size, gen, cfg.normal_noise_rad)
        losses = compute_losses(model(b, train.bank), b, train.bank, cfg)
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        step += 1
        for k, v in losses.items():
            acc[k] = acc.get(k, 0.0) + float(v.detach())
        acc["grad_norm"] = acc.get("grad_norm", 0.0) + float(gn)
        if step % cfg.log_every == 0:
            row = {"step": step, "lr": lr_at(step, cfg), "sec": round(time.time() - t0, 1)} | \
                  {f"train/{k}": v / cfg.log_every for k, v in acc.items()}
            acc = {}
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
            log.write(json.dumps(row) + "\n")
            if cfg.wandb:
                wandb.log(row, step=step)
        if step % cfg.eval_every == 0 or step == cfg.total_steps:
            row = {"step": step}
            for s, d in evals.items():
                row |= {f"{s}/{k}": v for k, v in evaluate(model, d, cfg).items()}
            log.write(json.dumps(row) + "\n")
            log.flush()
            if cfg.wandb:
                wandb.log(row, step=step)
            keys = ("model/contact_mm", "model/success", "wrist_only/success", "noop/success")
            print(" | ".join(f"{s}: " + " ".join(f"{k.split('/')[0][0]}.{k.split('/')[1]}={row[f'{s}/{k}']:.3g}"
                                                   for k in keys) for s in evals), flush=True)
            score = row.get("val/model/success", -1.0)
            if score > best:
                best = score
                save(out / "best.pt", model, opt, step, cfg, best)
            save(out / "last.pt", model, opt, step, cfg, best)  # after best is updated, so a resume keeps it
        elif step % cfg.ckpt_every == 0:
            save(out / "last.pt", model, opt, step, cfg, best)
        if stop["sig"] is not None:
            save(out / "last.pt", model, opt, step, cfg, best)
            log.close()
            print(f"signal {stop['sig']}: saved last.pt at step {step}, exiting for requeue", flush=True)
            raise SystemExit(0)
    log.close()
    (out / "DONE").write_text(f"{step}\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
