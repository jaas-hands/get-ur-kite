"""Train a KITE LEAP head on the team dataset's examples (B0-ds), saved in KITE's checkpoint format.

Gate G4 showed that the head trained on KITE's own LEAP data transfers poorly to the team dataset (on the
nominal hand: 5.2x better than no movement on KITE's test set, 1.4x on the dataset's). B0-ds continues
from such a checkpoint on examples generated with the dataset's recipe, so B0-B3 and the GET decoder are
trained and tested on the same kind of data. KITE's own loss (`compute_loss`, the released `full_mix`
setting, but on all joints: see `finetune_kite_head.kite_batch`), KITE's contact-normal noise (0.1 rad), AdamW 1e-4 with warm-up and cosine decay, gradient
clipping 1.0. Trainable: the per-hand parameters, plus the trunk with `--trunk` (ψ always frozen).

    python scripts/train_kite_head_dataset.py outputs/leap_head/local/b0-trunk/last.pt \
        --data outputs/leap_head/ds_nominal --out outputs/leap_head/local/b0-ds --epochs 20 --trunk

B4 (a test hand retrained, the upper bound) is the same recipe on one test hand: `--hand H --mode b1`
starts from B0-ds's B1 head for that hand (B0-ds's weights, the hand's joint rows), trains on the hand's
generated examples, picks the epoch on their val/, and scores it on the hand's official examples of
`--test-split`. Its checkpoints are the head's state dict (`best_head.pt`, `last_head.pt`), not a full KITE checkpoint.

    python scripts/train_kite_head_dataset.py B0DS/best.pt --hand I0-M2-R0-T2_g7 --mode b1 --trunk \
        --data $GETKITE_DATA/leap_head/b4 --test-split test_geometry --out RUN/b4/I0-M2-R0-T2_g7
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import torch

from getkite.kite_baselines import VariantRunner, kite_template
from getkite.kite_head import KiteHead
from getkite.leap.data import LeapSplit
from getkite.leap.intent import rotate_normals

import importlib.util


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ft = _load("finetune_kite_head")
_ekh = _load("eval_kite_head")
HAND = "I4-M4-R4-T4_g0"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", help="KITE checkpoint to start from (its dataset_meta.pt must sit next to it)")
    ap.add_argument("--data", type=Path, required=True, help="folder with train/ and val/ <hand>.npz")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--normal-noise", type=float, default=0.1)
    ap.add_argument("--trunk", action="store_true", help="also train the shared trunk (not ψ)")
    ap.add_argument("--masked-joints-only", action="store_true",
                    help="KITE's mask: supervise only the wrist and the joints upstream of a contact")
    ap.add_argument("--hand", default=HAND)
    ap.add_argument("--mode", default="b0", choices=["b0", "b1", "b2"], help="head to start from (b1/b2 for a variant)")
    ap.add_argument("--test-split", default=None, help="score the best epoch on this split's examples of the hand")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = torch.device(args.device)
    b0 = KiteHead.load(args.checkpoint, device=dev)
    runner = VariantRunner(b0, args.hand, args.mode, template=kite_template())
    train = LeapSplit.load("train", device=dev, hands=[args.hand], root=args.data / "train", full_template=True)
    val = LeapSplit.load("val", device=dev, hands=[args.hand], root=args.data / "val", full_template=True)
    vbatch = val.batch(torch.arange(len(val)), dtype=torch.float64)
    n = len(train)
    print(f"[B0-ds] {args.hand} ({args.mode}): {n} train / {len(val)} val examples from {args.data}; "
          f"trunk {'trained' if args.trunk else 'frozen'}", flush=True)

    head = runner.head.head
    frozen = set(_ft.SHARED) if not args.trunk else {"contact_only_encoder"}
    for name, p in head.named_parameters():
        p.requires_grad_(name.split(".")[0] not in frozen)
    params = [p for p in head.parameters() if p.requires_grad]
    print(f"[B0-ds] trainable {sum(p.numel() for p in params):,} of {sum(p.numel() for p in head.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-6)
    steps_per_epoch = n // args.batch_size
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    g = torch.Generator(device="cpu").manual_seed(0)

    def evaluate():
        head.eval()
        w, j = runner.predict(vbatch, val.bank)
        m = _ekh.contact_metrics(val, vbatch, w, j)
        head.train()
        return {k: float(v.mean()) for k, v in m.items()}

    args.out.mkdir(parents=True, exist_ok=True)
    floor = _ekh.contact_metrics(val, vbatch, vbatch["wrist_init"].double(), vbatch["joints_init"].double())
    floor = {k: float(v.mean()) for k, v in floor.items()}
    hist = [{"epoch": 0, **evaluate()}]
    print(f"[B0-ds] no movement {floor['contact_mm']:.2f} mm {100 * floor['success']:.1f}% | "
          f"epoch 0: {hist[0]['contact_mm']:.2f} mm {100 * hist[0]['success']:.1f}%", flush=True)
    best = hist[0]["contact_mm"]
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    shutil.copy2(Path(args.checkpoint).with_name("dataset_meta.pt"), args.out / "dataset_meta.pt")

    def save(path):
        if args.mode != "b0":  # a variant head: its own shapes, no KITE bank to put it in
            torch.save({"head": {k: v.detach().cpu() for k, v in head.state_dict().items()}, "hand": args.hand,
                        "mode": args.mode, "rows": runner.rows, "history": hist}, path.with_name(f"{path.stem}_head.pt"))
            return
        state = {k: v.detach().cpu() for k, v in b0.bank.state_dict().items()}
        out = dict(ckpt)
        out.update(model=state, ema_model=state, b0_ds={"data": str(args.data), "history": hist, "trunk": args.trunk})
        for k in ("optimizer", "lr_scheduler", "ema_state"):
            out.pop(k, None)
        torch.save(out, path)

    save(args.out / "best.pt")  # epoch 0, in case no epoch improves on it
    head.train()
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n, generator=g)
        run_loss = 0.0
        for s in range(steps_per_epoch):
            idx = perm[s * args.batch_size:(s + 1) * args.batch_size]
            raw = train.batch(idx, dtype=torch.float64)
            if args.normal_noise > 0:
                raw["contact_normal"] = rotate_normals(raw["contact_normal"], args.normal_noise)
            kb = _ft.kite_batch(runner, raw, train.bank, torch.arange(len(idx), device=raw["hand"].device),
                                all_joints=not args.masked_joints_only)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss, _ = head.compute_loss(kb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            run_loss += float(loss.detach())
        m = evaluate()
        hist.append({"epoch": epoch, "train_loss": run_loss / steps_per_epoch, "minutes": round((time.time() - t0) / 60, 1), **m})
        print(f"[B0-ds] epoch {epoch:3d}  loss {hist[-1]['train_loss']:.4f}  val {m['contact_mm']:.2f} mm "
              f"{100 * m['success']:.1f}%  ({hist[-1]['minutes']} min)", flush=True)
        save(args.out / "last.pt")
        if m["contact_mm"] < best:
            best = m["contact_mm"]
            save(args.out / "best.pt")
        (args.out / "history.json").write_text(json.dumps({"no_movement": floor, "history": hist}, indent=1))

    if args.test_split:  # the best epoch on the hand's official test examples
        if args.mode == "b0":
            best_ck = torch.load(args.out / "best.pt", map_location="cpu", weights_only=False)["model"]
            b0.bank.load_state_dict(best_ck)
        else:
            head.load_state_dict(torch.load(args.out / "best_head.pt", map_location="cpu", weights_only=False)["head"])
        test = LeapSplit.load(args.test_split, device=dev, hands=[args.hand], full_template=True)
        tbatch = test.batch(torch.arange(len(test)), dtype=torch.float64)
        head.eval()
        w, j = runner.predict(tbatch, test.bank)
        res = {k: float(v.mean()) for k, v in _ekh.contact_metrics(test, tbatch, w, j).items()}
        nm = {k: float(v.mean()) for k, v in _ekh.contact_metrics(
            test, tbatch, tbatch["wrist_init"].double(), tbatch["joints_init"].double()).items()}
        best_epoch = min(hist, key=lambda h: h["contact_mm"])["epoch"]
        out = {"hand": args.hand, "split": args.test_split, "examples": len(test), "best_epoch": best_epoch,
               "test": res, "no_movement": nm}
        (args.out / "test.json").write_text(json.dumps(out, indent=1))
        print(f"[B0-ds] test {args.test_split} {args.hand} (epoch {best_epoch}): {res['contact_mm']:.2f} mm "
              f"{100 * res['success']:.1f}%  (no movement {nm['contact_mm']:.2f} mm)", flush=True)


if __name__ == "__main__":
    main()
