"""Baseline B3: fine-tune the KITE head's per-hand parameters on one test hand; error vs. time and data.

Starts from the B1 head of the hand (B0 with that hand's joint rows, `getkite.kite_baselines`), trains
only the per-hand parameters (the shared trunk and ψ stay frozen) with KITE's own loss
(`compute_loss`: masked L2 on the normalized residual, the released `full_mix` setting) on the first
`--train` examples of the hand, and scores held-out examples of the same hand (the last `--eval`)
after 0, 15, 30 s, 1, 2, 5, 10, 20, 40 min of training.

With `--data DIR` (from `scripts/gen_kite_head_data.py`) the protocol is fair: training examples are the
first `--train` of DIR/train/<hand>.npz, the checkpoint is chosen on DIR/val/<hand>.npz (`--val`), and the
reported number is the hand's official test examples (`--eval` of them) at that checkpoint.

    python scripts/finetune_kite_head.py CKPT I4-M0-R4-T4_g0 --split test_finger_subset --train 1000
    python scripts/finetune_kite_head.py CKPT HAND --train 250 --minutes 5 --out outputs/leap_head/b3/HAND_250.json
    python scripts/finetune_kite_head.py CKPT HAND --split test_geometry --data $GETKITE_DATA/leap_head/b3 \
        --train 4000 --val 500 --eval 2000 --minutes 40 --lr 1e-5 --out RUN/b3/HAND_4000.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from getkite.kite_baselines import VariantRunner, kite_template, tokens_world
from getkite.kite_head import N_WRIST, KiteHead, to_kite_q
from getkite.leap.data import LeapSplit

import importlib.util

_spec = importlib.util.spec_from_file_location("eval_kite_head", Path(__file__).with_name("eval_kite_head.py"))
_ekh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ekh)

SHARED = ("hand_proj", "contact_proj", "scene_encoder", "contact_only_encoder", "contact_slot_encoder",
          "cls_unpacker", "slot_to_scene_proj", "bottleneck_scene_encoder", "non_contact_context_encoder",
          "time_mlp", "action_decoder")
CHECKPOINTS_S = (0, 15, 30, 60, 120, 300, 600, 1200, 2400)


def kite_batch(runner: VariantRunner, batch: dict, bank, idx: torch.Tensor, all_joints: bool = True) -> dict:
    """KITE's training batch for examples `idx` of a single-hand LeapSplit batch.

    `all_joints` supervises every joint (the label keeps the joints it does not need at their start
    value), not only the wrist and the joints upstream of a contact. KITE's masked loss leaves the other
    joints unsupervised, and they drift: trained that way on the dataset, the head moved them by
    0.016 rad and its contact error got worse while its loss fell (Nov 24).
    """
    dev = runner.b0.device
    T_palm = runner.b0.T_palm.to(dev)
    sub = {k: (v[idx] if torch.is_tensor(v) and v.shape[:1] == batch["hand"].shape else v) for k, v in batch.items()}
    n = runner.n_dof
    B = idx.shape[0]
    zeros = torch.zeros(B, 16, dtype=torch.float64, device=dev)
    w0 = to_kite_q(sub["wrist_init"].double().to(dev), zeros, T_palm)[:, :N_WRIST]
    w1 = to_kite_q(sub["wrist_goal"].double().to(dev), zeros, T_palm)[:, :N_WRIST]
    q0 = torch.cat([w0, sub["joints_init"].double().to(dev)[:, :n]], 1).float()
    q1 = torch.cat([w1, sub["joints_goal"].double().to(dev)[:, :n]], 1).float()
    # Euler angles of the goal on the same branch as the start (avoid 2*pi jumps in the residual)
    q1[:, 3:6] = q0[:, 3:6] + torch.remainder(q1[:, 3:6] - q0[:, 3:6] + torch.pi, 2 * torch.pi) - torch.pi
    T = bank.fk(sub["hand"], sub["joints_init"].double(), sub["wrist_init"].double())
    pts, nrm = tokens_world(runner.tokens, T.to(dev), runner.links)
    joints = torch.ones_like(sub["mask"][:, :n]).bool() if all_joints else sub["mask"][:, :n].bool()
    mask = torch.cat([torch.ones(B, N_WRIST, dtype=torch.bool, device=dev), joints.to(dev)], 1)
    valid = sub["contact_valid"].to(dev).bool()
    cloud = torch.cat([sub["contact_pos"], sub["contact_normal"]], -1).float().to(dev) * valid[..., None]
    return {
        "contact_cloud": cloud, "contact_valid_mask": valid,
        "hand_points_world": pts.float(), "hand_normals_world": nrm.float(),
        "q1_padded": q0, "action": q1, "action_valid_mask": mask, "contact_joint_mask": mask,
        "robot_index": torch.zeros(B, dtype=torch.long, device=dev),
        "local_dof": torch.full((B,), N_WRIST + n, device=dev),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint")
    ap.add_argument("hand")
    ap.add_argument("--split", default="test_finger_subset")
    ap.add_argument("--mode", default="b1", choices=["b1", "b2", "clamped", "b0"], help="head to start from")
    ap.add_argument("--train", type=int, default=1000, help="training examples of the hand")
    ap.add_argument("--eval", type=int, default=500, help="held-out (test) examples of the hand")
    ap.add_argument("--data", type=Path, default=None, help="train/ and val/ <hand>.npz to train and select on")
    ap.add_argument("--val", type=int, default=500, help="with --data: validation examples")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    b0 = KiteHead.load(args.checkpoint, device=args.device)
    runner = VariantRunner(b0, args.hand, args.mode, template=kite_template())
    def first(split, n):
        b = next(split.batches(n, shuffle=False, dtype=torch.float64))
        if b["hand"].shape[0] < n:
            raise SystemExit(f"{args.hand}: only {b['hand'].shape[0]} examples in {split.name}, {n} needed")
        return b

    load = dict(device=args.device, hands=[args.hand], full_template=True)
    if args.data is None:  # Nov 24 laptop protocol: train and held-out examples from the test split
        split = LeapSplit.load(args.split, max_examples=args.train + args.eval, seed=0, **load)
        batch = first(split, args.train + args.eval)
        N = batch["hand"].shape[0]
        train_idx = torch.arange(args.train, device=batch["hand"].device)
        eval_idx = torch.arange(N - args.eval, N, device=batch["hand"].device)
        eval_batch = {k: (v[eval_idx] if torch.is_tensor(v) and v.shape[:1] == batch["hand"].shape else v)
                      for k, v in batch.items()}
        val_split = val_batch = None
    else:
        split = LeapSplit.load("train", root=args.data / "train", **load)
        batch = first(split, args.train)
        train_idx = torch.arange(args.train, device=batch["hand"].device)
        val_split = LeapSplit.load("val", root=args.data / "val", **load)
        val_batch = first(val_split, args.val)
        test_split = LeapSplit.load(args.split, **load)
        eval_batch = first(test_split, args.eval)

    head = runner.head.head
    for name, p in head.named_parameters():
        p.requires_grad_(name.split(".")[0] not in SHARED)
    params = [p for p in head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-6)
    full = kite_batch(runner, batch, split.bank, train_idx)
    g = torch.Generator(device="cpu").manual_seed(0)

    eval_split = split if args.data is None else test_split

    def score(sp, b):
        w, j = runner.predict(b, sp.bank)
        return {k: float(v.mean()) for k, v in _ekh.contact_metrics(sp, b, w, j).items()}

    def evaluate():
        head.eval()
        m = score(eval_split, eval_batch)
        if val_batch is not None:
            m.update({f"val_{k}": v for k, v in score(val_split, val_batch).items()})
        head.train()
        return m

    floor = _ekh.contact_metrics(eval_split, eval_batch, eval_batch["wrist_init"].double(), eval_batch["joints_init"].double())
    curve = [{"seconds": 0.0, "steps": 0, **evaluate()}]
    print(f"{args.hand} {args.mode} train={args.train}: start {curve[0]['contact_mm']:.2f} mm "
          f"{100 * curve[0]['success']:.1f}%  (no movement {float(floor['contact_mm'].mean()):.2f} mm)", flush=True)
    head.train()
    t_train, steps, next_ck = 0.0, 0, 1
    while t_train < args.minutes * 60 and next_ck < len(CHECKPOINTS_S):
        t0 = time.time()
        sel = torch.randint(0, args.train, (args.batch_size,), generator=g).to(full["q1_padded"].device)
        b = {k: (v[sel] if torch.is_tensor(v) and v.shape[0] == args.train else v) for k, v in full.items()}
        loss, _ = head.compute_loss(b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if b0.device.type == "cuda":
            torch.cuda.synchronize()
        t_train += time.time() - t0
        steps += 1
        if t_train >= CHECKPOINTS_S[next_ck]:
            curve.append({"seconds": round(t_train, 1), "steps": steps, "loss": float(loss.detach()), **evaluate()})
            c = curve[-1]
            v = f"  (val {c['val_contact_mm']:.2f} mm)" if "val_contact_mm" in c else ""
            print(f"  {c['seconds']:7.1f} s  {steps:6d} steps  {c['contact_mm']:.2f} mm  {100 * c['success']:.1f}%{v}", flush=True)
            next_ck += 1
    out = {"checkpoint": str(args.checkpoint), "hand": args.hand, "split": args.split, "mode": args.mode,
           "train_examples": args.train, "eval_examples": args.eval,
           "lr": args.lr, "data": str(args.data) if args.data else None,
           "no_movement": {k: float(v.mean()) for k, v in floor.items()}, "curve": curve}
    if val_batch is not None:  # the test result at the checkpoint with the best validation error
        out["val_examples"] = args.val
        out["selected"] = min(curve, key=lambda c: c["val_contact_mm"])
        c = out["selected"]
        print(f"{args.hand} train={args.train}: selected at {c['seconds']} s on val: test {c['contact_mm']:.2f} mm "
              f"{100 * c['success']:.1f}%", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
