"""The zero-shot KITE baselines (clamped, B1 sliced, B2 rows reassigned) on the dataset's test splits.

    python scripts/eval_kite_baselines.py CKPT --splits test_finger_subset test_joint_count \
        --modes b1 b2 clamped --per-hand 200 --out outputs/leap_head/baselines.json

Metrics per split and mode (as `scripts/eval_kite_head.py`): contact error (mm), normal error (deg),
success (all contacts within 10 mm and 15 deg), and the "no movement" floor. Hands where a mode is
undefined (clamped on a hand without some finger's links) are counted as skipped. Per-hand rows go to
<out>.hands.jsonl.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from getkite.kite_baselines import VariantRunner, kite_template
from getkite.kite_head import KiteHead
from getkite.leap.data import LeapSplit, split_hand_ids

import importlib.util

_spec = importlib.util.spec_from_file_location("eval_kite_head", Path(__file__).with_name("eval_kite_head.py"))
_ekh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ekh)
contact_metrics = _ekh.contact_metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--splits", nargs="+", default=["test_finger_subset", "test_joint_count", "test_geometry", "test_combined"])
    ap.add_argument("--modes", nargs="+", default=["b1", "b2", "clamped"])
    ap.add_argument("--per-hand", type=int, default=200, help="examples per hand")
    ap.add_argument("--max-hands", type=int, default=0, help="first N hands of each split (0 = all)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    b0 = KiteHead.load(args.checkpoint, device=args.device)
    template = kite_template()
    results, hand_rows = {}, []
    for split_name in args.splits:
        hands = split_hand_ids(split_name)
        if args.max_hands:
            hands = hands[: args.max_hands]
        t0 = time.time()
        sums = {}
        for hid in hands:
            split = LeapSplit.load(split_name, device=args.device, hands=[hid], full_template=True,
                                   max_examples=args.per_hand, seed=0)
            batch = next(split.batches(args.per_hand, shuffle=False, dtype=torch.float64))
            n = batch["hand"].shape[0]
            rows = {"no_movement": contact_metrics(split, batch, batch["wrist_init"].double(), batch["joints_init"].double())}
            for mode in args.modes:
                try:
                    runner = VariantRunner(b0, hid, mode, template=template)
                except ValueError:
                    sums[(mode, "skipped_hands")] = sums.get((mode, "skipped_hands"), 0) + 1
                    continue
                wrist, joints = runner.predict(batch, split.bank)
                rows[mode] = contact_metrics(split, batch, wrist, joints)
            for who, m in rows.items():
                rec = {"split": split_name, "hand": hid, "mode": who, "examples": n}
                for k, v in m.items():
                    sums[(who, k)] = sums.get((who, k), 0.0) + float(v.sum())
                    rec[k] = float(v.mean())
                sums[(who, "examples")] = sums.get((who, "examples"), 0) + n
                hand_rows.append(rec)
        res = {"hands": len(hands), "seconds": round(time.time() - t0, 1)}
        for (who, k), v in sums.items():
            if k in ("examples", "skipped_hands"):
                res.setdefault(who, {})[k] = v
        for (who, k), v in sums.items():
            if k not in ("examples", "skipped_hands"):
                res[who][k] = v / max(1, sums.get((who, "examples"), 1))
        results[split_name] = res
        line = "  ".join(f"{who}: {m['contact_mm']:.2f} mm {100 * m['success']:.1f}%"
                         for who, m in res.items() if isinstance(m, dict) and "contact_mm" in m)
        print(f"{split_name} ({len(hands)} hands, {res['seconds']} s): {line}", flush=True)
    out = {"checkpoint": str(args.checkpoint), "per_hand": args.per_hand, "results": results}
    print(json.dumps(out, indent=1))
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1))
        with open(p.with_suffix(".hands.jsonl"), "w") as f:
            for r in hand_rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
