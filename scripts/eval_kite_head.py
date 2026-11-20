"""Open-loop metrics of a KITE decoder head on the team's LEAP data (gate G4, the evaluation matrix).

    python scripts/eval_kite_head.py outputs/leap_head/b0/last.pt                       # nominal hand
    python scripts/eval_kite_head.py CKPT --split val --hands I4-M4-R4-T4_g3 --dataset-tokens
    python scripts/eval_kite_head.py CKPT --max-examples 2000 --out outputs/leap_head/eval.json

Metrics (masked contacts and joints as in the dataset):
- contact error: distance between each target contact and the same template point on the predicted
  hand (mm); success: every contact within 10 mm and 15 deg (as docs/decoder.md);
- KITE's own test metrics, computed the way KITE's trainer does: `action_mse_valid` (squared error of the
  KITE configuration over the wrist and the masked joints) and `contact_point_mse` (per coordinate, m^2);
- "no movement": the same for the start pose, as a floor.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from getkite.kite_head import KiteHead, N_WRIST, to_kite_q
from getkite.leap.data import LeapSplit

NOMINAL = "I4-M4-R4-T4_g0"


def contact_metrics(split: LeapSplit, batch: dict, wrist: torch.Tensor, joints: torch.Tensor,
                    tol_mm: float = 10.0, tol_deg: float = 15.0) -> dict:
    bank, h = split.bank, batch["hand"]
    T = bank.fk(h, joints, wrist)
    idx = batch["template_idx"].clamp_min(0)
    pos, nrm = bank.template_points(h, T, idx)
    valid = batch["contact_valid"]
    d = (pos - batch["contact_pos"].to(pos)).norm(dim=-1) * 1000.0
    cos = (nrm * batch["contact_normal"].to(nrm)).sum(-1).clamp(-1, 1)
    ang = torch.rad2deg(torch.acos(cos))
    ok = ((d <= tol_mm) & (ang <= tol_deg)) | ~valid
    n = valid.sum(1).clamp_min(1)
    sq = ((pos - batch["contact_pos"].to(pos)) ** 2 * valid[..., None]).sum((1, 2)) / (3 * n)
    return {
        "contact_mm": ((d * valid).sum(1) / n),
        "normal_deg": ((ang * valid).sum(1) / n),
        "success": ok.all(1).float(),
        "contact_point_mse": sq,
    }


def action_mse_valid(q_pred: torch.Tensor, q_goal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """KITE's metric: squared error over the wrist dims and the masked joints (wrist angles wrapped)."""
    err = q_pred - q_goal
    err[:, 3:6] = torch.remainder(err[:, 3:6] + math.pi, 2 * math.pi) - math.pi
    m = torch.cat([torch.ones_like(mask[:, :N_WRIST]), mask[:, : q_pred.shape[1] - N_WRIST]], 1).to(err)
    return ((err ** 2) * m).sum(1) / m.sum(1).clamp_min(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--split", default="train", help="dataset split (the nominal hand is in train)")
    ap.add_argument("--hands", nargs="*", default=[NOMINAL])
    ap.add_argument("--max-examples", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--dataset-tokens", action="store_true", help="hand tokens from the dataset, not KITE's template")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    head = KiteHead.load(args.checkpoint, device=args.device)
    split = LeapSplit.load(args.split, device=args.device, hands=args.hands, full_template=True,
                           max_examples=args.max_examples, seed=0)
    sums, n = {}, 0
    z_gap = 0.0
    for batch in split.batches(args.batch_size, shuffle=False, dtype=torch.float64):
        wrist, joints = head.predict(batch, dataset_hand_tokens=args.dataset_tokens)
        T_palm = head.T_palm.to(wrist)
        q_goal = to_kite_q(batch["wrist_goal"].double(), batch["joints_goal"].double(), T_palm)
        q_pred = to_kite_q(wrist, joints, T_palm)
        q_init = to_kite_q(batch["wrist_init"].double(), batch["joints_init"].double(), T_palm)
        rows = {
            "pred": {**contact_metrics(split, batch, wrist, joints),
                     "action_mse_valid": action_mse_valid(q_pred, q_goal, batch["mask"])},
            "no_movement": {**contact_metrics(split, batch, batch["wrist_init"].double(), batch["joints_init"].double()),
                            "action_mse_valid": action_mse_valid(q_init, q_goal, batch["mask"])},
            "label": contact_metrics(split, batch, batch["wrist_goal"].double(), batch["joints_goal"].double()),
        }
        for who, metrics in rows.items():
            for k, v in metrics.items():
                sums[(who, k)] = sums.get((who, k), 0.0) + float(v.sum())
        n += batch["hand"].shape[0]
    res = {"checkpoint": str(args.checkpoint), "split": args.split, "hands": args.hands, "examples": n,
           "hand_tokens": "dataset" if args.dataset_tokens else "kite_template"}
    for (who, k), v in sums.items():
        res.setdefault(who, {})[k] = v / n
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
