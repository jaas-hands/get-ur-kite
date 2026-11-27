"""Run a LEAP-family decoder checkpoint on a kite_data folder and save its predictions with our metrics.

    python scripts/decoder/predict.py --config configs/decoder/kite_trunk_get_ft.json \
        --ckpt $GETKITE_RUNS/kite_trunk_get_ft/best.pt --data leap_sample_v1 --out predictions/

--data is a folder laid out like get-ur-kite-dataset's data/ (hands/, examples/<regime>/<split>/<hand_id>.npz), e.g.
the leap_sample_v1 sample. Every example of every split found there is predicted (no holdout filtering), using the
start pose stored in the file, exactly as in our evaluation. Writes, per input file,
<out>/<split>/<hand_id>.npz with

  joints_pred      (n, n_dof)  predicted finger joints (rad), canonical order, as output (NOT clamped)
  wrist_pos_pred   (n, 3)      predicted wrist position, world frame (m)
  wrist_quat_pred  (n, 4)      predicted wrist orientation, w, x, y, z with w >= 0
  model_* / noop_* (n,)        our per-example metrics for the prediction and for no movement (the start pose):
                               contact_mm, normal_deg, success, wrist_mm, wrist_deg, limit_violation
                               (joints are clamped to their limits before FK, as in our evaluation)

plus <out>/summary.json (means per split, checkpoint, config and data provenance).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from getkite.decoder.config import Config
from getkite.decoder.data import SplitData, _quat_from_matrix
from getkite.decoder.losses import contact_metrics
from getkite.decoder.model import build_model

METRICS = ("contact_mm", "normal_deg", "success", "wrist_mm", "wrist_deg", "limit_violation")


def sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="kite_data-style folder (hands/, examples/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    data = Path(a.data).resolve()
    cfg = Config.load(a.config, [f"data_dir={data}", *a.set])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(dev)
    ck = torch.load(a.ckpt, map_location=dev)
    model.load_state_dict(ck["model"])
    model.eval()

    out = Path(a.out)
    summary = {"checkpoint": {"path": str(a.ckpt), "step": ck.get("step"), "sha256_16": sha16(a.ckpt)},
               "config": a.config, "data": str(data), "frame": "world", "splits": {}}
    for sdir in sorted((data / "examples" / cfg.regime).iterdir()):
        if not sdir.is_dir() or not any(sdir.glob("*.npz")):
            continue
        split = sdir.name
        d = SplitData(cfg, split, dev, part="all")
        b = next(d.iterate(d.n))
        o = model(b, d.bank)
        m = contact_metrics(d.bank, b, o["q"], o["R"], o["t"], cfg, nearest=False)
        n0 = contact_metrics(d.bank, b, b["q_init"], b["R_init"], b["t_init"], cfg, nearest=False)
        h = b["h"].cpu().numpy()
        quat = _quat_from_matrix(o["R"]).cpu().numpy()
        for hi, hid in enumerate(d.bank.hand_ids):
            rows = np.flatnonzero(h == hi)
            nd = int(d.t["dof_valid"][rows[0]].sum())
            rec = {"hand_id": hid, "split": split,
                   "joints_pred": o["q"][rows, :nd].cpu().numpy().astype(np.float32),
                   "wrist_pos_pred": o["t"][rows].cpu().numpy().astype(np.float32),
                   "wrist_quat_pred": quat[rows].astype(np.float32)}
            for k in METRICS:
                rec[f"model_{k}"] = m[k][rows].cpu().numpy().astype(np.float32)
                rec[f"noop_{k}"] = n0[k][rows].cpu().numpy().astype(np.float32)
            p = out / split / f"{hid}.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(p, **rec)
        summary["splits"][split] = {"n": d.n, **{f"model_{k}": float(m[k].mean()) for k in METRICS},
                                    **{f"noop_{k}": float(n0[k].mean()) for k in METRICS}}
        print(split, d.n, f"model {summary['splits'][split]['model_contact_mm']:.2f} mm /"
              f" {100 * summary['splits'][split]['model_success']:.1f}%",
              f"no movement {summary['splits'][split]['noop_contact_mm']:.2f} mm")
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
