"""Run a J1 decoder checkpoint (e.g. frozen CF-3e) on a kite_data folder and save its predictions with our metrics.

J1 = frozen KITE psi (contacts -> z) + contact predictor + GET decoder (getkite.decoder.joint_contact_get). Canonical-frame (CF)
checkpoints are wrapped exactly as in our benchmarks (scripts/decoder/benchmark/transfer_benchmark.load_system), so predictions
come back in each hand's NATIVE base frame and the dataset's joint order. Only deployment-time inputs reach the model: the hand's
description files, the requested contacts (positions, normals) and the start pose; ground-truth labels are used only for scoring.

Data mode (default): --data is a folder laid out like get-ur-kite-dataset's data/ (hands/<hand_id>/ or prim/hands/<id>/,
examples/<regime>/<split>/<hand_id>.npz). Every example of every split is predicted from the START POSE STORED IN THE FILE
(joints_init, wrist_pos_init, wrist_quat_init); the target is the stored contact pose. Writes, per input file,
<out>/<split>/<hand_id>.npz with

  joints_pred      (n, n_dof)  predicted finger joints (rad), dataset order, as output (NOT clamped)
  wrist_pos_pred   (n, 3)      predicted wrist position, world frame (m)
  wrist_quat_pred  (n, 4)      predicted wrist orientation, w, x, y, z with w >= 0
  model_* / noop_* (n,)        our per-example metrics for the prediction and for no movement (the start pose):
                               corr_mm, corr_deg, corr_success (10 mm / 15 deg), compat_mm, compat_success, surf_mm,
                               wrist_mm, wrist_deg (joints clamped to their limits before FK, as in our evaluation)

plus <out>/summary.json (means per split, checkpoint sha256, psi / data provenance).

Check mode (--check-config CONFIG --check-pool KEY): reproduces our benchmark exactly — the pool cfg["extra_eval"][KEY] (or
"val") with the benchmark's seeded noisy start (val_start_seed, KITE noise) — and prints the means, to compare with the
benchmark's metrics.json for the same checkpoint.

    PYTHONPATH=src python scripts/decoder/predict_j1.py --ckpt $GETKITE_RUNS/frozen_cf3e_2026-11-27/cf3e_long_hf/best.pt \
        --data $GETKITE_DATASET --regime kite_code --splits val --out predictions/cf3e_s0
    PYTHONPATH=src python scripts/decoder/predict_j1.py --ckpt .../best.pt \
        --check-config configs/decoder/scratch/c3_psi50k_10x_wrist_leapval105.json --check-pool leap_heldout
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from getkite.decoder import paths
from getkite.decoder.multifamily import StartGoal, fixed_start, load_leap_records
from getkite.decoder.rotations import quat_to_matrix
from getkite.decoder.train_scratch import FrozenPsi, build_bank, load_config, load_pool, start_noise

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "benchmark")); sys.path.insert(0, str(HERE / "diagnostics"))
from crossbody_heldout_eval import ANG_DEG, POS_MM, sg_index, sg_to  # noqa: E402
from transfer_benchmark import load_system, score  # noqa: E402

METRICS = ("corr_mm", "corr_deg", "corr_success", "compat_mm", "compat_success", "surf_mm", "wrist_mm", "wrist_deg")
SCFG = {"success_pos_mm": POS_MM, "success_normal_deg": ANG_DEG}


def sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def quat_wxyz(R):
    """rotation matrices (N, 3, 3) -> unit quaternions w, x, y, z with w >= 0."""
    from scipy.spatial.transform import Rotation as Rot
    q = Rot.from_matrix(R.double().cpu().numpy()).as_quat()[:, [3, 0, 1, 2]]
    return (q * np.where(q[:, :1] < 0, -1.0, 1.0)).astype(np.float32)


def stored_start(bank, rec, z):
    """StartGoal from the start pose stored in a kite_data file; goal = stored contact pose (joints outside the mask stay at
    the start, as in draw_start)."""
    dev = rec.t2.device
    J = rec.qc2.shape[1]
    q1 = np.zeros((len(rec), J), np.float32)
    ji = z["joints_init"]; q1[:, :ji.shape[1]] = ji
    q1 = torch.as_tensor(q1, device=dev)
    cv = bank.coord_valid[rec.r]
    q1 = torch.where(cv, q1, 0.0)
    t1 = torch.as_tensor(z["wrist_pos_init"], dtype=torch.float32, device=dev)
    R1 = quat_to_matrix(torch.as_tensor(z["wrist_quat_init"], dtype=torch.float64)).float().to(dev)
    qg = torch.where(rec.active & cv, rec.qc2, q1)
    return StartGoal(q1, t1, R1, qg, rec.t2, rec.R2)


@torch.no_grad()
def run(S, psi, bank, rec, sg, dev):
    chunks = [slice(b, min(b + 256, len(rec))) for b in range(0, len(rec), 256)]
    out = {k: [] for k in ("qc", "R", "t")}; m = {k: [] for k in METRICS}; n0 = {k: [] for k in METRICS}
    for i in chunks:
        g = sg_index(sg, i)
        z = psi(rec.c_pos[i], rec.c_normal[i], rec.c_valid[i])
        o = S.predict(bank, rec.r[i], g, z)
        sc = score(bank, rec.index(i), rec.r[i], g, o["qc"], o["R"], o["t"], SCFG)
        s0 = score(bank, rec.index(i), rec.r[i], g, g.qc1, g.R1, g.t1, SCFG)
        for k in out:
            out[k].append(o[k])
        for k in METRICS:
            m[k].append(sc[k].float().cpu().numpy()); n0[k].append(s0[k].float().cpu().numpy())
    return ({k: torch.cat(v) for k, v in out.items()}, {k: np.concatenate(v) for k, v in m.items()},
            {k: np.concatenate(v) for k, v in n0.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data"); ap.add_argument("--regime", default="kite_code"); ap.add_argument("--splits", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--check-config"); ap.add_argument("--check-pool")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ccfg = paths.expand(ck["config"])
    S, info = load_system("model", {"kind": "j1", "ckpt": a.ckpt}, dev, "hand_frame")
    psi = FrozenPsi(ccfg["kite_ckpt"], ccfg["kite_root"]).to(dev).eval()

    if a.check_config:  # exact benchmark reproduction
        cfg = load_config(a.check_config, []); cfg["surface_sampling"] = "hand_frame"
        torch.manual_seed(0); noise = start_noise(cfg)
        pool = cfg["val"] if a.check_pool == "val" else cfg["extra_eval"][a.check_pool]
        bank = build_bank(cfg, dev)
        rec = load_pool(pool, bank).to(dev)
        sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
        _, m, n0 = run(S, psi, bank, rec, sg, dev)
        print(json.dumps({"pool": a.check_pool, "n": len(rec), "model": {k: round(float(v.mean()), 4) for k, v in m.items()},
                          "no_movement": {k: round(float(v.mean()), 4) for k, v in n0.items()}}, indent=1))
        return

    data = Path(paths.expand(a.data)).resolve(); out = Path(a.out)
    ex = data / "examples" / a.regime
    splits = a.splits or sorted(p.name for p in ex.iterdir() if p.is_dir())
    files = [(s, f) for s in splits for f in sorted((ex / s).glob("*.npz"))]
    hand_dirs = {}
    for s, f in files:
        hid = str(np.load(f)["hand_id"])
        hd = next((d for d in (data / "hands" / hid, data / "prim" / "hands" / hid) if (d / "hand.json").exists()), None)
        assert hd is not None, f"no hand description for {hid} under {data}/hands or {data}/prim/hands"
        hand_dirs[hid] = str(hd)
    bank = build_bank({"robots": [{"kind": "leap", "hand_dir": d} for d in hand_dirs.values()], "surface_sampling": "hand_frame"}, dev)
    summary = {"checkpoint": {"path": str(a.ckpt), "step": ck.get("step"), "sha256_16": sha16(a.ckpt), **info},
               "psi": {"kite_ckpt": str(ccfg["kite_ckpt"]), "sha256_16": sha16(ccfg["kite_ckpt"])},
               "data": str(data), "regime": a.regime, "start": "stored (joints_init / wrist_*_init)", "frame": "world, native base",
               "success": [POS_MM, ANG_DEG], "splits": {}}
    agg = {}
    for s, f in files:
        z = np.load(f); hid = str(z["hand_id"])
        rec = load_leap_records(f, bank).to(dev)
        sg = stored_start(bank, rec, z)
        o, m, n0 = run(S, psi, bank, rec, sg, dev)
        nd = z["joints_init"].shape[1]
        r = {"hand_id": hid, "split": s, "joints_pred": o["qc"][:, :nd].cpu().numpy().astype(np.float32),
             "wrist_pos_pred": o["t"].cpu().numpy().astype(np.float32), "wrist_quat_pred": quat_wxyz(o["R"])}
        for k in METRICS:
            r[f"model_{k}"] = m[k].astype(np.float32); r[f"noop_{k}"] = n0[k].astype(np.float32)
        p = out / s / f"{hid}.npz"; p.parent.mkdir(parents=True, exist_ok=True); np.savez(p, **r)
        A = agg.setdefault(s, {"n": 0, **{f"model_{k}": [] for k in METRICS}, **{f"noop_{k}": [] for k in METRICS}})
        A["n"] += len(rec)
        for k in METRICS:
            A[f"model_{k}"].append(m[k]); A[f"noop_{k}"].append(n0[k])
    for s, A in agg.items():
        summary["splits"][s] = {"n": A["n"], **{k: round(float(np.concatenate(v).mean()), 4) for k, v in A.items() if k != "n"}}
        print(s, summary["splits"][s]["n"], f"model corr {summary['splits'][s]['model_corr_mm']:.3f} mm,"
              f" success {100 * summary['splits'][s]['model_corr_success']:.1f}% | no movement {summary['splits'][s]['noop_corr_mm']:.3f} mm")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
