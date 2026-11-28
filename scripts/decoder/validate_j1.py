"""Run a small J1 validation sample through the team's predictor and Janice's evaluator.

Requires the evaluator branch of get-ur-kite-dataset on PYTHONPATH. Only val files
are selected. See docs/CF3E_VALIDATION.md for the command and scope.
"""
import argparse
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch

import predict_j1 as j1
from kite_data.eval_io import (export_predictions, new_directory, read_npz,
                               sha256, source_record, write_json)
from kite_data.eval_metrics import baseline, evaluate_hand
from kite_data.evaluate import load_sampler
from kite_data.prediction_audit import convert_prediction

MATCH = {"corr_mm": ("mean_position_mm", .001),
         "corr_deg": ("mean_normal_deg", .05),
         "corr_success": ("contact_success", 0),
         "limit_violation_preclamp": ("joint_limit_violation", 0)}


@torch.no_grad()
def check_input_isolation(system, psi, bank, rec, start):
    """Check real inference with missing goals and poisoned scoring-only fields."""
    def predict(r, g):
        return system.predict(bank, r.r, g, psi(r.c_pos, r.c_normal, r.c_valid))

    normal = predict(rec, start)
    inputs_only = SimpleNamespace(qc1=start.qc1, R1=start.R1, t1=start.t1)
    blind = replace(rec, qc2=torch.full_like(rec.qc2, float("nan")),
                    t2=torch.full_like(rec.t2, float("nan")),
                    R2=torch.full_like(rec.R2, float("nan")),
                    e2=torch.full_like(rec.e2, float("nan")),
                    active=~rec.active, c_tpl=torch.full_like(rec.c_tpl, -1))
    poisoned = replace(start, qcg=torch.full_like(start.qcg, float("nan")),
                       tg=torch.full_like(start.tg, float("nan")),
                       Rg=torch.full_like(start.Rg, float("nan")))
    variants = [predict(blind, inputs_only), predict(blind, poisoned)]
    return all(torch.isfinite(normal[k]).all().item() and torch.equal(normal[k], v[k])
               for v in variants for k in ("qc", "R", "t"))


def compare_rows(rows, raw, prefix, hand, stage):
    if len(rows) != len(raw[f"{prefix}_corr_mm"]):
        raise ValueError(f"row count mismatch: {hand}")
    checks = []
    for i, row in enumerate(rows):
        if int(row["example_id"]) != i:
            raise ValueError(f"row order mismatch: {hand}")
        for metric, (field, tolerance) in MATCH.items():
            actual = float(row[field])
            expected = float(raw[f"{prefix}_{metric}"][i])
            delta = abs(actual - expected)
            checks.append(dict(stage=stage, hand=hand, row=i, prediction=prefix, metric=metric,
                               abs_delta=delta, passed=bool(np.isfinite(delta) and delta <= tolerance)))
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--kite-root", type=Path, required=True)
    ap.add_argument("--kite-ckpt", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    for name in ("ckpt", "kite_root", "kite_ckpt", "data", "out"):
        setattr(a, name, getattr(a, name).resolve())
    files = sorted((a.data / "examples/kite_code/val").glob("*.npz"))
    if not files:
        raise ValueError("No kite_code validation files; test splits are not selected by this script")
    out = new_directory(a.out)
    commands = []
    setup = ["#!/bin/sh", "set -eu", "cd " + shlex.quote(str(Path.cwd()))]
    for key in ("PYTHONPATH", "KITE_ROOT", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if key in os.environ:
            setup.append(f"export {key}=" + shlex.quote(os.environ[key]))

    def run(*args):
        cmd = [sys.executable, *map(str, args)]
        commands.append(shlex.join(cmd))
        (out / "commands.sh").write_text("\n".join(setup + commands) + "\n")
        with (out / "run.log").open("a") as log:
            log.write("\n$ " + commands[-1] + "\n"); log.flush()
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)

    run(Path(__file__).with_name("predict_j1.py"), "--ckpt", a.ckpt,
        "--kite-root", a.kite_root, "--kite-ckpt", a.kite_ckpt,
        "--data", a.data, "--splits", "val", "--out", out / "raw")
    dev = torch.device("cpu")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    system, _ = j1.load_system("model", {"kind": "j1", "ckpt": str(a.ckpt)}, dev, "hand_frame")
    psi = j1.FrozenPsi(str(a.kite_ckpt), str(a.kite_root)).eval()
    bank = j1.build_bank({"robots": [{"kind": "leap", "hand_dir": str(a.data / "hands" / f.stem)}
                                    for f in files], "surface_sampling": "hand_frame"}, dev)
    checks, isolation, sources, goal_checks = [], {}, [], {}
    for f in files:
        ex, record = source_record(f, a.data / "examples/kite_code", a.data / "hands", contact_only=True)
        hid = record["hand_id"]
        sampler, _, names = load_sampler(a.data / "hands" / hid, contact_only=True)
        if names != bank.descs[bank.index[hid]].coords:
            raise ValueError(f"joint order mismatch: {hid}")
        if any(hid in str(s) for s in ck["config"]["train"]):
            raise ValueError(f"validation hand is listed in checkpoint training data: {hid}")
        raw = read_npz(out / "raw" / record["path"])
        pred = convert_prediction(raw, record, names)
        export_predictions(out / "adapted" / record["path"], pred, record, names)
        for prefix, pose in (("model", pred), ("noop", baseline(ex, "init"))):
            rows = evaluate_hand(sampler, ex, pose, joint_policy="clamp")
            checks.extend(compare_rows(rows, raw, prefix, hid, "in_memory"))
        goals = evaluate_hand(sampler, ex, baseline(ex, "goal"), joint_policy="clamp")
        goal_checks[hid] = all(r["contact_success"] and r["mean_position_mm"] < .001
                              and r["mean_normal_deg"] < .01 for r in goals)
        rec = j1.load_leap_records(f, bank).to(dev)
        isolation[hid] = check_input_isolation(system, psi, bank, rec, j1.stored_start(bank, rec, ex))
        sources.append(record)

    metadata = {"model": "J1 checkpoint", "checkpoint_sha256": sha256(a.ckpt),
                "encoder_sha256": sha256(a.kite_ckpt), "checkpoint_step": ck["step"],
                "training_seed": ck["config"]["seed"], "selected_on": ck.get("selected_on"),
                "canonical_frame": ck["config"].get("canonical_frame", False),
                "start": "stored", "scope": "validation integration check, not a full benchmark",
                "inputs": sources, "input_isolation_pass": isolation,
                "packages": {"torch": torch.__version__, "numpy": np.__version__},
                "prediction_code_sha256": {p.name: sha256(p) for p in
                                           (Path(__file__), Path(__file__).with_name("predict_j1.py"))}}
    write_json(out / "model.json", metadata)
    common = ["--data", a.data / "examples/kite_code", "--hands", a.data / "hands",
              "--splits", "val", "--contact-only", "--joint-policy", "clamp"]
    for label in ("goal", "init", "model"):
        extra = ["--predictions", out / "adapted", "--metadata", out / "model.json"] if label == "model" else ["--baseline", label]
        run("-m", "kite_data.evaluate", "run", *common, *extra, "--plot-count", "2" if label == "model" else "0",
            "--run-id", label, "--out", out / label)
    run("-m", "kite_data.evaluate", "compare", "--runs", f"CF-3e={out / 'model'}",
        f"No-motion={out / 'init'}", "--out", out / "comparison")
    # Check the persisted evaluator output too, including its prediction-import path.
    for label, prefix in (("model", "model"), ("init", "noop")):
        with (out / label / "examples.csv").open() as stream:
            saved = list(csv.DictReader(stream))
        for f in files:
            rows = [r for r in saved if r["hand_id"] == f.stem]
            for r in rows:
                for key in ("contact_success", "joint_limit_violation"):
                    r[key] = {"True": True, "False": False}[r[key]]
            checks.extend(compare_rows(rows, read_npz(out / "raw/val" / f.name), prefix, f.stem, "saved_import"))
    with (out / "crosscheck.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checks[0])); writer.writeheader(); writer.writerows(checks)
    verification = {"metric_checks": len(checks), "metric_failures": sum(not c["passed"] for c in checks),
                    "max_abs_delta": {k: max(c["abs_delta"] for c in checks if c["metric"] == k) for k in MATCH},
                    "tolerances": {k: v[1] for k, v in MATCH.items()},
                    "goal_checks": goal_checks, "input_isolation": isolation,
                    "independently_checked": list(MATCH),
                    "not_independently_checked": ["compat_success", "wrist errors"],
                    "not_evaluated": ["self-collision", "task success", "full validation pool", "test splits"]}
    write_json(out / "verification.json", verification)
    if verification["metric_failures"] or not all(goal_checks.values()) or not all(isolation.values()):
        raise ValueError(f"Validation checks failed; inspect {out / 'verification.json'}")
    print(json.dumps(verification, indent=2))
    print(f"Report: {out / 'comparison/report.md'}")


if __name__ == "__main__":
    main()
