"""Weight-0 reproduction check for the CF-3f-contact change (review item, 2026-11-28).

With contact_fk_weight = 0, a short run of the modified trainer must give identical losses and identical parameters to the
unmodified trainer, from the same config and seed; computing nothing extra means the random-number streams are untouched.
Runs both on CPU (deterministic) in throwaway output folders and compares every logged training value and every tensor in
last.pt exactly.

    python scripts/decoder/dlevels/check_contact_weight0.py --base 1244f9e --config configs/decoder/scratch/cf3e_long_hf.json \
        --steps 200 --work /tmp/fkw0 [--out checks/contact_weight0.json]

Needs the full CF-3e environment (data, KITE, teachers under $GETKITE_RUNS); CUDA is hidden so both runs use the CPU.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]


def run(src: Path, cfg: Path, out: Path, steps: int, cfg_over: dict):
    c = json.loads(cfg.read_text())
    c |= {"total_steps": steps, "log_every": 10, "eval_every": 10 ** 9, "ckpt_every": steps, "out_dir": str(out)} | cfg_over
    out.mkdir(parents=True, exist_ok=True)
    p = out.parent / f"{out.name}.json"
    p.write_text(json.dumps(c, indent=1))
    env = os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": f"{src / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    subprocess.run([sys.executable, "-m", "getkite.decoder.train_joint", "--config", str(p)], cwd=src, env=env, check=True)
    rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    return [r for r in rows if "train/loss" in r], torch.load(out / "last.pt", map_location="cpu", weights_only=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="unmodified commit (e.g. 1244f9e)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    base_src = work / "base_src"
    if not base_src.exists():
        base_src.mkdir()
        tar = subprocess.run(["git", "-C", str(REPO), "archive", a.base], check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(base_src)], input=tar, check=True)
    cfg = Path(a.config).resolve()
    rows_b, ck_b = run(base_src, cfg, work / "base_run", a.steps, {})
    rows_n, ck_n = run(REPO, cfg, work / "new_run", a.steps, {"contact_fk_weight": 0})
    keys = sorted({k for r in rows_b for k in r if k.startswith("train/")})
    loss_diff = max(abs(rb[k] - rn[k]) for rb, rn in zip(rows_b, rows_n) for k in keys)
    same_rows = len(rows_b) == len(rows_n) and all(rb["step"] == rn["step"] for rb, rn in zip(rows_b, rows_n))
    sd_b, sd_n = ck_b["model"], ck_n["model"]
    param_diff = max(float((sd_b[k].float() - sd_n[k].float()).abs().max()) for k in sd_b) if set(sd_b) == set(sd_n) else float("inf")
    res = {"base": a.base, "new": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                                                 text=True).stdout.strip(),
           "config": str(a.config), "steps": a.steps, "n_logged_rows": len(rows_b), "same_rows": same_rows,
           "max_abs_train_value_diff": loss_diff, "max_abs_param_diff": param_diff,
           "pass": bool(same_rows and loss_diff == 0.0 and param_diff == 0.0)}
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
