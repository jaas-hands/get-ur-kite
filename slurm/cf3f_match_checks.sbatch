#!/bin/bash
#SBATCH -J kr-cf3f-match-checks
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -o slurm/logs/%x-%j.out
# CF-3f matched teachers: implementation checks, then a 300-step smoke run (scratch out dir). Launches nothing else.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
echo "host=$(hostname -s) start=$(date -Is)"
getkite_require_cuda slurm/cf3f_match_checks.sbatch
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/dlevels/cf3f_matched_checks.py
SM="$GETKITE_RUNS/smoke_cf3f_match"; rm -rf "$SM"
getkite_run_forwarding "$GETKITE_PY" -m getkite.decoder.train_joint --config configs/decoder/scratch/cf3f_match_hf.json --set total_steps=300 out_dir="$SM" eval_every=300 log_every=100
cat "$SM/distill_matched.json" || true
echo "end=$(date -Is)"
