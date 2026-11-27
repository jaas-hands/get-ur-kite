#!/bin/bash
#SBATCH -J kr-frame-select
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 03:00:00
#SBATCH -o slurm/logs/%x-%j.out
# Diagnostic: frame selection at inference (rule E and its 90/180/270 turns about d), existing CF2 checkpoints (evaluation only).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
echo "host=$(hostname -s) start=$(date -Is)"
getkite_require_cuda slurm/frame_select_eval.sbatch
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/benchmark/frame_select_eval.py
echo "end=$(date -Is)"
