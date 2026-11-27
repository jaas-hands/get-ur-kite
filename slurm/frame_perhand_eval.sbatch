#!/bin/bash
#SBATCH -J kr-frame-perhand
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 05:00:00
#SBATCH -o slurm/logs/%x-%j.out
# Per-hand frame choice for frozen CF-3e (evaluation only; docs/cf3f_proposal.md).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
echo "host=$(hostname -s) start=$(date -Is)"
getkite_require_cuda slurm/frame_perhand_eval.sbatch
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/benchmark/frame_perhand_eval.py
echo "end=$(date -Is)"
