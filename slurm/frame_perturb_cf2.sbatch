#!/bin/bash
#SBATCH -J kr-cf2-perturb
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o slurm/logs/%x-%j.out
# Step 0 of docs/cf_frame_jitter_proposal.md (evaluation only).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
echo "host=$(hostname -s) start=$(date -Is)"
getkite_require_cuda slurm/frame_perturb_cf2.sbatch
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/benchmark/frame_perturb_eval.py
echo "end=$(date -Is)"
