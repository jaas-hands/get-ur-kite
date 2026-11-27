#!/bin/bash
#SBATCH -J kr-seed5-ensemble
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 03:00:00
#SBATCH -o slurm/logs/%x-%j.out
# Seed ensemble (evaluation only).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
echo "host=$(hostname -s) start=$(date -Is)"
getkite_require_cuda slurm/seed5_ensemble_eval.sbatch
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/benchmark/seed5_ensemble_eval.py --out '${GETKITE_RUNS}/seed5_ensemble_v1'
echo "end=$(date -Is)"
