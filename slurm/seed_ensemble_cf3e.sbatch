#!/bin/bash
#SBATCH -J kr-cf3e-ensemble
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 03:00:00
#SBATCH -o slurm/logs/%x-%j.out
# CF-3e ensembles vs CF2, D1-bv and the 6-checkpoint teacher (evaluation only).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
echo "host=$(hostname -s) start=$(date -Is)"
getkite_require_cuda slurm/seed_ensemble_eval.sbatch
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/benchmark/seed_ensemble_eval.py --out '${GETKITE_RUNS}/seed_ensemble_cf3e_v1' --arms cf2 d1bv cf3d cf3e --mix d1bv+cf2
echo "end=$(date -Is)"
