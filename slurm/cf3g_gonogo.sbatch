#!/bin/bash
#SBATCH -J kr-cf3g-gonogo
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 03:00:00
#SBATCH -o slurm/logs/%x-%j.out
# CF-3g go/no-go on the 9 fold teachers (evaluation only; docs/cf3g_split_teachers.md). Run after all 9 finish.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
echo "host=$(hostname -s) start=$(date -Is)"
getkite_require_cuda slurm/cf3g_gonogo.sbatch "$@"
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/benchmark/cf3g_gonogo.py
echo "end=$(date -Is)"
