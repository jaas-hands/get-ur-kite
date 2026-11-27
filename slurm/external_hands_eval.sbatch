#!/bin/bash
#SBATCH -J kr-external-eval
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 03:00:00
#SBATCH -o slurm/logs/%x-%j.out
# Decoders on the external evaluation hands (evaluation only). Args: out dir, then arms.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
echo "host=$(hostname -s) start=$(date -Is)"
getkite_require_cuda slurm/external_hands_eval.sbatch "$@"
OUT=$1; shift
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/benchmark/external_hands_eval.py --out "$OUT" --arms "$@"
echo "end=$(date -Is)"
