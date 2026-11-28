#!/bin/bash
#SBATCH -J kr-teacher-memo
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o slurm/logs/%x-%j.out
# D1-bv memorisation check on generated bodies (evaluation only; docs/cf3f_proposal.md).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
echo "host=$(hostname -s) start=$(date -Is)"
getkite_require_cuda slurm/teacher_memorization_eval.sbatch "$@"
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/benchmark/teacher_memorization_eval.py
echo "end=$(date -Is)"
