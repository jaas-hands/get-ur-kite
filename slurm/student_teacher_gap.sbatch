#!/bin/bash
#SBATCH -J kr-st-gap
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 03:00:00
#SBATCH -o slurm/logs/%x-%j.out
# Diagnostic: CF-3d student-teacher distance by pool (coverage check), existing checkpoints (evaluation only).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
echo "host=$(hostname -s) start=$(date -Is)"
getkite_require_cuda slurm/student_teacher_gap.sbatch
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/benchmark/student_teacher_gap.py
echo "end=$(date -Is)"
