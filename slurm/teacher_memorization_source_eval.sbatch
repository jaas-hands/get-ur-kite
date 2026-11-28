#!/bin/bash
#SBATCH -J kr-teacher-memo-src
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o slurm/logs/%x-%j.out
# Source-family (LEAP / Allegro / Barrett) teacher memorisation check (evaluation only; docs/cf3f_proposal.md).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
echo "host=$(hostname -s) start=$(date -Is)"
getkite_require_cuda slurm/teacher_memorization_source_eval.sbatch "$@"
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/benchmark/teacher_memorization_source_eval.py
echo "end=$(date -Is)"
