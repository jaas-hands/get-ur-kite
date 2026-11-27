#!/bin/bash
#SBATCH -J kr-frame-probe
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 03:00:00
#SBATCH -o slurm/logs/%x-%j.out
# Diagnostic: rule-E frame probe on external hands (evaluation only).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
echo "host=$(hostname -s) start=$(date -Is)"
getkite_require_cuda slurm/external_frame_probe.sbatch
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/benchmark/external_frame_probe.py
echo "end=$(date -Is)"
