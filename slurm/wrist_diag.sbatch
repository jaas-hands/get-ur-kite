#!/bin/bash
#SBATCH -J kr-wristdiag
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -p default_partition
#SBATCH --constraint=ampere|ada|2080ti
#SBATCH -o slurm/logs/%x-%j.out
# Phase 0 wrist diagnostics (scripts/decoder/benchmark/wrist_diag.py; evaluation only). Submit from the repo root:
#   sbatch --exclude=<list> slurm/wrist_diag.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
echo "host=$(hostname -s) start=$(date -Is)"
getkite_require_cuda slurm/wrist_diag.sbatch
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/benchmark/wrist_diag.py --bench configs/decoder/benchmark/transfer_v1_followups.json \
  --systems d0_g1 c1 d1 d1b n5 r4 --out '${GETKITE_RUNS}/wrist_diag_v1' --device cuda
echo "end=$(date -Is)"
