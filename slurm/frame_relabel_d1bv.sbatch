#!/bin/bash
#SBATCH -J kr-d1bv-relabel
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 03:00:00
#SBATCH -o slurm/logs/%x-%j.out
# Relabelling test on D1-bv vs D1-long (docs/variety_and_ensemble_checks.md, 1b; evaluation only).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
echo "host=$(hostname -s) start=$(date -Is)"
getkite_require_cuda slurm/frame_relabel_d1bv.sbatch
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/benchmark/frame_relabel_eval.py --bench configs/decoder/benchmark/transfer_v1_bv.json \
  --systems d1long_s0 d1long_s1 d1long_s2 d1bv_s0 d1bv_s1 d1bv_s2 --out '${GETKITE_RUNS}/d1bv_frame_relabel'
echo "end=$(date -Is)"
