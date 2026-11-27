#!/bin/bash
#SBATCH -J kr-cfobv-origin-shift
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 03:00:00
#SBATCH -o slurm/logs/%x-%j.out
# docs/multipass_origin_checks_proposal.md (evaluation only).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
echo "host=$(hostname -s) start=$(date -Is)"
getkite_require_cuda slurm/origin_shift_cfobv.sbatch
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/benchmark/origin_shift_eval.py --out '${GETKITE_RUNS}/origin_shift_cfobv' \
  --systems cfobv_s0 cfobv_s1 cfobv_s2 --bench configs/decoder/benchmark/transfer_v1_cfobv.json
echo "end=$(date -Is)"
