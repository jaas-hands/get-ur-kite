#!/bin/bash
#SBATCH -J kr-cfo-origin-shift
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
getkite_require_cuda slurm/origin_shift_cfo.sbatch
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/benchmark/origin_shift_eval.py --out '${GETKITE_RUNS}/origin_shift_cfo' \
  --systems cfo_s0 cfo_s1 cfo_s2 cf2_s0 cf2_s1 cf2_s2 --bench configs/decoder/benchmark/transfer_v1_cfo.json
echo "end=$(date -Is)"
