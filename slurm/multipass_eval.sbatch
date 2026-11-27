#!/bin/bash
#SBATCH -J kr-multipass
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
getkite_require_cuda slurm/multipass_eval.sbatch
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/benchmark/multipass_eval.py --out '${GETKITE_RUNS}/multipass_v1' \
  --systems d1_s0 d1_s1 d1_s2 d1long_s0 d1long_s1 d1long_s2 cf2_s0 cf2_s1 cf2_s2 d1bv_s0 d1bv_s1 d1bv_s2
echo "end=$(date -Is)"
