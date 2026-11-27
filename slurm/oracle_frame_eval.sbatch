#!/bin/bash
#SBATCH -J kr-oracle-frame
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 03:00:00
#SBATCH -o slurm/logs/%x-%j.out
# CF-3 step 0: oracle per-hand frame for CF2 on held-out LEAP (evaluation only; docs/canonical_origin_proposal.md).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
echo "host=$(hostname -s) start=$(date -Is)"
getkite_require_cuda slurm/oracle_frame_eval.sbatch
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/benchmark/oracle_frame_eval.py
echo "end=$(date -Is)"
