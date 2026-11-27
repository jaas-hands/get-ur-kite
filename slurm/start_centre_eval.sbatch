#!/bin/bash
#SBATCH -J kr-start-centre
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 03:00:00
#SBATCH -o slurm/logs/%x-%j.out
# Start-error rotation centre test for CF2 vs CF-O (evaluation only; docs/canonical_origin_proposal.md).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
echo "host=$(hostname -s) start=$(date -Is)"
getkite_require_cuda slurm/start_centre_eval.sbatch
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/benchmark/start_centre_eval.py --out '${GETKITE_RUNS}/start_centre_v1'
echo "end=$(date -Is)"
