#!/bin/bash
#SBATCH -J kr-bv-validate
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --constraint=avx2
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH --array=0-7
#SBATCH -o slurm/logs/%x-%A_%a.out
# B body-variety LEAP validation (scripts/decoder/dlevels/validate_leap_bv.py), 8 shards, CPU only.
#   sbatch -p default_partition slurm/bodyvar_validate.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
export GETKITE_LEAP_DATA=$GETKITE_DATASET
echo "host=$(hostname -s) start=$(date -Is) shard=$SLURM_ARRAY_TASK_ID"
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/dlevels/validate_leap_bv.py --shard "$SLURM_ARRAY_TASK_ID" --n_shards 8
echo "end=$(date -Is)"
