#!/bin/bash
#SBATCH -J kr-shadow-recs
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --constraint=avx2
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o slurm/logs/%x-%j.out
# Shadow eval-only records (reserved external target; never used for training/selection/tuning). CPU only.
#   sbatch -p default_partition slurm/shadow_records.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
echo "host=$(hostname -s) start=$(date -Is)"
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/shadow/make_shadow_records.py
echo "end=$(date -Is)"
