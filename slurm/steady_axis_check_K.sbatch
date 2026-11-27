#!/bin/bash
#SBATCH -J kr-axis-check-K
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --constraint=avx2
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o slurm/logs/%x-%j.out
# Steady axis rule (docs/steady_axis_rule_proposal.md): CPU check, CPU only.
#   sbatch -p default_partition slurm/steady_axis_check_K.sbatch
set -euo pipefail
export GETKITE_LEAP_DATA=${GETKITE_LEAP_DATA:-}
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh; export GETKITE_LEAP_DATA=$GETKITE_DATASET PYTHONPATH=src:../get-ur-kite-dataset
echo "host=$(hostname -s) start=$(date -Is)"
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/dlevels/steady_axis_check_K.py \
  --workers "${SLURM_CPUS_PER_TASK:-32}"
echo "end=$(date -Is)"
