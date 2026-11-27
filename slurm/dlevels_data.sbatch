#!/bin/bash
#SBATCH -J kr-dlevels-data
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --constraint=avx2
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o slurm/logs/%x-%j.out
# Morphology-diversity test data (CPU only): primitive-body records from configs/decoder/dlevels/record_plan.json,
# then per-hand validation of every body (FK adapter/native, records, coverage). Submit from the repo root:
#   sbatch -p default_partition slurm/dlevels_data.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
echo "host=$(hostname -s) start=$(date -Is)"
# old CPUs (no x86-64-v2 / no AVX2; MuJoCo dies with SIGILL there) are excluded by --constraint=avx2; guard kept as a backstop
if ! "$GETKITE_PY" -c "import numpy" 2>/dev/null; then
  X=${DLEVELS_DATA_EXCLUDE:-${GETKITE_EXCLUDE:-}}; X=${X:+$X,}$(hostname -s); export DLEVELS_DATA_EXCLUDE=$X
  echo "numpy unusable on $(hostname -s); resubmitting with --exclude=$X"
  sbatch --export=ALL -p "${SLURM_JOB_PARTITION:-default_partition}" -c "${SLURM_CPUS_PER_TASK:-4}" --mem=12G --constraint=avx2 --exclude="$X" slurm/dlevels_data.sbatch
  exit 0
fi
PRIM=$GETKITE_DATASET/prim
"$GETKITE_PY" -u -m kite_data.build_prim records --hands "$PRIM/hands" --out "$PRIM/examples/kite_code" \
  --plan configs/decoder/dlevels/record_plan.json --workers "${SLURM_CPUS_PER_TASK:-16}"
"$GETKITE_PY" -u scripts/decoder/dlevels/validate_prim.py --hands "$PRIM/hands" --records "$PRIM/examples/kite_code" \
  --out configs/decoder/dlevels/prim_validation.json
echo "end=$(date -Is)"
