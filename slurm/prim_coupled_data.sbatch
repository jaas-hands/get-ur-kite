#!/bin/bash
#SBATCH -J kr-prim-coupled-data
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --constraint=avx2
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o slurm/logs/%x-%j.out
# Coupled body draw "prim_c" (2026-11-21; docs/coupling_and_wrist_proposal.md, Phase 1), CPU only. Same pipeline as draw A:
#   * kite_data.build_prim couple: draw A's 900 validated bodies + coupled joints (prim_hand.sample_coupling, seed 20261119);
#   * records with the same plan / seed / regime as draw A (configs/decoder/dlevels/record_plan.json, seed 20261119);
#   * regression check (uncoupled bodies identical to draw A, records included) and validate_prim.py on every body.
#   sbatch -p default_partition slurm/prim_coupled_data.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
echo "host=$(hostname -s) start=$(date -Is)"
"$GETKITE_PY" -c "import numpy, mujoco" || { echo "numpy/mujoco unusable on $(hostname -s)"; exit 1; }
W=${SLURM_CPUS_PER_TASK:-16}; A=$GETKITE_DATASET/prim; C=$GETKITE_DATASET/prim_c
if [ ! -f "$C/hands/summary.json" ]; then
  mkdir -p "$C/hands"
  "$GETKITE_PY" -u -m kite_data.build_prim couple --src "$A/hands" --out "$C/hands" --seed 20261119 --workers "$W"
fi
"$GETKITE_PY" -u -m kite_data.build_prim records --hands "$C/hands" --out "$C/examples/kite_code" --seed 20261119 \
  --plan configs/decoder/dlevels/record_plan.json --workers "$W"
"$GETKITE_PY" -u scripts/decoder/dlevels/check_coupled_regression.py --src "$A" --dst "$C" --out configs/decoder/dlevels/prim_c_regression.json
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/dlevels/validate_prim.py --hands "$C/hands" --records "$C/examples/kite_code" \
  --out configs/decoder/dlevels/prim_c_validation.json
echo "end=$(date -Is)"
