#!/bin/bash
#SBATCH -J kr-prim-followup-data
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --constraint=avx2
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o slurm/logs/%x-%j.out
# Follow-up data for the morphology-diversity test (2026-11-20; docs/decoder_audit.md §59), CPU only. Same pipeline as the first
# draw (kite_data.build_prim hands -> records from a plan -> validate_prim.py):
#   * second body draw "prim_b": 900 bodies, generator seed 20261121, same record plan and record seed 20261121;
#   * control records from the first draw: P0003 (first 5-finger body) x 20000 (train_ctrl5), P0000-P0049 x 1600 (train_r4).
#   sbatch -p default_partition slurm/prim_followup_data.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
echo "host=$(hostname -s) start=$(date -Is)"
"$GETKITE_PY" -c "import numpy, mujoco" || { echo "numpy/mujoco unusable on $(hostname -s)"; exit 1; }
W=${SLURM_CPUS_PER_TASK:-16}; A=$GETKITE_DATASET/prim; B=$GETKITE_DATASET/prim_b
if [ ! -f "$B/hands/summary.json" ]; then
  mkdir -p "$B/hands"
  "$GETKITE_PY" -u -m kite_data.build_prim hands --n 900 --seed 20261121 --hands "$B/hands" --out "$B/hands" --workers "$W"
fi
"$GETKITE_PY" -u -m kite_data.build_prim records --hands "$B/hands" --out "$B/examples/kite_code" --seed 20261121 \
  --plan configs/decoder/dlevels/record_plan.json --workers "$W"
"$GETKITE_PY" -u -m kite_data.build_prim records --hands "$A/hands" --out "$A/examples/kite_code_ctrl5" \
  --plan configs/decoder/dlevels/record_plan_ctrl5.json --workers 1
"$GETKITE_PY" -u -m kite_data.build_prim records --hands "$A/hands" --out "$A/examples/kite_code_r4" \
  --plan configs/decoder/dlevels/record_plan_r4.json --workers "$W"
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/dlevels/validate_prim.py --hands "$B/hands" --records "$B/examples/kite_code" \
  --out configs/decoder/dlevels/prim_b_validation.json
R4IDS=$(seq -f "P%04g" 0 49 | tr '\n' ' ')
for X in "ctrl1:P0000" "ctrl5:P0003" "r4:$R4IDS"; do
  PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/dlevels/validate_prim.py --hands "$A/hands" --records "$A/examples/kite_code_${X%%:*}" \
    --ids ${X#*:} --out configs/decoder/dlevels/prim_${X%%:*}_validation.json
done
echo "end=$(date -Is)"
