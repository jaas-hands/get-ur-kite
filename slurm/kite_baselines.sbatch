#!/bin/bash
#SBATCH -J kite-baselines
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH -o slurm/logs/%x-%A_%a.out
# The KITE-head baselines on the team dataset (docs/training_plan.md, section 1), one stage per job.
# Submit all of them with slurm/submit_kite_baselines.sh, which sets resources and dependencies per stage.
#   gen     examples with the dataset's code: 1M nominal (B0-ds), 16k per B3 hand, 200k per B4 hand (CPU only)
#   b0ds    B0-ds at scale: the LEAP head from the laptop's b0-trunk, 1M dataset examples, trunk trained
#   matrix  B0-ds, clamped, B1, B2 on all hands of the four test splits, all 2,000 examples per hand; G4
#   b3      array task i: B3 on line i of b3_hands.txt, 1k / 4k / 16k examples, val-selected, official test
#   b4      array task i: B4 on line i of b4_hands.txt (B1 head of B0-ds retrained on 200k examples, trunk too)
# Inputs:  $GETKITE_RUNS/leap_head/init/b0-trunk/{last.pt,dataset_meta.pt}  (copied from the laptop)
# Outputs: data in $GETKITE_DATA/leap_head/, runs in $GETKITE_RUNS/leap_head/cluster/
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
STAGE=$1; shift
DATA=$GETKITE_DATA/leap_head
OUT=$GETKITE_RUNS/leap_head/cluster
INIT=${GETKITE_B0DS_INIT:-$GETKITE_RUNS/leap_head/init/b0-trunk/last.pt}
B0DS=$OUT/b0-ds/best.pt
# the dataset's hands and test examples (LeapSplit reads GETKITE_LEAP_DATA, default: the submodule's data/)
[ -z "${GETKITE_LEAP_DATA:-}" ] && [ -d "$GETKITE_DATASET/hands" ] && export GETKITE_LEAP_DATA=$GETKITE_DATASET
mkdir -p "$DATA" "$OUT"
TASK=${SLURM_ARRAY_TASK_ID:-0}
echo "host=$(hostname -s) start=$(date -Is) stage=$STAGE task=$TASK data=${GETKITE_LEAP_DATA:-submodule} extra=$*"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
[ "$STAGE" = gen ] || getkite_require_cuda slurm/kite_baselines.sbatch "$STAGE" "$@"

line() { sed -n "$((TASK + 1))p" "$1"; }   # "split hand" of this array task

case $STAGE in
gen)
  "$GETKITE_PY" -u scripts/gen_kite_head_data.py --out "$DATA" --workers "${SLURM_CPUS_PER_TASK:-8}" "$@"
  ;;
b0ds)
  [ -f "$INIT" ] || { echo "missing $INIT (copy the laptop's b0-trunk checkpoint, see submit_kite_baselines.sh)"; exit 1; }
  "$GETKITE_PY" -u scripts/train_kite_head_dataset.py "$INIT" --data "$DATA/nominal" --out "$OUT/b0-ds" \
    --epochs 20 --trunk "$@"
  "$GETKITE_PY" -u scripts/eval_kite_head.py "$B0DS" --max-examples 2000 --out "$OUT/b0-ds.g4.json" | tail -20
  ;;
matrix)
  "$GETKITE_PY" -u scripts/eval_kite_baselines.py "$B0DS" --per-hand 2000 --modes b1 b2 clamped \
    --out "$OUT/b0-ds.baselines.json" "$@" | grep -E "^test_|Error|Trace"
  ;;
b3)
  read -r SPLIT HAND <<< "$(line "$DATA/b3_hands.txt")"
  for N in 1000 4000 16000; do
    echo "== B3 $HAND ($SPLIT) train=$N $(date +%T)"
    "$GETKITE_PY" -u scripts/finetune_kite_head.py "$B0DS" "$HAND" --split "$SPLIT" --data "$DATA/b3" \
      --train $N --val 500 --eval 2000 --minutes 40 --lr 1e-5 --out "$OUT/b3/${HAND}_${N}.json" "$@"
  done
  ;;
b4)
  read -r SPLIT HAND <<< "$(line "$DATA/b4_hands.txt")"
  "$GETKITE_PY" -u scripts/train_kite_head_dataset.py "$B0DS" --hand "$HAND" --mode b1 --trunk \
    --data "$DATA/b4" --test-split "$SPLIT" --epochs 20 --out "$OUT/b4/$HAND" "$@"
  ;;
*) echo "unknown stage $STAGE"; exit 2 ;;
esac
echo "end=$(date -Is)"
