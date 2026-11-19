#!/bin/bash
#SBATCH -J getkite-eval
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -o slurm/logs/%x-%j.out
# Final evaluation of a finished LEAP run's val-selected checkpoint ($GETKITE_RUNS/<run>/best.pt) on every split,
# in full -> $GETKITE_RUNS/<run>/final_eval.json. Submit from the repo root:
#   sbatch -J kr-eval-<run> slurm/final_eval.sbatch configs/decoder/<config>.json <run>
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
CFG=$1; NAME=$2
OUT=$GETKITE_RUNS/$NAME
echo "host=$(hostname -s) start=$(date -Is) cfg=$CFG run=$NAME"
getkite_require_cuda slurm/final_eval.sbatch "$CFG" "$NAME"
"$GETKITE_PY" -m getkite.decoder.train --config "$CFG" --eval-only "$OUT/best.pt" --eval-out "$OUT/final_eval.json" \
  --set "kite_root=$KITE_ROOT" "kite_ckpt=$KITE_ROOT/checkpoints/decoder/release.pt" "data_dir=$GETKITE_DATASET" \
  "out_dir=$OUT" eval_max_examples=0 \
  'eval_splits=["val","train@holdout","test_geometry","test_joint_count","test_finger_subset","test_combined"]'
echo "end=$(date -Is)"
