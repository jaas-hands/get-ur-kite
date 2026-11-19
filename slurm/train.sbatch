#!/bin/bash
#SBATCH -J getkite-train
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH -o slurm/logs/%x-%j.out
# LEAP-family decoder training (getkite.decoder.train). Submit from the repo root:
#   sbatch -J kr-<run> slurm/train.sbatch configs/decoder/<config>.json <run> [key=value ...]
# Unicorn: add -p default_partition --constraint="ampere|ada|2080ti" (torch cu13x has no Pascal/Volta kernels).
# Output: $GETKITE_RUNS/<run>/ (config.json, log.jsonl, last.pt, best.pt, DONE). Resumes from last.pt after
# preemption or resubmission; slurm/watchdog.sh can resubmit automatically.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
CFG=$1; NAME=$2; shift 2
OUT=$GETKITE_RUNS/$NAME
echo "host=$(hostname -s) start=$(date -Is) restart_count=${SLURM_RESTART_COUNT:-0} cfg=$CFG run=$NAME extra=$*"
getkite_require_cuda slurm/train.sbatch "$CFG" "$NAME" "$@"
if [ -f "$OUT/DONE" ]; then echo "$OUT/DONE exists; nothing to do"; exit 0; fi
RESUME=(); if [ -f "$OUT/last.pt" ]; then RESUME=(--resume "$OUT/last.pt"); echo "resuming from $OUT/last.pt"; fi
getkite_run_forwarding "$GETKITE_PY" -m getkite.decoder.train --config "$CFG" "${RESUME[@]}" \
  --set "kite_root=$KITE_ROOT" "kite_ckpt=$KITE_ROOT/checkpoints/decoder/release.pt" "data_dir=$GETKITE_DATASET" \
  "out_dir=$OUT" "$@"
