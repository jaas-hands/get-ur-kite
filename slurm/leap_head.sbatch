#!/bin/bash
#SBATCH -J leap-head
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH -o slurm/logs/%x-%j.out
# Baseline B0: KITE's decoder head for LEAP, started from KITE's released trunk (scripts/train_leap_head.py).
# Submit from the repo root, after `python scripts/install_kite_leap.py` once on the cluster:
#   sbatch -J b0-trunk --export=ALL,GETKITE_B0_INIT=source,GETKITE_B0_UNFREEZE=trunk slurm/leap_head.sbatch b0-trunk
#   sbatch -J b0-frozen --export=ALL,GETKITE_B0_INIT=source slurm/leap_head.sbatch b0-frozen
#   sbatch -J b0-trunk-200k --mem=32G -t 08:00:00 --export=ALL,GETKITE_B0_INIT=source,GETKITE_B0_UNFREEZE=trunk \
#       slurm/leap_head.sbatch b0-trunk-200k synthetic_data.train_samples=200000
#   sbatch -J b0-smoke -t 00:30:00 slurm/leap_head.sbatch b0-smoke synthetic_data.train_samples=20000 \
#       synthetic_data.test_samples=500 training.epochs=2
# b0-trunk is the main run (trunk trained, ψ frozen: the laptop pilots showed a frozen trunk learns little
# for LEAP). LEAP's contact regions follow the dataset (scripts/train_leap_head.py). Defaults are KITE's released settings (1M examples, batch 128, ...) with an epoch cap of 100; anything after
# the run name is a Hydra override for KITE's script.
# Output: $GETKITE_RUNS/leap_head/<run>/ (KITE checkpoints last.pt, epoch_*.pt; buffers/ with the generated data).
# A requeued or resubmitted job reuses the saved data and resumes from last.pt.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
NAME=$1; shift
OUT=$GETKITE_RUNS/leap_head/$NAME
mkdir -p "$OUT"
echo "host=$(hostname -s) start=$(date -Is) restart_count=${SLURM_RESTART_COUNT:-0} run=$NAME init=${GETKITE_B0_INIT:-random} extra=$*"
getkite_require_cuda slurm/leap_head.sbatch "$NAME" "$@"
RESUME=(); if [ -f "$OUT/last.pt" ]; then RESUME=(training.resume=true "training.resume_checkpoint_path=$OUT/last.pt"); echo "resuming from $OUT/last.pt"; fi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
getkite_run_forwarding "$GETKITE_PY" -u scripts/train_leap_head.py \
  "training.save_dir=$OUT" training.epochs=100 dataset.num_workers=6 "hydra.run.dir=$OUT/hydra" \
  "${RESUME[@]}" "$@"
