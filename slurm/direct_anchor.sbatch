#!/bin/bash
#SBATCH -J getkite-anchor
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH -o slurm/logs/%x-%j.out
# direct_anchor_a1 learned-anchor diagnostic (getkite.decoder.train_direct_anchor). Submit from the repo root:
#   sbatch -J kr-<name> slurm/direct_anchor.sbatch configs/decoder/scratch/<config>.json [--set k=v model.k=v ...]
# The output dir is the config's out_dir (under $GETKITE_RUNS); resumes from its last.pt when requeued.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
CFG=$1; shift
echo "host=$(hostname -s) start=$(date -Is) restart_count=${SLURM_RESTART_COUNT:-0} cfg=$CFG extra=$*"
getkite_require_cuda slurm/direct_anchor.sbatch "$CFG" "$@"
getkite_run_forwarding "$GETKITE_PY" -m getkite.decoder.train_direct_anchor --config "$CFG" "$@"
