#!/bin/bash
#SBATCH -J kr-cfo-relabel
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 03:00:00
#SBATCH -p default_partition
#SBATCH --constraint=ampere|ada|2080ti
#SBATCH --requeue
#SBATCH -o slurm/logs/%x-%j.out
# W2-a frame-relabelling test (scripts/decoder/benchmark/frame_relabel_eval.py; evaluation only). Submit from the repo root:
#   sbatch --exclude=<list> slurm/frame_relabel.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
echo "host=$(hostname -s) start=$(date -Is) restart_count=${SLURM_RESTART_COUNT:-0}"
getkite_require_cuda slurm/frame_relabel_cfo.sbatch
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/benchmark/frame_relabel_eval.py --bench configs/decoder/benchmark/transfer_v1_cfo.json \
  --systems cfo_s0 cfo_s1 cfo_s2 --out '${GETKITE_RUNS}/cfo_frame_relabel'
echo "end=$(date -Is)"
