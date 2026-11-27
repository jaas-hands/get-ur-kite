#!/bin/bash
#SBATCH -J kr-benchmark
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH -o slurm/logs/%x-%j.out
# Frozen transfer benchmark (scripts/decoder/benchmark/transfer_benchmark.py). Submit from the repo root:
#   sbatch -J kr-bench-<name> slurm/benchmark.sbatch configs/decoder/benchmark/<config>.json
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
CFG=$1
echo "host=$(hostname -s) start=$(date -Is) restart_count=${SLURM_RESTART_COUNT:-0} cfg=$CFG"
getkite_require_cuda slurm/benchmark.sbatch "$CFG"
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/benchmark/transfer_benchmark.py --config "$CFG"
echo "end=$(date -Is)"
