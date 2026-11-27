#!/bin/bash
#SBATCH -J kr-external-hands
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o slurm/logs/%x-%j.out
# External evaluation hands: adapters, templates and 500 evaluation-only records each (scripts/decoder/external_hands/make_hand_adapter.py). CPU only.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
echo "host=$(hostname -s) start=$(date -Is)"
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/external_hands/make_hand_adapter.py --hand all --n 500
echo "end=$(date -Is)"
