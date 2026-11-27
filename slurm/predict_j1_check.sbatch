#!/bin/bash
#SBATCH -J kr-predict-j1-check
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -o slurm/logs/%x-%j.out
# predict_j1.py validation (evaluation only): (1) check mode on the 105-hand pool must reproduce the benchmark's CF-3e s0
# numbers; (2) data mode (stored start) on the dataset's val split, frozen CF-3e s0, to a scratch out dir.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
echo "host=$(hostname -s) start=$(date -Is)"
getkite_require_cuda slurm/predict_j1_check.sbatch
CK="$GETKITE_RUNS/cf3e_long_hf/best.pt"  # bash scripts/fetch_checkpoints.sh cf3e_long_hf
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/predict_j1.py --ckpt "$CK" --check-config configs/decoder/scratch/c3_psi50k_10x_wrist_leapval105.json --check-pool leap_heldout
OUT="$GETKITE_RUNS/predict_j1_check_cf3e_s0"; rm -rf "$OUT"
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/predict_j1.py --ckpt "$CK" --data "$GETKITE_DATASET" --regime kite_code --splits val --out "$OUT"
echo "end=$(date -Is)"
