#!/bin/bash
#SBATCH -J kr-cf3g-checks
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 01:30:00
#SBATCH -o slurm/logs/%x-%j.out
# CF-3g split-trained teachers: implementation checks, then 300-step smoke runs of one fold teacher and one student (stand-in
# teachers), both in scratch out dirs under $GETKITE_RUNS/smoke_cf3g. Launches nothing else.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
echo "host=$(hostname -s) start=$(date -Is) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
getkite_require_cuda slurm/cf3g_checks.sbatch
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/dlevels/cf3g_checks.py
SM="$GETKITE_RUNS/smoke_cf3g"; rm -rf "$SM/teacher" "$SM/student"
PYTHONPATH=src "$GETKITE_PY" -m getkite.decoder.train_joint --config configs/decoder/scratch/d1_bv_foldA_long_hf.json \
    --set total_steps=300 out_dir="$SM/teacher" eval_every=300 log_every=100
PYTHONPATH=src "$GETKITE_PY" -m getkite.decoder.train_joint --config "$SM/student_config.json" \
    --set total_steps=300 eval_every=300 log_every=100
cat "$SM/student/distill_groups.json"
grep -h examples_per_s "$SM/teacher/log.jsonl" | tail -1 | grep -o '"examples_per_s": [0-9.]*' || true
grep -h examples_per_s "$SM/student/log.jsonl" | tail -1 | grep -o '"examples_per_s": [0-9.]*' || true
echo "end=$(date -Is)"
