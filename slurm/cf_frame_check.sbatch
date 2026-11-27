#!/bin/bash
#SBATCH -J kr-cf-frame-check
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --constraint=avx2
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o slurm/logs/%x-%j.out
# CF step 1 (docs/cf_canonical_palm_frame_proposal.md): geometric check of the canonical palm frame, CPU only.
#   sbatch -p default_partition slurm/cf_frame_check.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
echo "host=$(hostname -s) start=$(date -Is)"
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/dlevels/canonical_frame_check.py --out configs/decoder/dlevels/cf_frame_check.json \
  --workers "${SLURM_CPUS_PER_TASK:-32}"
echo "end=$(date -Is)"
