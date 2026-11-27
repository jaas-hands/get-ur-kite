#!/bin/bash
# Waits for cf2_long_hf_s3/_s4 and d1_bv_long_hf_s3/_s4 DONE, then submits the 5-seed ensemble evaluation (retry once).
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
source slurm/env.sh
EXCL=${GETKITE_EXCLUDE_EVAL:-$GETKITE_EXCLUDE}
LOG=slurm/logs/seed5_chain.log
until [ -f "$GETKITE_RUNS/cf2_long_hf_s3/DONE" ] && [ -f "$GETKITE_RUNS/cf2_long_hf_s4/DONE" ] && [ -f "$GETKITE_RUNS/d1_bv_long_hf_s3/DONE" ] && [ -f "$GETKITE_RUNS/d1_bv_long_hf_s4/DONE" ]; do sleep 300; done
echo "$(date -Is) extra seeds DONE" >> "$LOG"
M=$GETKITE_RUNS/seed5_ensemble_v1/seed5_ensemble.json
for tries in 1 2; do
  J=$(sbatch --parsable -p default_partition --constraint="ampere|ada|2080ti" --exclude="$EXCL" slurm/seed5_ensemble_eval.sbatch)
  echo "$(date -Is) try $tries: seed5=[$J]" >> "$LOG"
  while squeue -h -j "$J" 2>/dev/null | grep -q .; do sleep 120; done
  [ -f "$M" ] && { echo "$(date -Is) seed5 evaluation complete" >> "$LOG"; exit 0; }
done
echo "$(date -Is) seed5 evaluation incomplete after 2 tries" >> "$LOG"
