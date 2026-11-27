#!/bin/bash
# After CF-3d training (all three DONE) and the external-hand records exist, score CF-3d with D1-long, CF2 and D1-bv on the
# external evaluation hands (reported, never gating). Run from the repo root: nohup bash slurm/cf3d_external_chain.sh &
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
source slurm/env.sh
EXCL=${GETKITE_EXCLUDE_EVAL:-$GETKITE_EXCLUDE}
LOG=slurm/logs/cf3d_external_chain.log
EXT=$GETKITE_EXTERNAL
until [ -f "$GETKITE_RUNS/cf3d_long_hf/DONE" ] && [ -f "$GETKITE_RUNS/cf3d_long_hf_s1/DONE" ] && [ -f "$GETKITE_RUNS/cf3d_long_hf_s2/DONE" ] \
      && [ -f "$EXT/aero_adapter/aero_eval.npz" ] && ! squeue -h -u "$USER" -n kr-external-hands | grep -q .; do sleep 300; done
J=$(sbatch --parsable -p default_partition --constraint="ampere|ada|2080ti" --exclude="$EXCL" slurm/external_hands_eval.sbatch '${GETKITE_RUNS}/external_hands_cf3d_v1' d1long cf2 d1bv cf3d)
echo "$(date -Is) submitted external-hands CF-3d eval [$J]" >> "$LOG"
