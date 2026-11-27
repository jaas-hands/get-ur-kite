#!/bin/bash
# Big continued 100k->200k (user request 2026-11-27): when all 3 seeds are DONE, copy each final last.pt to
# cf3f_bigextlast_hf[_sN]/best.pt, then submit the standard + 105-pool benchmarks and the external-hands eval
# (cf3e, big@100k, bigext selected, bigext final). Divergence: NaN loss -> logged. Launches no training.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
source slurm/env.sh
LOG=slurm/logs/cf3f_chain.log
EXCL=${GETKITE_EXCLUDE_EVAL:-$GETKITE_EXCLUDE}
sub() { sbatch --parsable -p default_partition --constraint="ampere|ada|2080ti" --exclude="$EXCL" "$@"; }
sfx() { [ "$1" = 0 ] || echo "_s$1"; }
echo "$(date -Is) bigext chain start" >> "$LOG"
declare -A INC NAN
while :; do
  n=0
  for s in 0 1 2; do
    d=$GETKITE_RUNS/cf3f_bigext_hf$(sfx $s)
    if [ -z "${NAN[$s]:-}" ] && grep -q '"train/loss": NaN' "$d/log.jsonl" 2>/dev/null; then echo "$(date -Is) bigext s$s DIVERGED nan" >> "$LOG"; NAN[$s]=1; fi
    if [ -f "$d/DONE" ]; then n=$((n + 1)); continue; fi
    if ! squeue -h -u "$USER" -n "kr-cf3f_bigext_hf$(sfx $s)" | grep -q . && [ -z "${INC[$s]:-}" ]; then
      echo "$(date -Is) INCOMPLETE: kr-cf3f_bigext_hf$(sfx $s) not in queue and no DONE" >> "$LOG"; INC[$s]=1; fi
  done
  [ "$n" = 3 ] && break
  sleep 300
done
for s in 0 1 2; do mkdir -p "$GETKITE_RUNS/cf3f_bigextlast_hf$(sfx $s)"; cp "$GETKITE_RUNS/cf3f_bigext_hf$(sfx $s)/last.pt" "$GETKITE_RUNS/cf3f_bigextlast_hf$(sfx $s)/best.pt"; done
B=$(sub -J kr-bench-cf3fbigext -t 03:00:00 slurm/benchmark.sbatch configs/decoder/benchmark/transfer_v1_cf3fbigext.json)
V=$(sub -J kr-bench-cf3fbigext105 -t 03:00:00 --mem=48G slurm/benchmark.sbatch configs/decoder/benchmark/transfer_v1_cf3fbigext_leapval105.json)
X=$(sub -J kr-ext-cf3fbigext -t 03:00:00 --mem=48G slurm/external_hands_eval.sbatch "\${GETKITE_RUNS}/external_hands_cf3fbigext_v1" cf3e cf3fbig cf3fbigext cf3fbigextlast)
echo "$(date -Is) bigext all DONE; benchmarks std=[$B] 105=[$V] external=[$X]" >> "$LOG"
