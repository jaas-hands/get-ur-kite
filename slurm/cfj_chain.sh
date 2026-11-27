#!/bin/bash
# CF-J auto-evaluation (docs/cf_frame_jitter_proposal.md). Waits until cfj_long_hf{,_s1,_s2} all have DONE, then submits
# transfer_v1_cfj (4-hand held-out, Shadow; Primaries 1-3), transfer_v1_cfj_leapval105 (105-hand held-out target) and the
# frame-relabelling test; resubmits any missing output once. Run in the background from the repo root: nohup bash slurm/cfj_chain.sh &
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
source slurm/env.sh
EXCL=${GETKITE_EXCLUDE_EVAL:-$GETKITE_EXCLUDE}
LOG=slurm/logs/cfj_chain.log
until [ -f "$GETKITE_RUNS/cfj_long_hf/DONE" ] && [ -f "$GETKITE_RUNS/cfj_long_hf_s1/DONE" ] && [ -f "$GETKITE_RUNS/cfj_long_hf_s2/DONE" ]; do sleep 300; done
echo "$(date -Is) all CF-J runs DONE" >> "$LOG"
sub() { sbatch --parsable -p default_partition --constraint="ampere|ada|2080ti" --exclude="$EXCL" "$@"; }
M1=$GETKITE_RUNS/transfer_benchmark_v1_cfj/metrics.json; M2=$GETKITE_RUNS/transfer_benchmark_v1_cfj_leapval105/metrics.json
M3=$GETKITE_RUNS/cfj_frame_relabel/frame_relabel.json
for tries in 1 2; do
  B=""; V=""; L=""
  [ -f "$M1" ] || B=$(sub -J kr-bench-cfj slurm/benchmark.sbatch configs/decoder/benchmark/transfer_v1_cfj.json)
  [ -f "$M2" ] || V=$(sub -J kr-bench-cfj105 -t 03:00:00 --mem=48G slurm/benchmark.sbatch configs/decoder/benchmark/transfer_v1_cfj_leapval105.json)
  [ -f "$M3" ] || L=$(sub slurm/frame_relabel_cfj.sbatch)
  echo "$(date -Is) try $tries: benchmark=[$B] leapval105=[$V] relabel=[$L]" >> "$LOG"
  while squeue -h -u "$USER" -n kr-bench-cfj,kr-bench-cfj105,kr-cfj-relabel | grep -q .; do sleep 120; done
  if [ -f "$M1" ] && [ -f "$M2" ] && [ -f "$M3" ]; then echo "$(date -Is) CF-J evaluation complete" >> "$LOG"; exit 0; fi
done
echo "$(date -Is) CF-J evaluation incomplete after 2 tries; see slurm/logs/kr-bench-cfj*-* and kr-cfj-relabel-*" >> "$LOG"
