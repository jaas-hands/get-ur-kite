#!/bin/bash
# CF-O + body variety auto-evaluation (docs/canonical_origin_proposal.md). Waits until cfo_bv_long_hf{,_s1,_s2} all have DONE, then submits
# transfer_v1_cfobv (4-hand held-out, Shadow; Primaries 1-3), transfer_v1_cfobv_leapval105 (105-hand held-out target) and the
# frame-relabelling test and the origin-shift test (CF-O spread must be ~0); resubmits any missing output once. Run in the background from the repo root: nohup bash slurm/cfj_chain.sh &
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
source slurm/env.sh
EXCL=${GETKITE_EXCLUDE_EVAL:-$GETKITE_EXCLUDE}
LOG=slurm/logs/cfobv_chain.log
until [ -f "$GETKITE_RUNS/cfo_bv_long_hf/DONE" ] && [ -f "$GETKITE_RUNS/cfo_bv_long_hf_s1/DONE" ] && [ -f "$GETKITE_RUNS/cfo_bv_long_hf_s2/DONE" ]; do sleep 300; done
echo "$(date -Is) all CF-O runs DONE" >> "$LOG"
sub() { sbatch --parsable -p default_partition --constraint="ampere|ada|2080ti" --exclude="$EXCL" "$@"; }
M1=$GETKITE_RUNS/transfer_benchmark_v1_cfobv/metrics.json; M2=$GETKITE_RUNS/transfer_benchmark_v1_cfobv_leapval105/metrics.json
M3=$GETKITE_RUNS/cfobv_frame_relabel/frame_relabel.json
M4=$GETKITE_RUNS/origin_shift_cfobv/origin_shift.json
for tries in 1 2; do
  B=""; V=""; L=""; S=""
  [ -f "$M1" ] || B=$(sub -J kr-bench-cfobv slurm/benchmark.sbatch configs/decoder/benchmark/transfer_v1_cfobv.json)
  [ -f "$M2" ] || V=$(sub -J kr-bench-cfobvbv105 -t 03:00:00 --mem=48G slurm/benchmark.sbatch configs/decoder/benchmark/transfer_v1_cfobv_leapval105.json)
  [ -f "$M3" ] || L=$(sub slurm/frame_relabel_cfobv.sbatch)
  [ -f "$M4" ] || S=$(sub slurm/origin_shift_cfobv.sbatch)
  echo "$(date -Is) try $tries: benchmark=[$B] leapval105=[$V] relabel=[$L] origin_shift=[$S]" >> "$LOG"
  while squeue -h -u "$USER" -n kr-bench-cfobv,kr-bench-cfobvbv105,kr-cfobv-relabel,kr-cfobv-origin-shift | grep -q .; do sleep 120; done
  if [ -f "$M1" ] && [ -f "$M2" ] && [ -f "$M3" ] && [ -f "$M4" ]; then echo "$(date -Is) CF-O evaluation complete" >> "$LOG"; exit 0; fi
done
echo "$(date -Is) CF-O evaluation incomplete after 2 tries; see slurm/logs/kr-bench-cfobv*-* and kr-cfobv-relabel-*" >> "$LOG"
