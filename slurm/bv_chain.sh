#!/bin/bash
# B body-variety auto-evaluation; also waits for d1_bv_long_hf*. Waits until cf_bv_long_hf{,_s1,_s2} all have DONE, then
# submits the transfer benchmark (transfer_v1_bv.json) and the frame-relabelling test; resubmits either once if its job
# disappears without output. Run in the background from the repo root: nohup bash slurm/cf_chain.sh &
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
source slurm/env.sh
EXCL=${GETKITE_EXCLUDE_EVAL:-$GETKITE_EXCLUDE}
LOG=slurm/logs/bv_chain.log
until [ -f "$GETKITE_RUNS/d1_long_hf/DONE" ] && [ -f "$GETKITE_RUNS/d1_long_hf_s1/DONE" ] && [ -f "$GETKITE_RUNS/d1_long_hf_s2/DONE" ] && [ -f "$GETKITE_RUNS/cf2_long_hf/DONE" ] && [ -f "$GETKITE_RUNS/cf2_long_hf_s1/DONE" ] && [ -f "$GETKITE_RUNS/cf2_long_hf_s2/DONE" ] && [ -f "$GETKITE_RUNS/d1_bv_long_hf/DONE" ] && [ -f "$GETKITE_RUNS/d1_bv_long_hf_s1/DONE" ] && [ -f "$GETKITE_RUNS/d1_bv_long_hf_s2/DONE" ] && [ -f "$GETKITE_RUNS/cf_bv_long_hf/DONE" ] && [ -f "$GETKITE_RUNS/cf_bv_long_hf_s1/DONE" ] && [ -f "$GETKITE_RUNS/cf_bv_long_hf_s2/DONE" ]; do sleep 300; done
echo "$(date -Is) all B runs DONE" >> "$LOG"
sub() { sbatch --parsable -p default_partition --constraint="ampere|ada|2080ti" --exclude="$EXCL" "$@"; }
for tries in 1 2; do
  B=""; L=""
  [ -f "$GETKITE_RUNS/transfer_benchmark_v1_bv/metrics.json" ] || B=$(sub -J kr-bench-bv slurm/benchmark.sbatch configs/decoder/benchmark/transfer_v1_bv.json)
  [ -f "$GETKITE_RUNS/cfbv_frame_relabel/frame_relabel.json" ] || L=$(sub slurm/frame_relabel_cfbv.sbatch)
  echo "$(date -Is) try $tries: benchmark=[$B] relabel=[$L]" >> "$LOG"
  while squeue -h -u "$USER" -n kr-bench-bv,kr-cfbv-relabel | grep -q .; do sleep 120; done
  if [ -f "$GETKITE_RUNS/transfer_benchmark_v1_bv/metrics.json" ] && [ -f "$GETKITE_RUNS/cfbv_frame_relabel/frame_relabel.json" ]; then
    echo "$(date -Is) CF evaluation complete" >> "$LOG"; exit 0; fi
done
echo "$(date -Is) CF evaluation incomplete after 2 tries; see slurm/logs/kr-bench-bv-* and kr-cfbv-relabel-*" >> "$LOG"
