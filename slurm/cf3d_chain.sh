#!/bin/bash
# CF-3d auto-evaluation (docs/cf3_distillation_proposal.md). Waits until cf3d_long_hf{,_s1,_s2} all have DONE, then submits
# transfer_v1_cf3d (standard pools, Shadow), transfer_v1_cf3d_leapval105 (105-hand held-out, gating), transfer_v1_cf3d_leap360
# (360-hand second held-out pool, reported) and the ensemble comparison; resubmits missing outputs once; then runs the readout
# (scripts/decoder/benchmark/cf3d_gate.py; launches nothing). Run from the repo root: nohup bash slurm/cf3d_chain.sh &
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
source slurm/env.sh
EXCL=${GETKITE_EXCLUDE_EVAL:-$GETKITE_EXCLUDE}
LOG=slurm/logs/cf3d_chain.log
until [ -f "$GETKITE_RUNS/cf3d_long_hf/DONE" ] && [ -f "$GETKITE_RUNS/cf3d_long_hf_s1/DONE" ] && [ -f "$GETKITE_RUNS/cf3d_long_hf_s2/DONE" ]; do sleep 300; done
echo "$(date -Is) all CF-3d runs DONE" >> "$LOG"
sub() { sbatch --parsable -p default_partition --constraint="ampere|ada|2080ti" --exclude="$EXCL" "$@"; }
M1=$GETKITE_RUNS/transfer_benchmark_v1_cf3d/metrics.json; M2=$GETKITE_RUNS/transfer_benchmark_v1_cf3d_leapval105/metrics.json
M3=$GETKITE_RUNS/transfer_benchmark_v1_cf3d_leap360/metrics.json; M4=$GETKITE_RUNS/seed_ensemble_cf3d_v1/seed_ensemble.json
for tries in 1 2; do
  B=""; V=""; W=""; E=""
  [ -f "$M1" ] || B=$(sub -J kr-bench-cf3d slurm/benchmark.sbatch configs/decoder/benchmark/transfer_v1_cf3d.json)
  [ -f "$M2" ] || V=$(sub -J kr-bench-cf3d105 -t 03:00:00 --mem=48G slurm/benchmark.sbatch configs/decoder/benchmark/transfer_v1_cf3d_leapval105.json)
  [ -f "$M3" ] || W=$(sub -J kr-bench-cf3d360 -t 04:00:00 --mem=64G slurm/benchmark.sbatch configs/decoder/benchmark/transfer_v1_cf3d_leap360.json)
  [ -f "$M4" ] || E=$(sub slurm/seed_ensemble_cf3d.sbatch)
  echo "$(date -Is) try $tries: benchmark=[$B] leapval105=[$V] leap360=[$W] ensembles=[$E]" >> "$LOG"
  while squeue -h -u "$USER" -n kr-bench-cf3d,kr-bench-cf3d105,kr-bench-cf3d360,kr-cf3d-ensemble | grep -q .; do sleep 120; done
  if [ -f "$M1" ] && [ -f "$M2" ]; then
    PYTHONPATH=src "$GETKITE_PY" scripts/decoder/benchmark/cf3d_gate.py >> "$LOG" 2>&1
    if [ -f "$M3" ] && [ -f "$M4" ]; then echo "$(date -Is) CF-3d evaluation complete" >> "$LOG"; exit 0; fi
  fi
done
echo "$(date -Is) CF-3d evaluation incomplete after 2 tries; see slurm/logs/kr-bench-cf3d*-*" >> "$LOG"
