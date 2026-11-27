#!/bin/bash
# CF-3e auto-evaluation (docs/cf3_distillation_proposal.md). Waits until cf3e_long_hf{,_s1,_s2} all have DONE, then submits
# transfer_v1_cf3e (standard pools, Shadow), transfer_v1_cf3e_leapval105 (105-hand held-out, gating), transfer_v1_cf3e_leap360
# (360-hand second held-out pool, reported) and the ensemble comparison; resubmits missing outputs once; then runs the readout
# (scripts/decoder/benchmark/cf3d_gate.py cf3e; launches nothing). Run from the repo root: nohup bash slurm/cf3e_chain.sh &
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
source slurm/env.sh
EXCL=${GETKITE_EXCLUDE_EVAL:-$GETKITE_EXCLUDE}
LOG=slurm/logs/cf3e_chain.log
until [ -f "$GETKITE_RUNS/cf3e_long_hf/DONE" ] && [ -f "$GETKITE_RUNS/cf3e_long_hf_s1/DONE" ] && [ -f "$GETKITE_RUNS/cf3e_long_hf_s2/DONE" ]; do sleep 300; done
echo "$(date -Is) all CF-3e runs DONE" >> "$LOG"
sub() { sbatch --parsable -p default_partition --constraint="ampere|ada|2080ti" --exclude="$EXCL" "$@"; }
M1=$GETKITE_RUNS/transfer_benchmark_v1_cf3e/metrics.json; M2=$GETKITE_RUNS/transfer_benchmark_v1_cf3e_leapval105/metrics.json
M3=$GETKITE_RUNS/transfer_benchmark_v1_cf3e_leap360/metrics.json; M4=$GETKITE_RUNS/seed_ensemble_cf3e_v1/seed_ensemble.json
for tries in 1 2; do
  B=""; V=""; W=""; E=""
  [ -f "$M1" ] || B=$(sub -J kr-bench-cf3e slurm/benchmark.sbatch configs/decoder/benchmark/transfer_v1_cf3e.json)
  [ -f "$M2" ] || V=$(sub -J kr-bench-cf3e105 -t 03:00:00 --mem=48G slurm/benchmark.sbatch configs/decoder/benchmark/transfer_v1_cf3e_leapval105.json)
  [ -f "$M3" ] || W=$(sbatch --parsable -p default_partition --gres=gpu:nvidia_rtx_a6000:1 --exclude="$EXCL" -J kr-bench-cf3e360 -t 04:00:00 --mem=64G slurm/benchmark.sbatch configs/decoder/benchmark/transfer_v1_cf3e_leap360.json)
  [ -f "$M4" ] || E=$(sub slurm/seed_ensemble_cf3e.sbatch)
  echo "$(date -Is) try $tries: benchmark=[$B] leapval105=[$V] leap360=[$W] ensembles=[$E]" >> "$LOG"
  while squeue -h -u "$USER" -n kr-bench-cf3e,kr-bench-cf3e105,kr-bench-cf3e360,kr-cf3e-ensemble | grep -q .; do sleep 120; done
  if [ -f "$M1" ] && [ -f "$M2" ]; then
    PYTHONPATH=src "$GETKITE_PY" scripts/decoder/benchmark/cf3d_gate.py cf3e >> "$LOG" 2>&1
    if [ -f "$M3" ] && [ -f "$M4" ]; then echo "$(date -Is) CF-3e evaluation complete" >> "$LOG"; exit 0; fi
  fi
done
echo "$(date -Is) CF-3e evaluation incomplete after 2 tries; see slurm/logs/kr-bench-cf3e*-*" >> "$LOG"
