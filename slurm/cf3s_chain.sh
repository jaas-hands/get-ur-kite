#!/bin/bash
# CF-3s chain (docs/lag_robustness_proposal.md; scripts/decoder/dlevels/make_cf3s.py). Run from the repo root with nohup:
#   nohup bash slurm/cf3s_chain.sh > slurm/logs/cf3s_chain.out 2>&1 &
# Every 5 min, per arm (leap, body), independently:
#   * a seed with no DONE and no queued/running job -> logged INCOMPLETE and resubmitted once (Slurm's requeue handles preemption);
#   * all 3 seeds DONE -> submit that arm's benchmarks (standard + Shadow, 105 pool), the lag eval (--arm) and its external-hands eval at once;
#   * benchmarks + external done -> run scripts/decoder/benchmark/cf3s_readout.py ARM if it exists (committed before any result).
set -uo pipefail
cd "$(dirname "$0")/.."; source slurm/env.sh
LOG=slurm/logs/cf3s_chain.log
EXCL=rocky,samitha-compute-01,sablab-gpu-12,portal-compute-04,cuvl-compute-05,kuleshov-compute-01,davis-compute-01,snavely-compute-01,snavely-compute-02,acadia,eas-shared-compute-01,nikola-compute-15,yosemite,ellis-compute-01,dean-compute-01,fang-compute-02,yu-compute-01,elor-compute-02,portal-compute-02,elor-compute-01,seo-compute-01,portal-compute-03
sub() { sbatch --parsable -p default_partition --exclude="$EXCL" "$@"; }
run() { echo "$GETKITE_RUNS/cf3s_$1_hf$([ "$2" = 0 ] || echo "_s$2")"; }
name() { echo "kr-cf3s_$1_hf$([ "$2" = 0 ] || echo "_s$2")"; }
cfg() { echo "configs/decoder/scratch/cf3s_$1_hf$([ "$2" = 0 ] || echo "_s$2").json"; }
declare -A RESUB BENCH READ
echo "$(date -Is) chain start $(git log --oneline -1)" >> "$LOG"
while :; do
  all=1
  for a in wide; do
    [ "${READ[$a]:-}" = 1 ] && continue
    all=0; nd=0
    for s in 0 1 2; do
      if [ -f "$(run $a $s)/DONE" ]; then nd=$((nd + 1)); continue; fi
      if ! squeue -h -u "$USER" -n "$(name $a $s)" | grep -q .; then
        if [ -z "${RESUB[$a$s]:-}" ]; then
          J=$(sub --constraint="ampere|ada|2080ti" -J "$(name $a $s)" -t 08:00:00 slurm/joint.sbatch "$(cfg $a $s)")
          echo "$(date -Is) INCOMPLETE $a s$s: resubmitted [$J]" >> "$LOG"; RESUB[$a$s]=1
        fi
      fi
    done
    if [ "$nd" = 3 ] && [ -z "${BENCH[$a]:-}" ]; then
      B=$(sub --constraint="ampere|ada|2080ti" -J "kr-bench-cf3s$a" -t 03:00:00 --requeue slurm/benchmark.sbatch "configs/decoder/benchmark/transfer_v1_cf3s$a.json")
      V=$(sub --constraint="ampere|ada|2080ti" -J "kr-bench-cf3s${a}105" -t 03:00:00 --mem=48G --requeue slurm/benchmark.sbatch "configs/decoder/benchmark/transfer_v1_cf3s${a}_leapval105.json")
      W=$(sub --gres=gpu:1 -J "kr-lag-cf3s$a" -c 4 --mem=64G -t 02:00:00 --requeue -o "slurm/logs/kr-lag-cf3s$a-%j.out" --wrap "exec bash -c 'set -eo pipefail; source slurm/env.sh; PYTHONPATH=src \$GETKITE_PY scripts/decoder/benchmark/lag_robustness_eval.py --arm cf3s$a'")
      X=$(sub --constraint="ampere|ada|2080ti" -J "kr-ext-cf3s$a" -t 04:00:00 --mem=48G --requeue slurm/external_hands_eval.sbatch "\${GETKITE_RUNS}/external_hands_cf3s${a}_v1" cf2 d1bv cf3e "cf3s$a")
      echo "$(date -Is) $a all DONE; std=[$B] 105=[$V] lag=[$W] ext=[$X]" >> "$LOG"; BENCH[$a]=1
    fi
    if [ "${BENCH[$a]:-}" = 1 ] && ! squeue -h -u "$USER" -n "kr-bench-cf3s$a,kr-bench-cf3s${a}105,kr-lag-cf3s$a,kr-ext-cf3s$a" | grep -q .; then
      if [ -f scripts/decoder/benchmark/cf3s_readout.py ]; then
        PYTHONPATH=src "$GETKITE_PY" scripts/decoder/benchmark/cf3s_readout.py "$a" >> "$LOG" 2>&1
        echo "$(date -Is) $a READOUT done (see above)" >> "$LOG"
      else
        echo "$(date -Is) $a benchmarks done; cf3s_readout.py missing" >> "$LOG"
      fi
      READ[$a]=1
    fi
  done
  [ "$all" = 1 ] && break
  sleep 300
done
echo "$(date -Is) chain end" >> "$LOG"
