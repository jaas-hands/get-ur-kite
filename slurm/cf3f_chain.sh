#!/bin/bash
# CF-3f test arms: auto-evaluation + divergence check (docs/cf3f_proposal.md). Launches no training. For each arm in ARMS:
#   * divergence check (predeclared): NaN loss, or source-val contact error at the same schedule fraction worse than CF-3e's by
#     > 1 mm (CF-3e at 25k: 7.349 / 7.395 / 7.267; the long arm is checked at 75k = 25 % of its schedule) -> logged as DIVERGED;
#     the arm keeps running (the user decides);
#   * job ended without DONE -> logged as INCOMPLETE (no automatic resubmission beyond Slurm's own requeue);
#   * all 3 seeds DONE -> submit the arm's benchmarks (standard pools + Shadow; 105-hand pool; 360 pool for the long arm), then
#     run scripts/decoder/benchmark/cf3f_gate.py ARM (writes ${GETKITE_RUNS}/cf3f_${ARM}_gate.json).
# When every arm is gated, the external-hands evaluation runs once for all arms (reported only).
# Run from the repo root: setsid nohup bash slurm/cf3f_chain.sh > /dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
source slurm/env.sh
EXCL=${GETKITE_EXCLUDE_EVAL:-$GETKITE_EXCLUDE}
LOG=slurm/logs/cf3f_chain.log
ARMS=(leap long teach big rec)
declare -A DONE_ARM CHECKED BENCH
sub() { sbatch --parsable -p default_partition --constraint="ampere|ada|2080ti" --exclude="$EXCL" "$@"; }
run() { local a=$1 s=$2; echo "$GETKITE_RUNS/cf3f_${a}_hf$([ "$s" = 0 ] || echo "_s$s")"; }
divcheck() {  # $1 arm $2 seed
  local d; d=$(run "$1" "$2"); [ -f "$d/log.jsonl" ] || return 0
  "$GETKITE_PY" - "$d/log.jsonl" "$1" "$2" <<'E'
import json, math, sys
p, arm, s = sys.argv[1], sys.argv[2], int(sys.argv[3])
ref = [7.349, 7.395, 7.267][s]; at = 75000 if arm == "long" else 25000
nan = False; v = None
for line in open(p):
    r = json.loads(line)
    if "train/loss" in r and not math.isfinite(r["train/loss"]): nan = True
    if "eval" in r and r["step"] == at: v = r["eval"]["all/model/contact_mm"]
if nan: print("DIVERGED nan")
elif v is not None: print(("DIVERGED" if v > ref + 1.0 else "OK") + f" at {at}: {v:.3f} vs CF-3e@25k {ref:.3f}")
E
}
echo "$(date -Is) chain start (arms: ${ARMS[*]})" >> "$LOG"
while :; do
  all=1
  for a in "${ARMS[@]}"; do
    [ "${DONE_ARM[$a]:-}" = 1 ] && continue
    all=0; ndone=0
    for s in 0 1 2; do
      d=$(run "$a" "$s")
      if [ -z "${CHECKED[$a$s]:-}" ]; then
        r=$(divcheck "$a" "$s"); [ -n "$r" ] && { echo "$(date -Is) $a s$s divergence check: $r" >> "$LOG"; CHECKED[$a$s]=1; }
      fi
      if [ -f "$d/DONE" ]; then ndone=$((ndone + 1)); continue; fi
      n="kr-cf3f_${a}_hf$([ "$s" = 0 ] || echo "_s$s")"
      if ! squeue -h -u "$USER" -n "$n" | grep -q . && [ -z "${BENCH[inc$a$s]:-}" ]; then
        echo "$(date -Is) INCOMPLETE: $n not in queue and no DONE" >> "$LOG"; BENCH[inc$a$s]=1
      fi
    done
    if [ "$ndone" = 3 ]; then
      M1=$GETKITE_RUNS/transfer_benchmark_v1_cf3f${a}/metrics.json; M2=$GETKITE_RUNS/transfer_benchmark_v1_cf3f${a}_leapval105/metrics.json
      M3=$GETKITE_RUNS/transfer_benchmark_v1_cf3f${a}_leap360/metrics.json
      if [ -z "${BENCH[$a]:-}" ] || { [ "${BENCH[$a]}" = 1 ] && ! squeue -h -u "$USER" -n "kr-bench-cf3f$a,kr-bench-cf3f${a}105,kr-bench-cf3f${a}360" | grep -q . && { [ ! -f "$M1" ] || [ ! -f "$M2" ]; }; }; then
        BENCH[$a]=$(( ${BENCH[$a]:-0} + 1 ))
        B=""; V=""; W=""
        [ -f "$M1" ] || B=$(sub -J "kr-bench-cf3f$a" -t 03:00:00 slurm/benchmark.sbatch "configs/decoder/benchmark/transfer_v1_cf3f${a}.json")
        [ -f "$M2" ] || V=$(sub -J "kr-bench-cf3f${a}105" -t 03:00:00 --mem=48G slurm/benchmark.sbatch "configs/decoder/benchmark/transfer_v1_cf3f${a}_leapval105.json")
        if [ "$a" = long ] && [ ! -f "$M3" ]; then
          W=$(sbatch --parsable -p default_partition --gres=gpu:nvidia_rtx_a6000:1 --exclude="$EXCL" -J "kr-bench-cf3f${a}360" -t 04:00:00 --mem=64G slurm/benchmark.sbatch "configs/decoder/benchmark/transfer_v1_cf3f${a}_leap360.json")
        fi
        echo "$(date -Is) $a all DONE; benchmarks try ${BENCH[$a]}: std=[$B] 105=[$V] 360=[$W]" >> "$LOG"
      elif [ -f "$M1" ] && [ -f "$M2" ] && ! squeue -h -u "$USER" -n "kr-bench-cf3f${a}360" | grep -q .; then
        PYTHONPATH=src "$GETKITE_PY" scripts/decoder/benchmark/cf3f_gate.py "$a" >> "$LOG" 2>&1
        echo "$(date -Is) $a GATED -> $GETKITE_RUNS/cf3f_${a}_gate.json" >> "$LOG"; DONE_ARM[$a]=1
      elif [ "${BENCH[$a]}" -ge 2 ] && ! squeue -h -u "$USER" -n "kr-bench-cf3f$a,kr-bench-cf3f${a}105" | grep -q .; then
        echo "$(date -Is) $a benchmarks failed twice; see slurm/logs/kr-bench-cf3f${a}*" >> "$LOG"; DONE_ARM[$a]=1
      fi
    fi
  done
  [ "$all" = 1 ] && break
  sleep 300
done
X=$(sub -J kr-ext-cf3f -t 04:00:00 --mem=48G slurm/external_hands_eval.sbatch "\${GETKITE_RUNS}/external_hands_cf3f_v1" cf2 d1bv cf3e cf3fleap cf3flong cf3fteach cf3fbig cf3frec)
echo "$(date -Is) all arms gated; external hands eval [$X]; chain end" >> "$LOG"
