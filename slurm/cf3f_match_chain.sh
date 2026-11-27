#!/bin/bash
# CF-3f matched teachers (user-approved 2026-11-27): divergence check at 25k (NaN, or source val > CF-3e@25k + 1 mm -> logged),
# INCOMPLETE detection; when all 3 seeds are DONE: standard + 105-pool benchmarks, then cf3f_gate.py match
# (-> ${GETKITE_RUNS}/cf3f_match_gate.json), then the external-hands eval (cf2 d1bv cf3e cf3fmatch). Launches no training.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
source slurm/env.sh
LOG=slurm/logs/cf3f_chain.log
EXCL=${GETKITE_EXCLUDE_EVAL:-$GETKITE_EXCLUDE}
sub() { sbatch --parsable -p default_partition --constraint="ampere|ada|2080ti" --exclude="$EXCL" "$@"; }
sfx() { [ "$1" = 0 ] || echo "_s$1"; }
echo "$(date -Is) match chain start" >> "$LOG"
declare -A CHK INC
while :; do
  n=0
  for s in 0 1 2; do
    d=$GETKITE_RUNS/cf3f_match_hf$(sfx $s)
    if [ -z "${CHK[$s]:-}" ] && [ -f "$d/log.jsonl" ]; then
      r=$("$GETKITE_PY" - "$d/log.jsonl" "$s" <<'P'
import json, math, sys
ref = [7.349, 7.395, 7.267][int(sys.argv[2])]; nan = False; v = None
for line in open(sys.argv[1]):
    r = json.loads(line)
    if "train/loss" in r and not math.isfinite(r["train/loss"]): nan = True
    if "eval" in r and r["step"] == 25000: v = r["eval"]["all/model/contact_mm"]
if nan: print("DIVERGED nan")
elif v is not None: print(("DIVERGED" if v > ref + 1.0 else "OK") + f" at 25000: {v:.3f} vs CF-3e@25k {ref:.3f}")
P
)
      [ -n "$r" ] && { echo "$(date -Is) match s$s divergence check: $r" >> "$LOG"; CHK[$s]=1; }
    fi
    if [ -f "$d/DONE" ]; then n=$((n + 1)); continue; fi
    if ! squeue -h -u "$USER" -n "kr-cf3f_match_hf$(sfx $s)" | grep -q . && [ -z "${INC[$s]:-}" ]; then
      echo "$(date -Is) INCOMPLETE: kr-cf3f_match_hf$(sfx $s) not in queue and no DONE" >> "$LOG"; INC[$s]=1; fi
  done
  [ "$n" = 3 ] && break
  sleep 300
done
B=$(sub -J kr-bench-cf3fmatch -t 03:00:00 slurm/benchmark.sbatch configs/decoder/benchmark/transfer_v1_cf3fmatch.json)
V=$(sub -J kr-bench-cf3fmatch105 -t 03:00:00 --mem=48G slurm/benchmark.sbatch configs/decoder/benchmark/transfer_v1_cf3fmatch_leapval105.json)
echo "$(date -Is) match all DONE; benchmarks std=[$B] 105=[$V]" >> "$LOG"
while squeue -h -j "$B,$V" 2>/dev/null | grep -q .; do sleep 120; done
if [ -f "$GETKITE_RUNS/transfer_benchmark_v1_cf3fmatch/metrics.json" ] && [ -f "$GETKITE_RUNS/transfer_benchmark_v1_cf3fmatch_leapval105/metrics.json" ]; then
  PYTHONPATH=src "$GETKITE_PY" scripts/decoder/benchmark/cf3f_gate.py match >> "$LOG" 2>&1
  echo "$(date -Is) match GATED -> $GETKITE_RUNS/cf3f_match_gate.json" >> "$LOG"
else echo "$(date -Is) match benchmarks failed; see slurm/logs/kr-bench-cf3fmatch*" >> "$LOG"; fi
X=$(sub -J kr-ext-cf3fmatch -t 03:00:00 --mem=48G slurm/external_hands_eval.sbatch "\${GETKITE_RUNS}/external_hands_cf3fmatch_v1" cf2 d1bv cf3e cf3fmatch)
echo "$(date -Is) match external hands eval [$X]" >> "$LOG"
