#!/bin/bash
# Stall / failure watcher for all of my kr-* slurm jobs (data, Shadow, smoke, D1/D3 training). Run from the repo root:
#   bash slurm/stall_watch.sh            # loops forever, every $STALL_EVERY s (default 120)
#   STALL_ONCE=1 bash slurm/stall_watch.sh
# Prints one line per ALERT to stdout (and appends to slurm/logs/stall_watch.log); silent when everything is healthy.
# Per running job:
#   * progress age = minutes since the newest write to its slurm .out or, for training, $GETKITE_RUNS/<run>/log.jsonl;
#   * if progress age >= $STALL_MIN (default 15), samples the summed %CPU of my processes inside the job
#     (srun --overlap ps) and alerts STALL when that is below 5% (an idle job, like the 66-min Pool hang on 2026-11-19);
#   * alerts ERROR on new crash signatures in its .out (Traceback, Illegal instruction, Killed, OOM, CUDA error, ...).
# Per finished job (left the queue since the last pass): alerts on any sacct state other than COMPLETED.
# Per job name: alerts MOVED when a run is requeued/preempted (Restarts count rises) or continues under a new job id.
# Any slurm log written since the last pass that contains 'CUDA unusable' alerts BADNODE (the job bounced off a node).
# Alerts are de-duplicated per job and kind. Read-only: it never cancels or resubmits (dlevels_watch.sh resubmits).
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
source slurm/env.sh
EVERY=${STALL_EVERY:-120}; STALL_MIN=${STALL_MIN:-15}; LOG=slurm/logs/stall_watch.log
SIG='Traceback|Illegal instruction|Killed|out of memory|OutOfMemory|CUDA error|Segmentation fault|core dumped|RuntimeError|BrokenProcessPool|CANCELLED|DUE TO TIME LIMIT|numpy unusable'
declare -A SEEN PREVJOBS LASTSIG
MARK=$(mktemp); touch "$MARK"
alert() { local key=$1; shift; [ -n "${SEEN[$key]:-}" ] && return; SEEN[$key]=1; local m="$(date '+%F %T') ALERT $*"; echo "$m"; echo "$m" >> "$LOG"; }
mtime() { stat -c %Y "$1" 2>/dev/null || echo 0; }
while true; do
  now=$(date +%s); declare -A CUR=()
  while read -r JID NAME STATE NODE; do
    [ -z "$JID" ] && continue; CUR[$JID]=$NAME
    sig="$JID:$(scontrol show job "$JID" 2>/dev/null | grep -o 'Restarts=[0-9]*' | cut -d= -f2)"
    if [ -n "${LASTSIG[$NAME]:-}" ] && [ "${LASTSIG[$NAME]}" != "$sig" ]; then
      alert "$NAME-moved-$sig" "$NAME MOVED: ${LASTSIG[$NAME]} -> $sig (job:restarts) state=$STATE node=${NODE:-none}"
    fi
    LASTSIG[$NAME]=$sig
    [ "$STATE" != RUNNING ] && continue
    OUT=$(ls slurm/logs/"$NAME"-"$JID".out 2>/dev/null)
    newest=$(mtime "$OUT"); run=${NAME#kr-}
    [ -f "$GETKITE_RUNS/$run/log.jsonl" ] && { t=$(mtime "$GETKITE_RUNS/$run/log.jsonl"); [ "$t" -gt "$newest" ] && newest=$t; }
    start=$(date -d "$(squeue -h -j "$JID" -o %S)" +%s 2>/dev/null || echo "$now"); [ "$newest" -lt "$start" ] && newest=$start
    age=$(( (now - newest) / 60 ))
    if [ -n "$OUT" ] && grep -qE "$SIG" "$OUT"; then
      alert "$JID-err" "$NAME ($JID on $NODE): $(grep -E "$SIG" "$OUT" | tail -1 | cut -c1-160)"
    fi
    if [ "$age" -ge "$STALL_MIN" ]; then
      cpu=$(timeout 60 srun --jobid "$JID" --overlap ps -u "$USER" -o pcpu= 2>/dev/null | awk '{s+=$1} END{printf "%d", s}')
      if [ -n "$cpu" ] && [ "$cpu" -lt 5 ]; then
        alert "$JID-stall-$(( age / 30 ))" "$NAME ($JID on $NODE): STALL - no output for ${age} min and CPU ${cpu}% (idle)"
      fi
    fi
  done < <(squeue -h -u "$USER" -o "%i %j %T %N" | awk '$2 ~ /^kr-/')
  for JID in "${!PREVJOBS[@]}"; do
    [ -n "${CUR[$JID]:-}" ] && continue
    st=$(sacct -n -X -j "$JID" -o State,ExitCode 2>/dev/null | head -1 | xargs)
    case "$st" in COMPLETED*|"") ;; *) alert "$JID-end" "${PREVJOBS[$JID]} ($JID) ended: $st";; esac
  done
  NEWMARK=$(mktemp); touch "$NEWMARK"
  for f in $(find slurm/logs -name 'kr-*.out' -newer "$MARK" 2>/dev/null); do
    l=$(grep -m1 "CUDA unusable" "$f") && alert "$f-badnode" "BADNODE $(basename "$f"): $l"
  done
  rm -f "$MARK"; MARK=$NEWMARK
  PREVJOBS=(); for J in "${!CUR[@]}"; do PREVJOBS[$J]=${CUR[$J]}; done
  [ -n "${STALL_ONCE:-}" ] && exit 0
  sleep "$EVERY"
done
