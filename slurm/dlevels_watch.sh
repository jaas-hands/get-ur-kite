#!/bin/bash
# Watcher for the morphology-diversity training runs (D1 / D3; slurm/joint.sbatch). Run from the repo root,
# in the background:
#   bash slurm/dlevels_watch.sh d1_prim50_hf d3_prim800_hf
# Every $WATCH_EVERY seconds (default 300), for each run:
#   * prints one status line: job state and node, last.pt step, best step, latest source-val key;
#   * if the run has no DONE and no queued or running job named kr-<run>, resubmits slurm/joint.sbatch with the
#     standard partition, GPU constraint and exclude list. The trainer resumes from $GETKITE_RUNS/<run>/last.pt.
#     Slurm also requeues preempted jobs itself (--requeue); this catches cancellations, node failures and crashes.
# A node whose job died without progress is added to the exclude list for that run's later submissions.
# The watcher gives up on a run after 3 resubmissions without checkpoint progress, and exits once every run is DONE.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
source slurm/env.sh
EXCL=${DLEVELS_EXCLUDE:-$GETKITE_EXCLUDE}
EVERY=${WATCH_EVERY:-300}
LOG=slurm/logs/dlevels_watch.log
declare -A PREV NOPROG
step_of() { "$GETKITE_PY" -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['step'])" "$1" 2>/dev/null || echo 0; }
while true; do
  all_done=1
  for RUN in "$@"; do
    OUT=$GETKITE_RUNS/$RUN; CFG=configs/decoder/scratch/$RUN.json
    if [ -f "$OUT/DONE" ]; then echo "$(date -Is) $RUN DONE $(cat "$OUT/DONE")" >> "$LOG"; continue; fi
    all_done=0
    J=$(squeue -h -u "$USER" -n "kr-$RUN" -o "%i %T %N" | head -1)
    S=$(step_of "$OUT/last.pt")
    V=$(tail -n 200 "$OUT/log.jsonl" 2>/dev/null | grep '"eval"' | tail -1 | "$GETKITE_PY" -c "import sys,json; l=sys.stdin.read().strip(); d=json.loads(l) if l else {}; print(d.get('step','-'), {k: round(v,3) for k,v in d.get('eval',{}).items() if k in ('all/model/contact_mm','prim/model/contact_mm','robot_macro/model/contact_mm')})" 2>/dev/null)
    echo "$(date -Is) $RUN job=[${J:-none}] last_step=$S eval=[$V]" >> "$LOG"
    if [ -z "$J" ]; then
      if [ "$S" = "${PREV[$RUN]:-x}" ]; then NOPROG[$RUN]=$(( ${NOPROG[$RUN]:-0} + 1 )); else NOPROG[$RUN]=0; fi
      PREV[$RUN]=$S
      if [ "${NOPROG[$RUN]}" -ge 3 ]; then echo "$(date -Is) $RUN: 3 resubmissions without progress; giving up" >> "$LOG"; continue; fi
      LASTNODE=$(sacct -n -X --name "kr-$RUN" -o NodeList --starttime now-1days 2>/dev/null | awk 'NF' | tail -1 | tr -d ' ')
      if [ "${NOPROG[$RUN]}" -ge 1 ] && [ -n "$LASTNODE" ] && [ "$LASTNODE" != "None" ]; then EXCL="$EXCL,$LASTNODE"; fi
      JID=$(sbatch --parsable -p default_partition --constraint="ampere|ada|2080ti" --exclude="$EXCL" -J "kr-$RUN" slurm/joint.sbatch "$CFG" 2>&1)
      echo "$(date -Is) $RUN: no job, not DONE (last.pt step $S); resubmitted -> $JID" >> "$LOG"
    fi
  done
  [ "$all_done" = 1 ] && { echo "$(date -Is) all runs DONE; watcher exits" >> "$LOG"; exit 0; }
  sleep "$EVERY"
done
