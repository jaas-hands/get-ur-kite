#!/bin/bash
# Keeps the runs listed in slurm/watch_runs.txt alive while nobody is online. Run it from cron, e.g.
#   */15 * * * * /path/to/get-ur-kite/slurm/watchdog.sh >> /path/to/get-ur-kite/slurm/logs/watchdog.cron.log 2>&1
# For each run with no DONE marker and no queued/running job named kr-<run>, it resubmits slurm/train.sbatch,
# which resumes from $GETKITE_RUNS/<run>/last.pt. Slurm requeues preempted jobs itself; this catches time limits,
# node failures, cancellations and crashes. It gives up on a run (writes <run>/WATCHDOG_GAVE_UP) after 3
# resubmissions in a row without checkpoint progress, so a real bug cannot loop forever. A finished run that has
# a FINAL_EVAL_WANTED file gets one full evaluation of best.pt (slurm/final_eval.sbatch), at most 3 attempts.
# Extra sbatch options for every submission (partition, constraint, excludes): WATCHDOG_SBATCH_ARGS in env.local.sh.
export PATH=/usr/local/slurm/current/bin:/usr/bin:/bin:$PATH
export USER=${USER:-$(id -un)}  # cron does not set USER; without it squeue -u "" sees no jobs
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
source slurm/env.sh
SBATCH=${WATCHDOG_SBATCH:-sbatch}  # tests: WATCHDOG_SBATCH=echo
read -r -a SARGS <<< "${WATCHDOG_SBATCH_ARGS:-}"
LIST=${WATCHDOG_LIST:-slurm/watch_runs.txt}
LOG=${WATCHDOG_LOG:-slurm/logs/watchdog.log}
[ -f "$LIST" ] || exit 0
exec 9>slurm/logs/.watchdog.lock
flock -n 9 || exit 0
date -Is > slurm/logs/watchdog.heartbeat
ts() { date -Is; }
squeue -h -u "$USER" >/dev/null 2>&1 || { echo "$(ts) squeue unavailable; skipping" >> "$LOG"; exit 0; }
grep -v "^\s*#" "$LIST" | while read -r NAME CFG EXTRA; do
  [ -z "$NAME" ] && continue
  OUT=$GETKITE_RUNS/$NAME; ST=$OUT/.watchdog_state
  if [ -f "$OUT/DONE" ]; then
    if [ -f "$OUT/FINAL_EVAL_WANTED" ] && [ ! -f "$OUT/final_eval.json" ] && [ -f "$OUT/best.pt" ] \
       && [ -z "$(squeue -h -u "$USER" -n "kr-eval-$NAME" -o %i)" ]; then
      N=$(cat "$OUT/.eval_attempts" 2>/dev/null || echo 0)
      if [ "$N" -lt 3 ]; then
        JID=$($SBATCH --parsable "${SARGS[@]}" -J "kr-eval-$NAME" slurm/final_eval.sbatch "$CFG" "$NAME" 2>&1)
        echo $((N + 1)) > "$OUT/.eval_attempts"
        echo "$(ts) $NAME: DONE; final evaluation of best.pt submitted (attempt $((N + 1))) -> $JID" >> "$LOG"
      fi
    fi
    continue
  fi
  [ -f "$OUT/WATCHDOG_GAVE_UP" ] && continue
  [ -n "$(squeue -h -u "$USER" -n "kr-$NAME" -o %i)" ] && continue
  STEP=$("$GETKITE_PY" -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['step'])" "$OUT/last.pt" 2>/dev/null || echo 0)
  read -r PREV NOPROG 2>/dev/null < "$ST" || { PREV=-1; NOPROG=0; }
  if [ "$STEP" = "$PREV" ]; then NOPROG=$((NOPROG + 1)); else NOPROG=0; fi
  if [ "$NOPROG" -ge 3 ]; then
    echo "$(ts) $NAME: 3 resubmissions without progress past step $STEP; giving up" | tee -a "$LOG" > "$OUT/WATCHDOG_GAVE_UP"
    continue
  fi
  mkdir -p "$OUT"
  JID=$($SBATCH --parsable "${SARGS[@]}" -J "kr-$NAME" slurm/train.sbatch "$CFG" "$NAME" $EXTRA 2>&1)
  echo "$STEP $NOPROG" > "$ST"
  echo "$(ts) $NAME: no job and not DONE (last.pt step $STEP); resubmitted -> $JID" >> "$LOG"
done
