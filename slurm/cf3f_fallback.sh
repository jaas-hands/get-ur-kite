#!/bin/bash
# CF-3f GPU fallback (user rule 2026-11-27): the long arm asks for fast GPUs; if a long job has been PENDING > 60 min, or has been
# requeued >= 2 times and is pending again, widen its constraint to "ampere|ada|2080ti". Logs to slurm/logs/cf3f_chain.log.
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
LOG=slurm/logs/cf3f_chain.log
declare -A WIDE
while :; do
  any=0
  for j in $(awk '$2 ~ /cf3f_long/ {print $1}' slurm/logs/cf3f_jobs.txt); do
    st=$(squeue -h -j "$j" -o "%T %V %f" 2>/dev/null); [ -z "$st" ] && continue; any=1
    [ -n "${WIDE[$j]:-}" ] && continue
    read -r state submit feat <<<"$st"
    rc=$(scontrol show job "$j" 2>/dev/null | grep -o 'Restarts=[0-9]*' | cut -d= -f2)
    age=$(( $(date +%s) - $(date -d "$submit" +%s) ))
    if [ "$state" = PENDING ] && { [ "$age" -gt 3600 ] || [ "${rc:-0}" -ge 2 ]; }; then
      if scontrol update job "$j" Features="ampere|ada|2080ti" 2>>"$LOG"; then
        WIDE[$j]=1; echo "$(date -Is) fallback: job $j pending ${age}s restarts=${rc:-0} -> constraint widened to ampere|ada|2080ti" >> "$LOG"
      fi
    fi
  done
  [ "$any" = 0 ] && break
  sleep 300
done
echo "$(date -Is) fallback watcher end" >> "$LOG"
