#!/bin/bash
# Long-arm mid-training snapshots (user-approved 2026-11-27; evaluation only, no training change). For each target step K in
# 175k / 225k / 275k and each seed, once the run's log passes K, copy its last.pt (saved every 1000 steps, atomic replace) to
# ${GETKITE_RUNS}/cf3f_long_snap${K}k_hf[_sN]/best.pt and record the checkpoint's step. When all 3 seeds have a snapshot at K,
# submit the external-hands evaluation (9 unseen-family hands incl. Wuji / Shadow) for cf3e + that snapshot.
# Run from the repo root: setsid nohup bash slurm/cf3f_long_snap.sh > /dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
source slurm/env.sh
LOG=slurm/logs/cf3f_long_snap.log
EXCL=${GETKITE_EXCLUDE_EVAL:-$GETKITE_EXCLUDE}
sfx() { [ "$1" = 0 ] || echo "_s$1"; }
declare -A SUB
echo "$(date -Is) snapshot watcher start" >> "$LOG"
while :; do
  left=0
  for K in 175 225 275; do
    [ -n "${SUB[$K]:-}" ] && continue
    left=1; n=0
    for s in 0 1 2; do
      src=$GETKITE_RUNS/cf3f_long_hf$(sfx $s); dst=$GETKITE_RUNS/cf3f_long_snap${K}k_hf$(sfx $s)
      if [ -f "$dst/best.pt" ]; then n=$((n + 1)); continue; fi
      st=$(tail -n 1 "$src/log.jsonl" | "$GETKITE_PY" -c "import json,sys; print(json.loads(sys.stdin.read())['step'])" 2>/dev/null || echo 0)
      if [ "$st" -ge $((K * 1000)) ]; then
        mkdir -p "$dst"; cp "$src/last.pt" "$dst/best.pt.tmp" && mv "$dst/best.pt.tmp" "$dst/best.pt"
        ck=$("$GETKITE_PY" -c "import torch; print(torch.load('$dst/best.pt', map_location='cpu', weights_only=False)['step'])")
        echo "{\"source\": \"cf3f_long_hf$(sfx $s)/last.pt\", \"ckpt_step\": $ck, \"copied\": \"$(date -Is)\"}" > "$dst/SNAPSHOT.json"
        echo "$(date -Is) snap ${K}k s$s: ckpt step $ck" >> "$LOG"; n=$((n + 1))
      fi
    done
    if [ "$n" = 3 ]; then
      j=$(sbatch --parsable -p default_partition --constraint="ampere|ada|2080ti" --exclude="$EXCL" -J "kr-ext-long${K}k" -t 02:00:00 --mem=48G \
          slurm/external_hands_eval.sbatch "\${GETKITE_RUNS}/external_hands_cf3flong${K}k_v1" cf3e "cf3flong${K}k")
      echo "$(date -Is) ${K}k: all 3 seeds snapped; external hands eval [$j]" >> "$LOG"; SUB[$K]=$j
    fi
  done
  [ "$left" = 0 ] && break
  sleep 120
done
echo "$(date -Is) snapshot watcher end" >> "$LOG"
