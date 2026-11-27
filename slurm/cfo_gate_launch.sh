#!/bin/bash
# Conditional launch (user approval 2026-11-25): after CF-O's evaluation completes, run scripts/decoder/benchmark/cfo_gate.py.
# Only if it says launch=true, submit the combined run (cfo_bv_long_hf*, 3 seeds) and its evaluation chain. Log: slurm/logs/cfo_gate.log
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
source slurm/env.sh
export PYTHONPATH=src:../get-ur-kite-dataset
PY=${GETKITE_PY_VENV:-$GETKITE_PY}
EXCL=$GETKITE_EXCLUDE
LOG=slurm/logs/cfo_gate.log
until grep -qE "evaluation (complete|incomplete)" slurm/logs/cfo_chain.log 2>/dev/null; do sleep 120; done
if grep -q "evaluation incomplete" slurm/logs/cfo_chain.log; then echo "$(date -Is) CF-O evaluation incomplete; NOT launching" >> "$LOG"; exit 0; fi
"$PY" scripts/decoder/benchmark/cfo_gate.py >> "$LOG" 2>&1 || { echo "$(date -Is) gate script failed; NOT launching" >> "$LOG"; exit 0; }
L=$("$PY" -c "import json,os; print(json.load(open(os.environ['GETKITE_RUNS']+'/cfo_gate.json'))['launch'])")
echo "$(date -Is) gate launch=$L" >> "$LOG"
if [ "$L" = "True" ]; then
  for s in "" _s1 _s2; do
    J=$(sbatch --parsable -p default_partition --constraint="ampere|ada|2080ti" --exclude="$EXCL" -J kr-cfo_bv_long_hf$s -t 05:00:00 slurm/joint.sbatch configs/decoder/scratch/cfo_bv_long_hf$s.json)
    echo "$(date -Is) submitted cfo_bv_long_hf$s job=$J" >> "$LOG"
  done
  nohup bash slurm/cfobv_chain.sh > /dev/null 2>&1 &
  echo "$(date -Is) combined evaluation chain started" >> "$LOG"
else
  echo "$(date -Is) CF-O did not pass the launch gate; combined run NOT launched" >> "$LOG"
fi
