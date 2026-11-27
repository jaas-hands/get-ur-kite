# Settings shared by the Slurm scripts. Sourced from the repo root; every value can be overridden by exporting it
# first or by putting the exports in slurm/env.local.sh (gitignored, machine-specific). See docs/decoder.md.
#   GETKITE_PY       python with torch + getkite + kite_data installed     default: python on PATH
#   KITE_ROOT        KITE checkout with assets/ and checkpoints/            default: third_party/KITE
#   GETKITE_DATASET  get-ur-kite-dataset's data/ (hands/, examples/)       default: ../get-ur-kite-dataset/data
#   GETKITE_DATA     data generated for the decoder (pilot data, ...)       default: ../getkite_data
#   GETKITE_RUNS     run outputs (checkpoints, logs, results)               default: ../getkite_runs
#   GETKITE_EXTERNAL third-party hand assets (external hands, Shadow)       default: ../external_assets
#   GETKITE_EXCLUDE  nodes to always avoid, comma-separated                 default: none
GETKITE_REPO=$(pwd)
[ -f "$GETKITE_REPO/slurm/env.local.sh" ] && source "$GETKITE_REPO/slurm/env.local.sh"
export GETKITE_REPO
export GETKITE_PY=${GETKITE_PY:-python}
export KITE_ROOT=${KITE_ROOT:-$("${GETKITE_PY:-python}" -c "import sys; sys.path.insert(0, 'src'); from getkite.decoder import paths; print(paths.get('KITE_ROOT'))" 2>/dev/null || echo "$GETKITE_REPO/third_party/KITE")}
export GETKITE_DATASET=${GETKITE_DATASET:-$(cd "$GETKITE_REPO/.." && pwd)/get-ur-kite-dataset/data}
export GETKITE_DATA=${GETKITE_DATA:-$(cd "$GETKITE_REPO/.." && pwd)/getkite_data}
export GETKITE_RUNS=${GETKITE_RUNS:-$(cd "$GETKITE_REPO/.." && pwd)/getkite_runs}
export GETKITE_EXTERNAL=${GETKITE_EXTERNAL:-$(cd "$GETKITE_REPO/.." && pwd)/external_assets}
export GETKITE_EXCLUDE=${GETKITE_EXCLUDE:-}
export PYTHONPATH=$GETKITE_REPO/src${PYTHONPATH:+:$PYTHONPATH}

# Exits 0 after resubmitting the job elsewhere if this node's GPU is unusable (e.g. a driver too old for the
# installed torch build: torch then silently falls back to CPU). Usage: getkite_require_cuda <sbatch args...>
getkite_require_cuda() {
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
  if ! "$GETKITE_PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    local host orig; host=$(hostname -s)
    # keep the job's own --exclude list (it was being dropped on resubmission, 2026-11-20)
    orig=$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | grep -o 'ExcNodeList=[^ ]*' | cut -d= -f2)
    [ "$orig" = "(null)" ] && orig=""
    export GETKITE_EXCLUDE=${GETKITE_EXCLUDE:-$orig}; export GETKITE_EXCLUDE=${GETKITE_EXCLUDE:+$GETKITE_EXCLUDE,}$host
    # keep the job's own time limit and partition too (dropped before 2026-11-27: resubmits fell back to the 2 h default)
    local tl part; tl=$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | grep -o 'TimeLimit=[^ ]*' | cut -d= -f2)
    part=$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | grep -o ' Partition=[^ ]*' | cut -d= -f2)
    echo "CUDA unusable on $host; resubmitting with --exclude=$GETKITE_EXCLUDE ${tl:+-t $tl} ${part:+-p $part}"
    sbatch --export=ALL --exclude="$GETKITE_EXCLUDE" ${tl:+-t "$tl"} ${part:+-p "$part"} -J "$SLURM_JOB_NAME" "$@"
    exit 0
  fi
}

# Runs a command in the background and forwards SIGTERM/USR1 (preemption, time limit) so the trainer can save
# last.pt before Slurm kills it. Usage: getkite_run_forwarding <command...>
getkite_run_forwarding() {
  "$@" &
  local pid=$! rc
  trap 'echo "got SIGTERM at $(date -Is), forwarding"; kill -TERM '"$pid"' 2>/dev/null' TERM USR1
  set +e
  wait $pid; rc=$?
  while kill -0 $pid 2>/dev/null; do wait $pid; rc=$?; done
  set -e
  echo "exit=$rc end=$(date -Is)"
  return $rc
}
