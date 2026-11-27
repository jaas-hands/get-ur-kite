#!/bin/bash
# Download checkpoints listed in checkpoints/manifest.tsv from Google Drive to ${GETKITE_RUNS}/<run_dir>/best.pt (where the
# configs and benchmarks expect them) and verify each SHA256. Checkpoints are never stored in git (*.pt is gitignored).
# Run from the repo root (GETKITE_RUNS comes from slurm/env.sh / slurm/env.local.sh):
#     bash scripts/fetch_checkpoints.sh                    # every row
#     bash scripts/fetch_checkpoints.sh cf3e_long_hf       # just the CF-3e delivery model (seed 0)
#     bash scripts/fetch_checkpoints.sh 'cf3e_*'           # shell-style pattern on run_dir
# Uses gdown when installed (pip install gdown), otherwise curl. Drive files must be shared as "anyone with the link".
# Branches add rows for their own runs; each run has its own folder, so rows never collide.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source slurm/env.sh >/dev/null
: "${GETKITE_RUNS:?GETKITE_RUNS not set}"
PATS=("$@"); [ $# -eq 0 ] && PATS=("*")
n=0
while IFS=$'\t' read -r run id sha note; do
  [[ -z "$run" || "$run" == \#* ]] && continue
  hit=0; for p in "${PATS[@]}"; do [[ "$run" == $p ]] && hit=1; done; [ $hit = 1 ] || continue
  n=$((n + 1)); out="$GETKITE_RUNS/$run/best.pt"
  if [ -f "$out" ] && echo "$sha  $out" | sha256sum -c --quiet - 2>/dev/null; then echo "$run: present, verified"; continue; fi
  [[ "$id" == REPLACE_* ]] && { echo "$run: Drive file id not filled in checkpoints/manifest.tsv" >&2; exit 1; }
  mkdir -p "$(dirname "$out")"; echo "$run: downloading ($note)"
  if command -v gdown >/dev/null; then gdown --quiet "$id" -O "$out.part"
  else curl -fsSL "https://drive.usercontent.google.com/download?id=$id&export=download&confirm=t" -o "$out.part"; fi
  echo "$sha  $out.part" | sha256sum -c --quiet - || { echo "$run: SHA256 mismatch, keeping $out.part for inspection" >&2; exit 1; }
  mv "$out.part" "$out"; echo "$run: verified -> $out"
done < checkpoints/manifest.tsv
[ $n -gt 0 ] || { echo "no manifest rows match: ${PATS[*]}" >&2; exit 1; }
