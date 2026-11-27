#!/bin/bash
# Fetch the third-party hand models used for evaluation only (7 external hands + Shadow) into $GETKITE_EXTERNAL
# (default: ../external_assets next to this repo), at the exact commits our results used. These files are NOT redistributed
# in our repos (SPIDER is CC-BY-NC, MuJoCo Menagerie is Apache-2.0); see docs/external_hands.md. Run from the repo root:
#     bash scripts/decoder/external_hands/fetch_assets.sh
# then build the adapters, templates and evaluation-only records (CPU jobs):
#     sbatch slurm/external_hands.sbatch      # Ability, Inspire, XHand, Schunk, MetaHand, Sharpa Wave, Tetheria Aero
#     sbatch slurm/shadow_records.sbatch      # Shadow (after: python scripts/decoder/shadow/make_shadow_urdf.py)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../../.."
source slurm/env.sh >/dev/null
D=${GETKITE_EXTERNAL:?}; mkdir -p "$D"
SPIDER_COMMIT=1d607af226dcb5aeb8c1013335732ab513fff125
MENAGERIE_COMMIT=370c43965323c0944b433b202aac5058b5f907a9
# fetch <dir> <url> <commit> <sparse paths...>: clone (blobless, sparse) if missing; check out the pinned commit only when needed,
# so an existing copy at the right commit is left exactly as it is.
fetch() {
  local dir=$1 url=$2 commit=$3; shift 3
  if [ ! -d "$dir/.git" ]; then
    git clone --quiet --filter=blob:none --no-checkout "$url" "$dir"
    git -C "$dir" sparse-checkout set "$@"
    git -C "$dir" checkout --quiet "$commit"
  elif [ "$(git -C "$dir" rev-parse HEAD)" != "$commit" ]; then
    git -C "$dir" checkout --quiet "$commit"
  fi
}
fetch "$D/spider_repo" https://github.com/facebookresearch/spider.git "$SPIDER_COMMIT" \
  spider/assets/robots/ability spider/assets/robots/inspire spider/assets/robots/metahand spider/assets/robots/schunk spider/assets/robots/xhand
fetch "$D/mujoco_menagerie" https://github.com/google-deepmind/mujoco_menagerie.git "$MENAGERIE_COMMIT" shadow_hand sharpa_wave tetheria_aero_hand_open
echo "SPIDER    $(git -C "$D/spider_repo" rev-parse --short HEAD)    -> $D/spider_repo ($(git -C "$D/spider_repo" sparse-checkout list | tr '\n' ' '))"
echo "Menagerie $(git -C "$D/mujoco_menagerie" rev-parse --short HEAD)    -> $D/mujoco_menagerie ($(git -C "$D/mujoco_menagerie" sparse-checkout list | tr '\n' ' '))"
