#!/usr/bin/env bash
# Create (or refill) the getyourkite conda environment.
#
#   bash scripts/setup_env.sh              # environment only
#   bash scripts/setup_env.sh --data       # also KITE's assets and the LEAP dataset (~1 GB, ~10 min)
#
# Needs: conda, git submodules checked out (git submodule update --init), an NVIDIA GPU and
# CUDA Toolkit 12.8 (for the pytorch3d extension). Set CUDA_HOME if it is not /usr/local/cuda-12.8,
# ENV_NAME for another environment name, MAX_JOBS to change pytorch3d's build parallelism.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${ENV_NAME:-getyourkite}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
MAX_JOBS="${MAX_JOBS:-4}"
WITH_DATA=0
[ "${1:-}" = "--data" ] && WITH_DATA=1

cd "$ROOT"
for sub in third_party/KITE third_party/get-ur-kite-dataset; do
  if [ ! -f "$sub/README.md" ]; then
    echo "missing $sub: run  git submodule update --init" >&2
    exit 1
  fi
done
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "no nvcc in $CUDA_HOME: install CUDA Toolkit 12.8 or set CUDA_HOME" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
if [ -d "$CONDA_BASE/envs/$ENV_NAME" ]; then
  echo "== $ENV_NAME exists: updating its packages"
else
  echo "== creating $ENV_NAME"
  conda env create -q -n "$ENV_NAME" -f environment.yml
fi
PY="$CONDA_BASE/envs/$ENV_NAME/bin/python"

echo "== pinned packages"
"$PY" -m pip install -q --no-deps -r requirements.txt

echo "== pytorch3d (CUDA extension, takes a while)"
if ! "$PY" -c "import pytorch3d" 2>/dev/null; then
  CUDA_HOME="$CUDA_HOME" CUB_HOME="$CUDA_HOME/include" MAX_JOBS="$MAX_JOBS" \
    "$PY" -m pip install -q --no-deps --no-build-isolation ./third_party/KITE/third_party/pytorch3d
fi

echo "== getkite and kite-data (editable)"
"$PY" -m pip install -q --no-deps --no-build-isolation -e . -e ./third_party/get-ur-kite-dataset

if [ "$WITH_DATA" = 1 ]; then
  echo "== KITE assets"
  (cd third_party/KITE && "$PY" -m pip install -q -r requirements-download.txt && "$PY" download_assets.py)
  echo "== LEAP dataset"
  (cd third_party/get-ur-kite-dataset && ./scripts/fetch_assets.sh \
    && "$PY" -m kite_data.build_hands && "$PY" -m kite_data.splits && "$PY" -m kite_data.generate)
fi

echo "== check"
MUJOCO_GL=egl "$PY" - <<'EOF'
import torch, mujoco, pytorch3d, pytorch_kinematics, kite_data, getkite
print(f"torch {torch.__version__} (CUDA available: {torch.cuda.is_available()}), mujoco {mujoco.__version__}, "
      f"pytorch3d {pytorch3d.__version__}, getkite {getkite.__version__}")
EOF
echo "done: conda activate $ENV_NAME"
