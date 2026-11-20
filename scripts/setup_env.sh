#!/usr/bin/env bash
# Create (or refill) the getyourkite conda environment.
#
#   bash scripts/setup_env.sh                # environment only
#   bash scripts/setup_env.sh --kite-assets  # + KITE's assets and LEAP's files for KITE (what B0 training needs)
#   bash scripts/setup_env.sh --data         # + also LEAP's surface template in KITE and the LEAP dataset
#                                            #   (~1 GB, ~15 min, ~10 GB RAM)
#   bash scripts/setup_env.sh --install-cuda --kite-assets
#                                            # no CUDA toolkit on the machine (e.g. the cluster): first install
#                                            # CUDA 12.8 with conda into its own environment (cuda128, ~5.4 GB)
#
# Needs: conda, git submodules checked out (git submodule update --init) and CUDA Toolkit 12.8 (nvcc, for
# the pytorch3d extension); a GPU is not needed to build. Environment variables:
#   CUDA_HOME             toolkit root (default: /usr/local/cuda-12.8, else the nvcc on PATH, else the
#                         conda environment cuda128 from --install-cuda)
#   TORCH_CUDA_ARCH_LIST  GPU architectures pytorch3d is compiled for (default: 2080 Ti / T4 up to
#                         Blackwell, "7.5;8.0;8.6;8.9;9.0;12.0+PTX", so one build runs on every node type)
#   ENV_NAME              environment name (default getyourkite); MAX_JOBS  build parallelism (default 4)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${ENV_NAME:-getyourkite}"
CUDA_ENV="${CUDA_ENV:-cuda128}"
WITH_DATA=0; WITH_KITE_ASSETS=0; INSTALL_CUDA=0
for arg in "$@"; do
  case "$arg" in
    --data) WITH_DATA=1; WITH_KITE_ASSETS=1 ;;
    --kite-assets) WITH_KITE_ASSETS=1 ;;
    --install-cuda) INSTALL_CUDA=1 ;;
    *) echo "unknown option $arg: use --kite-assets, --data, --install-cuda" >&2; exit 2 ;;
  esac
done
CONDA_BASE="$(conda info --base)"
if [ "$INSTALL_CUDA" = 1 ] && [ ! -x "$CONDA_BASE/envs/$CUDA_ENV/bin/nvcc" ]; then
  # NVIDIA's channel for the toolkit; conda-forge for the base libraries it depends on.
  echo "== CUDA 12.8 toolkit with conda into $CUDA_ENV (~5.4 GB, ~6 min)"
  conda create -y -q -n "$CUDA_ENV" -c nvidia/label/cuda-12.8.1 -c conda-forge --override-channels cuda-toolkit
fi
if [ -z "${CUDA_HOME:-}" ]; then
  CUDA_HOME=/usr/local/cuda-12.8
  if [ ! -x "$CUDA_HOME/bin/nvcc" ] && command -v nvcc >/dev/null; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  fi
  if [ ! -x "$CUDA_HOME/bin/nvcc" ] && [ -x "$CONDA_BASE/envs/$CUDA_ENV/bin/nvcc" ]; then
    CUDA_HOME="$CONDA_BASE/envs/$CUDA_ENV"
  fi
fi
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0;12.0+PTX}"
MAX_JOBS="${MAX_JOBS:-4}"

cd "$ROOT"
for sub in third_party/KITE third_party/get-ur-kite-dataset; do
  if [ ! -f "$sub/README.md" ]; then
    echo "missing $sub: run  git submodule update --init" >&2
    exit 1
  fi
done
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "no CUDA toolkit found (nvcc): rerun with --install-cuda, or set CUDA_HOME to a CUDA 12.8 toolkit" >&2
  exit 1
fi
NVCC_VERSION="$("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
echo "== CUDA $NVCC_VERSION in $CUDA_HOME; pytorch3d for architectures $TORCH_CUDA_ARCH_LIST"
case "$NVCC_VERSION" in
  12.8*|12.9*) ;;
  *) echo "warning: torch is built for CUDA 12.8; nvcc $NVCC_VERSION may fail to build pytorch3d" >&2 ;;
esac

# conda's CUDA packages keep headers and libraries under targets/<arch>/ (no include/, no lib64/), where
# torch's extension builder does not look; point the compiler there.
CUDA_INC="$CUDA_HOME/include"
if [ ! -f "$CUDA_INC/cuda_runtime.h" ] && [ -f "$CUDA_HOME/targets/x86_64-linux/include/cuda_runtime.h" ]; then
  CUDA_INC="$CUDA_HOME/targets/x86_64-linux/include"
  export CPATH="$CUDA_INC${CPATH:+:$CPATH}"
  export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi
export PATH="$CUDA_HOME/bin:$PATH"
echo "== CUDA headers in $CUDA_INC"

if [ -d "$CONDA_BASE/envs/$ENV_NAME" ]; then
  echo "== $ENV_NAME exists: updating its packages"
else
  echo "== creating $ENV_NAME"
  conda env create -q -n "$ENV_NAME" -f environment.yml
fi
PY="$CONDA_BASE/envs/$ENV_NAME/bin/python"

echo "== pinned packages"
"$PY" -m pip install -q --no-deps -r requirements.txt

echo "== pytorch3d (CUDA extension, takes a while; FORCE_CUDA builds it without a GPU on this node)"
if ! "$PY" -c "import pytorch3d._C" 2>/dev/null; then
  CUDA_HOME="$CUDA_HOME" CUB_HOME="$CUDA_INC" MAX_JOBS="$MAX_JOBS" FORCE_CUDA=1 \
    "$PY" -m pip install -q --no-deps --no-build-isolation ./third_party/KITE/third_party/pytorch3d
fi

echo "== getkite and kite-data (editable)"
"$PY" -m pip install -q --no-deps --no-build-isolation -e . -e ./third_party/get-ur-kite-dataset

if [ "$WITH_KITE_ASSETS" = 1 ]; then
  echo "== KITE assets"
  (cd third_party/KITE && "$PY" -m pip install -q -r requirements-download.txt && "$PY" download_assets.py)
fi
if [ "$WITH_DATA" = 1 ]; then
  echo "== LEAP in KITE's assets (DRO-Grasp URDF; surface template ~5 min, ~10 GB RAM)"
  "$PY" scripts/install_kite_leap.py
elif [ "$WITH_KITE_ASSETS" = 1 ]; then
  echo "== LEAP in KITE's assets (files only; the first training job builds the surface template)"
  "$PY" scripts/install_kite_leap.py --no-template
fi
if [ "$WITH_DATA" = 1 ]; then
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
