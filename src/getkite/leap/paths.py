"""Where the LEAP dataset and KITE's checkpoint live.

The dataset is the ``third_party/get-ur-kite-dataset`` submodule; its generated files are in its
``data/`` folder (``python -m kite_data.build_hands`` etc., or ``scripts/setup_env.sh --data``).
Set ``GETKITE_LEAP_DATA`` to use a ``data/`` folder elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = REPO_ROOT / "third_party" / "get-ur-kite-dataset"
KITE_ROOT = REPO_ROOT / "third_party" / "KITE"


def data_dir() -> Path:
    return Path(os.environ.get("GETKITE_LEAP_DATA", DATASET_ROOT / "data")).expanduser().resolve()


def hands_dir() -> Path:
    return data_dir() / "hands"


def examples_dir(regime: str = "kite") -> Path:
    return data_dir() / "examples" / regime


def splits_file() -> Path:
    return data_dir() / "splits.json"


def kite_checkpoint() -> Path:
    return Path(os.environ.get("GETKITE_KITE_CHECKPOINT", KITE_ROOT / "checkpoints" / "decoder" / "release.pt"))
