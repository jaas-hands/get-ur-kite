"""Where KITE, the datasets and the run outputs live.

Nothing large is stored in this repo. Configs write paths with these placeholders, expanded at load time
(environment variable first, then the default):

  ${KITE_ROOT}        KITE checkout with its downloaded assets/ and checkpoints/   default: <repo>/third_party/KITE
                      (or ../KITE when the submodule is not initialised)
  ${GETKITE_DATASET}  get-ur-kite-dataset's data/ (hands/, examples/, splits.json) default: kite_data.config.DATA_DIR
  ${GETKITE_DATA}     data generated here (e.g. the Allegro/Barrett pilot data)    default: <repo>/../getkite_data
  ${GETKITE_RUNS}     training runs: checkpoints, logs, results                    default: <repo>/../getkite_runs

To move data or runs (e.g. to a synced Google Drive folder), move the folder and point the variable at it.
`python -m getkite.decoder.paths` prints the resolved locations.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _dataset_default() -> str:
    try:
        from kite_data import config as kd_config
        return str(kd_config.DATA_DIR)
    except ImportError:
        return str(REPO.parent / "get-ur-kite-dataset" / "data")


def _kite_default() -> str:
    """The repo's KITE submodule once initialised, else a sibling ../KITE checkout if there is one."""
    sub, sib = REPO / "third_party" / "KITE", REPO.parent / "KITE"
    return str(sib if not (sub / "networks").is_dir() and (sib / "networks").is_dir() else sub)


DEFAULTS = {
    "KITE_ROOT": lambda: _kite_default(),
    "GETKITE_DATASET": _dataset_default,
    "GETKITE_DATA": lambda: str(REPO.parent / "getkite_data"),
    "GETKITE_RUNS": lambda: str(REPO.parent / "getkite_runs"),
}


def get(name: str) -> str:
    v = os.environ.get(name)
    return os.path.expanduser(v) if v else DEFAULTS[name]()


_VAR = re.compile(r"\$\{(" + "|".join(DEFAULTS) + r")\}")


def expand(obj):
    """Replace ${KITE_ROOT} / ${GETKITE_*} in every string of a (nested) config, and expand ~."""
    if isinstance(obj, str):
        s = _VAR.sub(lambda m: get(m.group(1)), obj)
        return os.path.expanduser(s) if s.startswith("~") else s
    if isinstance(obj, dict):
        return {k: expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand(v) for v in obj]
    return obj


def kite_ckpt() -> str:
    return str(Path(get("KITE_ROOT")) / "checkpoints" / "decoder" / "release.pt")


if __name__ == "__main__":
    for k in DEFAULTS:
        p = get(k)
        print(f"{k:16s} {p}  {'(exists)' if Path(p).exists() else '(missing)'}")
