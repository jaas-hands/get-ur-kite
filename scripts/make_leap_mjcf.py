"""Write floating, actuated MJCFs (KITE's simulator format) for LEAP hands of the dataset.

    python scripts/make_leap_mjcf.py I4-M4-R4-T4_g0                 # the nominal LEAP hand
    python scripts/make_leap_mjcf.py I3-M0-R2-T4_g2 --out outputs/leap_mjcf

Default output: assets/leap/<hand_id>_floating.xml. Needs the dataset's generated hands
(third_party/get-ur-kite-dataset/data/hands). See getkite/leap/mjcf.py for the format.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from getkite.leap.mjcf import write_floating_leap  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("hand_ids", nargs="+")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "assets" / "leap")
    args = p.parse_args()
    import mujoco

    for hid in args.hand_ids:
        path = write_floating_leap(hid, args.out)
        m = mujoco.MjModel.from_xml_path(str(path))
        print(f"{path}: {m.njnt} joints, {m.nu} actuators, {m.ngeom} geoms")


if __name__ == "__main__":
    main()
