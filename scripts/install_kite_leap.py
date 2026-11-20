"""Put the LEAP hand into KITE's assets, where KITE's robot registry expects it.

KITE already registers `leaphand_right` (robot_model/robot_model.py) and ships LEAP's fitted
collision primitives, but not the URDF and meshes. They come from DRO-Grasp's data release
(MIT license, github.com/zhenyuwei2003/DRO-Grasp, release v1.0), the same source as KITE's
Barrett. Only the LEAP files (~40 MB) are read from the 1 GB archive, with HTTP range requests.

    python scripts/install_kite_leap.py            # download + install, then build the surface template
    python scripts/install_kite_leap.py --no-template

Writes into third_party/KITE/assets/dro_urdf_robot/leaphand/ (git-ignored by KITE, like its other
downloaded assets). The surface template build takes ~5 min and ~10 GB RAM, once.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KITE = Path(os.environ.get("KITE_ROOT", REPO / "third_party" / "KITE")).resolve()
URL = "https://github.com/zhenyuwei2003/DRO-Grasp/releases/download/v1.0/data.zip"
MEMBER_PREFIX = "data_urdf/robot/leaphand/"
DEST = KITE / "assets" / "dro_urdf_robot" / "leaphand"


class _HttpFile(io.RawIOBase):
    """Seekable read-only view of a remote file (HTTP range requests)."""

    def __init__(self, url: str):
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.url, self.size = r.url, int(r.headers["Content-Length"])
        self.pos = 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or len(b) == 0:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def download() -> None:
    zf = zipfile.ZipFile(io.BufferedReader(_HttpFile(URL), buffer_size=1 << 20))
    members = [m for m in zf.infolist() if m.filename.startswith(MEMBER_PREFIX) and not m.is_dir()]
    if not members:
        sys.exit(f"no {MEMBER_PREFIX} in {URL}")
    tmp = DEST.with_name("leaphand.partial")
    shutil.rmtree(tmp, ignore_errors=True)
    for m in members:
        out = tmp / m.filename[len(MEMBER_PREFIX):]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(zf.read(m))
    for f in DEST.glob("surface_template_graph_*.pt") if DEST.exists() else []:
        shutil.copy2(f, tmp / f.name)  # keep an already built template
    shutil.rmtree(DEST, ignore_errors=True)
    tmp.rename(DEST)
    print(f"installed {len(members)} files from DRO-Grasp v1.0 into {DEST.relative_to(REPO)}")


def build_template() -> None:
    os.chdir(KITE)
    sys.path.insert(0, str(KITE))
    sys.path.insert(0, str(REPO / "scripts"))
    from train_leap_head import _patch_trimesh_closest_point

    _patch_trimesh_closest_point()
    import torch
    from robot_model.robot_model import create_robot_model

    m = create_robot_model("leaphand_right", device=torch.device("cpu"), num_points=4096)
    print(f"leaphand_right: dof={m.dof}, base translation {m.base_translation_indices}, "
          f"rotation {m.base_rotation_indices}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-template", action="store_true", help="skip building the surface template")
    ap.add_argument("--force", action="store_true", help="download again even if installed")
    args = ap.parse_args()
    if args.force or not (DEST / "leap_hand_right_extended.urdf").is_file():
        download()
    else:
        print(f"already installed: {DEST.relative_to(REPO)}")
    if not args.no_template:
        build_template()


if __name__ == "__main__":
    main()
