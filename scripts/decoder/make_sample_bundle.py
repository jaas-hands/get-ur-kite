"""Build a small, self-describing sample of the LEAP-family data for testing/eval code.

    source slurm/env.sh   # GETKITE_DATASET (get-ur-kite-dataset/data), GETKITE_DATA (pilot data, output)
    python scripts/decoder/make_sample_bundle.py [--rows 32] [--out $GETKITE_DATA/samples/leap_sample_v1]

One hand per split plus the 16-DoF training hand (7 hands): its hand assets as getkite.decoder reads them (hand.json, template.npz, byte-identical
to data/hands/) and the first --rows examples of its kite_code npz (same keys, dtypes and scalar metadata as the
full files, so kite_data.dataset.load_split(..., root=<bundle>/examples/kite_code) works unchanged). Plus a
split manifest restricted to these hands, the full splits.json, the generator summary and the Allegro/Barrett
pilot slices.
"""
import argparse
import os
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("GETKITE_DATASET", REPO.parent / "get-ur-kite-dataset" / "data"))
DS = DATA.parent
GDATA = Path(os.environ.get("GETKITE_DATA", REPO.parent / "getkite_data"))
PILOT = GDATA / "multifamily_pilot"
HANDS = {  # first hand of each split in splits.json, plus the full 16-DoF training hand
    "train": ["I0-M2-R0-T2_g0", "I4-M4-R4-T4_g0"],
    "val": ["I0-M2-R0-T4_g0"],
    "test_finger_subset": ["I0-M0-R2-T2_g0"],
    "test_combined": ["I0-M0-R2-T2_g1"],
    "test_joint_count": ["I3-M0-R0-T2_g0"],
    "test_geometry": ["I0-M2-R2-T0_g7"],
}
META = ("hand_id", "split", "regime", "seed", "kite_data")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--out", default=str(GDATA / "samples" / "leap_sample_v1"))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    splits = json.loads((DATA / "splits.json").read_text())
    files = {}

    for split, hid in ((s, h) for s, hs in HANDS.items() for h in hs):
        assert hid in splits["hands"][split], (split, hid)
        hd = out / "hands" / hid
        hd.mkdir(parents=True)
        for f in ("hand.json", "template.npz"):  # what getkite.decoder reads; URDF/MJCF are generator-only
            shutil.copy2(DATA / "hands" / hid / f, hd / f)
        z = np.load(DATA / "examples" / "kite_code" / split / f"{hid}.npz")
        ex = {k: (z[k] if k in META else z[k][: a.rows]) for k in z.files}
        p = out / "examples" / "kite_code" / split / f"{hid}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, **ex)
        files[hid] = {"split": split, "n_dof": int(z["joints_goal"].shape[1]), "rows": a.rows,
                      "rows_in_full_file": int(z["joints_goal"].shape[0]),
                      "full_file_sha256_16": sha(DATA / "examples" / "kite_code" / split / f"{hid}.npz")}

    manifest = {k: splits[k] for k in ("version", "seed", "rules")}
    manifest["hands"] = HANDS
    manifest["full_counts"] = {"topologies": {k: len(v) if isinstance(v, list) else v
                                              for k, v in splits["topologies"].items()},
                               "hands": {k: len(v) for k, v in splits["hands"].items()}}
    manifest["sample"] = {
        "regime": "kite_code", "rows_per_hand": a.rows, "files": files,
        "source": {"repo": "jaas-hands/get-ur-kite-dataset",
                   "branch": subprocess.run(["git", "-C", str(DS), "branch", "--show-current"],
                                            capture_output=True, text=True).stdout.strip(),
                   "commit": subprocess.run(["git", "-C", str(DS), "rev-parse", "--short", "HEAD"],
                                            capture_output=True, text=True).stdout.strip(),
                   "splits_json_sha256_16": sha(DATA / "splits.json")},
    }
    (out / "splits.sample.json").write_text(json.dumps(manifest, indent=1) + "\n")
    shutil.copy2(DATA / "splits.json", out / "splits.full.json")
    shutil.copy2(DATA / "examples" / "kite_code" / "summary.json", out / "examples" / "kite_code" / "summary.json")

    pd = out / "multifamily_pilot"
    pd.mkdir()
    for f in ("allegro_right_val.npz", "barrett_val.npz"):
        z = np.load(PILOT / f)
        np.savez(pd / f, **{k: (z[k] if k == "meta" else z[k][: a.rows]) for k in z.files})
    shutil.copy2(PILOT / "manifest.json", pd / "manifest.json")
    shutil.copy2(Path(__file__).with_name("leap_sample_README.md"), out / "README.md")
    print(out)


if __name__ == "__main__":
    main()
