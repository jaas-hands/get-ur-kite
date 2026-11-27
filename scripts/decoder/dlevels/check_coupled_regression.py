"""Regression check for the coupled body draw (2026-11-21; docs/coupling_and_wrist_proposal.md, Phase 1).

A body that drew no coupling must be identical to the source draw: hand.json, hand.xml, template.npz byte for byte,
and every record file array-equal (same generator code path, plan, seed and regime). A coupled body must keep the
source's template.npz byte for byte (coupling does not change the geometry). Writes a JSON summary.

    python scripts/decoder/dlevels/check_coupled_regression.py --src $GETKITE_DATASET/prim --dst $GETKITE_DATASET/prim_c \
        --out configs/decoder/dlevels/prim_c_regression.json
"""
import argparse
import csv
import filecmp
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--out", required=True); a = ap.parse_args()
    S, D = Path(a.src), Path(a.dst)
    rows = list(csv.DictReader(open(D / "hands/index.csv")))
    bad, n_rec_cmp, n_unc, n_cpl = [], 0, 0, 0
    for r in rows:
        h, coupled = r["id"], int(r["n_coupled"]) > 0
        if not filecmp.cmp(S / "hands" / h / "template.npz", D / "hands" / h / "template.npz", shallow=False):
            bad.append(f"{h}: template differs")
        if coupled:
            n_cpl += 1
            continue
        n_unc += 1
        for f in ("hand.json", "hand.xml"):
            if not filecmp.cmp(S / "hands" / h / f, D / "hands" / h / f, shallow=False):
                bad.append(f"{h}: {f} differs")
        for fs in sorted((S / "examples/kite_code").glob(f"*/{h}.npz")):
            fd = D / "examples/kite_code" / fs.parent.name / fs.name
            if not fd.exists():
                bad.append(f"{h}: missing {fs.parent.name}"); continue
            xs, xd = np.load(fs), np.load(fd)
            for k in xs.files:
                if k == "kite_data":
                    continue
                if k not in xd.files or not np.array_equal(xs[k], xd[k]):
                    bad.append(f"{h}/{fs.parent.name}: {k} differs")
            n_rec_cmp += 1
    res = {"pass": not bad, "n_bodies": len(rows), "n_uncoupled": n_unc, "n_coupled": n_cpl,
           "uncoupled_record_files_compared": n_rec_cmp, "problems": bad[:50], "n_problems": len(bad)}
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
