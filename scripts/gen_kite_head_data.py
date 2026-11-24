"""Examples for the KITE-head baselines, generated with the team dataset's own code (kite_data, `kite` regime).

Three sets, each written as <out>/<set>/{train,val}/<hand>.npz (LeapSplit's `root=` layout):
  nominal  B0-ds: the nominal hand, --nominal-train train + 2,000 val examples
  b3       per B3 hand: --b3-train train + --b3-val val examples (separate from the hand's official test
           examples, which stay the test set)
  b4       per B4 hand: --b4-train train + --b3-val val examples
The B3/B4 hands are picked here and listed in <out>/b3_hands.txt and <out>/b4_hands.txt ("split hand" per
line; the Slurm array tasks read their line). Seeds differ from the dataset's, so no example repeats a
test example. Chunks run in parallel; a set whose files exist is skipped.

    python scripts/gen_kite_head_data.py --out $GETKITE_DATA/leap_head --workers 32
    python scripts/gen_kite_head_data.py --out DIR --sets b3 --b3-train 16000      # one set only
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from getkite.leap.data import split_hand_ids
from getkite.leap.paths import hands_dir

NOMINAL = "I4-M4-R4-T4_g0"
SPLITS = ("test_finger_subset", "test_joint_count", "test_geometry", "test_combined")
CHUNK = 25_000


def pick_hands(per_split: int) -> list[tuple[str, str]]:
    """`per_split` hands of each test split, spread over its sorted list; the first is the one the laptop B3
    runs used (Nov 24), with the middle-finger-removed hand for test_finger_subset."""
    out = []
    for s in SPLITS:
        hands = split_hand_ids(s)
        first = "I4-M0-R4-T4_g0" if s == "test_finger_subset" and "I4-M0-R4-T4_g0" in hands else hands[0]
        rest = [h for h in hands if h != first]
        picks = [first] + [rest[int(i)] for i in np.linspace(0, len(rest) - 1, per_split)[1:].round()] if per_split > 1 else [first]
        out += [(s, h) for h in dict.fromkeys(picks)]
    return out


def _job(hand: str, tag: str, n: int, seed: int, tmp: Path):
    from kite_data.generate import generate_one

    return generate_one((hand, tag, n, seed, "kite", str(hands_dir()), str(tmp)))


def generate(pool, hand: str, n: int, seed: int, dest: Path, tag: str) -> None:
    """n examples of `hand` -> dest (one .npz), in chunks of CHUNK generated in parallel."""
    if dest.is_file():
        print(f"  {dest} exists, skipped", flush=True)
        return
    tmp = dest.parent / f".tmp_{tag}_{hand}"
    sizes = [CHUNK] * (n // CHUNK) + ([n % CHUNK] if n % CHUNK else [])
    futs = [pool.submit(_job, hand, f"{tag}{i}", m, seed, tmp) for i, m in enumerate(sizes)]
    for f in futs:
        f.result()
    parts = [dict(np.load(tmp / f"{tag}{i}" / f"{hand}.npz")) for i in range(len(sizes))]
    merged = {k: (np.concatenate([p[k] for p in parts]) if parts[0][k].ndim else parts[0][k]) for k in parts[0]}
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dest.with_suffix(".tmp.npz"), **merged)
    os.replace(dest.with_suffix(".tmp.npz"), dest)
    shutil.rmtree(tmp)
    print(f"  {dest}: {len(merged['k'])} examples", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sets", nargs="+", default=["nominal", "b3", "b4"], choices=["nominal", "b3", "b4"])
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)))
    ap.add_argument("--nominal-train", type=int, default=1_000_000)
    ap.add_argument("--b3-per-split", type=int, default=3)
    ap.add_argument("--b3-train", type=int, default=16_000, help="B3 uses the first 1k / 4k / 16k of these")
    ap.add_argument("--b3-val", type=int, default=500)
    ap.add_argument("--b4-train", type=int, default=200_000)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    b3 = pick_hands(args.b3_per_split)
    b4 = pick_hands(1)
    (args.out / "b3_hands.txt").write_text("".join(f"{s} {h}\n" for s, h in b3))
    (args.out / "b4_hands.txt").write_text("".join(f"{s} {h}\n" for s, h in b4))
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        if "nominal" in args.sets:
            print(f"[gen] nominal: {args.nominal_train} train + 2000 val ({args.workers} workers)", flush=True)
            generate(pool, NOMINAL, 2000, 2027, args.out / "nominal" / "val" / f"{NOMINAL}.npz", "val")
            generate(pool, NOMINAL, args.nominal_train, 2026, args.out / "nominal" / "train" / f"{NOMINAL}.npz", "train")
        for name, hands, n in (("b3", b3, args.b3_train), ("b4", b4, args.b4_train)):
            if name not in args.sets:
                continue
            print(f"[gen] {name}: {len(hands)} hands, {n} train + {args.b3_val} val each", flush=True)
            for _, h in hands:
                generate(pool, h, args.b3_val, 3031, args.out / name / "val" / f"{h}.npz", "val")
                generate(pool, h, n, 3030, args.out / name / "train" / f"{h}.npz", "train")
    print(f"[gen] done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
