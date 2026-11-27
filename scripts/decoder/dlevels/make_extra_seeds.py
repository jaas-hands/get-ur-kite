"""Seeds 3 and 4 for D1-bv and CF2 (2026-11-25, user go-ahead): to test whether a 5-seed inference ensemble beats the 3-seed one.
Only seed / out_dir / _note change against the seed-2 config (asserted).

    python scripts/decoder/dlevels/make_extra_seeds.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for base in ("cf2_long_hf", "d1_bv_long_hf"):
    src = json.loads((REPO / f"configs/decoder/scratch/{base}_s2.json").read_text())
    for s in (3, 4):
        c = dict(src); c["seed"] = s; c["out_dir"] = f"${{GETKITE_RUNS}}/{base}_s{s}"
        c["_note"] = f"{base} seed {s} (2026-11-25): extra seed for the 5-seed ensemble test (docs/variety_and_ensemble_checks.md §3). Made by scripts/decoder/dlevels/make_extra_seeds.py."
        ch = sorted(k for k in set(src) | set(c) if src.get(k) != c.get(k)); assert ch == ["_note", "out_dir", "seed"], ch
        (REPO / f"configs/decoder/scratch/{base}_s{s}.json").write_text(json.dumps(c, indent=1)); print(f"{base}_s{s}", ch)
