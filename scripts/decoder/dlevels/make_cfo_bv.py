"""Combined run (approved 2026-11-25, conditional on CF-O passing its launch gate): CF-O + B-scale body variety.
= cf_bv_long_hf* (CF2 with 414 LEAP hands x 48 records, 800 generated bodies x 25; same record budget) + canonical_origin.
Only canonical_origin / out_dir / _note change against cf_bv_long_hf* (asserted).

    python scripts/decoder/dlevels/make_cfo_bv.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for sfx in ("", "_s1", "_s2"):
    src = json.loads((REPO / f"configs/decoder/scratch/cf_bv_long_hf{sfx}.json").read_text())
    c = dict(src); c["canonical_origin"] = True; c["out_dir"] = f"${{GETKITE_RUNS}}/cfo_bv_long_hf{sfx}"
    c["_note"] = (f"CF-O + body variety (2026-11-25): cf_bv_long_hf{sfx} + canonical_origin. Launched only if CF-O passes "
                  "scripts/decoder/benchmark/cfo_gate.py. docs/canonical_origin_proposal.md. Made by scripts/decoder/dlevels/make_cfo_bv.py.")
    ch = sorted(k for k in set(src) | set(c) if src.get(k) != c.get(k)); assert ch == ["_note", "canonical_origin", "out_dir"], ch
    (REPO / f"configs/decoder/scratch/cfo_bv_long_hf{sfx}.json").write_text(json.dumps(c, indent=1)); print("cfo_bv_long_hf" + (sfx or "_s0"), ch)
