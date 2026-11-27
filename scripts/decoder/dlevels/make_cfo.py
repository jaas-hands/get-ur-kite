"""CF-O (docs/canonical_origin_proposal.md, Step 2): CF2 (cf2_long_hf*, 100k steps, A data) plus one change, canonical base
origin (rule O1, palm-point bounding-box centre along the canonical axes).

    python scripts/decoder/dlevels/make_cfo.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for sfx in ("", "_s1", "_s2"):
    src = json.loads((REPO / f"configs/decoder/scratch/cf2_long_hf{sfx}.json").read_text())
    c = dict(src)
    c["canonical_origin"] = True
    c["out_dir"] = f"${{GETKITE_RUNS}}/cfo_long_hf{sfx}"
    c["_note"] = (f"CF-O (2026-11-25): cf2_long_hf{sfx} + canonical_origin (rule O1). docs/canonical_origin_proposal.md. "
                  "Made by scripts/decoder/dlevels/make_cfo.py.")
    (REPO / f"configs/decoder/scratch/cfo_long_hf{sfx}.json").write_text(json.dumps(c, indent=1))
    print("cfo_long_hf" + (sfx or "_s0"), "changed:", sorted(k for k in set(src) | set(c) if src.get(k) != c.get(k)))
