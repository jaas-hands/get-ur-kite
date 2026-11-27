"""CF-J (docs/cf_frame_jitter_proposal.md, Step 1): CF2 (cf2_long_hf*, 100k steps, A data) plus one change, training-time
canonical-frame jitter (k = 32 frames, block 0 = identity, others a random axis by an angle uniform in [0, 15] deg).

    python scripts/decoder/dlevels/make_cfj.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for sfx in ("", "_s1", "_s2"):
    src = json.loads((REPO / f"configs/decoder/scratch/cf2_long_hf{sfx}.json").read_text())
    c = dict(src)
    c["canonical_jitter"] = {"k": 32, "max_deg": 15.0, "seed": 20261125 + int(src.get("seed", 0))}
    c["out_dir"] = f"${{GETKITE_RUNS}}/cfj_long_hf{sfx}"
    c["_note"] = (f"CF-J (2026-11-25): cf2_long_hf{sfx} + canonical_jitter (training only; evaluation / selection unjittered). "
                  "docs/cf_frame_jitter_proposal.md. Made by scripts/decoder/dlevels/make_cfj.py.")
    (REPO / f"configs/decoder/scratch/cfj_long_hf{sfx}.json").write_text(json.dumps(c, indent=1))
    print("cfj_long_hf" + (sfx or "_s0"), "seed", src.get("seed", 0), "changed:", sorted(k for k in set(src) | set(c) if src.get(k) != c.get(k)))
