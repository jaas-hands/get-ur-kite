"""CF configs: D1 (d1_prim50_hf, _s1, _s2) + canonical palm frame (docs/cf_canonical_palm_frame_proposal.md). One change:
"canonical_frame": every robot is expressed in its canonical palm frame (robots.canonical_frame, rule E; computed from the
robot description + template) for training, validation, selection and the benchmark. base_frame_aug stays off.

    python scripts/decoder/dlevels/make_cf.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for sfx in ("", "_s1", "_s2"):
    d1 = json.loads((REPO / f"configs/decoder/scratch/d1_prim50_hf{sfx}.json").read_text())
    c = dict(d1)
    c["canonical_frame"] = True
    c["out_dir"] = f"${{GETKITE_RUNS}}/cf_canon_hf{sfx}"
    c["_note"] = (f"CF (2026-11-22, docs/cf_canonical_palm_frame_proposal.md): D1 (d1_prim50_hf{sfx}) + canonical palm frame "
                  "(rule E) for every robot. Everything else is D1's. Made by scripts/decoder/dlevels/make_cf.py.")
    (REPO / f"configs/decoder/scratch/cf_canon_hf{sfx}.json").write_text(json.dumps(c, indent=1))
    print(sfx or "_s0", sorted(k for k in set(d1) | set(c) if d1.get(k) != c.get(k)))
