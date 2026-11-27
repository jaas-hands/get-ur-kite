"""W2-b configs: D1 (d1_prim50_hf, _s1, _s2) + base-frame augmentation (docs/w2_wrist_frame_proposal.md). One change:
"base_frame_aug": every training example's robot base frame is re-expressed by one of k = 32 fixed rotations (identity +
31 uniform SO(3) draws, seed 20261121), chosen uniformly per example. Evaluation / selection unchanged (identity frame).

    python scripts/decoder/dlevels/make_w2b.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for sfx in ("", "_s1", "_s2"):
    d1 = json.loads((REPO / f"configs/decoder/scratch/d1_prim50_hf{sfx}.json").read_text())
    c = dict(d1)
    c["base_frame_aug"] = {"k": 32, "seed": 20261121}
    c["out_dir"] = f"${{GETKITE_RUNS}}/w2b_frameaug_hf{sfx}"
    c["_note"] = (f"W2-b (2026-11-21, docs/w2_wrist_frame_proposal.md): D1 (d1_prim50_hf{sfx}) + base-frame augmentation "
                  "(k=32 fixed base rotations, identity + 31 uniform SO(3), per example). Everything else is D1's. Made by "
                  "scripts/decoder/dlevels/make_w2b.py.")
    (REPO / f"configs/decoder/scratch/w2b_frameaug_hf{sfx}.json").write_text(json.dumps(c, indent=1))
    print(sfx or "_s0", sorted(k for k in set(d1) | set(c) if d1.get(k) != c.get(k)))
