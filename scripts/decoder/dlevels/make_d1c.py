"""D1c configs: D1 (d1_prim50_hf, _s1, _s2) with the generated bodies taken from the coupled draw prim_c (2026-11-21;
docs/coupling_and_wrist_proposal.md, Phase 2). prim_c = draw A + coupled joints (kite_data.build_prim couple), so the
bodies, templates and the records of uncoupled bodies are draw A's; only coupled bodies and their records differ.
Every "/prim/" path -> "/prim_c/"; everything else is D1's. Not launched by this script.

    python scripts/decoder/dlevels/make_d1c.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

for sfx in ("", "_s1", "_s2"):
    d1 = json.loads((REPO / f"configs/decoder/scratch/d1_prim50_hf{sfx}.json").read_text())
    c = json.loads(json.dumps(d1).replace("/prim/", "/prim_c/"))
    c["out_dir"] = f"${{GETKITE_RUNS}}/d1c_prim50c_hf{sfx}"
    c["_note"] = (f"Coupled-joint arm (2026-11-21, docs/coupling_and_wrist_proposal.md Phase 2): D1 (d1_prim50_hf{sfx}) with the "
                  "generated bodies from prim_c (draw A + coupled joints). Everything else is D1's (4 families x 25%). Wuji and "
                  "Shadow excluded from training. Made by scripts/decoder/dlevels/make_d1c.py.")
    (REPO / f"configs/decoder/scratch/d1c_prim50c_hf{sfx}.json").write_text(json.dumps(c, indent=1))
    diff = sorted(k for k in set(d1) | set(c) if d1.get(k) != c.get(k))
    print(f"d1c_prim50c_hf{sfx}: keys differing from D1: {diff}")
