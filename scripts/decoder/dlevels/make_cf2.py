"""CF2 configs (docs/cf2_longer_training_proposal.md): CF (cf_canon_hf, _s1, _s2) trained twice as long, plus the matching D1
control. One change for both: total_steps 50000 -> 100000 (the cosine schedule stretches with it). Frame rule E, data,
losses, model and every other setting are unchanged.

    python scripts/decoder/dlevels/make_cf2.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
STEPS = 100000
for src, dst, lab in (("cf_canon_hf", "cf2_long_hf", "CF2"), ("d1_prim50_hf", "d1_long_hf", "D1-long control")):
    for sfx in ("", "_s1", "_s2"):
        a = json.loads((REPO / f"configs/decoder/scratch/{src}{sfx}.json").read_text())
        c = dict(a)
        c["total_steps"] = STEPS
        c["out_dir"] = f"${{GETKITE_RUNS}}/{dst}{sfx}"
        c["_note"] = (f"{lab} (2026-11-25, docs/cf2_longer_training_proposal.md): {src}{sfx} with total_steps {STEPS} "
                      "(was 50000). Everything else unchanged. Made by scripts/decoder/dlevels/make_cf2.py.")
        (REPO / f"configs/decoder/scratch/{dst}{sfx}.json").write_text(json.dumps(c, indent=1))
        print(dst + (sfx or "_s0"), sorted(k for k in set(a) | set(c) if a.get(k) != c.get(k)))
