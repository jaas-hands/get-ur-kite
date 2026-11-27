"""CF-3d (docs/cf3_distillation_proposal.md, step 1): CF2 (cf2_long_hf*, 100k steps, A data) plus one change, ensemble
distillation from the six D1-bv + CF2 checkpoints (alpha 0.5).

    python scripts/decoder/dlevels/make_cf3d.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TEACHERS = [f"${{GETKITE_RUNS}}/{a}{s}/best.pt" for a in ("d1_bv_long_hf", "cf2_long_hf") for s in ("", "_s1", "_s2")]
for sfx in ("", "_s1", "_s2"):
    src = json.loads((REPO / f"configs/decoder/scratch/cf2_long_hf{sfx}.json").read_text())
    c = dict(src)
    c["distill_teachers"] = TEACHERS
    c["distill_alpha"] = 0.5
    c["out_dir"] = f"${{GETKITE_RUNS}}/cf3d_long_hf{sfx}"
    c["_note"] = (f"CF-3d (2026-11-26): cf2_long_hf{sfx} + ensemble distillation (6 teachers D1-bv+CF2, alpha 0.5). "
                  "docs/cf3_distillation_proposal.md. Made by scripts/decoder/dlevels/make_cf3d.py.")
    (REPO / f"configs/decoder/scratch/cf3d_long_hf{sfx}.json").write_text(json.dumps(c, indent=1))
    print("cf3d_long_hf" + (sfx or "_s0"), "changed:", sorted(k for k in set(src) | set(c) if src.get(k) != c.get(k)))
