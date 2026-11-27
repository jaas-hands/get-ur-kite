"""Evaluation-only: held-out LEAP pool widened from 4 to all 105 dataset val-split LEAP hands (2026-11-25, user go-ahead).

No training, no selection. Writes a copy of the benchmark base config with the 101 other val hands appended to robots (after
the existing ones, so robot indices and every other pool are unchanged) and extra_eval.leap_heldout = 105 hands x rows [0, 125)
(the 4 original hands keep their exact spec, so their numbers must reproduce transfer_benchmark_v1_cf2 / _bv). The test_*
splits stay untouched. None of the 105 hands is in any training config (asserted).

    python scripts/decoder/benchmark/make_leapval105.py
"""
import json
import os
from pathlib import Path
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())

REPO = Path(__file__).resolve().parents[3]
DS = Path(os.environ.get("GETKITE_DATASET", _REPO.parent / "get-ur-kite-dataset" / "data"))
LEAPD = "${GETKITE_DATASET}"
ROWS = [0, 125]

s = json.loads((DS / "splits.json").read_text())["hands"]
val = sorted(s["val"])
base = json.loads((REPO / "configs/decoder/scratch/c3_psi50k_10x_wrist.json").read_text())
orig = base["extra_eval"]["leap_heldout"]
orig_h = [Path(e["npz"]).stem for e in orig]
assert all(h in val for h in orig_h) and all(e["rows"] == ROWS for e in orig)
# no val hand trained on by any arm in the comparison
for c in ["d1_prim50_hf", "d1_long_hf", "cf2_long_hf", "d1_bv_long_hf", "cf_bv_long_hf"]:
    for sfx in ("", "_s1", "_s2"):
        tr = {Path(t["npz"]).stem for t in json.loads((REPO / f"configs/decoder/scratch/{c}{sfx}.json").read_text())["train"]}
        assert not (tr & set(val)), c + sfx
have = {Path(r["hand_dir"]).name for r in base["robots"] if "hand_dir" in r}
new = [h for h in val if h not in have]
base["robots"] = base["robots"] + [{"kind": "leap", "hand_dir": f"{LEAPD}/hands/{h}"} for h in new]
base["extra_eval"]["leap_heldout"] = orig + [{"kind": "leap", "npz": f"{LEAPD}/examples/kite_code/val/{h}.npz", "rows": ROWS}
                                             for h in val if h not in orig_h]
base["_note"] = ("Evaluation-only copy of c3_psi50k_10x_wrist: held-out LEAP pool = all 105 val-split LEAP hands x 125 records "
                 "(original 4 first, unchanged). Made by scripts/decoder/benchmark/make_leapval105.py.")
(REPO / "configs/decoder/scratch/c3_psi50k_10x_wrist_leapval105.json").write_text(json.dumps(base, indent=1))
B = json.loads((REPO / "configs/decoder/benchmark/transfer_v1_bv.json").read_text())
B["base_config"] = "configs/decoder/scratch/c3_psi50k_10x_wrist_leapval105.json"
B["out_dir"] = "${GETKITE_RUNS}/transfer_benchmark_v1_leapval105"
B["shadow_pool"] = False
B["_note"] = ("Held-out LEAP widened to 105 val-split hands (evaluation only). Systems as transfer_v1_bv: frozen D1 (50k), D1-long, "
              "CF-long (CF2), D1-bv, CF-bv x 3 seeds. Shadow off (not needed; unchanged from transfer_v1_bv).")
(REPO / "configs/decoder/benchmark/transfer_v1_leapval105.json").write_text(json.dumps(B, indent=1))
print("robots", len(base["robots"]), "new robots", len(new), "held-out entries", len(base["extra_eval"]["leap_heldout"]))
