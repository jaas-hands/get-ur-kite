"""CF-3e (docs/cf3_distillation_proposal.md, "CF-3e"): CF-3d (cf3d_long_hf*) plus one change, generated-body variety: the
generated-body family's training data becomes D1-bv's set, P0000..P0799 x 25 records (CF-3d: P0000..P0049 x 400; same record
budget, same 25 % family share under balanced sampling). These bodies get the same loss as every record (alpha 0.5 ground truth +
teacher), so the student sees the six-checkpoint teacher on 800 diverse bodies instead of 50.
LEAP data stays CF2's 48 hands (D1-bv's extra LEAP topologies are NOT added, so the 105- and 360-hand held-out LEAP pools stay
held out for CF-3e). Allegro / Barrett, teachers, alpha, model, optimiser, 100k steps, seeds, selection unchanged.
Never used: P0800..P0899 (prim_heldout eval), Wuji, Shadow, external hands, test_* splits.

    python scripts/decoder/dlevels/make_cf3e.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BV = json.loads((REPO / "configs/decoder/scratch/d1_bv_long_hf.json").read_text())
BV_PRIM_TRAIN = [t for t in BV["train"] if "/prim/" in t["npz"]]
assert len(BV_PRIM_TRAIN) == 800 and all(t["rows"] == [0, 25] for t in BV_PRIM_TRAIN)
assert all(int(Path(t["npz"]).stem[1:]) < 800 for t in BV_PRIM_TRAIN)
BV_PRIM_ROBOTS = [r for r in BV["robots"] if "/prim/" in r.get("hand_dir", "")]
for sfx in ("", "_s1", "_s2"):
    src = json.loads((REPO / f"configs/decoder/scratch/cf3d_long_hf{sfx}.json").read_text())
    c = dict(src)
    have = {r.get("hand_dir") for r in src["robots"]}
    c["robots"] = src["robots"] + [r for r in BV_PRIM_ROBOTS if r["hand_dir"] not in have]
    old_prim = [t for t in src["train"] if "/prim/" in t["npz"]]
    assert len(old_prim) == 50 and sum(t["rows"][1] - t["rows"][0] for t in old_prim) == 20000
    c["train"] = [t for t in src["train"] if "/prim/" not in t["npz"]] + BV_PRIM_TRAIN
    leap_src = [t for t in src["train"] if t["kind"] == "leap" and "/prim/" not in t["npz"]]
    assert [t for t in c["train"] if t["kind"] == "leap" and "/prim/" not in t["npz"]] == leap_src and len(leap_src) == 48
    c["out_dir"] = f"${{GETKITE_RUNS}}/cf3e_long_hf{sfx}"
    c["_note"] = (f"CF-3e (2026-11-26): cf3d_long_hf{sfx} + generated-body variety (P0000-P0799 x 25, D1-bv's set; same budget). "
                  "LEAP data unchanged (48 hands). docs/cf3_distillation_proposal.md. Made by scripts/decoder/dlevels/make_cf3e.py.")
    (REPO / f"configs/decoder/scratch/cf3e_long_hf{sfx}.json").write_text(json.dumps(c, indent=1))
    print("cf3e_long_hf" + (sfx or "_s0"), "changed:", sorted(k for k in set(src) | set(c) if src.get(k) != c.get(k)),
          "robots", len(src["robots"]), "->", len(c["robots"]), "train entries", len(src["train"]), "->", len(c["train"]))
