"""Evaluation-only: a second held-out LEAP pool (2026-11-26, user go-ahead to widen evaluation).

The 360 train-split LEAP hands that no training run used AND whose topology none of CF2 / D1-long / CF-3d trained on
(CF2 and D1-long share one training set; CF-3d uses CF2's). D1-bv trained on all of these topologies (not these hands), so the
pool is held-out at topology level only for CF2 / D1-long / CF-3d; D1-bv numbers on it are reported with that caveat.
Writes c3_psi50k_10x_wrist_leap360.json (copy of the 105-hand base config with these 360 hands appended to robots and
extra_eval.leap_heldout replaced by 360 hands x rows [0, 125)) and transfer_v1_leap360.json (D1-long, CF2, D1-bv x 3 seeds).
Test splits untouched; hands asserted absent from every training config.

    python scripts/decoder/benchmark/make_leap360.py
"""
import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DS = Path.home() / "code/uni-kite/get-ur-kite-dataset/data"
LEAPD = "${GETKITE_DATASET}"
ROWS = [0, 125]
top = lambda h: h.rsplit("_", 1)[0]  # noqa: E731

s = json.loads((DS / "splits.json").read_text())["hands"]
used = set()
for f in glob.glob(str(REPO / "configs/decoder/scratch/*.json")):
    used |= {Path(t["npz"]).stem for t in json.loads(Path(f).read_text()).get("train", []) if "npz" in t}
cf2_tops = {top(Path(t["npz"]).stem) for sfx in ("", "_s1", "_s2")
            for t in json.loads((REPO / f"configs/decoder/scratch/cf2_long_hf{sfx}.json").read_text())["train"] if t["kind"] == "leap"}
d1_tops = {top(Path(t["npz"]).stem) for sfx in ("", "_s1", "_s2")
           for t in json.loads((REPO / f"configs/decoder/scratch/d1_long_hf{sfx}.json").read_text())["train"] if t["kind"] == "leap"}
assert cf2_tops == d1_tops
pool = sorted(h for h in s["train"] if h not in used and top(h) not in cf2_tops)
assert all((DS / f"examples/kite_code/train/{h}.npz").exists() for h in pool)
base = json.loads((REPO / "configs/decoder/scratch/c3_psi50k_10x_wrist_leapval105.json").read_text())
have = {Path(r["hand_dir"]).name for r in base["robots"] if "hand_dir" in r}
assert not (set(pool) & have)
base["robots"] = base["robots"] + [{"kind": "leap", "hand_dir": f"{LEAPD}/hands/{h}"} for h in pool]
base["extra_eval"]["leap_heldout"] = [{"kind": "leap", "npz": f"{LEAPD}/examples/kite_code/train/{h}.npz", "rows": ROWS} for h in pool]
base["_note"] = ("Evaluation-only: held-out LEAP pool = 360 unused train-split hands of topologies CF2/D1-long never trained on "
                 "(D1-bv trained on these topologies). Made by scripts/decoder/benchmark/make_leap360.py.")
(REPO / "configs/decoder/scratch/c3_psi50k_10x_wrist_leap360.json").write_text(json.dumps(base, indent=1))
B = json.loads((REPO / "configs/decoder/benchmark/transfer_v1_leapval105.json").read_text())
B["base_config"] = "configs/decoder/scratch/c3_psi50k_10x_wrist_leap360.json"
B["out_dir"] = "${GETKITE_RUNS}/transfer_benchmark_v1_leap360"
B["systems"] = {k: v for k, v in B["systems"].items() if k.split("_s")[0] in ("d1long", "cf2", "d1bv")}
B["_note"] = "Second held-out LEAP pool (360 hands, evaluation only): D1-long, CF2, D1-bv x 3 seeds. Shadow off."
(REPO / "configs/decoder/benchmark/transfer_v1_leap360.json").write_text(json.dumps(B, indent=1))
print("pool", len(pool), "topologies", len({top(h) for h in pool}), "robots", len(base["robots"]), "systems", list(B["systems"]))
