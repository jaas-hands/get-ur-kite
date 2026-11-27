"""Morphology-diversity test (2026-11-19): record plan and the D1 / D3 training configs. No training is run here.

Bodies: primitive hands P0000..P0899 from kite_data.build_prim (seed 20261119), in $GETKITE_DATASET/prim/hands.
  * P0000..P0799: training pool, nested. D1 trains on P0000..P0049; D3 on P0000..P0799.
  * P0800..P0849: held-out development bodies (evaluation only, never trained on or used for selection).
  * P0850..P0899: reserved test bodies (records generated, not used anywhere yet).

Records (kite_code regime), file split names and rows per body:
  * train: 420 rows for every training-pool body.
    - Training rows: [0, 20000 / N), so 400 per body for D1 (N = 50) and 25 per body for D3 (N = 800).
      Both levels get 20,000 primitive-body training records.
    - Source-val rows: [410, 420) of P0000..P0049, i.e. 500 records. They are identical for D1 and D3, so the
      selection set does not change between levels.
  * heldout: 10 rows per body, i.e. 500.
  * test: 10 rows per body, i.e. 500.

Configs: G1 (g1_leapdiv50k_hf.json) unchanged, plus one family "prim":
  * robots: the level's training bodies + the 50 held-out development bodies;
  * train: the primitive training rows;
  * val: G1's val + 500 primitive source-val records;
  * extra_eval: G1's + "prim_heldout" (500).
Everything else is G1's: model, loss, optimiser, 50k steps, seed, starts, selection rule.

    python scripts/decoder/dlevels/make_dlevels.py
"""
import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DSD = "${GETKITE_DATASET}/prim"
TRAIN_POOL, HELDOUT, TEST = range(0, 800), range(800, 850), range(850, 900)
N_TRAIN_ROWS, VAL_ROWS, N_PRIM_TRAIN = 420, (410, 420), 20000
LEVELS = {"d1": 50, "d3": 800}
pid = lambda i: f"P{i:04d}"  # noqa: E731


def main():
    out = REPO / "configs/decoder/dlevels"; out.mkdir(parents=True, exist_ok=True)
    plan = {"train": {"hands": [pid(i) for i in TRAIN_POOL], "n": N_TRAIN_ROWS},
            "heldout": {"hands": [pid(i) for i in HELDOUT], "n": 10},
            "test": {"hands": [pid(i) for i in TEST], "n": 10}}
    (out / "record_plan.json").write_text(json.dumps(plan, indent=1))
    g1 = json.loads((REPO / "configs/decoder/scratch/g1_leapdiv50k_hf.json").read_text())
    for lvl, n in LEVELS.items():
        per = N_PRIM_TRAIN // n
        assert per * n == N_PRIM_TRAIN and per <= VAL_ROWS[0]
        c = copy.deepcopy(g1)
        bodies = [pid(i) for i in range(n)]
        c["robots"] += [{"kind": "leap", "hand_dir": f"{DSD}/hands/{b}"} for b in bodies + [pid(i) for i in HELDOUT]]
        c["train"] += [{"kind": "leap", "npz": f"{DSD}/examples/kite_code/train/{b}.npz", "rows": [0, per]} for b in bodies]
        c["val"] += [{"kind": "leap", "npz": f"{DSD}/examples/kite_code/train/{pid(i)}.npz", "rows": list(VAL_ROWS)} for i in range(50)]
        c["extra_eval"] = dict(c["extra_eval"])
        c["extra_eval"]["prim_heldout"] = [{"kind": "leap", "npz": f"{DSD}/examples/kite_code/heldout/{pid(i)}.npz"} for i in HELDOUT]
        c["out_dir"] = f"${{GETKITE_RUNS}}/{lvl}_prim{n}_hf"
        c["_note"] = (f"Morphology-diversity {lvl.upper()} (2026-11-19): G1 (g1_leapdiv50k_hf) unchanged + family 'prim' = {n} "
                      f"primitive bodies P0000-{pid(n - 1)} x {per} records ({N_PRIM_TRAIN} total). Val = G1 val + 500 prim "
                      f"source-val records (rows 410-420 of P0000-P0049, same for D1/D3). Held-out bodies P0800-P0849 are eval "
                      f"only. Wuji and Shadow are excluded from training. Plan: docs/morphology_diversity_proposal.md. Made by "
                      f"scripts/decoder/dlevels/make_dlevels.py.")
        p = REPO / f"configs/decoder/scratch/{lvl}_prim{n}_hf.json"
        p.write_text(json.dumps(c, indent=1))
        print("wrote", p, "robots", len(c["robots"]), "train specs", len(c["train"]), "val specs", len(c["val"]))
    print("wrote", out / "record_plan.json")


if __name__ == "__main__":
    main()
