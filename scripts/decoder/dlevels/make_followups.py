"""Follow-up arms of the morphology-diversity test (2026-11-20; docs/decoder_audit.md §59). No training is run here.

Every arm is the D1 config of the same training seed (d1_prim50_hf, _s1, _s2) with ONLY the generated-body data changed;
family shares (4 x 25%), model, loss, optimiser, steps, starts and the selection rule are D1's.
  * d1b  second body draw: generator + record seed 20261121 ($GETKITE_DATASET/prim_b); bodies P0000-P0049 x 400, its own
         source-val rows [410, 420) and held-out bodies P0800-P0849 (every "/prim/" path -> "/prim_b/").
  * c5   one 5-finger body: P0003 (the first 5-finger body by index in draw A) x 20000 records (split train_ctrl5). Pairs with C1
         (one 2-finger body) to separate layout resemblance from variety.
  * n5   50 bodies with no 5-finger layout: the first 50 non-5-finger bodies of draw A's training pool (P0000-P0064) x 400 rows.
  * r4   records per body x4: draw A's P0000-P0049 x 1600 fresh records (split train_r4; 80000 prim records instead of 20000).
c5 / n5 / r4 keep D1's val (500 records of P0000-P0049) and held-out bodies, so selection data are identical to D1's.

    python scripts/decoder/dlevels/make_followups.py
"""
import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DSD = "${GETKITE_DATASET}/prim"
SEEDS = {0: "", 1: "_s1", 2: "_s2"}


def prim_free(train):
    return [t for t in train if "/prim/" not in t["npz"] and "/prim_b/" not in t["npz"]]


def main():
    no5 = json.loads((REPO / "configs/decoder/dlevels/no5f_bodies.json").read_text())
    arms = {
        "d1b_prim50_hf": ("second body draw (seed 20261121), P0000-P0049 x 400", None),
        "c5_prim1x5f_hf": ("one 5-finger body P0003 x 20000 (train_ctrl5)",
                           [{"kind": "leap", "npz": f"{DSD}/examples/kite_code_ctrl5/train_ctrl5/P0003.npz"}]),
        "n5_prim50no5f_hf": (f"50 non-5-finger bodies ({no5[0]}-{no5[-1]}) x 400",
                             [{"kind": "leap", "npz": f"{DSD}/examples/kite_code/train/{b}.npz", "rows": [0, 400]} for b in no5]),
        "r4_prim50x1600_hf": ("P0000-P0049 x 1600 fresh records (train_r4)",
                              [{"kind": "leap", "npz": f"{DSD}/examples/kite_code_r4/train_r4/P{i:04d}.npz"} for i in range(50)]),
    }
    for s, sfx in SEEDS.items():
        d1 = json.loads((REPO / f"configs/decoder/scratch/d1_prim50_hf{sfx}.json").read_text())
        for arm, (desc, prim_train) in arms.items():
            if prim_train is None:
                c = json.loads(json.dumps(d1).replace("/prim/", "/prim_b/"))
            else:
                c = copy.deepcopy(d1)
                c["train"] = prim_free(c["train"]) + prim_train
                have = {r.get("hand_dir") for r in c["robots"]}
                for t in prim_train:
                    hd = f"{DSD}/hands/{Path(t['npz']).stem}"
                    if hd not in have:
                        c["robots"].append({"kind": "leap", "hand_dir": hd}); have.add(hd)
            c["out_dir"] = f"${{GETKITE_RUNS}}/{arm}{sfx}"
            c["_note"] = (f"Morphology-diversity follow-up (2026-11-20, §59): D1 (d1_prim50_hf{sfx}) with the generated-body data "
                          f"replaced by: {desc}. Everything else is D1's (4 families x 25%). Wuji and Shadow excluded from training. "
                          f"Made by scripts/decoder/dlevels/make_followups.py.")
            (REPO / f"configs/decoder/scratch/{arm}{sfx}.json").write_text(json.dumps(c, indent=1))
            print(arm + sfx, "train files", len(c["train"]), "robots", len(c["robots"]))


if __name__ == "__main__":
    main()
