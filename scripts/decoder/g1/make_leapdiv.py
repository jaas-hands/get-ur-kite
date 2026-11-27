"""G1 (2026-11-19): LEAP-diversity scaling experiment -- manifest + training config. No data generated.

Expands the J1 source LEAP list from 12 to 48 hands (all from the dataset's TRAIN split; 48 distinct topologies = 48 distinct
graphs; geometry draws 0-6), keeping Allegro/Barrett nominal, J1's architecture/loss/budget and ~60k total goals:
Allegro 20,000 + Barrett 20,000 (as J1) + LEAP 48 x 417 = 20,016 (J1: 12 x 1,667). Records come from the existing seed-0
kite_code pool (1,000 per train hand, KITE-code recipe); rows [0, 417) for training, rows [900, 942) are the (unchanged)
source-val rows of the 12 J1 hands and the fresh-goal eval rows of the 36 new hands.
Selection rule (fixed, seeded): the 36 new topologies are drawn from the train topologies not used by J1, round-robin over
(finger count, DOF) strata (sorted), uniformly within a stratum; each gets one geometry draw uniform in 0-6.
Never used for training or selection: dataset val split (incl. the 4 held-out LEAP dev hands), all test_* splits (reserved
final within-family tests), Wuji (dev), Shadow E3M5 (reserved external).

    python scripts/decoder/g1/make_leapdiv.py
"""
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DS = Path.home() / "code/uni-kite/get-ur-kite-dataset/data"
SEED, N_NEW, N_PER = 20261119, 36, 417


def topo(h):
    return h.rsplit("_g", 1)[0]


def feats(t):
    c = [int(x) for x in re.findall(r"[IMRT](\d)", t)]
    return sum(x > 0 for x in c), sum(c)


def main():
    s = json.loads((DS / "splits.json").read_text())
    j1 = json.loads((REPO / "configs/decoder/scratch/j1_joint50k_hf.json").read_text())
    j1_hands = [Path(r["hand_dir"]).name for r in j1["robots"] if r["kind"] == "leap"]
    heldout = j1_hands[12:]; j1_train = j1_hands[:12]
    assert all(h in s["hands"]["train"] for h in j1_train) and all(h in s["hands"]["val"] for h in heldout)
    used = {topo(h) for h in j1_train}
    cand = sorted(set(s["topologies"]["train"]) - used)
    strata = {}
    for t in cand:
        strata.setdefault(feats(t), []).append(t)
    rng = np.random.default_rng(SEED)
    for k in strata:
        rng.shuffle(strata[k])
    keys = sorted(strata); new = []
    while len(new) < N_NEW:
        for k in keys:
            if strata[k] and len(new) < N_NEW:
                new.append(strata[k].pop())
    new_hands = [f"{t}_g{int(rng.integers(0, 7))}" for t in new]
    assert all(h in s["hands"]["train"] for h in new_hands)
    hands = j1_train + new_hands
    for h in hands:
        assert (DS / "examples/kite_code/train" / f"{h}.npz").exists() and (DS / "hands" / h / "hand.json").exists()
    forbidden = set().union(*[set(v) for k, v in s["hands"].items() if k != "train"])
    assert not (set(hands) & forbidden)
    man = {"experiment": "G1 LEAP-diversity scaling (option b), 2026-11-19", "seed": SEED,
           "source_bodies": {"allegro": ["allegro_right (nominal)"], "barrett": ["barrett (nominal)"], "leap": hands},
           "n_bodies": 2 + len(hands), "leap_j1": j1_train, "leap_new": new_hands,
           "records": {"allegro": 20000, "barrett": 20000, "leap_per_hand": N_PER, "leap_total": N_PER * len(hands),
                       "total": 40000 + N_PER * len(hands), "j1_leap_per_hand": 1667, "rows": [0, N_PER]},
           "graphs_vs_geometry": "48 distinct LEAP topologies (graphs); each one geometry draw (g0 nominal, g1-6 random link "
                                 "stretch 0.8-1.2 / mount shift +-4 mm). Allegro/Barrett unchanged.",
           "coverage": {"leap_fingers": sorted({feats(topo(h))[0] for h in hands}), "leap_dof": sorted({feats(topo(h))[1] for h in hands}),
                        "max_dof_any_source": 16, "max_fingers_any_source": 4,
                        "note": "no source body has 5 fingers or >16 DOF: Wuji (20 DOF, 5 fingers) and Shadow are extrapolation"},
           "selection": "unchanged J1 source val (Allegro/Barrett 500 each, 12 J1 LEAP hands rows 900-942)",
           "development": ["leap_heldout (4 val-split hands, already inspected)", "wuji (unseen by our decoder, seen by psi)"],
           "reserved_final": ["dataset test_joint_count / test_combined / test_finger_subset / test_geometry hands (never loaded)",
                              "Shadow Hand E3M5 (external; adapter not built)"],
           "psi_saw_leap": False}
    out = REPO / "configs/decoder/g1"
    (out / "leapdiv_manifest.json").write_text(json.dumps(man, indent=1))
    c = dict(j1)
    kite = [r for r in j1["robots"] if r["kind"] == "kite"]
    c["robots"] = kite + [{"kind": "leap", "hand_dir": "${GETKITE_DATASET}/hands/" + h} for h in hands + heldout]
    c["train"] = [t for t in j1["train"] if t["kind"] == "kite"] + [
        {"kind": "leap", "npz": "${GETKITE_DATASET}/examples/kite_code/train/" + h + ".npz", "rows": [0, N_PER]} for h in hands]
    c["extra_eval"] = dict(j1["extra_eval"])
    c["extra_eval"]["new_hands_fresh"] = [{"kind": "leap", "npz": "${GETKITE_DATASET}/examples/kite_code/train/" + h + ".npz",
                                           "rows": [900, 942]} for h in new_hands]
    c["out_dir"] = "${GETKITE_RUNS}/g1_leapdiv50k_hf"
    c["_note"] = ("G1 LEAP-diversity scaling (2026-11-19): J1 architecture/loss/optimiser/budget/seed/selection unchanged; source LEAP "
                  "12 -> 48 train-split hands (48 distinct topologies), 417 seed-0 records each (rows 0-417); Allegro/Barrett "
                  "nominal 20k each as J1; Wuji excluded. Manifest configs/decoder/g1/leapdiv_manifest.json. new_hands_fresh = "
                  "eval only (not selection).")
    (REPO / "configs/decoder/scratch/g1_leapdiv50k_hf.json").write_text(json.dumps(c, indent=1))
    print(json.dumps({k: man[k] for k in ("n_bodies", "records", "coverage")}, indent=1))
    print("new:", new_hands)


if __name__ == "__main__":
    main()
