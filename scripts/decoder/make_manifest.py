"""Pilot manifest (Stage C) and the C1/C2/C3 configs for get_universal_scratch.

Families: allegro_right, barrett (KITE-native records, get-ur-kite-dataset scripts/kite_family/gen_kite_family.py) and LEAP variants
(kite_data kite_code records). LEAP training morphologies: 12 hands of 12 distinct topologies from the
existing LEAP *train* split; held-out LEAP morphologies: 4 hands of 4 distinct topologies from the LEAP
*val* split (first structural test; never trained on, never used for selection). Rows of each LEAP hand
file: training goals [0, 167), fresh validation goals [900, 942).
"""
import json
from pathlib import Path

import numpy as np
from getkite.decoder import paths

# Paths are written as ${...} placeholders (expanded at load time by getkite.decoder.paths), so the
# generated configs work on any machine.
DATA, PD, K, RUNS = "${GETKITE_DATASET}", "${GETKITE_DATA}/multifamily_pilot", "${KITE_ROOT}", "${GETKITE_RUNS}"
CFG = paths.REPO / "configs/decoder/scratch"
PD_LOCAL = Path(paths.expand(PD))
PD_LOCAL.mkdir(parents=True, exist_ok=True)

splits = json.loads((Path(paths.expand(DATA)) / "splits.json").read_text())["hands"]
rng = np.random.default_rng(0)


def pick(split, n):
    by_topo = {}
    for h in splits[split]:
        by_topo.setdefault(h.rsplit("_g", 1)[0], []).append(h)
    topos = sorted(by_topo)
    chosen = rng.choice(len(topos), n, replace=False)
    return [sorted(by_topo[topos[i]])[rng.integers(len(by_topo[topos[i]]))] for i in sorted(chosen)]


leap_train, leap_held = pick("train", 12), pick("val", 4)
kite_robots = [
    {"kind": "kite", "robot": "allegro_right", "family": "allegro",
     "urdf": (K + "/assets/ohra_urdf_robot/allegro/allegro_hand_right_extended.urdf"),
     "template": (K + "/assets/ohra_urdf_robot/allegro/surface_template_graph_n4096_ddcc74fe8fec.pt"),
     "coords_npz": (PD + "/allegro_right_train.npz"), "ignore_mimic": True},
    {"kind": "kite", "robot": "barrett", "family": "barrett",
     "urdf": (K + "/assets/dro_urdf_robot/barrett/model_extended.urdf"),
     "template": (K + "/assets/dro_urdf_robot/barrett/surface_template_graph_n4096_3e0ba32e061c.pt"),
     "coords_npz": (PD + "/barrett_train.npz"), "ignore_mimic": True},
]
robots = kite_robots + [{"kind": "leap", "hand_dir": DATA + "/hands/" + h} for h in leap_train + leap_held]
leap_npz = lambda h, split: f"{DATA}/examples/kite_code/{split}/{h}.npz"  # noqa: E731


def kite_pool(split, rows=None):
    return [{"kind": "kite", "robot": r, "npz": (PD + f"/{r}_{split}.npz"), **({"rows": rows} if rows else {})}
            for r in ("allegro_right", "barrett")]


manifest = {"leap_train_hands": leap_train, "leap_heldout_hands": leap_held,
            "leap_rows": {"train": [0, 167], "val": [900, 942], "heldout": [0, 125]},
            "kite": {"allegro_right": {"train_seed": 101, "n_train": 2000, "val_seed": 202, "n_val": 500},
                     "barrett": {"train_seed": 101, "n_train": 2000, "val_seed": 202, "n_val": 500}},
            "families": ["allegro", "barrett", "leap"], "external_candidates_not_used": ["shadow", "inspire"]}
(PD_LOCAL / "manifest.json").write_text(json.dumps(manifest, indent=1))

common = {"kite_root": K, "kite_ckpt": (K + "/checkpoints/decoder/release.pt"), "robots": robots,
          "model": {"d_model": 128, "n_layers": 4, "n_heads": 4, "mlp_ratio": 4}}
# C1: 16 fixed examples per family (LEAP: 4 hands x 4), fixed q1, no input noise; scored on the same records
c1_train = kite_pool("train", [0, 16]) + [{"kind": "leap", "npz": leap_npz(h, "train"), "rows": [0, 4]} for h in leap_train[:4]]
c1 = common | {"out_dir": (RUNS + "/scratch_c1_overfit"), "train": c1_train, "start": "fixed",
               "train_start_seed": 1, "val_start_seed": 1, "normal_noise_rad": 0.0, "batch_size": 48,
               "lr": 3e-4, "warmup_steps": 100, "total_steps": 3000, "eval_every": 250, "log_every": 50,
               "ckpt_every": 500, "weight_decay": 0.0}
# C2: same fixed contact/goal sets, q1 redrawn every batch (KITE recipe); scored on fresh fixed q1 draws
c2 = c1 | {"out_dir": (RUNS + "/scratch_c2_resample"), "start": "resample", "val_start_seed": 12345,
           "total_steps": 6000, "eval_every": 500}
# C3: small independent dataset, fresh validation goals, held-out LEAP morphologies reported separately
c3 = common | {"out_dir": (RUNS + "/scratch_c3_small"), "start": "resample", "normal_noise_rad": 0.1,
               "train": kite_pool("train") + [{"kind": "leap", "npz": leap_npz(h, "train"), "rows": [0, 167]} for h in leap_train],
               "val": kite_pool("val") + [{"kind": "leap", "npz": leap_npz(h, "train"), "rows": [900, 942]} for h in leap_train],
               "extra_eval": {"leap_heldout": [{"kind": "leap", "npz": leap_npz(h, "val"), "rows": [0, 125]} for h in leap_held]},
               "batch_size": 256, "lr": 3e-4, "warmup_steps": 500, "total_steps": 10000, "eval_every": 1000,
               "log_every": 100, "ckpt_every": 500, "weight_decay": 1e-4}
for name, c in [("c1_overfit", c1), ("c2_resample", c2), ("c3_small", c3)]:
    (CFG / f"{name}.json").write_text(json.dumps(c, indent=1))
print(json.dumps(manifest, indent=1))
