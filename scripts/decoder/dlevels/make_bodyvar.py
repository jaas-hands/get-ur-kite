"""B: body-variety arm (2026-11-25, docs/cf2_longer_training_proposal.md, "B"). Manifest + training configs; no training here.

One change against A (d1_long_hf* / cf2_long_hf*, 100k steps): more distinct training bodies at the same record budget and
family shares (balanced family sampling is unchanged, so each family keeps 25% of batches).
  * LEAP: every train-split topology (138) x 3 geometry draws = 414 hands, 48 records each (rows [0, 48)) = 19,872
    records (A: 48 hands x 417 = 20,016). The 48 A hands are kept (their draw is one of the 3); the other draws per
    topology are chosen uniformly from the remaining train-split draws with a fixed seed.
  * Generated bodies: D3's set, P0000..P0799 x 25 records (rows [0, 25)) = 20,000 (A: P0000..P0049 x 400).
  * Allegro / Barrett unchanged.
Unchanged: source val (selection set), extra_eval pools (leap_heldout, prim_heldout, new_hands_fresh, ...), model, losses,
optimiser, 100k steps, seeds, starts, frame (D1 native / CF canonical), selection rule.
Never used for training or selection: dataset val split (incl. the 4 held-out LEAP hands), all test_* splits, Wuji, Shadow,
P0800..P0899. Training rows never overlap the val / fresh-eval rows (900-942 for LEAP, 410-420 for generated bodies).

    python scripts/decoder/dlevels/make_bodyvar.py
"""
import json
import os
from pathlib import Path

import numpy as np
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())

REPO = Path(__file__).resolve().parents[3]
DS = Path(os.environ.get("GETKITE_DATASET", _REPO.parent / "get-ur-kite-dataset" / "data"))
SEED, DRAWS, LEAP_ROWS, PRIM_N, PRIM_ROWS = 20261125, 3, 48, 800, 25
LEAPD = "${GETKITE_DATASET}"


def topo(h):
    return h.rsplit("_g", 1)[0]


def main():
    s = json.loads((DS / "splits.json").read_text())
    train_hands = sorted(s["hands"]["train"])
    forbidden = set().union(*[set(v) for k, v in s["hands"].items() if k != "train"])
    a = json.loads((REPO / "configs/decoder/scratch/d1_long_hf.json").read_text())
    a_leap = [Path(r["hand_dir"]).name for r in a["robots"] if r["kind"] == "leap" and "/prim/" not in r["hand_dir"]]
    a_train_leap = [Path(t["npz"]).stem for t in a["train"] if t["kind"] == "leap" and "/prim/" not in t["npz"]]
    heldout = [h for h in a_leap if h not in a_train_leap]
    assert len(a_train_leap) == 48 and all(h in s["hands"]["val"] for h in heldout)
    by_topo = {}
    for h in train_hands:
        by_topo.setdefault(topo(h), []).append(h)
    rng = np.random.default_rng(SEED)
    hands = []
    for t in sorted(by_topo):
        keep = [h for h in a_train_leap if topo(h) == t]
        rest = sorted(set(by_topo[t]) - set(keep)); rng.shuffle(rest)
        hands += keep + rest[: DRAWS - len(keep)]
    assert len(hands) == len(set(hands)) == DRAWS * len(by_topo) and not (set(hands) & forbidden)
    assert set(a_train_leap) <= set(hands)
    for h in hands:
        assert (DS / "examples/kite_code/train" / f"{h}.npz").exists() and (DS / "hands" / h / "hand.json").exists()
    prims = [f"P{i:04d}" for i in range(PRIM_N)]
    man = {"experiment": "B body variety (2026-11-25)", "seed": SEED,
           "leap_train": hands, "leap_new_vs_A": sorted(set(hands) - set(a_train_leap)), "leap_heldout_eval": heldout,
           "records": {"leap": len(hands) * LEAP_ROWS, "leap_per_hand": LEAP_ROWS, "prim": PRIM_N * PRIM_ROWS,
                       "prim_per_body": PRIM_ROWS, "allegro": 20000, "barrett": 20000},
           "n_topologies": len(by_topo), "draws_per_topology": DRAWS}
    out = REPO / "configs/decoder/dlevels"
    (out / "bodyvar_manifest.json").write_text(json.dumps(man, indent=1))

    for src, dst in (("d1_long_hf", "d1_bv_long_hf"), ("cf2_long_hf", "cf_bv_long_hf")):
        for sfx in ("", "_s1", "_s2"):
            c = json.loads((REPO / f"configs/decoder/scratch/{src}{sfx}.json").read_text())
            kite_r = [r for r in c["robots"] if r["kind"] == "kite"]
            held_prim = [r for r in c["robots"] if "/prim/" in r.get("hand_dir", "")
                         and int(Path(r["hand_dir"]).name[1:]) >= 800]
            c["robots"] = (kite_r + [{"kind": "leap", "hand_dir": f"{LEAPD}/hands/{h}"} for h in hands + heldout]
                           + [{"kind": "leap", "hand_dir": f"{LEAPD}/prim/hands/{p}"} for p in prims] + held_prim)
            c["train"] = ([t for t in c["train"] if t["kind"] == "kite"]
                          + [{"kind": "leap", "npz": f"{LEAPD}/examples/kite_code/train/{h}.npz", "rows": [0, LEAP_ROWS]} for h in hands]
                          + [{"kind": "leap", "npz": f"{LEAPD}/prim/examples/kite_code/train/{p}.npz", "rows": [0, PRIM_ROWS]} for p in prims])
            c["out_dir"] = f"${{GETKITE_RUNS}}/{dst}{sfx}"
            c["_note"] = (f"B body variety (2026-11-25): {src}{sfx} with LEAP 414 hands x 48 records (138 topologies x 3 draws) and "
                          "generated bodies P0000-P0799 x 25 records; same record budget and family shares. Everything else "
                          f"unchanged. Manifest configs/decoder/dlevels/bodyvar_manifest.json. Made by scripts/decoder/dlevels/make_bodyvar.py.")
            (REPO / f"configs/decoder/scratch/{dst}{sfx}.json").write_text(json.dumps(c, indent=1))
            ref = json.loads((REPO / f"configs/decoder/scratch/{src}{sfx}.json").read_text())
            print(dst + (sfx or "_s0"), "robots", len(c["robots"]), "train files", len(c["train"]),
                  "changed:", sorted(k for k in set(ref) | set(c) if ref.get(k) != c.get(k)))
    print("leap hands", len(hands), "new", len(man["leap_new_vs_A"]), "records", man["records"])


if __name__ == "__main__":
    main()
