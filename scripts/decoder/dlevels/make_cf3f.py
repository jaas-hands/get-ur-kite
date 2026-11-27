"""CF-3f test arms (docs/cf3f_proposal.md): each = frozen CF-3e (cf3e_long_hf*) + ONE change, 3 seeds, judged against CF-3e.

    leap  : LEAP data = D1-bv's 414 hands x 48 records (3 geometry variants of all 138 train topologies; same LEAP budget,
            same 25 % family share). Everything else as CF-3e.
    long  : 300k-step schedule (500-step warm-up, cosine to 0 at 300k). Everything else as CF-3e.
    teach : teacher group + the three frozen CF-3e checkpoints (9 teachers, equal weight). Everything else as CF-3e.
    big   : model 256 wide x 6 layers x 8 heads (CF-3e: 128 x 4 x 4). Everything else as CF-3e.
    rec   : generated bodies P0000-P0799 x 100 records (CF-3e: x 25). Everything else as CF-3e.

Each output asserts that only the keys belonging to its change differ from CF-3e (plus _note / out_dir), then the leakage audit
(audit_cf3f.py) is run separately. Never used for training: val topologies (105 pool), Wuji, Shadow, external hands,
P0800-P0899, test_* splits.

    python scripts/decoder/dlevels/make_cf3f.py
"""
import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SC = REPO / "configs/decoder/scratch"
BV = json.loads((SC / "d1_bv_long_hf.json").read_text())
FROZEN = "${GETKITE_RUNS}/frozen_cf3e_2026-11-27"
CHANGED = {"leap": {"robots", "train"}, "long": {"total_steps"}, "teach": {"distill_teachers"}, "big": {"model"}, "rec": {"train"}}


def is_leap(t):
    return t["kind"] == "leap" and "/prim/" not in t.get("npz", t.get("hand_dir", ""))


def make(arm, src, sfx):
    c = copy.deepcopy(src)
    if arm == "leap":
        bv_leap = [t for t in BV["train"] if is_leap(t)]
        assert len(bv_leap) == 414 and all(t["rows"] == [0, 48] for t in bv_leap)
        c["train"] = [t for t in src["train"] if not is_leap(t)] + bv_leap
        have = {r.get("hand_dir") for r in src["robots"]}
        c["robots"] = src["robots"] + [r for r in BV["robots"] if r.get("kind") == "leap" and "/prim/" not in r.get("hand_dir", "")
                                       and r["hand_dir"] not in have]
        assert {Path(t["npz"]).stem for t in src["train"] if is_leap(t)} <= {Path(t["npz"]).stem for t in bv_leap}
        assert [t for t in c["train"] if not is_leap(t)] == [t for t in src["train"] if not is_leap(t)]
    elif arm == "long":
        c["total_steps"] = 300000
    elif arm == "teach":
        c["distill_teachers"] = src["distill_teachers"] + [f"{FROZEN}/cf3e_long_hf{s}/best.pt" for s in ("", "_s1", "_s2")]
    elif arm == "big":
        c["model"] = dict(src["model"]) | {"d_model": 256, "n_layers": 6, "n_heads": 8}
    elif arm == "rec":
        n = 0
        for t in c["train"]:
            if "/prim/" in t["npz"]:
                assert t["rows"] == [0, 25]; t["rows"] = [0, 100]; n += 1
        assert n == 800
    c["out_dir"] = f"${{GETKITE_RUNS}}/cf3f_{arm}_hf{sfx}"
    c["_note"] = (f"CF-3f-{arm} (2026-11-27): frozen CF-3e (cf3e_long_hf{sfx}) + one change ({arm}); docs/cf3f_proposal.md. "
                  "Made by scripts/decoder/dlevels/make_cf3f.py.")
    diff = {k for k in set(src) | set(c) if src.get(k) != c.get(k)} - {"_note", "out_dir"}
    assert diff == CHANGED[arm], (arm, diff)
    return c


def main():
    for sfx in ("", "_s1", "_s2"):
        src = json.loads((SC / f"cf3e_long_hf{sfx}.json").read_text())
        frozen = REPO.parent / "getkite_runs/frozen_cf3e_2026-11-27/configs" / f"cf3e_long_hf{sfx}.json"
        if frozen.exists():
            assert json.loads(frozen.read_text()) == src, f"cf3e_long_hf{sfx}.json differs from the frozen copy"
        for arm in CHANGED:
            c = make(arm, src, sfx)
            (SC / f"cf3f_{arm}_hf{sfx}.json").write_text(json.dumps(c, indent=1))
            print(f"cf3f_{arm}_hf{sfx or '_s0'}: changed {sorted(CHANGED[arm])}; train entries {len(src['train'])} -> {len(c['train'])}; "
                  f"robots {len(src['robots'])} -> {len(c['robots'])}")


if __name__ == "__main__":
    main()
