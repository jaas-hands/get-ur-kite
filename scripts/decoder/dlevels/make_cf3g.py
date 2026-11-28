"""CF-3g split-trained teachers (docs/cf3g_split_teachers.md). Writes the fold plan, the 9 fold-teacher configs and the 2 student
arms (3 seeds each). Nothing here changes CF-3e or any existing config.

Why (teacher_memorization_v1 / teacher_memorization_source_v1): the CF-3e teachers memorised the records they trained on, so on
those records their targets are near-copies of the label. Fold teachers make every generated body and every CF-3e LEAP record get
a D1-bv-style target from teachers that never saw it.

Fold plan (seeded, fixed before any training): the 800 generated training bodies P0000-P0799 and the 48 LEAP hands CF-3e trains on
are each split into folds A / B / C (sorted names, rng permutation, rank % 3).

Fold teachers d1_bv_fold{A,B,C}_long_hf{,_s1,_s2} = D1-bv (d1_bv_long_hf*) with ONE change: drop the fold's generated bodies and
the fold's LEAP-hand entries (rows 0-48; D1-bv uses those 48 hands at rows 0-48 only, among its 414 LEAP entries). Everything else
(Allegro / Barrett, the other LEAP entries, model, steps, seed, selection) as D1-bv.

Student arms (each = CF-3e cf3e_long_hf* + per-hand teacher choice; distill_teachers lists 15 checkpoints:
0-2 original D1-bv s0-2, 3-5 CF2 s0-2, 6-8 fold A s0-2, 9-11 fold B, 12-14 fold C):
  split      : bodies and LEAP hands in fold k -> fold-k teachers (3) + CF2 (3). Every other robot (Allegro, Barrett) -> the
               original six (exactly CF-3e's targets).
  splitfresh : as split, but LEAP hands in fold k -> fold-k teachers only (CF2 dropped there; Aryan's arm).
Everything else as CF-3e (data, alpha 0.5, model, 100k steps, seeds, selection on source validation).

    python scripts/decoder/dlevels/make_cf3g.py
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SC = REPO / "configs/decoder/scratch"
FOLD_FILE = REPO / "configs/decoder/dlevels/cf3g_folds.json"
FOLD_SEED = 20261128
FOLDS = ("A", "B", "C")
SFX = ("", "_s1", "_s2")


def is_prim(t):
    return "/prim/" in t["npz"]


def is_leap(t):
    return t["kind"] == "leap" and not is_prim(t)


def stem(t):
    return Path(t["npz"]).stem


def fold_plan(bv, e):
    bodies = sorted(stem(t) for t in bv["train"] if is_prim(t))
    hands = sorted(stem(t) for t in e["train"] if is_leap(t))
    assert len(bodies) == 800 and len(hands) == 48
    rng = np.random.default_rng(FOLD_SEED)
    pb, ph = rng.permutation(len(bodies)), rng.permutation(len(hands))
    plan = {"seed": FOLD_SEED, "method": "sorted names; numpy default_rng(seed).permutation (bodies first, then hands); fold = rank % 3",
            "bodies": {f: sorted(bodies[i] for k, i in enumerate(pb) if k % 3 == j) for j, f in enumerate(FOLDS)},
            "leap_hands": {f: sorted(hands[i] for k, i in enumerate(ph) if k % 3 == j) for j, f in enumerate(FOLDS)}}
    for key, allv in (("bodies", bodies), ("leap_hands", hands)):
        got = [x for f in FOLDS for x in plan[key][f]]
        assert sorted(got) == allv and len(set(got)) == len(allv)
    return plan


def make_teacher(bv, plan, f, sfx):
    c = copy.deepcopy(bv)
    drop_b, drop_h = set(plan["bodies"][f]), set(plan["leap_hands"][f])
    leap_h = [t for t in bv["train"] if is_leap(t) and stem(t) in drop_h]
    assert len(leap_h) == len(drop_h) and all(t["rows"] == [0, 48] for t in leap_h), (f, len(leap_h))
    c["train"] = [t for t in bv["train"] if not (is_prim(t) and stem(t) in drop_b) and not (is_leap(t) and stem(t) in drop_h)]
    assert len(bv["train"]) - len(c["train"]) == len(drop_b) + len(drop_h)
    c["out_dir"] = f"${{GETKITE_RUNS}}/d1_bv_fold{f}_long_hf{sfx}"
    c["_note"] = (f"CF-3g fold teacher {f} (2026-11-28): d1_bv_long_hf{sfx} minus fold {f} ({len(drop_b)} generated bodies, "
                  f"{len(drop_h)} LEAP-hand entries at rows 0-48); fold plan configs/decoder/dlevels/cf3g_folds.json. "
                  "docs/cf3g_split_teachers.md. Made by scripts/decoder/dlevels/make_cf3g.py.")
    diff = {k for k in set(bv) | set(c) if bv.get(k) != c.get(k)} - {"_note", "out_dir"}
    assert diff == {"train"}, diff
    return c


def make_student(e, plan, arm, sfx):
    c = copy.deepcopy(e)
    fold_t = [f"${{GETKITE_RUNS}}/d1_bv_fold{f}_long_hf{s}/best.pt" for f in FOLDS for s in SFX]
    assert len(e["distill_teachers"]) == 6
    c["distill_teachers"] = e["distill_teachers"] + fold_t
    c["distill_default"] = [0, 1, 2, 3, 4, 5]
    prim = {stem(t): t["npz"] for t in e["train"] if is_prim(t)}
    leap = {stem(t): t["npz"] for t in e["train"] if is_leap(t)}
    groups = []
    for j, f in enumerate(FOLDS):
        ft = [6 + 3 * j, 7 + 3 * j, 8 + 3 * j]
        b_npz = [prim[b] for b in plan["bodies"][f]]
        h_npz = [leap[h] for h in plan["leap_hands"][f]]
        if arm == "split":
            groups.append({"name": f"fold{f}", "teachers": ft + [3, 4, 5], "npz": b_npz + h_npz})
        else:
            groups.append({"name": f"fold{f}_bodies", "teachers": ft + [3, 4, 5], "npz": b_npz})
            groups.append({"name": f"fold{f}_leap", "teachers": ft, "npz": h_npz})
    c["distill_groups"] = groups
    c["out_dir"] = f"${{GETKITE_RUNS}}/cf3g_{arm}_hf{sfx}"
    c["_note"] = (f"CF-3g-{arm} (2026-11-28): frozen CF-3e (cf3e_long_hf{sfx}) + per-hand teacher choice from fold teachers "
                  "(docs/cf3g_split_teachers.md). Made by scripts/decoder/dlevels/make_cf3g.py.")
    diff = {k for k in set(e) | set(c) if e.get(k) != c.get(k)} - {"_note", "out_dir"}
    assert diff == {"distill_teachers", "distill_default", "distill_groups"}, diff
    covered = [n for g in groups for n in g["npz"]]
    assert len(covered) == len(set(covered)) == 800 + 48
    return c


def main():
    bv0 = json.loads((SC / "d1_bv_long_hf.json").read_text())
    e0 = json.loads((SC / "cf3e_long_hf.json").read_text())
    plan = fold_plan(bv0, e0)
    txt = json.dumps(plan, indent=1)
    if FOLD_FILE.exists():
        assert FOLD_FILE.read_text() == txt, "fold plan changed; it is fixed once written"
    FOLD_FILE.write_text(txt)
    print("fold plan", FOLD_FILE.name, "sha256", hashlib.sha256(txt.encode()).hexdigest(),
          {f: (len(plan["bodies"][f]), len(plan["leap_hands"][f])) for f in FOLDS})
    for sfx in SFX:
        bv = json.loads((SC / f"d1_bv_long_hf{sfx}.json").read_text())
        e = json.loads((SC / f"cf3e_long_hf{sfx}.json").read_text())
        frozen = REPO.parent / "getkite_runs/frozen_cf3e_2026-11-27/configs" / f"cf3e_long_hf{sfx}.json"
        if frozen.exists():
            assert json.loads(frozen.read_text()) == e, f"cf3e_long_hf{sfx}.json differs from the frozen copy"
        assert {t["npz"] for t in bv["train"]} == {t["npz"] for t in bv0["train"]}  # seeds share data
        for f in FOLDS:
            c = make_teacher(bv, plan, f, sfx)
            (SC / f"d1_bv_fold{f}_long_hf{sfx}.json").write_text(json.dumps(c, indent=1))
            print(f"d1_bv_fold{f}_long_hf{sfx or '_s0'}: train entries {len(bv['train'])} -> {len(c['train'])}")
        for arm in ("split", "splitfresh"):
            c = make_student(e, plan, arm, sfx)
            (SC / f"cf3g_{arm}_hf{sfx}.json").write_text(json.dumps(c, indent=1))
            print(f"cf3g_{arm}_hf{sfx or '_s0'}: {len(c['distill_teachers'])} teachers, {len(c['distill_groups'])} groups")


if __name__ == "__main__":
    main()
