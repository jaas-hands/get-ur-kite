"""Implementation checks for CF-3g split-trained teachers (train_joint distill_groups; docs/cf3g_split_teachers.md). Must all pass
before launch. The fold-teacher checkpoints do not exist yet, so checks 4-5 use 9 distinct existing checkpoints as stand-ins.

1. Fold-teacher configs: each drops exactly its fold's generated bodies (no other change from D1-bv; all LEAP, Allegro and
   Barrett entries kept), and the three folds together cover all 800 bodies once.
2. teacher_groups on CF-3e's bank (both arms, revised 2026-11-28): generated bodies in fold k -> fold-k teachers + CF2 (3-5);
   Allegro, Barrett and all 48 LEAP hands -> the original six (0-5).
3. record_teacher_mask on the full CF-3e training pool: split = per-robot table for every record; splitfresh = the same except
   LEAP rows 48-417 of the 48 hands -> original D1-bv only (0-2), rows 0-48 -> the original six. Expected per record from an
   independent enumeration (one load_pool per train entry).
4. teacher_pose_sparse == the dense weighted average (every teacher on the whole batch, per-example weights) on a mixed batch
   with record-level weights (max abs diff <= 1e-5), i.e. running each teacher on its assigned examples only gives the same
   targets (also shows teachers have no cross-example operations).
5. On Allegro / Barrett and on LEAP rows 0-48, the targets equal CF-3e's six-teacher targets (max abs diff <= 1e-5).
Then writes a stand-in student config for the smoke run.

    PYTHONPATH=src python scripts/decoder/dlevels/cf3g_checks.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from getkite.decoder import paths
from getkite.decoder.multifamily import fixed_start
from getkite.decoder.train_joint import chordal_mean, load_teacher, record_teacher_mask, teacher_groups, teacher_pose, teacher_pose_sparse
from getkite.decoder.train_scratch import FrozenPsi, build_bank, load_config, load_pool, start_noise

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/decoder/diagnostics"))
from crossbody_heldout_eval import sg_to  # noqa: E402

R = os.environ["GETKITE_RUNS"]
SC = REPO / "configs/decoder/scratch"
FOLDS = ("A", "B", "C")
STANDIN = ["d1_bv_long_hf_s3", "d1_bv_long_hf_s4", "cf2_long_hf_s3", "cf2_long_hf_s4", "frozen_cf3e_2026-11-27/cf3e_long_hf",
           "frozen_cf3e_2026-11-27/cf3e_long_hf_s1", "frozen_cf3e_2026-11-27/cf3e_long_hf_s2", "cf3f_rec_hf", "cf3f_leap_hf"]


def hid(npz):
    return str(np.load(paths.expand(npz))["hand_id"])


def main():
    ok = True
    plan = json.loads((REPO / "configs/decoder/dlevels/cf3g_folds.json").read_text())
    # 1. fold-teacher configs
    for sfx in ("", "_s1", "_s2"):
        bv = json.loads((SC / f"d1_bv_long_hf{sfx}.json").read_text())
        for f in FOLDS:
            c = json.loads((SC / f"d1_bv_fold{f}_long_hf{sfx}.json").read_text())
            dropped = [t for t in bv["train"] if t not in c["train"]]
            exp = set(plan["bodies"][f])
            good = ({Path(t["npz"]).stem for t in dropped} == exp and len(dropped) == len(exp)
                    and all(t in bv["train"] for t in c["train"])
                    and {k for k in set(bv) | set(c) if bv.get(k) != c.get(k)} == {"train", "out_dir", "_note"})
            ok &= good
            if not good:
                print("check 1 FAILED", f, sfx)
    allb = [b for f in FOLDS for b in plan["bodies"][f]]; allh = [h for f in FOLDS for h in plan["leap_hands"][f]]
    ok &= len(allb) == len(set(allb)) == 800 and len(allh) == len(set(allh)) == 48
    print("check 1: fold-teacher configs", "ok" if ok else "FAILED")

    dev = torch.device("cuda")
    cfg = load_config(str(SC / "cf3g_split_hf.json"), [])
    cfg_f = load_config(str(SC / "cf3g_splitfresh_hf.json"), [])
    bank = build_bank(cfg, dev)
    body_fold = {}
    for t in cfg["train"]:
        if "/prim/" in t["npz"]:
            st = Path(t["npz"]).stem
            body_fold[hid(t["npz"])] = next(f for f in FOLDS if st in plan["bodies"][f])
    train_names = [t["robot"] if t["kind"] == "kite" else hid(t["npz"]) for t in cfg["train"]]

    def pat(ix):
        return "".join("1" if i in ix else "0" for i in range(15))

    def exp_robot(n):
        if n in body_fold:
            j = FOLDS.index(body_fold[n])
            return pat([6 + 3 * j, 7 + 3 * j, 8 + 3 * j, 3, 4, 5])
        return pat([0, 1, 2, 3, 4, 5])

    Ms = {}
    for name, c in (("split", cfg), ("splitfresh", cfg_f)):
        M = teacher_groups(c, bank, dev).cpu(); Ms[name] = M
        got = {n: "".join("1" if b else "0" for b in M[:, bank.index[n]].tolist()) for n in train_names}
        bad = {n: (got[n], exp_robot(n)) for n in got if got[n] != exp_robot(n)}
        print(f"check 2: {name} per-robot patterns, {len(got)} training robots, mismatches {len(bad)}", list(bad.items())[:3])
        ok &= not bad and len(body_fold) == 800

    # 3. record-level table vs an independent enumeration
    pools = [load_pool([t], bank) for t in cfg["train"]]
    train = type(pools[0]).cat(pools)
    W = {n: record_teacher_mask(c, train, Ms[n].to(dev)).cpu() for n, c in (("split", cfg), ("splitfresh", cfg_f))}
    exp = {"split": [], "splitfresh": []}
    for t, pl in zip(cfg["train"], pools):
        name = t["robot"] if t["kind"] == "kite" else hid(t["npz"])
        base = exp_robot(name)
        rows = list(range(*t["rows"])) if "rows" in t else list(range(len(pl.r)))
        assert len(rows) == len(pl.r)
        leap48 = t["kind"] == "leap" and "/prim/" not in t["npz"]
        for row in rows:
            exp["split"].append(base)
            exp["splitfresh"].append(pat([0, 1, 2]) if (leap48 and 48 <= row < 417) else base)
    for n in W:
        got = ["".join("1" if b else "0" for b in w.tolist()) for w in W[n]]
        nbad = sum(g != e for g, e in zip(got, exp[n]))
        cnt = {}
        for g in got:
            cnt[g] = cnt.get(g, 0) + 1
        print(f"check 3: {n} record table, {len(got)} records, mismatches {nbad}; patterns {cnt}")
        ok &= nbad == 0 and len(got) == len(exp[n])
    ok &= sum(e == pat([0, 1, 2]) for e in exp["splitfresh"]) == 48 * 369

    # 4-5 with stand-in teachers, record-level weights (splitfresh)
    tp = cfg["distill_teachers"][:6] + [f"{R}/{s}/best.pt" for s in STANDIN]
    teachers = [load_teacher(paths.expand(p), dev) for p in tp]
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    torch.manual_seed(0); noise = start_noise(cfg)
    g = torch.Generator().manual_seed(0)
    leap_idx = [i for i, e in enumerate(exp["splitfresh"])]
    idx = torch.randperm(len(train.r), generator=g)[:400]
    # make sure the batch holds every kind of record: LEAP rows 0-48 and 48-417, bodies, kite
    kinds = {}
    for i in idx.tolist():
        kinds.setdefault(exp["splitfresh"][i], 0); kinds[exp["splitfresh"][i]] += 1
    rec = train.index(idx).to(dev)
    wm = W["splitfresh"][idx].to(dev)
    sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
    with torch.no_grad():
        z = psi(rec.c_pos, rec.c_normal, rec.c_valid)
        sparse = teacher_pose_sparse(teachers, bank, rec.r, sg, z, Ms["splitfresh"].to(dev), wm)
        outs = [T(bank, rec.r, sg.qc1, sg.R1, sg.t1, z) for T in teachers]
        w = wm.T.float(); w = w / w.sum(0, keepdim=True)
        dense = (sum(wi[:, None] * o["qc"] for wi, o in zip(w, outs)), chordal_mean([o["R"] for o in outs], w),
                 sum(wi[:, None] * o["t"] for wi, o in zip(w, outs)))
        six = teacher_pose(teachers[:6], bank, rec.r, sg, z)
    d4 = max(float((a - b).abs().max()) for a, b in zip(dense, sparse))
    print(f"check 4: sparse vs dense record-weighted teacher pose, max abs diff {d4:.2e} (n={len(idx)}; record kinds {kinds})")
    ok &= d4 <= 1e-5 and len(kinds) >= 4
    is6 = torch.as_tensor([exp["splitfresh"][i] == pat([0, 1, 2, 3, 4, 5]) for i in idx.tolist()], device=dev)
    d5 = max(float((a[is6] - b[is6]).abs().max()) for a, b in zip(sparse, six))
    print(f"check 5: six-teacher records (Allegro/Barrett, LEAP rows 0-48) vs CF-3e targets {d5:.2e} (n={int(is6.sum())})")
    ok &= d5 <= 1e-5 and int(is6.sum()) > 0

    sm = Path(f"{R}/smoke_cf3g_v2"); sm.mkdir(parents=True, exist_ok=True)
    raw = json.loads((SC / "cf3g_splitfresh_hf.json").read_text())
    raw["distill_teachers"] = tp
    raw["out_dir"] = str(sm / "student")
    (sm / "student_config.json").write_text(json.dumps(raw, indent=1))
    print("wrote stand-in smoke config", sm / "student_config.json")
    print("ALL CHECKS PASS" if ok else "CHECK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
