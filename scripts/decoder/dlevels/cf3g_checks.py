"""Implementation checks for CF-3g split-trained teachers (train_joint distill_groups; docs/cf3g_split_teachers.md). Must all pass
before launch. The fold-teacher checkpoints do not exist yet, so checks 4-5 use 9 distinct existing checkpoints as stand-ins.

1. Fold-teacher configs: each drops exactly its fold's generated bodies and LEAP-hand entries (no other change from D1-bv), and
   the three folds together cover all 800 bodies and the 48 CF-3e LEAP hands once.
2. teacher_groups on CF-3e's bank, split arm: Allegro / Barrett -> the original six (0-5); bodies and LEAP hands in fold k ->
   fold-k teachers + CF2 (3-5); 800 + 48 robots covered.
3. Same, splitfresh arm: LEAP hands in fold k -> fold-k teachers only; bodies as split.
4. teacher_pose_sparse == teacher_pose(match=assign) on a mixed batch (max abs diff <= 1e-5), i.e. running each teacher on its
   assigned examples only gives the same targets (also shows teachers have no cross-example operations).
5. On Allegro / Barrett, the split-arm targets equal CF-3e's six-teacher targets (max abs diff <= 1e-5).
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
from getkite.decoder.train_joint import load_teacher, teacher_groups, teacher_pose, teacher_pose_sparse
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
            exp = set(plan["bodies"][f]) | set(plan["leap_hands"][f])
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
    fold_of = {}
    for t in cfg["train"]:
        if t["kind"] == "leap":
            st = Path(t["npz"]).stem
            for f in FOLDS:
                if st in plan["bodies"][f] or st in plan["leap_hands"][f]:
                    fold_of[hid(t["npz"])] = (f, "/prim/" in t["npz"])
    kite = [t["robot"] for t in cfg["train"] if t["kind"] == "kite"]

    def pat(ix):
        return "".join("1" if i in ix else "0" for i in range(15))

    for name, c, fresh in (("split", cfg, False), ("splitfresh", cfg_f, True)):
        M = teacher_groups(c, bank, dev).cpu()
        got = {n: "".join("1" if b else "0" for b in M[:, bank.index[n]].tolist()) for n in list(fold_of) + kite}
        exp = {n: pat([0, 1, 2, 3, 4, 5]) for n in kite}
        for n, (f, prim) in fold_of.items():
            j = FOLDS.index(f)
            ft = [6 + 3 * j, 7 + 3 * j, 8 + 3 * j]
            exp[n] = pat(ft if (fresh and not prim) else ft + [3, 4, 5])
        bad = {n: (got[n], exp[n]) for n in exp if got[n] != exp[n]}
        good = not bad and len(fold_of) == 848
        print(f"check {3 if fresh else 2}: {name} patterns, {len(fold_of)} fold robots, mismatches {len(bad)}", list(bad.items())[:3])
        ok &= good

    # 4-5 with stand-in teachers
    tp = cfg["distill_teachers"][:6] + [f"{R}/{s}/best.pt" for s in STANDIN]
    teachers = [load_teacher(paths.expand(p), dev) for p in tp]
    M = teacher_groups(cfg, bank, dev)
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    torch.manual_seed(0); noise = start_noise(cfg)
    leap = [t for t in cfg["train"] if t["kind"] == "leap"]
    pick = [dict(t, rows=[0, 20]) for t in leap[:4] + leap[-6:]] + [dict(t, rows=[0, 40]) for t in cfg["train"] if t["kind"] == "kite"][:2]
    rec = load_pool(pick, bank).to(dev)
    sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
    with torch.no_grad():
        z = psi(rec.c_pos, rec.c_normal, rec.c_valid)
        torch.cuda.synchronize(); t0 = time.time()
        dense = teacher_pose(teachers, bank, rec.r, sg, z, M)
        torch.cuda.synchronize(); t1 = time.time()
        sparse = teacher_pose_sparse(teachers, bank, rec.r, sg, z, M)
        torch.cuda.synchronize(); t2 = time.time()
        six = teacher_pose(teachers[:6], bank, rec.r, sg, z)
    d4 = max(float((a - b).abs().max()) for a, b in zip(dense, sparse))
    print(f"check 4: sparse vs dense weighted teacher pose, max abs diff {d4:.2e} (n={len(rec)}; dense {t1 - t0:.2f}s, sparse {t2 - t1:.2f}s)")
    ok &= d4 <= 1e-5
    isk = torch.as_tensor([bank.names[i] in kite for i in rec.r.tolist()], device=dev)
    d5 = max(float((a[isk] - b[isk]).abs().max()) for a, b in zip(sparse, six))
    print(f"check 5: Allegro/Barrett split targets vs CF-3e six-teacher targets {d5:.2e} (n={int(isk.sum())})")
    ok &= d5 <= 1e-5 and int(isk.sum()) > 0

    sm = Path(f"{R}/smoke_cf3g"); sm.mkdir(parents=True, exist_ok=True)
    raw = json.loads((SC / "cf3g_splitfresh_hf.json").read_text())
    raw["distill_teachers"] = tp
    raw["out_dir"] = str(sm / "student")
    (sm / "student_config.json").write_text(json.dumps(raw, indent=1))
    print("wrote stand-in smoke config", sm / "student_config.json")
    print("ALL CHECKS PASS" if ok else "CHECK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
