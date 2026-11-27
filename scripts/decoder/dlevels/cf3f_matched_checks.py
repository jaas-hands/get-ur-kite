"""Implementation checks for CF-3f matched teachers (train_joint distill_matched; docs/cf3f_proposal.md). Must all pass before launch.

1. teacher_match on CF-3e's bank: every CF-3e training robot is matched as expected — Allegro, Barrett, the 48 LEAP hands and
   CF2's 50 generated bodies -> all 6 teachers; the other 750 generated bodies -> the 3 D1-bv teachers only.
2. With an all-True match, teacher_pose equals the unmatched (CF-3e) teacher_pose (max abs diff <= 1e-5).
3. On generated bodies outside CF2's set, the matched pose equals the plain 3-teacher D1-bv mean (max abs diff <= 1e-5); on CF2's
   bodies it equals the 6-teacher mean.

    PYTHONPATH=src python scripts/decoder/dlevels/cf3f_matched_checks.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from getkite.decoder import paths
from getkite.decoder.multifamily import fixed_start
from getkite.decoder.train_joint import load_teacher, teacher_match, teacher_pose
from getkite.decoder.train_scratch import FrozenPsi, build_bank, load_config, load_pool, start_noise

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "diagnostics"))
from crossbody_heldout_eval import sg_to  # noqa: E402

R = os.environ["GETKITE_RUNS"]


def main():
    dev = torch.device("cuda")
    cfg = load_config(f"{R}/frozen_cf3e_2026-11-27/cf3e_long_hf/config.json", [])
    bank = build_bank(cfg, dev)
    M = teacher_match(cfg["distill_teachers"], bank, dev).cpu()
    cf2 = json.load(open(f"{R}/cf2_long_hf/config.json"))
    cf2_prim = {str(np.load(paths.expand(t["npz"]))["hand_id"]) for t in cf2["train"] if "/prim/" in t["npz"]}
    ok = True
    exp = {}
    for t in cfg["train"]:
        name = t["robot"] if t["kind"] == "kite" else str(np.load(paths.expand(t["npz"]))["hand_id"])
        prim = "/prim/" in t["npz"]
        exp[name] = "111000" if (prim and name not in cf2_prim) else "111111"
    got = {n: "".join("1" if b else "0" for b in M[:, bank.index[n]].tolist()) for n in exp}
    bad = {n: (got[n], exp[n]) for n in exp if got[n] != exp[n]}
    cnt = {}
    for v in got.values():
        cnt[v] = cnt.get(v, 0) + 1
    print("check 1: training-robot patterns", cnt, "mismatches", len(bad), list(bad.items())[:5])
    ok &= not bad and cnt.get("111000") == 750 and cnt.get("111111") == 100

    teachers = [load_teacher(paths.expand(p), dev) for p in cfg["distill_teachers"]]
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    torch.manual_seed(0); noise = start_noise(cfg)
    specs = [t for t in cfg["train"] if "/prim/" in t["npz"]]
    pick = [t for t in specs if str(np.load(paths.expand(t["npz"]))["hand_id"]) in cf2_prim][:3] + \
           [t for t in specs if str(np.load(paths.expand(t["npz"]))["hand_id"]) not in cf2_prim][:5] + \
           [t for t in cfg["train"] if t["kind"] != "leap"][:2]
    rec = load_pool(pick, bank).to(dev)
    sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
    with torch.no_grad():
        z = psi(rec.c_pos, rec.c_normal, rec.c_valid)
        base = teacher_pose(teachers, bank, rec.r, sg, z)
        ones = teacher_pose(teachers, bank, rec.r, sg, z, torch.ones_like(M).to(dev))
        matched = teacher_pose(teachers, bank, rec.r, sg, z, M.to(dev))
        d1 = teacher_pose(teachers[:3], bank, rec.r, sg, z)
    d2 = max(float((a - b).abs().max()) for a, b in zip(base, ones))
    print(f"check 2: all-True match vs unmatched, max abs diff {d2:.2e}")
    ok &= d2 <= 1e-5
    names = np.array([bank.names[i] for i in rec.r.cpu().numpy()])
    m750 = torch.as_tensor(np.array([(n not in cf2_prim) and bank.family[bank.index[n]] == "prim" for n in names]), device=dev)
    m6 = ~m750
    d3a = max(float((a[m750] - b[m750]).abs().max()) for a, b in zip(matched, d1))
    d3b = max(float((a[m6] - b[m6]).abs().max()) for a, b in zip(matched, base))
    print(f"check 3: matched vs D1-bv-only on the 750-type bodies {d3a:.2e} (n={int(m750.sum())}); matched vs 6-teacher elsewhere {d3b:.2e} (n={int(m6.sum())})")
    ok &= d3a <= 1e-5 and d3b <= 1e-5 and int(m750.sum()) > 0 and int(m6.sum()) > 0
    print("ALL CHECKS PASS" if ok else "CHECK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
