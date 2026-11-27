"""Teacher-target quality on training records: mode "leap" = the CF-3f leap arm's LEAP records; mode "prim" = CF-3e's own
generated-body records (seen = CF2's 50 bodies, unseen = the other 750; same predeclared reading).
Original docstring for mode "leap" -- teacher-target quality on the CF-3f "leap" arm's LEAP training records (evaluation only; no training; docs/cf3f_proposal.md).

Question (Aryan, 2026-11-27): did the leap arm get noisier distillation targets on the 366 of its 414 LEAP hands that the CF2
teachers never trained on? CF-3e's 6 teachers = D1-bv s0-2 (trained on exactly these 414 x 48 records) + CF2 s0-2 (trained on
CF-3e's 48 hands, rows 0-417, which include these rows). So on the 48 hands every teacher trained on the record; on the 366 only
the D1-bv half did.

Targets are computed exactly as in training (getkite.decoder.train_joint.teacher_pose: mean joint coordinates, mean translation,
chordal-mean rotation, native frame), from a fixed noisy start (val_start_seed, KITE noise), and scored against the ground-truth
goal with the benchmark's score() (corr_mm = contact error with correspondences, wrist_mm, corr_success).

Predeclared reading (fixed before running): teacher mismatch is a real factor if BOTH
  (a) on the 366 hands, the 6-teacher target has significantly higher corr_mm than the D1-bv-only (3-teacher) target
      (paired bootstrap over records), and the same comparison on the 48 hands is not significantly higher; and
  (b) the 6-teacher target's corr_mm on the 366 exceeds that on the 48 (bootstrap over hands, 95 % CI above 0).
Otherwise: not supported by this check. Everything else is descriptive.

    PYTHONPATH=src python scripts/decoder/benchmark/teacher_target_eval.py [leap|prim]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from getkite.decoder import paths
from getkite.decoder.multifamily import fixed_start
from getkite.decoder.train_joint import chordal_mean, load_teacher
from getkite.decoder.train_scratch import FrozenPsi, build_bank, load_config, load_pool, start_noise

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "diagnostics"))
from crossbody_heldout_eval import ANG_DEG, POS_MM, sg_index, sg_to  # noqa: E402
from transfer_benchmark import score  # noqa: E402
from wuji_per_example import paired_boot  # noqa: E402

R = os.environ["GETKITE_RUNS"]
METRICS = ("corr_mm", "wrist_mm", "corr_success")


def boot_hands(a, b, n=2000, seed=0):
    """mean(a) - mean(b) over per-hand means, bootstrap over hands (unpaired)."""
    g = np.random.default_rng(seed)
    d = [g.choice(a, len(a)).mean() - g.choice(b, len(b)).mean() for _ in range(n)]
    return {"mean_diff": round(float(a.mean() - b.mean()), 4), "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]}


def main():
    dev = torch.device("cuda")
    mode = sys.argv[1] if len(sys.argv) > 1 else "leap"
    cf3e = json.load(open(f"{R}/frozen_cf3e_2026-11-27/cf3e_long_hf/config.json"))
    if mode == "leap":  # leap arm's 414 LEAP hands; "seen" = CF-3e's (= CF2's) 48 hands
        out = Path(f"{R}/teacher_target_leap_v1")
        cfg = load_config(f"{R}/cf3f_leap_hf/config.json", [])
        specs = [t for t in cfg["train"] if t["kind"] == "leap" and "/prim/" not in t["npz"]]
        seen = {os.path.basename(t["npz"]) for t in cf3e["train"] if t["kind"] == "leap" and "/prim/" not in t["npz"]}
        assert len(specs) == 414 and len(seen) == 48
    else:  # "prim": CF-3e's own 800 generated bodies x 25 records; "seen" = the 50 bodies CF2 trained on (D1-bv trained on all 800)
        out = Path(f"{R}/teacher_target_prim_v1")
        cfg = load_config(f"{R}/frozen_cf3e_2026-11-27/cf3e_long_hf/config.json", [])
        specs = [t for t in cfg["train"] if "/prim/" in t["npz"]]
        cf2 = json.load(open(f"{R}/cf2_long_hf/config.json"))
        seen = {os.path.basename(t["npz"]) for t in cf2["train"] if "/prim/" in t["npz"]}
        d1 = json.load(open(f"{R}/d1_bv_long_hf/config.json"))
        assert len(specs) == 800 and len(seen) == 50
        assert {os.path.basename(t["npz"]) for t in specs} <= {os.path.basename(t["npz"]) for t in d1["train"] if "/prim/" in t["npz"]}
    assert seen <= {os.path.basename(t["npz"]) for t in specs}
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); noise = start_noise(cfg)
    bank = build_bank(cfg, dev)
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    leap_specs, seen48 = specs, seen
    tpaths = json.load(open(f"{R}/frozen_cf3e_2026-11-27/cf3e_long_hf/distill.json"))["teachers"]
    assert tpaths == cfg["distill_teachers"]
    teachers = [load_teacher(paths.expand(p), dev) for p in tpaths]
    groups = {"all6": list(range(6)), "d1bv3": [0, 1, 2], "cf2_3": [3, 4, 5]} | {f"t{i}": [i] for i in range(6)}
    scfg = {"success_pos_mm": POS_MM, "success_normal_deg": ANG_DEG}

    per = {g: {m: [] for m in METRICS} for g in groups}
    hand_of, in48 = [], []
    for k, sp in enumerate(leap_specs):
        rec = load_pool([sp], bank).to(dev)
        sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
        with torch.no_grad():
            z = psi(rec.c_pos, rec.c_normal, rec.c_valid)
            outs = [T(bank, rec.r, sg.qc1, sg.R1, sg.t1, z) for T in teachers]
            for g, ix in groups.items():
                qc = sum(outs[i]["qc"] for i in ix) / len(ix)
                Rm = chordal_mean([outs[i]["R"] for i in ix]) if len(ix) > 1 else outs[ix[0]]["R"]
                t = sum(outs[i]["t"] for i in ix) / len(ix)
                sc = score(bank, rec, rec.r, sg, qc, Rm, t, scfg)
                for m in METRICS:
                    per[g][m].append(sc[m].float().cpu().numpy())
        name = os.path.basename(sp["npz"])
        hand_of += [k] * len(rec); in48 += [name in seen48] * len(rec)
        if k % 50 == 0:
            print(k, name, len(rec), flush=True)
    per = {g: {m: np.concatenate(v) for m, v in d.items()} for g, d in per.items()}
    hand_of, in48 = np.array(hand_of), np.array(in48)
    masks = {"seen": in48, "unseen": ~in48}

    res = {"n_records": {k: int(v.sum()) for k, v in masks.items()},
           "n_hands": {k: int(len(np.unique(hand_of[v]))) for k, v in masks.items()},
           "means": {k: {g: {m: round(float(per[g][m][v].mean()), 4) for m in METRICS} for g in groups} for k, v in masks.items()},
           "paired_all6_minus_d1bv3": {k: {m: paired_boot(per["all6"][m][v], per["d1bv3"][m][v]) for m in ("corr_mm", "wrist_mm")}
                                       for k, v in masks.items()},
           "paired_cf2_3_minus_d1bv3": {k: {m: paired_boot(per["cf2_3"][m][v], per["d1bv3"][m][v]) for m in ("corr_mm", "wrist_mm")}
                                        for k, v in masks.items()}}
    hm = {k: np.array([per["all6"]["corr_mm"][hand_of == h].mean() for h in np.unique(hand_of[v])]) for k, v in masks.items()}
    res["all6_unseen_minus_seen_corr_mm_over_hands"] = boot_hands(hm["unseen"], hm["seen"])
    a = res["paired_all6_minus_d1bv3"]
    cond_a = a["unseen"]["corr_mm"]["ci95"][0] > 0 and not a["seen"]["corr_mm"]["ci95"][0] > 0
    cond_b = res["all6_unseen_minus_seen_corr_mm_over_hands"]["ci95"][0] > 0
    res["predeclared"] = {"a": bool(cond_a), "b": bool(cond_b), "teacher_mismatch_supported": bool(cond_a and cond_b)}
    (out / "teacher_target.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print("wrote", out / "teacher_target.json")


if __name__ == "__main__":
    main()
