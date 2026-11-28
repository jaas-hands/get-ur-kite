"""Did D1-bv memorise its generated-body training records? (evaluation only; no training; user-approved 2026-11-27;
docs/cf3f_proposal.md, "match arm").

Why: the matched-teacher arm (D1-bv-only targets on the 750 generated bodies CF2 never saw) trails CF-3e (6-teacher mean) on
held-out generated bodies mid-training, although D1-bv-only targets were closer to ground truth on those training records
(teacher_target_prim_v1). Hypothesis: D1-bv's targets on its own training records are near-copies of the label, so they add little
beyond the ground-truth term, while CF2's (which never saw these bodies) carry a generalisation signal.

Sets (generated bodies, family "prim"):
  seen_rec   = 150 of the 750 bodies (every 5th), rows 0-25   -> D1-bv trained on these exact records; CF2 never saw the body.
  unseen_rec = the same 150 bodies, rows 25-50                 -> no teacher trained on these records (same bodies).
  heldout    = P0800-P0849 (50 bodies never in any training), rows 0-25 (descriptive).
Targets are computed exactly as in training (mean qc, mean t, chordal-mean R; fixed noisy start val_start_seed) and scored with the
benchmark's score() against ground truth.

Predeclared reading (fixed before running): memorisation is supported if BOTH
  (a) D1-bv-only (3 teachers) corr_mm on unseen_rec minus seen_rec, paired by body, bootstrap over bodies: 95 % CI above 0; and
  (b) D1-bv's advantage over CF2-only shrinks off its training records:
      [cf2_3 - d1bv3]_seen_rec - [cf2_3 - d1bv3]_unseen_rec, paired by body, bootstrap over bodies: 95 % CI above 0.
Otherwise: not supported by this check. Everything else is descriptive.

    PYTHONPATH=src python scripts/decoder/benchmark/teacher_memorization_eval.py
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
from crossbody_heldout_eval import ANG_DEG, POS_MM, sg_to  # noqa: E402
from transfer_benchmark import score  # noqa: E402

R = os.environ["GETKITE_RUNS"]
METRICS = ("corr_mm", "wrist_mm")
GROUPS = {"all6": [0, 1, 2, 3, 4, 5], "d1bv3": [0, 1, 2], "cf2_3": [3, 4, 5]}


def boot_bodies(d, n=4000, seed=0):
    """mean of per-body values d, bootstrap over bodies."""
    g = np.random.default_rng(seed)
    b = [g.choice(d, len(d)).mean() for _ in range(n)]
    return {"mean": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)],
            "n_bodies": int(len(d))}


def main():
    dev = torch.device("cuda")
    out = Path(f"{R}/teacher_memorization_v1"); out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(f"{R}/frozen_cf3e_2026-11-27/cf3e_long_hf/config.json", [])
    cf2 = json.load(open(f"{R}/cf2_long_hf/config.json"))
    d1 = json.load(open(f"{R}/d1_bv_long_hf/config.json"))
    cf2_prim = {os.path.basename(t["npz"]) for t in cf2["train"] if "/prim/" in t["npz"]}
    d1_rows = {os.path.basename(t["npz"]): tuple(t["rows"]) for t in d1["train"] if "/prim/" in t["npz"]}
    prim = [t for t in cfg["train"] if "/prim/" in t["npz"] and os.path.basename(t["npz"]) not in cf2_prim]
    assert len(prim) == 750
    bodies = prim[::5]
    for t in bodies:
        assert d1_rows[os.path.basename(t["npz"])] == (0, 25) and tuple(t["rows"]) == (0, 25)
    hdir = Path(bodies[0]["npz"]).parent.parent / "heldout"
    held = sorted(hdir.glob("P*.npz"))
    assert len(held) == 50
    trained_names = {os.path.basename(t["npz"]) for c in (cfg, cf2, d1) for t in c["train"]}
    assert not any(h.name in trained_names for h in held)
    sets = {"seen_rec": [{**t, "rows": [0, 25]} for t in bodies],
            "unseen_rec": [{**t, "rows": [25, 50]} for t in bodies],
            "heldout": [{"kind": "leap", "npz": str(h), "rows": [0, 25]} for h in held]}

    torch.manual_seed(0); noise = start_noise(cfg)
    extra = [{"kind": "leap", "npz": str(h), "rows": [0, 25]} for h in held]
    bank = build_bank({**cfg, "train": cfg["train"] + extra}, dev)
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    tpaths = json.load(open(f"{R}/frozen_cf3e_2026-11-27/cf3e_long_hf/distill.json"))["teachers"]
    assert tpaths == cfg["distill_teachers"]
    teachers = [load_teacher(paths.expand(p), dev) for p in tpaths]
    scfg = {"success_pos_mm": POS_MM, "success_normal_deg": ANG_DEG}

    per = {}  # set -> group -> metric -> per-body mean array
    for sn, specs in sets.items():
        acc = {g: {m: [] for m in METRICS} for g in GROUPS}
        for k, sp in enumerate(specs):
            rec = load_pool([sp], bank).to(dev)
            sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
            with torch.no_grad():
                z = psi(rec.c_pos, rec.c_normal, rec.c_valid)
                outs = [T(bank, rec.r, sg.qc1, sg.R1, sg.t1, z) for T in teachers]
                for g, ix in GROUPS.items():
                    qc = sum(outs[i]["qc"] for i in ix) / len(ix)
                    Rm = chordal_mean([outs[i]["R"] for i in ix])
                    t = sum(outs[i]["t"] for i in ix) / len(ix)
                    sc = score(bank, rec, rec.r, sg, qc, Rm, t, scfg)
                    for m in METRICS:
                        acc[g][m].append(float(sc[m].float().mean()))
            if k % 50 == 0:
                print(sn, k, os.path.basename(sp["npz"]), len(rec), flush=True)
        per[sn] = {g: {m: np.array(v) for m, v in d.items()} for g, d in acc.items()}

    res = {"means": {sn: {g: {m: round(float(per[sn][g][m].mean()), 4) for m in METRICS} for g in GROUPS} for sn in sets}}
    d1s, d1u = per["seen_rec"]["d1bv3"]["corr_mm"], per["unseen_rec"]["d1bv3"]["corr_mm"]
    c2s, c2u = per["seen_rec"]["cf2_3"]["corr_mm"], per["unseen_rec"]["cf2_3"]["corr_mm"]
    a6s, a6u = per["seen_rec"]["all6"]["corr_mm"], per["unseen_rec"]["all6"]["corr_mm"]
    res["d1bv3_unseen_minus_seen"] = boot_bodies(d1u - d1s)
    res["cf2_3_unseen_minus_seen"] = boot_bodies(c2u - c2s)
    res["cf2_minus_d1bv_seen"] = boot_bodies(c2s - d1s)
    res["cf2_minus_d1bv_unseen"] = boot_bodies(c2u - d1u)
    res["did_cf2_minus_d1bv_seen_minus_unseen"] = boot_bodies((c2s - d1s) - (c2u - d1u))
    res["all6_minus_d1bv_seen"] = boot_bodies(a6s - d1s)
    res["all6_minus_d1bv_unseen"] = boot_bodies(a6u - d1u)
    res["all6_minus_d1bv_heldout"] = boot_bodies(per["heldout"]["all6"]["corr_mm"] - per["heldout"]["d1bv3"]["corr_mm"])
    ca = res["d1bv3_unseen_minus_seen"]["ci95"][0] > 0
    cb = res["did_cf2_minus_d1bv_seen_minus_unseen"]["ci95"][0] > 0
    res["predeclared"] = {"a": bool(ca), "b": bool(cb), "memorisation_supported": bool(ca and cb)}
    (out / "teacher_memorization.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print("wrote", out / "teacher_memorization.json")


if __name__ == "__main__":
    main()
