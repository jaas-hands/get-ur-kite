"""CF-3g go/no-go on the 9 fold teachers, BEFORE any student run (evaluation only; docs/cf3g_split_teachers.md). Rule fixed
2026-11-28 with Aryan, before any fold teacher trained. "Teacher" = mean of its 3 seeds (mean qc, mean t, chordal-mean R), the
same target the student would see; fixed noisy start val_start_seed; scored with the benchmark's score() (corr_mm) against
ground truth; paired; bootstrap over bodies / hands (4000 draws, seed 0).

GO only if ALL hold:
  1. Freshness pays: on each fold's left-out generated bodies, excluding CF2's 50 (rows 0-25, the records the student trains on),
     fold-k teachers - CF2, per body, pooled over folds: 95 % CI upper bound < 0.
  2. Not much weaker on new bodies: on P0800-P0849 (rows 0-25; no teacher trained on them), mean over folds of
     (fold-k teachers - original D1-bv), per body: point estimate <= +0.10 mm.
  3. Not much weaker on new LEAP records: on rows 900-942 of the 48 CF-3e LEAP hands (no teacher trained on them), for the hand's
     fold k: fold-k teachers - original D1-bv, per hand: point estimate <= +0.10 mm.
Descriptive only (not decided on): left-out LEAP rows 0-48 (the original D1-bv trained on these), and on LEAP rows 48-96
fold-only vs fold+CF2 targets (bears on the splitfresh arm).

    PYTHONPATH=src python scripts/decoder/benchmark/cf3g_gonogo.py [--runs-suffix ""]
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

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/decoder/benchmark")); sys.path.insert(0, str(REPO / "scripts/decoder/diagnostics"))
from crossbody_heldout_eval import ANG_DEG, POS_MM, sg_index, sg_to  # noqa: E402
from transfer_benchmark import score  # noqa: E402

R = os.environ["GETKITE_RUNS"]
FOLDS = ("A", "B", "C")
SFX = ("", "_s1", "_s2")
MARGIN_MM = 0.10


def boot(d, n=4000, seed=0):
    g = np.random.default_rng(seed)
    b = [g.choice(d, len(d)).mean() for _ in range(n)]
    return {"mean": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)],
            "n": int(len(d))}


def main():
    dev = torch.device("cuda")
    out = Path(f"{R}/cf3g_gonogo_v1"); out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((REPO / "configs/decoder/dlevels/cf3g_folds.json").read_text())
    cfg = load_config(str(REPO / "configs/decoder/scratch/cf3g_split_hf.json"), [])
    cf2 = json.load(open(f"{R}/cf2_long_hf/config.json"))
    cf2_prim = {Path(t["npz"]).stem for t in cf2["train"] if "/prim/" in t["npz"]}
    prim = {Path(t["npz"]).stem: t["npz"] for t in cfg["train"] if "/prim/" in t["npz"]}
    leap = {Path(t["npz"]).stem: t["npz"] for t in cfg["train"] if t["kind"] == "leap" and "/prim/" not in t["npz"]}
    hdir = Path(paths.expand(next(iter(prim.values())))).parent.parent / "heldout"
    held = sorted(hdir.glob("P*.npz"))
    assert len(held) == 50 and len(prim) == 800 and len(leap) == 48
    for f in FOLDS:  # every fold-teacher checkpoint must exist and be the fold config's run
        for s in SFX:
            ck = torch.load(f"{R}/d1_bv_fold{f}_long_hf{s}/best.pt", map_location="cpu", weights_only=False)
            assert ck["config"]["out_dir"].endswith(f"d1_bv_fold{f}_long_hf{s}"), (f, s)

    extra = [{"kind": "leap", "npz": str(h), "rows": [0, 25]} for h in held]
    bank = build_bank({**cfg, "train": cfg["train"] + extra}, dev)
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    torch.manual_seed(0); noise = start_noise(cfg)
    T = {"orig": [f"{R}/d1_bv_long_hf{s}/best.pt" for s in SFX], "cf2": [f"{R}/cf2_long_hf{s}/best.pt" for s in SFX],
         **{f: [f"{R}/d1_bv_fold{f}_long_hf{s}/best.pt" for s in SFX] for f in FOLDS}}
    models = {k: [load_teacher(paths.expand(p), dev) for p in v] for k, v in T.items()}
    scfg = {"success_pos_mm": POS_MM, "success_normal_deg": ANG_DEG}

    def err(spec, groups):
        """mean corr_mm per group of the group-mean target on one spec; groups = list of keys, or ("mix", [keys]) for a union."""
        rec = load_pool([spec], bank).to(dev)
        sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
        res = {}
        with torch.no_grad():
            z = psi(rec.c_pos, rec.c_normal, rec.c_valid)
            cache = {}
            for g in groups:
                keys = g[1] if isinstance(g, tuple) else [g]
                outs = []
                for k in keys:
                    if k not in cache:
                        cache[k] = [m(bank, rec.r, sg.qc1, sg.R1, sg.t1, z) for m in models[k]]
                    outs += cache[k]
                qc = sum(o["qc"] for o in outs) / len(outs)
                Rm = chordal_mean([o["R"] for o in outs])
                t = sum(o["t"] for o in outs) / len(outs)
                res[g[0] if isinstance(g, tuple) else g] = float(score(bank, rec, rec.r, sg, qc, Rm, t, scfg)["corr_mm"].float().mean())
        return res

    res = {"rule": {"margin_mm": MARGIN_MM}}
    # 1. left-out bodies, fold teachers vs CF2
    d1 = []
    for f in FOLDS:
        for b in plan["bodies"][f]:
            if b in cf2_prim:
                continue
            e = err({"kind": "leap", "npz": prim[b], "rows": [0, 25]}, [f, "cf2"])
            d1.append(e[f] - e["cf2"])
    res["c1_fold_minus_cf2_leftout_bodies"] = boot(np.array(d1))
    print("c1", res["c1_fold_minus_cf2_leftout_bodies"], flush=True)
    # 2. held-out bodies, fold teachers (mean over folds) vs original D1-bv
    d2, d2f = [], {f: [] for f in FOLDS}
    for h in held:
        e = err({"kind": "leap", "npz": str(h), "rows": [0, 25]}, list(FOLDS) + ["orig"])
        for f in FOLDS:
            d2f[f].append(e[f] - e["orig"])
        d2.append(np.mean([e[f] for f in FOLDS]) - e["orig"])
    res["c2_fold_minus_orig_heldout_bodies"] = boot(np.array(d2))
    res["c2_per_fold"] = {f: boot(np.array(v)) for f, v in d2f.items()}
    print("c2", res["c2_fold_minus_orig_heldout_bodies"], flush=True)
    # 3. LEAP rows 900-942, hand's fold teachers vs original; descriptive: rows 0-48 and 48-96
    d3, d3_left, d_fresh = [], [], []
    for f in FOLDS:
        for h in plan["leap_hands"][f]:
            e = err({"kind": "leap", "npz": leap[h], "rows": [900, 942]}, [f, "orig"])
            d3.append(e[f] - e["orig"])
            e = err({"kind": "leap", "npz": leap[h], "rows": [0, 48]}, [f, "orig"])
            d3_left.append(e[f] - e["orig"])
            e = err({"kind": "leap", "npz": leap[h], "rows": [48, 96]}, [f, ("mix", [f, "cf2"])])
            d_fresh.append(e[f] - e["mix"])
    res["c3_fold_minus_orig_leap_unseen_rows"] = boot(np.array(d3))
    res["desc_fold_minus_orig_leap_leftout_rows_0_48"] = boot(np.array(d3_left))
    res["desc_foldonly_minus_fold_plus_cf2_leap_rows_48_96"] = boot(np.array(d_fresh))
    print("c3", res["c3_fold_minus_orig_leap_unseen_rows"], flush=True)
    c1 = res["c1_fold_minus_cf2_leftout_bodies"]["ci95"][1] < 0
    c2 = res["c2_fold_minus_orig_heldout_bodies"]["mean"] <= MARGIN_MM
    c3 = res["c3_fold_minus_orig_leap_unseen_rows"]["mean"] <= MARGIN_MM
    res["decision"] = {"c1": bool(c1), "c2": bool(c2), "c3": bool(c3), "GO": bool(c1 and c2 and c3)}
    (out / "cf3g_gonogo.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["decision"]))
    print("wrote", out / "cf3g_gonogo.json")


if __name__ == "__main__":
    main()
