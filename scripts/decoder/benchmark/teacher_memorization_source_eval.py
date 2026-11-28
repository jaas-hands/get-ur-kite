"""Did the CF-3e teachers memorise their LEAP / Allegro / Barrett training records? (evaluation only; no training;
user-approved 2026-11-28; follows teacher_memorization_eval.py, which found memorisation on generated bodies; docs/cf3f_proposal.md).

Why: all six teachers (D1-bv s0-2, CF2 s0-2) trained on source-family records too. If their targets there are near-copies of the
label, distillation adds little on 3/4 of every batch, and split-trained teachers would need record-level folds for these families.

Sets
  LEAP (the 48 hands CF-3e trains on; paired by hand, bootstrap over hands):
    A = rows 0-48     -> all six teachers trained on these records
    B = rows 48-96    -> CF2 trained on these, D1-bv did not (D1-bv used rows 0-48 only)
    C = rows 900-942  -> no teacher trained on these (CF-3e source validation rows)
  Allegro / Barrett (single hands; every teacher trained on all training records; NO teacher-level control):
    train = first 500 records of the s101 training file; new = the 500 validation records (s202). Unpaired, bootstrap over records.
Targets exactly as in training (mean qc, mean t, chordal-mean R; fixed noisy start val_start_seed), scored with the benchmark's
score() against ground truth.

Predeclared reading (fixed before running):
  LEAP memorisation supported if BOTH
    (a) CF2-only (3 teachers): C - B, paired by hand, 95 % CI above 0; and
    (b) controlled for record difficulty by D1-bv (unseen in both B and C):
        (cf2_C - cf2_B) - (d1bv_C - d1bv_B), paired by hand, 95 % CI above 0.
  Allegro / Barrett: "gap present" if all6 new - train 95 % CI above 0, per robot. Weaker evidence (no control: a gap could also
  mean the validation records are harder; both come from the same sampler at different seeds).
Everything else is descriptive.

    PYTHONPATH=src python scripts/decoder/benchmark/teacher_memorization_source_eval.py
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

R = os.environ["GETKITE_RUNS"]
METRICS = ("corr_mm", "wrist_mm")
GROUPS = {"all6": [0, 1, 2, 3, 4, 5], "d1bv3": [0, 1, 2], "cf2_3": [3, 4, 5]}


def boot(d, n=4000, seed=0):
    g = np.random.default_rng(seed)
    b = [g.choice(d, len(d)).mean() for _ in range(n)]
    return {"mean": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)],
            "n": int(len(d))}


def boot_unpaired(x, y, n=4000, seed=0):
    """mean(y) - mean(x), bootstrap each set over records."""
    g = np.random.default_rng(seed)
    b = [g.choice(y, len(y)).mean() - g.choice(x, len(x)).mean() for _ in range(n)]
    return {"mean": round(float(y.mean() - x.mean()), 4),
            "ci95": [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)], "n": [len(x), len(y)]}


def main():
    dev = torch.device("cuda")
    out = Path(f"{R}/teacher_memorization_source_v1"); out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(f"{R}/frozen_cf3e_2026-11-27/cf3e_long_hf/config.json", [])
    cf2 = json.load(open(f"{R}/cf2_long_hf/config.json"))
    d1 = json.load(open(f"{R}/d1_bv_long_hf/config.json"))

    def leap(c):
        return {t["npz"]: tuple(t["rows"]) for t in c["train"] if t["kind"] == "leap" and "/prim/" not in t["npz"]}
    L3, Lc, Ld = leap(cfg), leap(cf2), leap(d1)
    assert len(L3) == 48
    for k in L3:
        assert L3[k] == (0, 417) and Lc[k] == (0, 417) and Ld[k] == (0, 48), k
    # rows 900-942 must not be used for training by any teacher (only as validation)
    for c in (cfg, cf2, d1):
        for t in c["train"]:
            if t["npz"] in L3:
                assert tuple(t["rows"])[1] <= 417
    for c in (cfg, cf2, d1):
        kite = [t for t in c["train"] if t["kind"] == "kite"]
        assert {os.path.basename(t["npz"]) for t in kite} == {"allegro_right_train.npz", "barrett_train.npz",
                                                              "allegro_right_train_extra_s303.npz", "barrett_train_extra_s303.npz"}
        assert all("rows" not in t for t in kite)
    hands = sorted(L3)
    leap_sets = {"A_all_seen": [0, 48], "B_cf2_seen": [48, 96], "C_none_seen": [900, 942]}
    kv = {v["robot"]: v["npz"] for v in cfg["val"] if v["kind"] == "kite"}
    kt = {t["robot"]: t["npz"] for t in cfg["train"] if t["kind"] == "kite" and "extra" not in t["npz"]}
    kite_sets = {rb: {"train": {"kind": "kite", "robot": rb, "npz": kt[rb], "rows": [0, 500]},
                      "new": {"kind": "kite", "robot": rb, "npz": kv[rb]}} for rb in ("allegro_right", "barrett")}

    torch.manual_seed(0); noise = start_noise(cfg)
    bank = build_bank(cfg, dev)
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    tpaths = json.load(open(f"{R}/frozen_cf3e_2026-11-27/cf3e_long_hf/distill.json"))["teachers"]
    assert tpaths == cfg["distill_teachers"]
    teachers = [load_teacher(paths.expand(p), dev) for p in tpaths]
    scfg = {"success_pos_mm": POS_MM, "success_normal_deg": ANG_DEG}

    def per_record(spec):
        """group -> metric -> per-record array."""
        rec = load_pool([spec], bank).to(dev)
        sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
        acc = {g: {m: [] for m in METRICS} for g in GROUPS}
        for b in range(0, len(rec), 256):
            i = slice(b, min(b + 256, len(rec)))
            g_ = sg_index(sg, i)
            sub = rec.index(i)
            with torch.no_grad():
                z = psi(rec.c_pos[i], rec.c_normal[i], rec.c_valid[i])
                outs = [T(bank, rec.r[i], g_.qc1, g_.R1, g_.t1, z) for T in teachers]
                for g, ix in GROUPS.items():
                    qc = sum(outs[j]["qc"] for j in ix) / len(ix)
                    Rm = chordal_mean([outs[j]["R"] for j in ix])
                    t = sum(outs[j]["t"] for j in ix) / len(ix)
                    sc = score(bank, sub, sub.r, g_, qc, Rm, t, scfg)
                    for m in METRICS:
                        acc[g][m].append(sc[m].float().cpu().numpy())
        return {g: {m: np.concatenate(v) for m, v in d.items()} for g, d in acc.items()}

    res = {"leap": {}, "kite": {}}
    # LEAP: per-hand means
    per = {s: {g: {m: [] for m in METRICS} for g in GROUPS} for s in leap_sets}
    for k, h in enumerate(hands):
        for s, rows in leap_sets.items():
            pr = per_record({"kind": "leap", "npz": h, "rows": rows})
            for g in GROUPS:
                for m in METRICS:
                    per[s][g][m].append(float(pr[g][m].mean()))
        if k % 8 == 0:
            print("leap", k, os.path.basename(h), flush=True)
    per = {s: {g: {m: np.array(v) for m, v in d.items()} for g, d in dd.items()} for s, dd in per.items()}
    res["leap"]["means"] = {s: {g: {m: round(float(per[s][g][m].mean()), 4) for m in METRICS} for g in GROUPS} for s in leap_sets}
    A, B, C = (per[s] for s in leap_sets)
    for g in GROUPS:
        res["leap"][f"{g}_C_minus_A"] = boot(C[g]["corr_mm"] - A[g]["corr_mm"])
        res["leap"][f"{g}_C_minus_B"] = boot(C[g]["corr_mm"] - B[g]["corr_mm"])
    res["leap"]["did_cf2_vs_d1bv_C_minus_B"] = boot((C["cf2_3"]["corr_mm"] - B["cf2_3"]["corr_mm"])
                                                    - (C["d1bv3"]["corr_mm"] - B["d1bv3"]["corr_mm"]))
    la = res["leap"]["cf2_3_C_minus_B"]["ci95"][0] > 0
    lb = res["leap"]["did_cf2_vs_d1bv_C_minus_B"]["ci95"][0] > 0

    # Allegro / Barrett: per-record, unpaired
    kp = {}
    for rb, ss in kite_sets.items():
        tr, nw = per_record(ss["train"]), per_record(ss["new"])
        res["kite"][rb] = {"means": {s: {g: {m: round(float(x[g][m].mean()), 4) for m in METRICS} for g in GROUPS}
                                     for s, x in (("train", tr), ("new", nw))}}
        for g in GROUPS:
            res["kite"][rb][f"{g}_new_minus_train"] = boot_unpaired(tr[g]["corr_mm"], nw[g]["corr_mm"])
        kp[rb] = res["kite"][rb]["all6_new_minus_train"]["ci95"][0] > 0
        print(rb, res["kite"][rb]["all6_new_minus_train"], flush=True)

    res["predeclared"] = {"leap_a": bool(la), "leap_b": bool(lb), "leap_memorisation_supported": bool(la and lb),
                          **{f"{rb}_gap_present": bool(v) for rb, v in kp.items()}}
    (out / "teacher_memorization_source.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print("wrote", out / "teacher_memorization_source.json")


if __name__ == "__main__":
    main()
