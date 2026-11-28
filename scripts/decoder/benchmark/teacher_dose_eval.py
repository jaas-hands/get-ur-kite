"""Teacher-memorisation v2, Stage 0: dose check (evaluation only; no training; docs/cf3g_v2_paired_proposal.md; user-approved
2026-11-28). How different are memorised vs fresh / honest teacher targets on the exact records each planned arm would train on?
If the difference is tiny, a student trained on it cannot show an effect worth measuring, so that arm stops before training.

Targets exactly as in training and as teacher_memorization_source_eval.py (mean qc, mean t, chordal-mean R over the arm's teacher
set; fixed noisy start val_start_seed), scored with the benchmark's score() against ground truth (corr_mm; wrist_mm reported).

Doses (fresh/honest minus memorised; positive = memorised targets look better):
  A-body : the six CF-3e teachers on generated bodies P0050-P0799 (P0000-P0049 excluded: CF2 trained their rows 0-400),
           rows 25-50 (no teacher saw them) minus rows 0-25 (D1-bv trained them). Paired by body, bootstrap over bodies.
  A-leap : the six CF-3e teachers on the 48 LEAP training hands, rows 417-834 (no teacher saw them) minus rows 0-417 (CF2 trained
           all of them; D1-bv rows 0-48). Paired by hand, bootstrap over hands.
  B-pair : generated bodies P0000-P0799 rows 0-25, honest arm targets (own fold-out teacher x3 + CF2 x3) minus memorised arm
           targets (cyclic fold-in teacher A->B, B->C, C->A, x3 + CF2 x3). Paired by body, bootstrap over bodies.
Rule (fixed before running): an arm proceeds to training only if its dose point estimate is >= 0.10 mm AND its 95 % CI lower
bound is > 0. Otherwise it is reported and stopped. Uses teacher targets only, never student or new-family results.
Also asserted before scoring: no planned training row range overlaps the validation rows (LEAP 900-942 of the training hands;
bodies 410-420 of P0000-P0049), and the fold plan's sha256 prefix matches the one CF-3g v1 used.

    PYTHONPATH=src python scripts/decoder/benchmark/teacher_dose_eval.py
"""
from __future__ import annotations

import hashlib
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
REPO = Path(paths.REPO)
METRICS = ("corr_mm", "wrist_mm")
FOLDS = ("A", "B", "C")
CYC = {"A": "B", "B": "C", "C": "A"}
FOLD_SHA_PREFIX = "56bdcd94"
MIN_DOSE = 0.10


def boot(d, n=4000, seed=0):
    g = np.random.default_rng(seed)
    b = d[g.integers(0, len(d), (n, len(d)))].mean(1)
    return {"mean": round(float(d.mean()), 4), "ci95": [round(float(np.quantile(b, .025)), 4), round(float(np.quantile(b, .975)), 4)],
            "n": int(len(d))}


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def main():
    dev = torch.device("cuda")
    out = Path(f"{R}/teacher_dose_v1"); out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(f"{R}/frozen_cf3e_2026-11-27/cf3e_long_hf/config.json", [])
    fp = REPO / "configs/decoder/dlevels/cf3g_folds.json"
    assert hashlib.sha256(fp.read_bytes()).hexdigest().startswith(FOLD_SHA_PREFIX)
    plan = json.load(open(fp))["bodies"]
    fold_of = {b: f for f in FOLDS for b in plan[f]}
    assert len(fold_of) == 800

    prim = {Path(t["npz"]).stem: t["npz"] for t in cfg["train"] if "/prim/" in t["npz"]}
    leap = sorted(t["npz"] for t in cfg["train"] if t["kind"] == "leap" and "/prim/" not in t["npz"])
    assert len(prim) == 800 and len(leap) == 48
    # leakage: planned training rows vs validation rows
    for v in cfg["val"]:
        if v["npz"] in leap:
            assert tuple(v["rows"]) == (900, 942) and not overlaps((417, 834), v["rows"])
        if "/prim/" in v["npz"]:
            assert not overlaps((25, 50), v["rows"]) and not overlaps((0, 25), v["rows"])

    torch.manual_seed(0); noise = start_noise(cfg)
    bank = build_bank(cfg, dev)
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    six = [load_teacher(paths.expand(p), dev) for p in cfg["distill_teachers"]]  # D1-bv s0-2, CF2 s0-2
    cf2 = six[3:]
    sfx = ("", "_s1", "_s2")
    foldt = {f: [load_teacher(f"{R}/d1_bv_fold{f}_long_hf{s}/best.pt", dev) for s in sfx] for f in FOLDS}
    scfg = {"success_pos_mm": POS_MM, "success_normal_deg": ANG_DEG}
    TS = {"six": six, **{f"fold{f}+cf2": foldt[f] + cf2 for f in FOLDS}}

    def fold_key(n):
        st = n if n in fold_of else Path(n).stem
        assert st in fold_of, n
        return fold_of[st]

    def targets_score(specs, sets):
        """sets: name -> function(robot_name) -> teacher list. Returns names array and name -> metric -> per-record array."""
        rec = load_pool(specs, bank).to(dev)
        names = np.array([bank.names[i] for i in rec.r.cpu().numpy().astype(int)])
        sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
        acc = {s: {m: [] for m in METRICS} for s in sets}
        for b in range(0, len(rec), 256):
            i = slice(b, min(b + 256, len(rec)))
            g_ = sg_index(sg, i); sub = rec.index(i)
            with torch.no_grad():
                z = psi(rec.c_pos[i], rec.c_normal[i], rec.c_valid[i])
                for s, pick in sets.items():
                    # every record in a chunk may need a different teacher set (B-pair): group by teacher set
                    nm = names[i]
                    qc = torch.empty_like(g_.qc1); t = torch.empty_like(g_.t1); Rm = torch.empty_like(g_.R1)
                    for key in sorted(set(pick(n) for n in nm)):
                        ix = torch.as_tensor(np.flatnonzero([pick(n) == key for n in nm]), device=dev)
                        T = TS[key]
                        outs = [M(bank, sub.r[ix], g_.qc1[ix], g_.R1[ix], g_.t1[ix], z[ix]) for M in T]
                        qc[ix] = sum(o["qc"] for o in outs) / len(outs)
                        t[ix] = sum(o["t"] for o in outs) / len(outs)
                        Rm[ix] = chordal_mean([o["R"] for o in outs])
                    sc = score(bank, sub, sub.r, g_, qc, Rm, t, scfg)
                    for m in METRICS:
                        acc[s][m].append(sc[m].float().cpu().numpy())
        return names, {s: {m: np.concatenate(v) for m, v in d.items()} for s, d in acc.items()}

    def by_unit(names, x):
        u = sorted(set(names))
        return np.array([x[names == n].mean() for n in u])

    def dose(names_f, xf, names_m, xm):
        a, b = by_unit(names_f, xf), by_unit(names_m, xm)
        assert len(a) == len(b)
        return boot(a - b)

    res = {"rule": {"min_dose_mm": MIN_DOSE, "ci_lower_gt": 0}}
    six_only = {"six": lambda n: "six"}
    # A-body
    bodies = [b for b in sorted(prim) if int(b[1:]) >= 50]
    nf, xf = targets_score([{"kind": "leap", "npz": prim[b], "rows": [25, 50]} for b in bodies], six_only)
    nm, xm = targets_score([{"kind": "leap", "npz": prim[b], "rows": [0, 25]} for b in bodies], six_only)
    res["A_body"] = {m: dose(nf, xf["six"][m], nm, xm["six"][m]) for m in METRICS}
    res["A_body"]["means"] = {"fresh": round(float(xf["six"]["corr_mm"].mean()), 4), "memorised": round(float(xm["six"]["corr_mm"].mean()), 4)}
    print("A_body", res["A_body"], flush=True)
    # A-leap
    nf, xf = targets_score([{"kind": "leap", "npz": h, "rows": [417, 834]} for h in leap], six_only)
    nm, xm = targets_score([{"kind": "leap", "npz": h, "rows": [0, 417]} for h in leap], six_only)
    res["A_leap"] = {m: dose(nf, xf["six"][m], nm, xm["six"][m]) for m in METRICS}
    res["A_leap"]["means"] = {"fresh": round(float(xf["six"]["corr_mm"].mean()), 4), "memorised": round(float(xm["six"]["corr_mm"].mean()), 4)}
    print("A_leap", res["A_leap"], flush=True)
    # B-pair
    nb, xb = targets_score([{"kind": "leap", "npz": prim[b], "rows": [0, 25]} for b in sorted(prim)],
                           {"honest": lambda n: f"fold{fold_key(n)}+cf2", "memorised": lambda n: f"fold{CYC[fold_key(n)]}+cf2"})
    res["B_pair"] = {m: boot(by_unit(nb, xb["honest"][m]) - by_unit(nb, xb["memorised"][m])) for m in METRICS}
    res["B_pair"]["means"] = {s: round(float(xb[s]["corr_mm"].mean()), 4) for s in ("honest", "memorised")}
    print("B_pair", res["B_pair"], flush=True)

    res["proceed"] = {k: bool(res[k]["corr_mm"]["mean"] >= MIN_DOSE and res[k]["corr_mm"]["ci95"][0] > 0)
                      for k in ("A_body", "A_leap", "B_pair")}
    (out / "teacher_dose.json").write_text(json.dumps(res, indent=1))
    print("PROCEED", res["proceed"], flush=True)


if __name__ == "__main__":
    main()
