"""Did CF-3e learn the mis-framed training bodies in their (wrong) rule-E frame? Evaluation only, frozen CF-3e s0/s1/s2;
user-approved 2026-11-28. Purpose: decide whether training-frame re-framing (Step B) is a variable worth adding or testing before
the larger-data run, so the big run does not change more variables than needed.

Bodies: CF-3e's training bodies P0050-P0799 (P0000-P0049 excluded: CF2 trained on their rows 0-400). Rule-E angle = angle between
the body's rule-E frame and its native palm frame (the generator's convention, which rule E targets). Mis-framed: > 30 deg;
well-framed: < 15 deg; 15-30 deg reported separately.
Records: rows 25-57 of each body's training file (32 records per body) -- never used by CF-3e (rows 0-25) or by any of its
teachers (D1-bv rows 0-25; CF2 only P0000-P0049), so no memorised records.
Each record is scored twice: in the body's rule-E frame (what CF-3e trained and is evaluated with) and in its native frame.

Statistic (per seed): D = corr_mm(rule E) - corr_mm(native) on the mis-framed bodies' records, paired bootstrap over records
(4000 draws); also reported with a bootstrap over bodies. Reading fixed before running:
  * "mixed convention learned": D's 95 % CI upper bound < -0.25 mm (rule-E frame clearly better on unseen records of these bodies)
    in >= 2 of 3 seeds -> CF-3e encodes the flipped frame as a second convention; re-framing could matter -> test Step B on its own
    before it goes into any larger run.
  * "wrong frames not absorbed": D point estimate >= 0 (native as good or better) in >= 2 of 3 seeds -> the mis-framed examples
    barely shaped the model; expected Step-B effect small -> keep CF-3e's convention in the big run (no re-framing variable) and
    only require that the new generator does not raise the mis-frame rate.
  * otherwise: inconclusive -> default as the second case (no extra variable), mis-frame rate check kept.
Reported: the same D for well-framed bodies (sanity: ~0), corr / wrist / success means per group and frame, the 15-30 deg band,
counts. Not measurable here: the effect of the mis-framed examples on the well-framed majority (needs a retrain).

    PYTHONPATH=src python scripts/decoder/benchmark/misframed_training_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from getkite.decoder import paths
from getkite.decoder.multifamily import fixed_start
from getkite.decoder.robots import RobotBank, canonical_bank, relabel_base
from getkite.decoder.train_scratch import FrozenPsi, build_bank, load_config, load_pool, start_noise

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "diagnostics"))
from crossbody_heldout_eval import ANG_DEG, POS_MM, sg_index, sg_to  # noqa: E402
from transfer_benchmark import load_system, score  # noqa: E402
from wuji_per_example import paired_boot  # noqa: E402
from frame_robustness_eval import angle_deg, ckpt  # noqa: E402

METRICS = ("corr_mm", "corr_success", "wrist_mm")
FROZEN = "${GETKITE_RUNS}/frozen_cf3e_2026-11-27"
ROWS = [25, 57]


def body_boot(d, bodies, n=4000, seed=0):
    ub = np.unique(bodies)
    per = np.array([d[bodies == b].mean() for b in ub])
    g = np.random.default_rng(seed)
    m = per[g.integers(0, len(per), (n, len(per)))].mean(1)
    return {"mean_diff": round(float(per.mean()), 4), "ci95": [round(float(np.quantile(m, .025)), 4),
                                                              round(float(np.quantile(m, .975)), 4)], "n_bodies": int(len(ub))}


def main():
    dev = torch.device("cuda")
    out = Path(paths.expand("${GETKITE_RUNS}/misframed_training_v1")); out.mkdir(parents=True, exist_ok=True)
    B = json.loads((Path(paths.REPO) / "configs/decoder/benchmark/transfer_v1_leapval105.json").read_text())
    ss = B["surface_sampling"]
    cfg = load_config(paths.expand(f"{FROZEN}/cf3e_long_hf/config.json"), []); cfg["surface_sampling"] = ss
    torch.manual_seed(0); noise = start_noise(cfg)
    bank = build_bank(cfg, dev)
    _, Q, _ = canonical_bank(bank); Qn = Q.double().cpu().numpy()
    idx = {nm: i for i, nm in enumerate(bank.names)}
    files = [t["npz"] for t in cfg["train"] if "/prim/" in t["npz"] and t.get("rows") == [0, 25]]
    files = [f for f in files if int(Path(f).stem[1:]) >= 50]
    assert len(files) == 750, len(files)
    pool = [{"kind": "leap", "npz": f, "rows": ROWS} for f in files]
    rec = load_pool(pool, bank).to(dev)
    names = np.array([bank.names[i] for i in rec.r.cpu().numpy().astype(int)])
    ang = {h: angle_deg(Qn[idx[h]]) for h in np.unique(names)}
    grp = np.array(["mis" if ang[h] > 30 else ("well" if ang[h] < 15 else "mid") for h in names])
    be = RobotBank([relabel_base(d, q) for d, q in zip(bank.descs, Qn)], device=dev, surface_sampling=bank.surface_sampling)
    variants = {"E": (be, torch.as_tensor(Qn, dtype=torch.float32, device=dev))}  # as frame_robustness_eval's "E" (Qn @ I)
    eye = np.broadcast_to(np.eye(3), Qn.shape).copy()
    bn = RobotBank([relabel_base(d, q) for d, q in zip(bank.descs, eye)], device=dev, surface_sampling=bank.surface_sampling)
    variants["native"] = (bn, torch.as_tensor(eye, dtype=torch.float32, device=dev))
    key = (id(bank), tuple(bank.names), str(bank.lower.device))
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
    chunks = [slice(b, min(b + 256, len(rec))) for b in range(0, len(rec), 256)]
    with torch.no_grad():
        z = torch.cat([psi(rec.c_pos[i], rec.c_normal[i], rec.c_valid[i]) for i in chunks])
    scfg = {"success_pos_mm": POS_MM, "success_normal_deg": ANG_DEG}
    R = {"rows": ROWS, "n_records": int(len(rec)),
         "n_bodies": {g: int(len({h for h in ang if (ang[h] > 30 if g == "mis" else ang[h] < 15 if g == "well" else 15 <= ang[h] <= 30)}))
                      for g in ("mis", "well", "mid")}, "seeds": {}}
    print("bodies", R["n_bodies"], flush=True)
    for s in range(3):
        S, _ = load_system(f"cf3e_s{s}", {"kind": "j1", "ckpt": ckpt("cf3e", s)}, dev, ss)
        per = {}
        for lab, v in variants.items():
            S.model._cache = {key: v}
            acc = {}
            with torch.no_grad():
                for i in chunks:
                    o = S.predict(bank, rec.r[i], sg_index(sg, i), z[i])
                    sc = score(bank, rec.index(i), rec.r[i], sg_index(sg, i), o["qc"], o["R"], o["t"], scfg)
                    for k in METRICS:
                        acc.setdefault(k, []).append(sc[k].float().cpu().numpy())
            per[lab] = {k: np.concatenate(v_) for k, v_ in acc.items()}
        S.model._cache = {}
        res = {}
        for g in ("mis", "well", "mid"):
            m = grp == g
            if not m.any():
                continue
            res[g] = {"means": {lab: {k: round(float(per[lab][k][m].mean()), 4) for k in METRICS} for lab in per},
                      "D_corr_records": paired_boot(per["E"]["corr_mm"][m], per["native"]["corr_mm"][m]),
                      "D_wrist_records": paired_boot(per["E"]["wrist_mm"][m], per["native"]["wrist_mm"][m]),
                      "D_corr_bodies": body_boot(per["E"]["corr_mm"][m] - per["native"]["corr_mm"][m], names[m])}
        d = res["mis"]["D_corr_records"]
        res["learned"] = bool(d["ci95"][1] < -0.25); res["not_absorbed"] = bool(d["mean_diff"] >= 0)
        R["seeds"][f"s{s}"] = res
        print(f"s{s} mis D", d, "well D", res["well"]["D_corr_records"], flush=True)
    nl = sum(R["seeds"][k]["learned"] for k in R["seeds"]); na = sum(R["seeds"][k]["not_absorbed"] for k in R["seeds"])
    R["reading"] = ("mixed convention learned" if nl >= 2 else "wrong frames not absorbed" if na >= 2 else "inconclusive")
    (out / "misframed_training.json").write_text(json.dumps(R, indent=1, default=str))
    print("READING", R["reading"], flush=True)


if __name__ == "__main__":
    main()
