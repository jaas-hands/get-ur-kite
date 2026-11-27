"""Frame selection at inference (evaluation only, existing CF2 checkpoints; docs/cf3_distillation_proposal.md, "Rule-E 90° flips").

Rule E's palm frame Q = [n a d] is sometimes ~90° off about d (n and a swapped): 9/360 held-out LEAP hands and MetaHand. Candidate
fix with no training: per request, run CF2 under Q R_d(θ) for θ in {0, 90, 180, 270} deg and keep the prediction whose hand surface
lies closest to the requested contact positions: surf_mm = mean over the requested contacts of the distance to the nearest predicted
surface point (no correspondences, no normals; only deployment-time inputs: contacts + the robot description). Fixed in advance,
no free parameters, nothing tuned on any pool.

Reported per seed (never pooled), paired bootstrap over records:
  ruleE   : CF2 as deployed (θ = 0; must reproduce the benchmark numbers);
  select  : the selection rule above (primary metric corr_mm; compat_* is near the selection score, read with care);
  oracle4 : best of the 4 by corr_mm (uses correspondences; a ceiling, not a method);
  D1-long, D1-bv (native) on the same records, for reference.
Pools: source val (Allegro, Barrett, LEAP), 105 held-out LEAP, 360 held-out LEAP (split: 9 flip hands > 45° from the training-LEAP
mean frame vs the rest; the split uses oracle knowledge and is for reporting only), Wuji, Shadow, the 7 external hands (evaluation
only, never used to choose anything).

    PYTHONPATH=src python scripts/decoder/benchmark/frame_select_eval.py
"""
from __future__ import annotations

import json
import math
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
from crossbody_heldout_eval import ANG_DEG, HD, POS_MM, WUJI, sg_index, sg_to  # noqa: E402
from transfer_benchmark import SHADOW, SHADOW_AD, load_system, score  # noqa: E402
from wuji_per_example import paired_boot  # noqa: E402
from external_hands_eval import EXT, HANDS, spec  # noqa: E402

REPO = Path(paths.REPO)
THETAS = (0, 90, 180, 270)
METRICS = ("corr_mm", "compat_success", "surf_mm", "wrist_mm", "wrist_deg")
RUNS = {"cf2": "cf2_long_hf", "d1long": "d1_long_hf", "d1bv": "d1_bv_long_hf"}
FLIP_DEG = 45.0


def rot_d(th):
    c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def ckpt(arm, s):
    return paths.expand(f"${{GETKITE_RUNS}}/{RUNS[arm]}{'' if s == 0 else f'_s{s}'}/best.pt")


def run_group(base_cfg, pools, extra_robots, dev, flip_names=None):
    B = json.loads((REPO / "configs/decoder/benchmark/transfer_v1_leapval105.json").read_text())
    cfg = load_config(str(REPO / base_cfg), []); cfg["surface_sampling"] = B["surface_sampling"]
    torch.manual_seed(0); noise = start_noise(cfg)
    bank = build_bank({**cfg, "robots": cfg["robots"] + extra_robots}, dev)
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    systems = {(a, s): load_system(f"{a}_s{s}", {"kind": "j1", "ckpt": ckpt(a, s)}, dev, B["surface_sampling"])[0]
               for a in RUNS for s in range(3)}
    _, Q, _ = canonical_bank(bank); Qn = Q.double().cpu().numpy()
    key = (id(bank), tuple(bank.names), str(bank.lower.device))
    variants = {}
    for th in THETAS:
        Qc = Qn @ rot_d(th)
        bc = RobotBank([relabel_base(d, q) for d, q in zip(bank.descs, Qc)], device=dev, surface_sampling=bank.surface_sampling)
        variants[th] = (bc, torch.as_tensor(Qc, dtype=torch.float32, device=dev))
    scfg = {"success_pos_mm": POS_MM, "success_normal_deg": ANG_DEG}
    fam = np.array([str(x) for x in bank.family], dtype=object)
    res = {}
    for pn, sp in (pools(cfg) if callable(pools) else pools).items():
        rec = load_pool(sp, bank).to(dev)
        sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
        chunks = [slice(b, min(b + 256, len(rec))) for b in range(0, len(rec), 256)]
        with torch.no_grad():
            z = torch.cat([psi(rec.c_pos[i], rec.c_normal[i], rec.c_valid[i]) for i in chunks])

        def predict(S):
            with torch.no_grad():
                o = [S.predict(bank, rec.r[i], sg_index(sg, i), z[i]) for i in chunks]
            return {k: torch.cat([x[k] for x in o]) for k in ("qc", "R", "t")}

        def scored(o):
            per = {}
            with torch.no_grad():
                for i in chunks:
                    sc = score(bank, rec.index(i), rec.r[i], sg_index(sg, i), o["qc"][i], o["R"][i], o["t"][i], scfg)
                    for k, v in sc.items():
                        per.setdefault(k, []).append(v.float().cpu().numpy())
            return {k: np.concatenate(v) for k, v in per.items()}

        names = np.array([bank.names[i] for i in rec.r.cpu().numpy().astype(int)])
        groups = {"all": np.ones(len(rec), bool)}
        if pn == "val":
            groups = {f: fam[rec.r.cpu().numpy().astype(int)] == f for f in sorted(set(fam[rec.r.cpu().numpy().astype(int)]))}
        if flip_names is not None and pn == "leap360":
            fl = np.isin(names, list(flip_names))
            groups = {"all": np.ones(len(rec), bool), "flip": fl, "rest": ~fl}
        per = {}
        for a in ("d1long", "d1bv"):
            for s in range(3):
                per[(a, s)] = scored(predict(systems[(a, s)]))
        chosen = {}
        for s in range(3):
            S = systems[("cf2", s)]
            outs, sc = {}, {}
            for th in THETAS:
                S.model._cache = {key: variants[th]}
                outs[th] = predict(S); sc[th] = scored(outs[th])
            S.model._cache = {}
            surf = np.stack([sc[th]["surf_mm"] for th in THETAS]); corr = np.stack([sc[th]["corr_mm"] for th in THETAS])
            pick, best = surf.argmin(0), corr.argmin(0)
            per[("ruleE", s)] = sc[0]
            per[("select", s)] = {k: np.stack([sc[th][k] for th in THETAS])[pick, np.arange(len(rec))] for k in sc[0]}
            per[("oracle4", s)] = {k: np.stack([sc[th][k] for th in THETAS])[best, np.arange(len(rec))] for k in sc[0]}
            chosen[s] = pick
        G = {}
        for gname, m in groups.items():
            if m.sum() == 0:
                continue
            F = {"n": int(m.sum()), "n_hands": int(len(set(names[m]))), "seeds": {}}
            for s in range(3):
                D = {"means": {k: {mm: round(float(per[(k, s)][mm][m].mean()), 4) for mm in METRICS}
                               for k in ("ruleE", "select", "oracle4", "d1long", "d1bv")},
                     "non_identity_frac": round(float((chosen[s][m] != 0).mean()), 4),
                     "theta_counts": {th: int((chosen[s][m] == j).sum()) for j, th in enumerate(THETAS)}}
                D["paired"] = {f"select-ruleE/{mm}": paired_boot(per[("select", s)][mm][m], per[("ruleE", s)][mm][m]) for mm in ("corr_mm", "wrist_mm")}
                for ref in ("d1long", "d1bv"):
                    for k in ("ruleE", "select"):
                        D["paired"][f"{k}-{ref}/corr_mm"] = paired_boot(per[(k, s)]["corr_mm"][m], per[(ref, s)]["corr_mm"][m])
                F["seeds"][s] = D
            G[gname] = F
            print(pn, gname, F["n"], {s: (F["seeds"][s]["means"]["ruleE"]["corr_mm"], F["seeds"][s]["means"]["select"]["corr_mm"],
                                          F["seeds"][s]["means"]["oracle4"]["corr_mm"], F["seeds"][s]["non_identity_frac"],
                                          F["seeds"][s]["paired"]["select-ruleE/corr_mm"]["ci95"]) for s in range(3)}, flush=True)
        res[pn] = G
    return res


def main():
    dev = torch.device("cuda")
    out = Path(paths.expand("${GETKITE_RUNS}/frame_select_v1")); out.mkdir(parents=True, exist_ok=True)
    R = {"thetas": THETAS, "rule": "argmin surf_mm over Q R_d(theta)", "flip_deg": FLIP_DEG}
    ext = [spec(h) for h in HANDS]
    pools_a = lambda cfg: {"val": cfg["val"], "leap105": cfg["extra_eval"]["leap_heldout"],
                           "wuji": [{"kind": "kite", "robot": "wujihand_right", "npz": f"{HD}/wujihand_right_eval.npz"}],
                           "shadow": [{"kind": "kite", "robot": "shadow_hand_right", "npz": str(SHADOW_AD / "shadow_eval.npz")}],
                           **{h: [{"kind": "kite", "robot": f"{h}_right", "npz": str(EXT / f"{h}_adapter" / f"{h}_eval.npz")}] for h in HANDS}}
    R["pools"] = run_group("configs/decoder/scratch/c3_psi50k_10x_wrist_leapval105.json", pools_a, [WUJI, SHADOW] + ext, dev)
    (out / "frame_select.json").write_text(json.dumps(R, indent=1))
    ang = json.load(open(paths.expand("${GETKITE_RUNS}/transfer_benchmark_v1_leap360/leap_frame_angles.json")))
    flips = {h for h, a in ang.items() if a > FLIP_DEG}
    R["flip_hands_360"] = sorted(flips)
    R["pools"] |= run_group("configs/decoder/scratch/c3_psi50k_10x_wrist_leap360.json",
                            lambda cfg: {"leap360": cfg["extra_eval"]["leap_heldout"]}, [], dev, flip_names=flips)
    (out / "frame_select.json").write_text(json.dumps(R, indent=1))
    print("wrote", out / "frame_select.json")


if __name__ == "__main__":
    main()
