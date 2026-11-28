"""Per-hand frame choice v2 for CF-3e (evaluation only, frozen CF-3e; docs/frame_robustness_proposal.md, Step A; user-approved
2026-11-28).

v1 (frame_perhand_eval.py) searched rule E's frame and its 90/180/270-degree turns about d only. Verified 2026-11-28: of the 9 flip
hands in the 360 pool, 7 are ~80-degree rotations about n (d and a swapped) and 2 are ~90 degrees about d, so v1 could fix only 2
(it switched exactly 2/9). ONE change vs v1: the candidate set adds the two 90-degree turns about n (6 candidates):
rule E; 90/180/270 about d; +90/-90 about n.

Rule (fixed before running; no parameter tuned on any pool): for each hand, its first K = 32 records in pool order are sample
requests; keep the candidate with the lowest mean surf_mm over them (deployment-time inputs only, no correspondences). The hand's
remaining records are scored with that frame. Margin 0 (predeclared); margins 0.25 / 0.5 mm reported descriptively only.

Predeclared gate G1 (per seed, paired bootstrap over the scored records, seeds never pooled): corr_mm and wrist_mm not significantly
higher than rule E on the 105 held-out LEAP pool and on source-validation LEAP, in any seed.
Reported only after G1: the 9 flip hands (7 about n, 2 about d) and the rest of the 360 pool, MetaHand, Wuji, Shadow and the other
external hands (new-family validation; never used to choose anything), fraction of hands switched and the candidate picked,
corr_success, median and p90 corr_mm; D1-bv on the same records for reference.

    PYTHONPATH=src python scripts/decoder/benchmark/frame_perhand_v2_eval.py
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
from frame_select_eval import rot_d, FLIP_DEG  # noqa: E402


def rot_n(th):
    """rotation about the canonical frame's n axis (first column; d <-> a for +-90)."""
    c, s_ = math.cos(math.radians(th)), math.sin(math.radians(th))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s_], [0.0, s_, c]])


CANDS = [("E", rot_d(0)), ("d90", rot_d(90)), ("d180", rot_d(180)), ("d270", rot_d(270)), ("n90", rot_n(90)), ("n270", rot_n(270))]
THETAS = [c[0] for c in CANDS]

REPO = Path(paths.REPO)
K = 32
MARGINS = (0.0, 0.25, 0.5)
METRICS = ("corr_mm", "corr_success", "surf_mm", "wrist_mm")
FROZEN = "${GETKITE_RUNS}/frozen_cf3e_2026-11-27"


def ckpt(arm, s):
    if arm == "cf3e":
        return paths.expand(f"{FROZEN}/cf3e_long_hf{'' if s == 0 else f'_s{s}'}/best.pt")
    return paths.expand(f"${{GETKITE_RUNS}}/d1_bv_long_hf{'' if s == 0 else f'_s{s}'}/best.pt")


def run_group(base_cfg, pools, extra_robots, dev, out_npz, flip_names=None):
    B = json.loads((REPO / "configs/decoder/benchmark/transfer_v1_leapval105.json").read_text())
    cfg = load_config(str(REPO / base_cfg), []); cfg["surface_sampling"] = B["surface_sampling"]
    torch.manual_seed(0); noise = start_noise(cfg)
    bank = build_bank({**cfg, "robots": cfg["robots"] + extra_robots}, dev)
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    systems = {(a, s): load_system(f"{a}_s{s}", {"kind": "j1", "ckpt": ckpt(a, s)}, dev, B["surface_sampling"])[0]
               for a in ("cf3e", "d1bv") for s in range(3)}
    _, Q, _ = canonical_bank(bank); Qn = Q.double().cpu().numpy()
    key = (id(bank), tuple(bank.names), str(bank.lower.device))
    variants = {}
    for th, Rc in CANDS:
        Qc = Qn @ Rc
        bc = RobotBank([relabel_base(d, q) for d, q in zip(bank.descs, Qc)], device=dev, surface_sampling=bank.surface_sampling)
        variants[th] = (bc, torch.as_tensor(Qc, dtype=torch.float32, device=dev))
    scfg = {"success_pos_mm": POS_MM, "success_normal_deg": ANG_DEG}
    fam = np.array([str(x) for x in bank.family], dtype=object)
    res, dump = {}, {}
    for pn, sp in (pools(cfg) if callable(pools) else pools).items():
        rec = load_pool(sp, bank).to(dev)
        sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
        chunks = [slice(b, min(b + 256, len(rec))) for b in range(0, len(rec), 256)]
        with torch.no_grad():
            z = torch.cat([psi(rec.c_pos[i], rec.c_normal[i], rec.c_valid[i]) for i in chunks])

        def scored(S):
            per = {}
            with torch.no_grad():
                for i in chunks:
                    o = S.predict(bank, rec.r[i], sg_index(sg, i), z[i])
                    sc = score(bank, rec.index(i), rec.r[i], sg_index(sg, i), o["qc"], o["R"], o["t"], scfg)
                    for k in METRICS:
                        per.setdefault(k, []).append(sc[k].float().cpu().numpy())
            return {k: np.concatenate(v) for k, v in per.items()}

        ri = rec.r.cpu().numpy().astype(int)
        names = np.array([bank.names[i] for i in ri])
        # calibration records: first K of each hand in pool order; the rest are scored
        calib = np.zeros(len(rec), bool)
        for h in np.unique(names):
            calib[np.flatnonzero(names == h)[:K]] = True
        evalm = ~calib
        ref = {s: scored(systems[("d1bv", s)]) for s in range(3)}
        per, picks = {}, {}
        for s in range(3):
            S = systems[("cf3e", s)]
            sc = {}
            for th in THETAS:
                S.model._cache = {key: variants[th]}
                sc[th] = scored(S)
            S.model._cache = {}
            dump[f"{pn}/s{s}"] = {f"{k}_{th}": sc[th][k] for th in THETAS for k in METRICS}
            per[("ruleE", s)] = sc["E"]
            for mg in MARGINS:
                choice = np.zeros(len(rec), int)
                hp = {}
                for h in np.unique(names):
                    m = names == h
                    ms = np.array([sc[th]["surf_mm"][m & calib].mean() for th in THETAS])
                    j = int(ms.argmin())
                    if ms[0] - ms[j] <= mg:
                        j = 0
                    choice[m] = j; hp[h] = THETAS[j]
                per[(f"perhand_m{mg}", s)] = {k: np.stack([sc[th][k] for th in THETAS])[choice, np.arange(len(rec))] for k in METRICS}
                picks[(mg, s)] = hp
        groups = {"all": np.ones(len(rec), bool)}
        if pn == "val":
            groups = {f: fam[ri] == f for f in sorted(set(fam[ri]))}
        if flip_names is not None and pn == "leap360":
            fl = np.isin(names, list(flip_names)); groups = {"all": groups["all"], "flip": fl, "rest": ~fl}
        G = {}
        for gname, gm in groups.items():
            m = gm & evalm
            if m.sum() == 0:
                continue
            hands_g = sorted(set(names[gm]))
            F = {"n_scored": int(m.sum()), "n_hands": len(hands_g), "seeds": {}}
            for s in range(3):
                D = {"median_p90_corr_mm": {k: [round(float(np.median(v["corr_mm"][m])), 4), round(float(np.percentile(v["corr_mm"][m], 90)), 4)]
                                            for k, v in [("ruleE", per[("ruleE", s)]), ("perhand_m0.0", per[("perhand_m0.0", s)]), ("d1bv", ref[s])]},
                     "means": {k: {mm: round(float(v[mm][m].mean()), 4) for mm in METRICS}
                               for k, v in [("ruleE", per[("ruleE", s)])] + [(f"perhand_m{mg}", per[(f"perhand_m{mg}", s)]) for mg in MARGINS]
                               + [("d1bv", ref[s])]},
                     "switched_frac_hands": {f"m{mg}": round(float(np.mean([picks[(mg, s)][h] != "E" for h in hands_g])), 4) for mg in MARGINS},
                     "picked_m0": {c: int(sum(picks[(0.0, s)][h] == c for h in hands_g)) for c in THETAS},
                     "paired": {}}
                for mg in MARGINS:
                    for mm in ("corr_mm", "wrist_mm"):
                        D["paired"][f"perhand_m{mg}-ruleE/{mm}"] = paired_boot(per[(f"perhand_m{mg}", s)][mm][m], per[("ruleE", s)][mm][m])
                    D["paired"][f"perhand_m{mg}-d1bv/corr_mm"] = paired_boot(per[(f"perhand_m{mg}", s)]["corr_mm"][m], ref[s]["corr_mm"][m])
                if len(hands_g) <= 12:
                    D["picks_m0"] = {h: picks[(0.0, s)][h] for h in hands_g}
                F["seeds"][s] = D
            G[gname] = F
            print(pn, gname, F["n_scored"], {s: (F["seeds"][s]["means"]["ruleE"]["corr_mm"], F["seeds"][s]["means"]["perhand_m0.0"]["corr_mm"],
                                                 F["seeds"][s]["means"]["d1bv"]["corr_mm"], F["seeds"][s]["switched_frac_hands"]["m0.0"],
                                                 F["seeds"][s]["paired"]["perhand_m0.0-ruleE/corr_mm"]["ci95"]) for s in range(3)}, flush=True)
        res[pn] = G
    np.savez_compressed(out_npz, **{f"{k}/{kk}": v for k, d in dump.items() for kk, v in d.items()})
    return res


def gate(R):
    out = {}
    for s in range(3):
        ok = True
        for pool, grp in (("leap105", "all"), ("val", "leap")):
            P = R["pools"][pool][grp]["seeds"][s]["paired"]
            for mm in ("corr_mm", "wrist_mm"):
                ok &= not (P[f"perhand_m0.0-ruleE/{mm}"]["ci95"][0] > 0)
        out[s] = bool(ok)
    return {"G1_per_seed": out, "G1": all(out.values())}


def main():
    dev = torch.device("cuda")
    out = Path(paths.expand("${GETKITE_RUNS}/frame_perhand_cf3e_v2")); out.mkdir(parents=True, exist_ok=True)
    R = {"K": K, "margins": MARGINS, "candidates": THETAS, "rule": "per hand: argmin mean surf_mm over its first K records among 6 candidate frames; scored on the rest"}
    ext = [spec(h) for h in HANDS]
    pools_a = lambda cfg: {"val": cfg["val"], "leap105": cfg["extra_eval"]["leap_heldout"],
                           "wuji": [{"kind": "kite", "robot": "wujihand_right", "npz": f"{HD}/wujihand_right_eval.npz"}],
                           "shadow": [{"kind": "kite", "robot": "shadow_hand_right", "npz": str(SHADOW_AD / "shadow_eval.npz")}],
                           **{h: [{"kind": "kite", "robot": f"{h}_right", "npz": str(EXT / f"{h}_adapter" / f"{h}_eval.npz")}] for h in HANDS}}
    R["pools"] = run_group("configs/decoder/scratch/c3_psi50k_10x_wrist_leapval105.json", pools_a, [WUJI, SHADOW] + ext, dev, out / "per_record_a.npz")
    R["gate"] = gate(R)
    (out / "frame_perhand.json").write_text(json.dumps(R, indent=1))
    print("GATE", R["gate"], flush=True)
    ang = json.load(open(paths.expand("${GETKITE_RUNS}/transfer_benchmark_v1_leap360/leap_frame_angles.json")))
    flips = {h for h, a in ang.items() if a > FLIP_DEG}
    R["flip_hands_360"] = sorted(flips)
    R["pools"] |= run_group("configs/decoder/scratch/c3_psi50k_10x_wrist_leap360.json",
                            lambda cfg: {"leap360": cfg["extra_eval"]["leap_heldout"]}, [], dev, out / "per_record_360.npz", flip_names=flips)
    (out / "frame_perhand.json").write_text(json.dumps(R, indent=1))
    print("wrote", out / "frame_perhand.json")


if __name__ == "__main__":
    main()
