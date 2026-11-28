"""Robustness of per-hand frame choice (evaluation only; frozen CF-3e and CF2; docs/frame_robustness_proposal.md; user-approved
2026-11-28). Why: in v2 the about-n candidates were added after analysing the 9 flip hands, and margin 0.25 mm was picked after v1,
so the 360-pool result shows the method CAN work, not that it generalises. Four checks, rules fixed before running (this file is
committed before the job starts).

Per-hand choice (as v2): for each hand, K calibration records; per candidate frame, mean surf_mm over them (deployment-time inputs
only); switch from rule E to the best candidate only if it is lower by more than the margin (0.25 mm); score the hand's
non-calibration records. Paired bootstrap over scored records (4000 draws), seeds never pooled.

Candidate sets: C6 = rule E; 90/180/270 deg about d; +-90 deg about n (v2). C24 = all 24 proper axis-aligned rotations of the
canonical frame (signed permutation matrices, det +1; includes C6).

CHECK 1 (generic candidates): C24, margin 0.25, K = 32 (first 32 records of each hand), CF-3e s0/s1/s2. PASS if, in every seed:
  (a) 9 flip hands of the 360 pool: perhand - rule E corr_mm, 95 % CI upper bound < 0;
  (b) 105 pool, source-validation LEAP and the other 351 hands of the 360 pool: perhand - rule E corr_mm AND wrist_mm, 95 % CI upper
      bound < +0.05 mm.
  Reported: switches per pool, external-hand switches and corr_mm, C24 vs C6 picks.
CHECK 2 (never-used pool with a known correct frame): the 50 held-out generated bodies P0800-P0849 with a fresh 130-record set
  (split "heldout_frame", deterministic generator, configs/decoder/frame/heldout_frame_plan.json); never used for any frame decision.
  Correct frame = the bodies' native convention (x palmar, y across, z distal), the frame rule E targets. A candidate is "correct"
  if its frame is within 30 deg of native. Misframed = rule E > 30 deg from native. C6 primary, C24 reported; margin 0.25, K = 32.
  PASS if, in every seed: (a) at least 80 % of switched bodies land on a correct candidate (when >= 1 switch); (b) at most 10 % of
  well-framed bodies are switched. Reported: misframed bodies switched / corrected, corr_mm change on misframed and well-framed.
  Caveat (written before running): only ~8 % of bodies are misframed (~4 of 50), so (a) rests on few bodies.
CHECK 3 (calibration robustness): C6, margin 0.25, CF-3e s0/s1/s2, on the 9 flip hands and MetaHand: 5 random calibration sets of
  32 records per hand (rng seed 1000 + i, drawn from all of the hand's records). PASS if, in every seed, at least 80 % of these 10
  hands get the same pick in >= 4 of the 5 random sets as with the first-32 default. Reported: K = 16 and K = 64 (first records)
  agreement with the default, per hand.
CHECK 4 (another model): C6, margin 0.25, K = 32, CF2 s0/s1/s2 (same canonical frame). PASS if rule M's conditions hold in every CF2
  seed: 9 flip hands corr CI upper < 0; 351 other hands, 105 pool and source LEAP corr and wrist CI upper < +0.05 mm.
Overall reading: the per-hand choice is called robust if checks 1-4 all pass. The sealed new-family set remains the clean test.

    PYTHONPATH=src python scripts/decoder/benchmark/frame_robustness_eval.py
"""
from __future__ import annotations

import itertools
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

REPO = Path(paths.REPO)
MARGIN = 0.25
TOL = 0.05
METRICS = ("corr_mm", "corr_success", "surf_mm", "wrist_mm")
FROZEN = "${GETKITE_RUNS}/frozen_cf3e_2026-11-27"


def rot_n(th):
    c, s_ = math.cos(math.radians(th)), math.sin(math.radians(th))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s_], [0.0, s_, c]])


def cube_rotations():
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i, (p, sg) in enumerate(zip(perm, signs)):
                M[i, p] = sg
            if np.linalg.det(M) > 0:
                out.append(M)
    return out


C6 = [("E", rot_d(0)), ("d90", rot_d(90)), ("d180", rot_d(180)), ("d270", rot_d(270)), ("n90", rot_n(90)), ("n270", rot_n(270))]
_c24 = cube_rotations()
assert len(_c24) == 24


def _label(M):
    for n, R in C6:
        if np.allclose(R, M, atol=1e-9):
            return n
    return "r" + "".join(f"{int(round(x)):+d}"[0] + str(abs(int(round(x)))) for x in M.flatten())


C24 = [("E", np.eye(3))] + [(_label(M), M) for M in _c24 if not np.allclose(M, np.eye(3))]
assert all(any(np.allclose(R, R2) for _, R2 in C24) for _, R in C6)


def ckpt(arm, s):
    sfx = "" if s == 0 else f"_s{s}"
    if arm == "cf3e":
        return paths.expand(f"{FROZEN}/cf3e_long_hf{sfx}/best.pt")
    return paths.expand(f"${{GETKITE_RUNS}}/{arm}_long_hf{sfx}/best.pt")


def angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def scores_group(base_cfg, pools, extra_robots, arms, dev, surface_sampling, extra_train=None):
    """per pool: dict(names, fam, rec_r, per[(arm,s)][cand] -> metric arrays), plus rule-E frames Qn."""
    cfg = load_config(str(base_cfg), []); cfg["surface_sampling"] = surface_sampling
    torch.manual_seed(0); noise = start_noise(cfg)
    bcfg = {**cfg, "robots": cfg["robots"] + extra_robots}
    if extra_train:
        bcfg["train"] = cfg["train"] + extra_train
    bank = build_bank(bcfg, dev)
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    systems = {(a, s): load_system(f"{a}_s{s}", {"kind": "j1", "ckpt": ckpt(a, s)}, dev, surface_sampling)[0] for a in arms for s in range(3)}
    _, Q, _ = canonical_bank(bank); Qn = Q.double().cpu().numpy()
    key = (id(bank), tuple(bank.names), str(bank.lower.device))
    variants = {}
    for lab, Rc in C24:
        Qc = Qn @ Rc
        bc = RobotBank([relabel_base(d, q) for d, q in zip(bank.descs, Qc)], device=dev, surface_sampling=bank.surface_sampling)
        variants[lab] = (bc, torch.as_tensor(Qc, dtype=torch.float32, device=dev))
    scfg = {"success_pos_mm": POS_MM, "success_normal_deg": ANG_DEG}
    out = {}
    for pn, sp in (pools(cfg) if callable(pools) else pools).items():
        rec = load_pool(sp, bank).to(dev)
        sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
        chunks = [slice(b, min(b + 256, len(rec))) for b in range(0, len(rec), 256)]
        with torch.no_grad():
            z = torch.cat([psi(rec.c_pos[i], rec.c_normal[i], rec.c_valid[i]) for i in chunks])
        per = {}
        for (a, s), S in systems.items():
            per[(a, s)] = {}
            for lab, _ in C24:
                S.model._cache = {key: variants[lab]}
                acc = {}
                with torch.no_grad():
                    for i in chunks:
                        o = S.predict(bank, rec.r[i], sg_index(sg, i), z[i])
                        sc = score(bank, rec.index(i), rec.r[i], sg_index(sg, i), o["qc"], o["R"], o["t"], scfg)
                        for k in METRICS:
                            acc.setdefault(k, []).append(sc[k].float().cpu().numpy())
                per[(a, s)][lab] = {k: np.concatenate(v) for k, v in acc.items()}
            S.model._cache = {}
        ri = rec.r.cpu().numpy().astype(int)
        out[pn] = {"names": np.array([bank.names[i] for i in ri]), "fam": np.array([str(bank.family[i]) for i in ri]),
                   "ri": ri, "per": per}
        print("scored", pn, len(rec), flush=True)
    return out, bank, Qn


def choose(P, arm_s, cands, names, calib, margin=MARGIN):
    """per-hand pick (label) and per-record metrics of the picked frame."""
    labs = [c for c, _ in cands]
    per = P[arm_s]
    pick, choice = {}, np.zeros(len(names), int)
    for h in np.unique(names):
        m = names == h
        cm = m & calib
        ms = np.array([per[c]["surf_mm"][cm].mean() for c in labs])
        j = int(ms.argmin())
        if ms[0] - ms[j] <= margin:
            j = 0
        pick[h] = labs[j]; choice[m] = j
    stacked = {k: np.stack([per[c][k] for c in labs])[choice, np.arange(len(names))] for k in METRICS}
    return pick, stacked


def first_k(names, k):
    calib = np.zeros(len(names), bool)
    for h in np.unique(names):
        calib[np.flatnonzero(names == h)[:k]] = True
    return calib


def compare(sel, base, m):
    return {mm: paired_boot(sel[mm][m], base[mm][m]) for mm in ("corr_mm", "wrist_mm")}


def main():
    dev = torch.device("cuda")
    out = Path(paths.expand("${GETKITE_RUNS}/frame_robustness_v1")); out.mkdir(parents=True, exist_ok=True)
    B = json.loads((REPO / "configs/decoder/benchmark/transfer_v1_leapval105.json").read_text())
    ss = B["surface_sampling"]
    ang = json.load(open(paths.expand("${GETKITE_RUNS}/transfer_benchmark_v1_leap360/leap_frame_angles.json")))
    flips = sorted(h for h, a in ang.items() if a > FLIP_DEG)
    assert len(flips) == 9
    ext = [spec(h) for h in HANDS]
    pools_a = lambda cfg: {"val": cfg["val"], "leap105": cfg["extra_eval"]["leap_heldout"],
                           "wuji": [{"kind": "kite", "robot": "wujihand_right", "npz": f"{HD}/wujihand_right_eval.npz"}],
                           "shadow": [{"kind": "kite", "robot": "shadow_hand_right", "npz": str(SHADOW_AD / "shadow_eval.npz")}],
                           **{h: [{"kind": "kite", "robot": f"{h}_right", "npz": str(EXT / f"{h}_adapter" / f"{h}_eval.npz")}] for h in HANDS}}
    cache = out / "scores_cache.pt"
    if cache.exists() and "--from-cache" in sys.argv:
        C = torch.load(cache, weights_only=False)
        pools, Qn_b, bank_names = C["pools"], C["Qn_b"], C["bank_names"]
    else:
        pools, Qn_b, bank_names = None, None, None
    if pools is None:
        A, _, _ = scores_group(REPO / "configs/decoder/scratch/c3_psi50k_10x_wrist_leapval105.json", pools_a, [WUJI, SHADOW] + ext,
                               ("cf3e", "cf2"), dev, ss)
        L, _, _ = scores_group(REPO / "configs/decoder/scratch/c3_psi50k_10x_wrist_leap360.json",
                               lambda cfg: {"leap360": cfg["extra_eval"]["leap_heldout"]}, [], ("cf3e", "cf2"), dev, ss)
        hf = sorted(Path(paths.expand("${GETKITE_DATA}/prim_frame_check/heldout_frame")).glob("P08*.npz"))
        assert len(hf) == 50
        bodies = [{"kind": "leap", "npz": str(p)} for p in hf]
        Pb, bank_b, Qn_b = scores_group(paths.expand(f"{FROZEN}/cf3e_long_hf/config.json"), {"bodies": bodies}, [], ("cf3e",), dev, ss)
        pools = {**A, **L, **Pb}
        bank_names = list(bank_b.names)
        torch.save({"pools": pools, "Qn_b": Qn_b, "bank_names": bank_names}, cache)

    def masks(pn):
        n = pools[pn]["names"]
        if pn == "val":
            return {"source_leap": pools[pn]["fam"] == "leap"}
        if pn == "leap360":
            fl = np.isin(n, flips)
            return {"flip": fl, "rest": ~fl}
        return {"all": np.ones(len(n), bool)}

    R = {"margin": MARGIN, "tol": TOL, "flip_hands": flips, "C24": [c for c, _ in C24]}

    def rule_m_checks(arm, cands, tag):
        res, ok = {}, {}
        for s in range(3):
            okc = True
            for pn, gname, kind in (("leap360", "flip", "better"), ("leap360", "rest", "tol"), ("leap105", "all", "tol"), ("val", "source_leap", "tol")):
                P = pools[pn]; n = P["names"]; calib = first_k(n, 32); mk = masks(pn)[gname] & ~calib
                pick, sel = choose(P["per"], (arm, s), cands, n, calib)
                c = compare(sel, P["per"][(arm, s)]["E"], mk)
                hands_g = sorted(set(n[masks(pn)[gname]]))
                sw = int(sum(pick[h] != "E" for h in hands_g))
                if kind == "better":
                    good = c["corr_mm"]["ci95"][1] < 0
                else:
                    good = c["corr_mm"]["ci95"][1] < TOL and c["wrist_mm"]["ci95"][1] < TOL
                okc &= good
                res[f"s{s}/{pn}/{gname}"] = {"paired": c, "switched_hands": sw, "n_hands": len(hands_g), "pass": bool(good),
                                             "picks_switched": {h: pick[h] for h in hands_g if pick[h] != "E"}}
            for pn in ["wuji", "shadow"] + list(HANDS):
                P = pools[pn]; n = P["names"]; calib = first_k(n, 32); m = ~calib
                pick, sel = choose(P["per"], (arm, s), cands, n, calib)
                res[f"s{s}/{pn}"] = {"pick": pick[n[0]], "ruleE_corr": round(float(P["per"][(arm, s)]["E"]["corr_mm"][m].mean()), 4),
                                     "perhand_corr": round(float(sel["corr_mm"][m].mean()), 4)}
            ok[s] = bool(okc)
        res["pass_per_seed"] = ok; res["pass"] = all(ok.values())
        print(tag, "pass", res["pass"], ok, flush=True)
        return res

    R["check1_C24_cf3e"] = rule_m_checks("cf3e", C24, "check1")
    R["ref_C6_cf3e"] = rule_m_checks("cf3e", C6, "ref C6")
    R["check4_C6_cf2"] = rule_m_checks("cf2", C6, "check4")

    # check 2: held-out bodies with known native frame
    P = pools["bodies"]; n = P["names"]
    idx = {nm: i for i, nm in enumerate(bank_names)}
    natang = {h: angle_deg(Qn_b[idx[h]]) for h in np.unique(n)}
    misf = {h for h, a in natang.items() if a > 30}
    c2 = {"n_bodies": len(natang), "misframed": sorted(misf), "rule_e_angle_deg": {h: round(a, 1) for h, a in natang.items()}}
    ok2 = {}
    for cands, tag in ((C6, "C6"), (C24, "C24")):
        cd = dict(cands)
        for s in range(3):
            calib = first_k(n, 32)
            pick, sel = choose(P["per"], ("cf3e", s), cands, n, calib)
            sw = [h for h in pick if pick[h] != "E"]
            correct = {h: angle_deg(Qn_b[idx[h]] @ cd[pick[h]]) <= 30 for h in pick}
            prec = (sum(correct[h] for h in sw) / len(sw)) if sw else None
            wf = [h for h in pick if h not in misf]
            wf_sw = sum(pick[h] != "E" for h in wf) / max(len(wf), 1)
            mm = ~calib
            mis_m = mm & np.isin(n, list(misf)); wf_m = mm & ~np.isin(n, list(misf))
            e = P["per"][("cf3e", s)]["E"]
            D = {"switched": {h: pick[h] for h in sw}, "switch_precision": prec, "wellframed_switched_frac": round(wf_sw, 4),
                 "misframed_corrected": sum(correct[h] for h in misf), "misframed_n": len(misf),
                 "corr_misframed_ruleE_vs_perhand": [round(float(e["corr_mm"][mis_m].mean()), 4) if mis_m.any() else None,
                                                     round(float(sel["corr_mm"][mis_m].mean()), 4) if mis_m.any() else None],
                 "paired_wellframed": compare(sel, e, wf_m)}
            if tag == "C6":
                ok2[s] = bool(((prec is None) or prec >= 0.8) and wf_sw <= 0.10)
            c2[f"{tag}/s{s}"] = D
    c2["pass_per_seed"] = ok2; c2["pass"] = all(ok2.values())
    R["check2_bodies"] = c2
    print("check2 pass", c2["pass"], ok2, flush=True)

    # check 3: calibration robustness on flip hands + MetaHand
    c3, ok3 = {}, {}
    for s in range(3):
        agree = {}
        for pn, hands in (("leap360", flips), ("metahand", None)):
            P = pools[pn]; n = P["names"]
            default, _ = choose(P["per"], ("cf3e", s), C6, n, first_k(n, 32))
            hs = hands if hands is not None else sorted(set(n))
            rnd = []
            for i in range(5):
                g = np.random.default_rng(1000 + i)
                calib = np.zeros(len(n), bool)
                for h in np.unique(n):
                    ix = np.flatnonzero(n == h)
                    calib[g.choice(ix, min(32, len(ix)), replace=False)] = True
                rnd.append(choose(P["per"], ("cf3e", s), C6, n, calib)[0])
            k16, _ = choose(P["per"], ("cf3e", s), C6, n, first_k(n, 16))
            k64, _ = choose(P["per"], ("cf3e", s), C6, n, first_k(n, 64))
            for h in hs:
                agree[h] = {"default": default[h], "random_same": int(sum(r[h] == default[h] for r in rnd)),
                            "random_picks": [r[h] for r in rnd], "k16": k16[h], "k64": k64[h]}
        stable = sum(v["random_same"] >= 4 for v in agree.values()) / len(agree)
        ok3[s] = bool(stable >= 0.8)
        c3[f"s{s}"] = {"hands": agree, "stable_frac": round(stable, 3)}
    c3["pass_per_seed"] = ok3; c3["pass"] = all(ok3.values())
    R["check3_calibration"] = c3
    print("check3 pass", c3["pass"], ok3, flush=True)

    R["overall_robust"] = bool(R["check1_C24_cf3e"]["pass"] and c2["pass"] and c3["pass"] and R["check4_C6_cf2"]["pass"])
    (out / "frame_robustness.json").write_text(json.dumps(R, indent=1, default=str))
    print("OVERALL", R["overall_robust"], {k: R[k]["pass"] for k in ("check1_C24_cf3e", "check2_bodies", "check3_calibration", "check4_C6_cf2")})
    print("wrote", out / "frame_robustness.json")


if __name__ == "__main__":
    main()
