"""Recovery from lag, Step 1 (evaluation only; frozen models; docs/lag_robustness_proposal.md; user-approved 2026-11-28).
Training and every gate draw the start pose around the goal with KITE's recipe (draw_start / start_noise: wrist N(0, 5 mm),
rotation N(0, 5 deg) per Euler angle, joints N(0, 5 % of range) on contact-relevant joints). Janice's closed-loop failure began
when the hand lagged beyond ~5 mm / 5 deg / 5 % -- one standard deviation. Here the start noise is scaled by k = 1, 2, 3, 4, 6
(k = 1 reproduces the gates), same records, same fixed start seed (val_start_seed) for every k, so the random direction is the
same and only its size changes (paired across k).

Systems: CF-3e s0-2 (primary); D1-bv s0-2 and CF2 s0-2 (the teachers; descriptive -- Step 2's design question); big s0-2
(descriptive -- does capacity help?). Pools: source val (Allegro, Barrett, LEAP), the 105 pool, Wuji, Shadow.
Readouts: (a) single call: corr_mm, corr_success, wrist_mm per k; (b) repeated calls, CF-3e only: the prediction is fed back as
the next start (qc, R, t), up to 4 calls, per k.

Reading (fixed before running):
  * "lag gap confirmed": CF-3e 105-pool corr_mm at k = 3 minus at k = 1 > 1.0 mm, OR corr_success drop > 0.10, in every seed.
  * "iteration recovers": CF-3e 105-pool corr_mm at k = 3 after the best of calls 1-4 is within 0.5 mm of the single call at
    k = 1, in every seed.
  * If the gap is not confirmed, Step 2 (wider-start training) is not launched and we wait for Janice's closed-loop trace.
  * Teachers: their k = 3 minus k = 1 gap on the 105 pool is reported; if it exceeds CF-3e's, wide-start examples in Step 2 train
    on ground truth only (the design default); otherwise that choice stays as designed (not relaxed after seeing this).

    PYTHONPATH=src python scripts/decoder/benchmark/lag_robustness_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from getkite.decoder import paths
from getkite.decoder.multifamily import fixed_start
from getkite.decoder.train_scratch import FrozenPsi, build_bank, load_config, load_pool, start_noise

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "diagnostics"))
from crossbody_heldout_eval import ANG_DEG, HD, POS_MM, WUJI, sg_index, sg_to  # noqa: E402
from transfer_benchmark import SHADOW, SHADOW_AD, load_system, score  # noqa: E402
from wuji_per_example import paired_boot  # noqa: E402

REPO = Path(paths.REPO)
KS = (1, 2, 3, 4, 6)
CALLS = 4
METRICS = ("corr_mm", "corr_success", "wrist_mm")
CK = {"cf3e": "${GETKITE_RUNS}/frozen_cf3e_2026-11-27/cf3e_long_hf{sfx}/best.pt",
      "d1bv": "${GETKITE_RUNS}/d1_bv_long_hf{sfx}/best.pt", "cf2": "${GETKITE_RUNS}/cf2_long_hf{sfx}/best.pt",
      "big": "${GETKITE_RUNS}/cf3f_big_hf{sfx}/best.pt"}


def main():
    dev = torch.device("cuda")
    out = Path(paths.expand("${GETKITE_RUNS}/lag_robustness_v1")); out.mkdir(parents=True, exist_ok=True)
    B = json.loads((REPO / "configs/decoder/benchmark/transfer_v1_leapval105.json").read_text())
    ss = B["surface_sampling"]
    cfg = load_config(str(REPO / "configs/decoder/scratch/c3_psi50k_10x_wrist_leapval105.json"), []); cfg["surface_sampling"] = ss
    torch.manual_seed(0); base = start_noise(cfg)
    bank = build_bank({**cfg, "robots": cfg["robots"] + [WUJI, SHADOW]}, dev)
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    pools = {"val": cfg["val"], "leap105": cfg["extra_eval"]["leap_heldout"],
             "wuji": [{"kind": "kite", "robot": "wujihand_right", "npz": f"{HD}/wujihand_right_eval.npz"}],
             "shadow": [{"kind": "kite", "robot": "shadow_hand_right", "npz": str(SHADOW_AD / "shadow_eval.npz")}]}
    systems = {(a, s): load_system(f"{a}_s{s}", {"kind": "j1", "ckpt": paths.expand(CK[a].format(sfx="" if s == 0 else f"_s{s}"))},
                                   dev, ss)[0] for a in CK for s in range(3)}
    scfg = {"success_pos_mm": POS_MM, "success_normal_deg": ANG_DEG}
    R = {"ks": KS, "base_noise": base, "calls": CALLS, "pools": {}}
    per = {}
    for pn, sp in pools.items():
        rec = load_pool(sp, bank).to(dev)
        fam = np.array([str(bank.family[i]) for i in rec.r.cpu().numpy().astype(int)])
        chunks = [slice(b, min(b + 256, len(rec))) for b in range(0, len(rec), 256)]
        with torch.no_grad():
            z = torch.cat([psi(rec.c_pos[i], rec.c_normal[i], rec.c_valid[i]) for i in chunks])
        res = {}
        for k in KS:
            nk = {"xyz_std": base["xyz_std"] * k, "rot_std_deg": base["rot_std_deg"] * k, "joint_frac": base["joint_frac"] * k}
            sg0 = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **nk), dev)
            for (a, s), S in systems.items():
                ncalls = CALLS if a == "cf3e" else 1
                acc = {c: {m: [] for m in METRICS} for c in range(1, ncalls + 1)}
                with torch.no_grad():
                    for i in chunks:
                        g = sg_index(sg0, i)
                        for c in range(1, ncalls + 1):
                            o = S.predict(bank, rec.r[i], g, z[i])
                            sc = score(bank, rec.index(i), rec.r[i], g, o["qc"], o["R"], o["t"], scfg)
                            for m in METRICS:
                                acc[c][m].append(sc[m].float().cpu().numpy())
                            g = type(g)(**{**{f: getattr(g, f) for f in g.__dataclass_fields__},
                                           "qc1": o["qc"].to(g.qc1.dtype), "R1": o["R"].to(g.R1.dtype), "t1": o["t"].to(g.t1.dtype)})
                for c in acc:
                    per[(pn, k, a, s, c)] = {m: np.concatenate(v) for m, v in acc[c].items()}
            print(pn, "k", k, "done", flush=True)
        groups = {"val": {f: fam == f for f in ("allegro", "barrett", "leap")}}.get(pn, {"all": np.ones(len(rec), bool)})
        for gname, m_ in groups.items():
            if not m_.any():
                print("empty group", pn, gname, flush=True); continue
            key = f"{pn}:{gname}"
            res = {}
            for (pn2, k, a, s, c), v in per.items():
                if pn2 != pn:
                    continue
                res.setdefault(f"{a}_s{s}", {}).setdefault(f"k{k}", {})[f"call{c}"] = {m: round(float(v[m][m_].mean()), 4) for m in METRICS}
            R["pools"][key] = res
    # readings on the 105 pool
    rd = {"gap": {}, "iter": {}, "teacher_gap": {}}
    for s in range(3):
        k1 = per[("leap105", 1, "cf3e", s, 1)]; k3 = per[("leap105", 3, "cf3e", s, 1)]
        d = paired_boot(k3["corr_mm"], k1["corr_mm"])
        drop = float(k1["corr_success"].mean() - k3["corr_success"].mean())
        rd["gap"][f"s{s}"] = {"corr_k3_minus_k1": d, "success_drop": round(drop, 4),
                              "confirmed": bool(d["mean_diff"] > 1.0 or drop > 0.10)}
        best = min(float(per[("leap105", 3, "cf3e", s, c)]["corr_mm"].mean()) for c in range(1, CALLS + 1))
        rd["iter"][f"s{s}"] = {"k3_best_of_calls": round(best, 4), "k1_single": round(float(k1["corr_mm"].mean()), 4),
                               "recovers": bool(best - float(k1["corr_mm"].mean()) <= 0.5)}
        for a in ("d1bv", "cf2", "big"):
            rd["teacher_gap"].setdefault(a, {})[f"s{s}"] = round(float(per[("leap105", 3, a, s, 1)]["corr_mm"].mean()
                                                                       - per[("leap105", 1, a, s, 1)]["corr_mm"].mean()), 4)
    rd["lag_gap_confirmed"] = all(v["confirmed"] for v in rd["gap"].values())
    rd["iteration_recovers"] = all(v["recovers"] for v in rd["iter"].values())
    R["reading"] = rd
    (out / "lag_robustness.json").write_text(json.dumps(R, indent=1, default=str))
    print("LAG GAP CONFIRMED", rd["lag_gap_confirmed"], "| ITERATION RECOVERS", rd["iteration_recovers"], flush=True)
    print(json.dumps(rd, indent=1))


if __name__ == "__main__":
    main()
