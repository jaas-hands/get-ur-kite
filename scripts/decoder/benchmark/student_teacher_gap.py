"""Is CF-3d's Wuji shortfall a coverage problem? (evaluation only, existing checkpoints; docs/cf3_distillation_proposal.md)

For each pool, the six-checkpoint teacher pose (D1-bv s0-s2 + CF2 s0-s2; mean qc, mean t, chordal-mean R, as scored) is compared
with each student. Distance = mean displacement of the hand's surface template points between the two predicted poses (mm; joints
clamped; frame-free). Reported per seed: dist(CF-3d s, teacher), dist(CF2 s, teacher), and the reduction CF2 -> CF-3d, plus the
teachers' own spread (mean distance of each teacher to the teacher mean).

Predeclared reading (before running): coverage is supported if the CF2 -> CF-3d reduction in distance to the teacher is clearly
smaller on Wuji and Shadow than on source validation and the 105 held-out LEAP hands (relative reduction at most half the LEAP one),
in at least 2/3 seeds. Wuji, Shadow and the external hands are measured only; nothing is chosen on them.

    PYTHONPATH=src python scripts/decoder/benchmark/student_teacher_gap.py
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
from crossbody_heldout_eval import HD, WUJI, clamp_q, sg_index, sg_to, surface_points  # noqa: E402
from transfer_benchmark import SHADOW, SHADOW_AD, load_system  # noqa: E402
from seed_ensemble_eval import chordal_mean  # noqa: E402
from external_hands_eval import EXT, HANDS, spec  # noqa: E402

REPO = Path(paths.REPO)
RUNS = {"cf2": "cf2_long_hf", "d1bv": "d1_bv_long_hf", "cf3d": "cf3d_long_hf"}


def ckpt(a, s):
    return paths.expand(f"${{GETKITE_RUNS}}/{RUNS[a]}{'' if s == 0 else f'_s{s}'}/best.pt")


def main():
    dev = torch.device("cuda")
    B = json.loads((REPO / "configs/decoder/benchmark/transfer_v1_leapval105.json").read_text())
    cfg = load_config(str(REPO / B["base_config"]), []); cfg["surface_sampling"] = B["surface_sampling"]
    torch.manual_seed(0); noise = start_noise(cfg)
    bank = build_bank({**cfg, "robots": cfg["robots"] + [WUJI, SHADOW] + [spec(h) for h in HANDS]}, dev)
    fam = np.array([str(x) for x in bank.family], dtype=object)
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    S = {(a, s): load_system(f"{a}_s{s}", {"kind": "j1", "ckpt": ckpt(a, s)}, dev, B["surface_sampling"])[0] for a in RUNS for s in range(3)}
    pools = {"val": cfg["val"], "leap105": cfg["extra_eval"]["leap_heldout"],
             "wuji": [{"kind": "kite", "robot": "wujihand_right", "npz": f"{HD}/wujihand_right_eval.npz"}],
             "shadow": [{"kind": "kite", "robot": "shadow_hand_right", "npz": str(SHADOW_AD / "shadow_eval.npz")}],
             **{h: [{"kind": "kite", "robot": f"{h}_right", "npz": str(EXT / f"{h}_adapter" / f"{h}_eval.npz")}] for h in HANDS}}
    res = {}
    for pn, sp in pools.items():
        rec = load_pool(sp, bank).to(dev)
        sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
        ri = rec.r.cpu().numpy().astype(int)
        D = {k: [] for k in [f"{a}_s{s}" for a in ("cf2", "cf3d") for s in range(3)] + ["teacher_spread"]}
        for b in range(0, len(rec), 256):
            i = slice(b, min(b + 256, len(rec))); r = rec.r[i]
            with torch.no_grad():
                z = psi(rec.c_pos[i], rec.c_normal[i], rec.c_valid[i])
                o = {k: S[k].predict(bank, r, sg_index(sg, i), z) for k in S}
                T = [o[(a, s)] for a in ("d1bv", "cf2") for s in range(3)]
                tq = sum(x["qc"] for x in T) / 6; tt = sum(x["t"] for x in T) / 6; tR = chordal_mean([x["R"] for x in T])

                def pts(qc, R, t):
                    return surface_points(bank, r, clamp_q(bank, r, qc)[0], R, t)[0]

                PT = pts(tq, tR, tt)
                dist = lambda x: ((pts(x["qc"], x["R"], x["t"]) - PT).norm(dim=-1).mean(-1) * 1000).cpu().numpy()
                for a in ("cf2", "cf3d"):
                    for s in range(3):
                        D[f"{a}_s{s}"].append(dist(o[(a, s)]))
                D["teacher_spread"].append(np.mean([dist(x) for x in T], 0))
        D = {k: np.concatenate(v) for k, v in D.items()}
        groups = {f: fam[ri] == f for f in sorted(set(fam[ri]))} if pn == "val" else {"all": np.ones(len(rec), bool)}
        for g, m in groups.items():
            name = f"val/{g}" if pn == "val" else pn
            F = {"n": int(m.sum()), "teacher_spread_mm": round(float(D["teacher_spread"][m].mean()), 3), "seeds": {}}
            for s in range(3):
                c2, c3 = float(D[f"cf2_s{s}"][m].mean()), float(D[f"cf3d_s{s}"][m].mean())
                F["seeds"][s] = {"cf2_to_teacher_mm": round(c2, 3), "cf3d_to_teacher_mm": round(c3, 3),
                                 "reduction_mm": round(c2 - c3, 3), "reduction_rel": round((c2 - c3) / c2, 3)}
            res[name] = F
            print(name, F["n"], "spread", F["teacher_spread_mm"],
                  {s: (v["cf2_to_teacher_mm"], v["cf3d_to_teacher_mm"], v["reduction_rel"]) for s, v in F["seeds"].items()}, flush=True)
    out = Path(paths.expand("${GETKITE_RUNS}/student_teacher_gap")); out.mkdir(parents=True, exist_ok=True)
    (out / "gap.json").write_text(json.dumps(res, indent=1))
    print("wrote", out / "gap.json")


if __name__ == "__main__":
    main()
