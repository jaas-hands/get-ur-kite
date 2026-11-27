"""Where does the start error rotate around? (evaluation only, existing checkpoints; docs/canonical_origin_proposal.md, "Why CF-O failed").

Hypothesis: KITE's start noise (multifamily.draw_start) rotates the wrist about the NATIVE base origin, so the native frame gets the
rotation centre for free, while CF-O (origin at the palm centre o) has to infer a family-specific lever-arm translation (R2 - R1) o.
Scores CF2 and CF-O (seeds 0-2) on the same records under three start-noise conditions, same seed and noise draws:
  native : KITE's recipe (rotation about the native origin) - replicates the benchmark;
  norot  : rotation noise 0 (translation + joint noise unchanged);
  palm   : the same rotation draws, but about the palm centre o (CF-O rule O1, native frame): t1 = t2 + (R2 - R1) o + noise.
Prediction if the hypothesis holds: CF-O's Wuji/Shadow deficit vs CF2 (native) mostly vanishes under norot and reverses under palm.

    PYTHONPATH=src python scripts/decoder/benchmark/start_centre_eval.py --out '${GETKITE_RUNS}/start_centre_v1'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from getkite.decoder import paths
from getkite.decoder.multifamily import fixed_start
from getkite.decoder.robots import canonical_bank
from getkite.decoder.train_scratch import FrozenPsi, build_bank, load_config, load_pool, start_noise

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "diagnostics"))
from crossbody_heldout_eval import ANG_DEG, HD, POS_MM, WUJI, sg_index, sg_to  # noqa: E402
from transfer_benchmark import SHADOW, SHADOW_AD, load_system, score  # noqa: E402
from wuji_per_example import paired_boot  # noqa: E402

REPO = Path(paths.REPO)
ARMS = {"cf2": "cf2_long_hf", "cfo": "cfo_long_hf"}
SEEDS = (0, 1, 2)
METRICS = ("corr_mm", "compat_success", "wrist_mm", "wrist_deg")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--max-records", type=int, default=0, help="smoke test: cap records per pool")
    a = ap.parse_args()
    B = json.loads((REPO / "configs/decoder/benchmark/transfer_v1_leapval105.json").read_text())
    out = Path(paths.expand(a.out)); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(str(REPO / B["base_config"]), []); cfg["surface_sampling"] = B["surface_sampling"]
    torch.manual_seed(0)
    noise = start_noise(cfg)
    bank = build_bank({**cfg, "robots": cfg["robots"] + [WUJI, SHADOW]}, dev)
    _, _, infos = canonical_bank(bank, origin=True)
    O = torch.tensor(np.array([i["o"] for i in infos]), dtype=torch.float32, device=dev)  # palm centre, native base frame
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    sysm = {(arm, s): load_system(f"{arm}_s{s}", {"kind": "j1", "ckpt": paths.expand(f"${{GETKITE_RUNS}}/{run}{'' if s == 0 else f'_s{s}'}/best.pt")},
                                  dev, B["surface_sampling"])[0] for arm, run in ARMS.items() for s in SEEDS}
    scfg = {"success_pos_mm": POS_MM, "success_normal_deg": ANG_DEG}
    pools = {"val": cfg["val"], "leap_heldout": cfg["extra_eval"]["leap_heldout"],
             "wuji": [{"kind": "kite", "robot": "wujihand_right", "npz": f"{HD}/wujihand_right_eval.npz"}],
             "shadow": [{"kind": "kite", "robot": "shadow_hand_right", "npz": str(SHADOW_AD / "shadow_eval.npz")}]}
    fam = np.array([str(x) for x in bank.family], dtype=object)
    Rres = {"arms": list(ARMS), "seeds": list(SEEDS), "palm_centre_norm_cm": {}, "pools": {}}
    for pn, spec in pools.items():
        rec = load_pool(spec, bank).to(dev)
        if a.max_records:
            rec = rec.index(torch.arange(min(a.max_records, len(rec)), device=rec.r.device))
        sg_nat = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
        sg_nor = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **{**noise, "rot_std_deg": 0.0}), dev)
        o = O[rec.r]
        t_palm = sg_nat.tg + torch.einsum("nij,nj->ni", sg_nat.Rg - sg_nat.R1, o) + (sg_nat.t1 - sg_nat.tg)
        sg_palm = type(sg_nat)(**{**{k: getattr(sg_nat, k) for k in sg_nat.__dataclass_fields__}, "t1": t_palm})
        conds = {"native": sg_nat, "norot": sg_nor, "palm": sg_palm}
        with torch.no_grad():
            z = torch.cat([psi(rec.c_pos[s:s + 256], rec.c_normal[s:s + 256], rec.c_valid[s:s + 256]) for s in range(0, len(rec), 256)])
        fo = fam[rec.r.cpu().numpy().astype(int)]
        per = {}
        for cn, sg in conds.items():
            with torch.no_grad():
                per[(cn, "none")] = {k: v.float().cpu().numpy() for k, v in score(bank, rec, rec.r, sg, sg.qc1, sg.R1, sg.t1, scfg).items()}
            for (arm, s), S in sysm.items():
                oo = []
                with torch.no_grad():
                    for b in range(0, len(rec), 256):
                        i = slice(b, b + 256)
                        oo.append(S.predict(bank, rec.r[i], sg_index(sg, i), z[i]))
                    oo = {k: torch.cat([x[k] for x in oo]) for k in ("qc", "R", "t")}
                    per[(cn, arm, s)] = {k: v.float().cpu().numpy() for k, v in score(bank, rec, rec.r, sg, oo["qc"], oo["R"], oo["t"], scfg).items()}
        res = {}
        for f in sorted(set(fo)):
            sel = fo == f
            rr = np.unique(rec.r.cpu().numpy()[sel])
            Rres["palm_centre_norm_cm"][f] = round(float(O[rr].norm(dim=1).mean() * 100), 2)
            F = {"n": int(sel.sum())}
            for cn in conds:
                C = {"no_movement": {m: round(float(per[(cn, "none")][m][sel].mean()), 4) for m in METRICS}}
                for arm in ARMS:
                    C[arm] = {m: [round(float(per[(cn, arm, s)][m][sel].mean()), 4) for s in SEEDS] for m in METRICS}
                C["cfo_minus_cf2"] = {m: [paired_boot(per[(cn, "cfo", s)][m][sel], per[(cn, "cf2", s)][m][sel]) for s in SEEDS]
                                      for m in ("corr_mm", "wrist_mm")}
                F[cn] = C
                print(pn, f, cn, "cf2", C["cf2"]["wrist_mm"], "cfo", C["cfo"]["wrist_mm"], "none", C["no_movement"]["wrist_mm"], flush=True)
            res[f] = F
        Rres["pools"][pn] = res
        (out / "start_centre.json").write_text(json.dumps(Rres, indent=1))
    print("wrote", out / "start_centre.json")


if __name__ == "__main__":
    main()
