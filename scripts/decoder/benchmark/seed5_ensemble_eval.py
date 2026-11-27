"""Five-seed vs three-seed inference ensemble (docs/variety_and_ensemble_checks.md §3; evaluation only).

For each arm (cf2, d1bv): single seeds s0..s4, the 3-seed ensemble (s0-s2), the 5-seed ensemble (s0-s4) and the 3-seed ensembles
of every seed triple. Same averaging as seed_ensemble_eval.py (mean joints / translation, chordal-mean rotation). Rule (§3):
5 seeds are used at inference if the 5-seed ensemble has significantly lower corr_mm than the 3-seed ensemble on the source pools
and the 105 held-out hands and is not significantly worse on Wuji or Shadow.

    PYTHONPATH=src python scripts/decoder/benchmark/seed5_ensemble_eval.py --out '${GETKITE_RUNS}/seed5_ensemble_v1'
"""
from __future__ import annotations

import argparse
import itertools
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
from seed_ensemble_eval import chordal_mean  # noqa: E402

REPO = Path(paths.REPO)
ARMS = {"cf2": "cf2_long_hf", "d1bv": "d1_bv_long_hf"}
SEEDS = (0, 1, 2, 3, 4)


def ens(outs):
    return {"qc": sum(o["qc"] for o in outs) / len(outs), "t": sum(o["t"] for o in outs) / len(outs), "R": chordal_mean([o["R"] for o in outs])}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    B = json.loads((REPO / "configs/decoder/benchmark/transfer_v1_leapval105.json").read_text())
    out = Path(paths.expand(a.out)); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    cfg = load_config(str(REPO / B["base_config"]), []); cfg["surface_sampling"] = B["surface_sampling"]
    torch.manual_seed(0)
    noise = start_noise(cfg)
    bank = build_bank({**cfg, "robots": cfg["robots"] + [WUJI, SHADOW]}, dev)
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    sysm = {(arm, s): load_system(f"{arm}_s{s}", {"kind": "j1", "ckpt": paths.expand(f"${{GETKITE_RUNS}}/{run}{'' if s == 0 else f'_s{s}'}/best.pt")},
                                  dev, B["surface_sampling"])[0] for arm, run in ARMS.items() for s in SEEDS}
    scfg = {"success_pos_mm": POS_MM, "success_normal_deg": ANG_DEG}
    pools = {"val": cfg["val"], "leap_heldout": cfg["extra_eval"]["leap_heldout"],
             "wuji": [{"kind": "kite", "robot": "wujihand_right", "npz": f"{HD}/wujihand_right_eval.npz"}],
             "shadow": [{"kind": "kite", "robot": "shadow_hand_right", "npz": str(SHADOW_AD / "shadow_eval.npz")}]}
    fam = np.array([str(x) for x in bank.family], dtype=object)
    R = {"arms": list(ARMS), "seeds": list(SEEDS), "pools": {}}
    for pn, spec in pools.items():
        rec = load_pool(spec, bank).to(dev)
        sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
        with torch.no_grad():
            z = torch.cat([psi(rec.c_pos[s:s + 256], rec.c_normal[s:s + 256], rec.c_valid[s:s + 256]) for s in range(0, len(rec), 256)])
        fo = fam[rec.r.cpu().numpy().astype(int)]
        res = {f: {"n": int((fo == f).sum())} for f in sorted(set(fo))}
        for arm in ARMS:
            outs = {}
            for s in SEEDS:
                o = []
                with torch.no_grad():
                    for b in range(0, len(rec), 256):
                        i = slice(b, b + 256)
                        o.append(sysm[(arm, s)].predict(bank, rec.r[i], sg_index(sg, i), z[i]))
                outs[s] = {k: torch.cat([x[k] for x in o]) for k in ("qc", "R", "t")}
            sc = {}
            with torch.no_grad():
                for s in SEEDS:
                    sc[f"s{s}"] = score(bank, rec, rec.r, sg, outs[s]["qc"], outs[s]["R"], outs[s]["t"], scfg)
                for tri in itertools.combinations(SEEDS, 3):
                    e = ens([outs[s] for s in tri]); sc["ens" + "".join(map(str, tri))] = score(bank, rec, rec.r, sg, e["qc"], e["R"], e["t"], scfg)
                e = ens([outs[s] for s in SEEDS]); sc["ens5"] = score(bank, rec, rec.r, sg, e["qc"], e["R"], e["t"], scfg)
            sc = {k: {m: v.float().cpu().numpy() for m, v in d.items()} for k, d in sc.items()}
            for f in res:
                sel = fo == f
                E = {k: {m: round(float(d[m][sel].mean()), 4) for m in ("corr_mm", "compat_success", "wrist_mm")} for k, d in sc.items()}
                E["ens5_vs_ens012"] = {m: paired_boot(sc["ens5"][m][sel], sc["ens012"][m][sel]) for m in ("corr_mm", "wrist_mm")}
                res[f][arm] = E
                print(pn, f, arm, {k: E[k]["corr_mm"] for k in ("s0", "s1", "s2", "s3", "s4", "ens012", "ens5")}, E["ens5_vs_ens012"]["corr_mm"], flush=True)
        R["pools"][pn] = res
        (out / "seed5_ensemble.json").write_text(json.dumps(R, indent=1))
    print("wrote", out / "seed5_ensemble.json")


if __name__ == "__main__":
    main()
