"""Seed ensemble (evaluation only, existing checkpoints): average the three seeds' predictions of one arm.

For each arm, the s0/s1/s2 decoders predict from the same start and request; the ensemble output is the mean joint coordinates,
the mean wrist translation and the chordal mean of the wrist rotations (SVD projection of the summed matrices). Outputs are in the
native frame (CF systems map back first), so averaging is like-for-like. Scored with transfer_benchmark.score / wrist_fit.
Reported per arm: ensemble vs each seed (paired bootstrap over records), vs no movement, and ensemble-vs-ensemble between arms.
This is not seed selection: all three seeds are used, always.
Pools: source val, held-out LEAP (105 val-split hands), Wuji, Shadow.

    PYTHONPATH=src python scripts/decoder/benchmark/seed_ensemble_eval.py --out '${GETKITE_RUNS}/seed_ensemble_v1'
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
from transfer_benchmark import SHADOW, SHADOW_AD, load_system, score, wrist_fit  # noqa: E402
from wuji_per_example import paired_boot  # noqa: E402

REPO = Path(paths.REPO)
ARMS = {"d1": "d1_prim50_hf", "d1long": "d1_long_hf", "cf2": "cf2_long_hf", "d1bv": "d1_bv_long_hf", "cfbv": "cf_bv_long_hf",
        "cfj": "cfj_long_hf", "cf3d": "cf3d_long_hf", "cf3e": "cf3e_long_hf"}
METRICS = ("corr_mm", "compat_success", "wrist_mm", "wrist_deg")


def chordal_mean(Rs):
    U, _, Vt = torch.linalg.svd(sum(Rs))
    d = torch.sign(torch.linalg.det(U @ Vt))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    return U @ D @ Vt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--mix", nargs="*", default=[], help="mixed ensembles over all seeds of several arms, e.g. d1bv+cf2")
    ap.add_argument("--timing", action="store_true", help="also time 1 / 3 / 6-checkpoint inference")
    a = ap.parse_args()
    B = json.loads((REPO / "configs/decoder/benchmark/transfer_v1_leapval105.json").read_text())
    out = Path(paths.expand(a.out)); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    cfg = load_config(str(REPO / B["base_config"]), []); cfg["surface_sampling"] = B["surface_sampling"]
    torch.manual_seed(0)
    noise = start_noise(cfg)
    bank = build_bank({**cfg, "robots": cfg["robots"] + [WUJI, SHADOW]}, dev)
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    sysm = {}
    for arm in a.arms:
        for s, sfx in zip(range(3), ("", "_s1", "_s2")):
            sysm[(arm, s)] = load_system(f"{arm}_s{s}", {"kind": "j1", "ckpt": paths.expand(f"${{GETKITE_RUNS}}/{ARMS[arm]}{sfx}/best.pt")},
                                         dev, B["surface_sampling"])[0]
    scfg = {"success_pos_mm": POS_MM, "success_normal_deg": ANG_DEG}
    pools = {"val": cfg["val"], "leap_heldout": cfg["extra_eval"]["leap_heldout"],
             "wuji": [{"kind": "kite", "robot": "wujihand_right", "npz": f"{HD}/wujihand_right_eval.npz"}],
             "shadow": [{"kind": "kite", "robot": "shadow_hand_right", "npz": str(SHADOW_AD / "shadow_eval.npz")}]}
    fam = np.array([str(x) for x in bank.family], dtype=object)
    Rres = {"arms": a.arms, "pools": {}}
    for pn, spec in pools.items():
        rec = load_pool(spec, bank).to(dev)
        sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
        with torch.no_grad():
            z = torch.cat([psi(rec.c_pos[s:s + 256], rec.c_normal[s:s + 256], rec.c_valid[s:s + 256]) for s in range(0, len(rec), 256)])
            nm = {k: v.float().cpu().numpy() for k, v in score(bank, rec, rec.r, sg, sg.qc1, sg.R1, sg.t1, scfg).items()}
        fo = fam[rec.r.cpu().numpy().astype(int)]
        per = {}
        outs_all = {}
        seed_outs = {}
        for arm in a.arms:
            outs = []
            for s in range(3):
                S = sysm[(arm, s)]
                o = []
                with torch.no_grad():
                    for b in range(0, len(rec), 256):
                        i = slice(b, b + 256)
                        o.append(S.predict(bank, rec.r[i], sg_index(sg, i), z[i]))
                o = {k: torch.cat([x[k] for x in o]) for k in ("qc", "R", "t")}
                outs.append(o)
                with torch.no_grad():
                    per[(arm, s)] = {k: v.float().cpu().numpy() for k, v in score(bank, rec, rec.r, sg, o["qc"], o["R"], o["t"], scfg).items()}
            with torch.no_grad():
                e = {"qc": sum(o["qc"] for o in outs) / 3, "t": sum(o["t"] for o in outs) / 3, "R": chordal_mean([o["R"] for o in outs])}
                per[(arm, "ens")] = {k: v.float().cpu().numpy() for k, v in score(bank, rec, rec.r, sg, e["qc"], e["R"], e["t"], scfg).items()}
            outs_all[arm] = e
            seed_outs[arm] = outs
        for mx in a.mix:
            parts = [o for arm in mx.split("+") for o in seed_outs[arm]]
            with torch.no_grad():
                e = {"qc": sum(o["qc"] for o in parts) / len(parts), "t": sum(o["t"] for o in parts) / len(parts),
                     "R": chordal_mean([o["R"] for o in parts])}
                per[(mx, "ens")] = {k: v.float().cpu().numpy() for k, v in score(bank, rec, rec.r, sg, e["qc"], e["R"], e["t"], scfg).items()}
            outs_all[mx] = e
        res = {}
        for f in sorted(set(fo)):
            sel = fo == f
            ss = torch.as_tensor(sel, device=dev)
            F = {"n": int(sel.sum()), "no_movement": {m: round(float(nm[m][sel].mean()), 4) for m in METRICS}}
            for arm in a.arms:
                e = outs_all[arm]
                wf = wrist_fit(*(x[ss].cpu().numpy() for x in (sg.R1, sg.Rg, e["R"], sg.t1, sg.tg, e["t"])))
                F[arm] = {"seeds": {m: [round(float(per[(arm, s)][m][sel].mean()), 4) for s in range(3)] for m in METRICS},
                          "ens": {m: round(float(per[(arm, "ens")][m][sel].mean()), 4) for m in METRICS} | {"trans_R2_id": wf["trans_R2_id"]},
                          "ens_vs_seed_corr_mm": [paired_boot(per[(arm, "ens")]["corr_mm"][sel], per[(arm, s)]["corr_mm"][sel]) for s in range(3)],
                          "ens_vs_no_movement": {m: paired_boot(per[(arm, "ens")][m][sel], nm[m][sel]) for m in ("corr_mm", "wrist_mm")}}
            for mx in a.mix:
                e = outs_all[mx]
                wf = wrist_fit(*(x[ss].cpu().numpy() for x in (sg.R1, sg.Rg, e["R"], sg.t1, sg.tg, e["t"])))
                F[mx] = {"ens": {m: round(float(per[(mx, "ens")][m][sel].mean()), 4) for m in METRICS} | {"trans_R2_id": wf["trans_R2_id"]},
                         "ens_vs_no_movement": {m: paired_boot(per[(mx, "ens")][m][sel], nm[m][sel]) for m in ("corr_mm", "wrist_mm")}}
            F["_ens_pairs"] = {f"{x}-{y}": {m: paired_boot(per[(x, "ens")][m][sel], per[(y, "ens")][m][sel]) for m in ("corr_mm", "wrist_mm")}
                               for x, y in itertools.combinations(list(a.arms) + list(a.mix), 2)}
            res[f] = F
            print(pn, f, {arm: (F[arm]["seeds"]["corr_mm"], F[arm]["ens"]["corr_mm"]) for arm in a.arms},
                  {mx: F[mx]["ens"]["corr_mm"] for mx in a.mix}, flush=True)
        Rres["pools"][pn] = res
        (out / "seed_ensemble.json").write_text(json.dumps(Rres, indent=1))
    if a.timing:  # wall-clock decoder inference (psi computed once and excluded), CUDA-synchronised, after warm-up
        import time
        rec = load_pool(pools["val"], bank).to(dev)
        sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
        with torch.no_grad():
            z = psi(rec.c_pos[:256], rec.c_normal[:256], rec.c_valid[:256])
        T = {}
        for bs in (1, 256):
            i = slice(0, bs)
            for k, members in (("1", [sysm[(a.arms[0], 0)]]), ("3", [sysm[(a.arms[0], s)] for s in range(3)]),
                               ("6", [sysm[(arm, s)] for arm in a.arms[:2] for s in range(3)])):
                with torch.no_grad():
                    for _ in range(5):
                        [S.predict(bank, rec.r[i], sg_index(sg, i), z[i]) for S in members]
                    torch.cuda.synchronize(); t0 = time.perf_counter(); n = 30
                    for _ in range(n):
                        [S.predict(bank, rec.r[i], sg_index(sg, i), z[i]) for S in members]
                    torch.cuda.synchronize()
                T[f"batch{bs}_ckpts{k}_ms"] = round((time.perf_counter() - t0) / n * 1000, 2)
        T["gpu"] = torch.cuda.get_device_name(); T["arms_timed"] = a.arms[:2]
        Rres["timing"] = T; print("timing", T, flush=True)
        (out / "seed_ensemble.json").write_text(json.dumps(Rres, indent=1))
    print("wrote", out / "seed_ensemble.json")


if __name__ == "__main__":
    main()
