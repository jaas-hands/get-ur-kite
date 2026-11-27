"""Item 1 of docs/multipass_origin_checks_proposal.md: multi-pass inference (evaluation only, existing checkpoints).

Pass 1 = the standard benchmark prediction. Pass k+1 re-runs the same decoder with pass k's output as the start (joints
clamped to limits, wrist pose as predicted); the request z, the robot and the goal are unchanged. Every pass is scored with
transfer_benchmark.score / wrist_fit against the ORIGINAL start and goal in the native bank, so pass 1 reproduces the
benchmark. Pools: source val, held-out LEAP (105 val-split hands, transfer_v1_leapval105 base config), Wuji, Shadow.
Results per seed, paired within seed; never pooled across runs.

    PYTHONPATH=src python scripts/decoder/benchmark/multipass_eval.py --systems d1_s0 ... --out '${GETKITE_RUNS}/multipass_v1'
"""
from __future__ import annotations

import argparse
import dataclasses
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
from crossbody_heldout_eval import ANG_DEG, HD, POS_MM, WUJI, clamp_q, sg_index, sg_to  # noqa: E402
from transfer_benchmark import SHADOW, SHADOW_AD, load_system, score, wrist_fit  # noqa: E402
from wuji_per_example import paired_boot  # noqa: E402

REPO = Path(paths.REPO)
METRICS = ("corr_mm", "compat_mm", "corr_success", "compat_success", "wrist_mm", "wrist_deg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="configs/decoder/benchmark/transfer_v1_leapval105.json")
    ap.add_argument("--systems", nargs="+", required=True); ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    B = json.loads((REPO / a.bench).read_text())
    out = Path(paths.expand(a.out)); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    cfg = load_config(str(REPO / B["base_config"]), []); cfg["surface_sampling"] = B["surface_sampling"]
    torch.manual_seed(0)
    noise = start_noise(cfg)
    bank = build_bank({**cfg, "robots": cfg["robots"] + [WUJI, SHADOW]}, dev)
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    systems = {n: load_system(n, {k: (paths.expand(v) if isinstance(v, str) else v) for k, v in B["systems"][n].items()}, dev,
                              B["surface_sampling"])[0] for n in a.systems}
    scfg = {"success_pos_mm": POS_MM, "success_normal_deg": ANG_DEG}
    pools = {"val": cfg["val"], "leap_heldout": cfg["extra_eval"]["leap_heldout"],
             "wuji": [{"kind": "kite", "robot": "wujihand_right", "npz": f"{HD}/wujihand_right_eval.npz"}],
             "shadow": [{"kind": "kite", "robot": "shadow_hand_right", "npz": str(SHADOW_AD / "shadow_eval.npz")}]}
    fam = np.array([str(x) for x in bank.family], dtype=object)
    R = {"bench": a.bench, "systems": a.systems, "passes": a.passes, "pools": {}}
    PE = {}
    for pn, spec in pools.items():
        rec = load_pool(spec, bank).to(dev)
        sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
        with torch.no_grad():
            z = torch.cat([psi(rec.c_pos[s:s + 256], rec.c_normal[s:s + 256], rec.c_valid[s:s + 256]) for s in range(0, len(rec), 256)])
        fo = fam[rec.r.cpu().numpy().astype(int)]
        with torch.no_grad():
            nm = {k: v.float().cpu().numpy() for k, v in score(bank, rec, rec.r, sg, sg.qc1, sg.R1, sg.t1, scfg).items()}
        res = {f: {"n": int((fo == f).sum()), "no_movement": {m: round(float(nm[m][fo == f].mean()), 4) for m in METRICS}}
               for f in sorted(set(fo))}
        for n, S in systems.items():
            cur = sg
            per_pass = []
            for k in range(1, a.passes + 1):
                outs = []
                with torch.no_grad():
                    for s in range(0, len(rec), 256):
                        i = slice(s, s + 256)
                        outs.append(S.predict(bank, rec.r[i], sg_index(cur, i), z[i]))
                    o = {kk: torch.cat([x[kk] for x in outs]) for kk in ("qc", "R", "t")}
                    per = {kk: v.float().cpu().numpy() for kk, v in score(bank, rec, rec.r, sg, o["qc"], o["R"], o["t"], scfg).items()}
                per_pass.append(per)
                for f in res:
                    sel = fo == f
                    ss = torch.as_tensor(sel, device=dev)
                    wf = wrist_fit(*(x[ss].cpu().numpy() for x in (sg.R1, sg.Rg, o["R"], sg.t1, sg.tg, o["t"])))
                    e = {m: round(float(per[m][sel].mean()), 4) for m in METRICS} | {k2: wf[k2] for k2 in ("trans_R2_id", "rot_R2_id", "trans_slope")}
                    e["vs_pass1"] = {m: paired_boot(per[m][sel], per_pass[0][m][sel]) for m in METRICS} if k > 1 else None
                    e["vs_prev"] = {m: paired_boot(per[m][sel], per_pass[-2][m][sel]) for m in METRICS} if k > 1 else None
                    e["vs_no_movement"] = {m: paired_boot(per[m][sel], nm[m][sel]) for m in METRICS}
                    res[f].setdefault(n, []).append(e)
                qc_c, _ = clamp_q(bank, rec.r, o["qc"])
                cur = dataclasses.replace(sg, qc1=qc_c, R1=o["R"], t1=o["t"])
            for f in res:
                print(pn, f, n, [res[f][n][k]["corr_mm"] for k in range(a.passes)], flush=True)
            PE[f"{pn}/{n}"] = {"corr_mm": np.stack([p["corr_mm"] for p in per_pass]), "names": np.array([bank.names[i] for i in rec.r.cpu().numpy().astype(int)])}
        R["pools"][pn] = res
        (out / "multipass.json").write_text(json.dumps(R, indent=1))
    torch.save(PE, out / "per_example_corr.pt")
    print("wrote", out / "multipass.json")


if __name__ == "__main__":
    main()
