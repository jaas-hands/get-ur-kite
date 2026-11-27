"""Item 2 of docs/multipass_origin_checks_proposal.md: base-origin shift (evaluation only, existing checkpoints).

The same physical task with every robot's base ORIGIN moved to o (a fixed offset in the native base frame), axes unchanged:
  base-link template points p -> p - o; every joint attached to the base link: origin translation -> T - o;
  start / goal wrist translations t -> t + R o (rotations, joints and contacts unchanged, so psi's z is unchanged).
The decoder's output translation is mapped back (t = t' - R' o) and scored in the ORIGINAL bank with transfer_benchmark.score
and wrist_fit. World link poses are identical by construction; checked numerically per offset (start pose). o = 0 must
reproduce the benchmark. CF systems recompute their canonical frame from the shifted description (rotation unchanged; checked).
Offsets: 1, 2, 5 cm along +-x, +-y, +-z. Results per seed, never pooled.

    PYTHONPATH=src python scripts/decoder/benchmark/origin_shift_eval.py --systems d1_s0 ... --out '${GETKITE_RUNS}/origin_shift_v1'
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
from getkite.decoder.canonical import CanonicalFrame
from getkite.decoder.robots import RobotBank, canonical_bank, canonical_frame, relabel_base
from getkite.decoder.train_scratch import FrozenPsi, build_bank, load_config, load_pool, start_noise

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "diagnostics"))
from crossbody_heldout_eval import ANG_DEG, HD, POS_MM, WUJI, sg_index, sg_to  # noqa: E402
from transfer_benchmark import SHADOW, SHADOW_AD, load_system, score, wrist_fit  # noqa: E402
from wuji_per_example import paired_boot  # noqa: E402

REPO = Path(paths.REPO)
METRICS = ("corr_mm", "compat_success", "wrist_mm", "wrist_deg")


def shift_origin(desc, o):
    """Same robot with its base origin moved to o (base frame); axes unchanged."""
    o = np.asarray(o, float)
    def mv(T):
        T = np.array(T, float); T[:3, 3] = T[:3, 3] - o; return T
    joints = [dataclasses.replace(j, T=mv(j.T)) if j.parent == desc.base_link else j for j in desc.joints]
    base_i = [i for i, n in enumerate(desc.tpl_link_names) if n == desc.base_link]
    on_base = np.isin(np.asarray(desc.tpl_link), base_i)
    pos = np.array(desc.tpl_pos, dtype=np.float64)
    pos[on_base] = pos[on_base] - o
    return dataclasses.replace(desc, joints=joints, tpl_pos=pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="configs/decoder/benchmark/transfer_v1_bv.json")
    ap.add_argument("--systems", nargs="+", required=True); ap.add_argument("--out", required=True)
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
    conds = [("none", np.zeros(3))] + [(f"{'xyz'[ax]}{'+' if s > 0 else '-'}{cm}cm", np.eye(3)[ax] * s * cm / 100.0)
                                       for cm in (1, 2, 5) for ax in range(3) for s in (1, -1)]
    banks, checks = [], {}
    for name, o in conds:
        bo = bank if name == "none" else RobotBank([shift_origin(d, o) for d in bank.descs], device=dev, surface_sampling=bank.surface_sampling)
        checks[name] = {"surf_idx_equal": bool(torch.equal(bo.surf_idx, bank.surf_idx))}
        banks.append(bo)
    # CF rotation must not depend on the origin (checked on a few robots)
    idx = [0, 1, len(bank.descs) - 2, len(bank.descs) - 1]
    checks["cf_rotation_max_diff"] = max(float(np.abs(canonical_frame(bank.descs[i])[0] - canonical_frame(shift_origin(bank.descs[i], conds[-1][1]))[0]).max()) for i in idx)
    print("checks", checks["cf_rotation_max_diff"], flush=True)
    cf_banks = {}
    if any(isinstance(S.model, CanonicalFrame) and not S.model.origin for S in systems.values()):  # canonical bank per shifted bank, Q computed once
        _, Qc, _ = canonical_bank(bank); Qn = Qc.double().cpu().numpy()
        for (name, o), bo in zip(conds, banks):
            cf_banks[id(bo)] = (RobotBank([relabel_base(d, q) for d, q in zip(bo.descs, Qn)], device=dev,
                                          surface_sampling=bank.surface_sampling), Qc)
    palm = {}
    R = {"conditions": [c[0] for c in conds], "systems": a.systems, "checks": checks, "pools": {}}
    for pn, spec in pools.items():
        rec = load_pool(spec, bank).to(dev)
        sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
        with torch.no_grad():
            z = torch.cat([psi(rec.c_pos[s:s + 256], rec.c_normal[s:s + 256], rec.c_valid[s:s + 256]) for s in range(0, len(rec), 256)])
        fo = fam[rec.r.cpu().numpy().astype(int)]
        res = {f: {n: {} for n in a.systems} for f in sorted(set(fo))}
        base_per = {}
        for (name, o), bo in zip(conds, banks):
            ot = torch.as_tensor(o, dtype=torch.float32, device=dev)
            sgo = dataclasses.replace(sg, t1=sg.t1 + sg.R1 @ ot, tg=sg.tg + sg.Rg @ ot)
            with torch.no_grad():  # world-pose identity check at the start pose
                i = slice(0, min(256, len(rec)))
                P0, _ = bank.points(rec.r[i], *bank.fk(rec.r[i], bank.expand(rec.r[i], sg.qc1[i]), sg.R1[i], sg.t1[i]), rec.c_tpl[i].clamp_min(0))
                P1, _ = bo.points(rec.r[i], *bo.fk(rec.r[i], bo.expand(rec.r[i], sg.qc1[i]), sgo.R1[i], sgo.t1[i]), rec.c_tpl[i].clamp_min(0))
                checks.setdefault(name, {})[f"world_points_max_abs_m/{pn}"] = float((P0 - P1).abs().max())
            for n, S in systems.items():
                if isinstance(S.model, CanonicalFrame) and not S.model.origin:  # CF-O computes its own (tests invariance)
                    S.model._cache = {(id(bo), tuple(bo.names), str(bo.lower.device)): cf_banks[id(bo)]}
                outs = []
                with torch.no_grad():
                    for s in range(0, len(rec), 256):
                        i = slice(s, s + 256)
                        oo = S.predict(bo, rec.r[i], sg_index(sgo, i), z[i])
                        outs.append({"qc": oo["qc"], "R": oo["R"], "t": oo["t"] - oo["R"] @ ot})
                    oo = {k: torch.cat([x[k] for x in outs]) for k in ("qc", "R", "t")}
                    per = {k: v.float().cpu().numpy() for k, v in score(bank, rec, rec.r, sg, oo["qc"], oo["R"], oo["t"], scfg).items()}
                if name == "none":
                    base_per[n] = per
                for f in res:
                    sel = fo == f
                    ss = torch.as_tensor(sel, device=dev)
                    wf = wrist_fit(*(x[ss].cpu().numpy() for x in (sg.R1, sg.Rg, oo["R"], sg.t1, sg.tg, oo["t"])))
                    e = {m: round(float(per[m][sel].mean()), 4) for m in METRICS} | {"trans_R2_id": wf["trans_R2_id"]}
                    if name != "none":
                        e["vs_none_corr_mm"] = paired_boot(per["corr_mm"][sel], base_per[n]["corr_mm"][sel])
                    res[f][n][name] = e
            print(pn, name, json.dumps({f: {n: res[f][n][name]["corr_mm"] for n in a.systems} for f in res}), flush=True)
        R["pools"][pn] = res
        (out / "origin_shift.json").write_text(json.dumps(R, indent=1))
    # context: distance from each robot's native origin to its palm-point centroid (template points on the base link)
    for d in bank.descs:
        base_i = [i for i, nme in enumerate(d.tpl_link_names) if nme == d.base_link]
        on = np.isin(np.asarray(d.tpl_link), base_i)
        if on.any():
            palm[d.name] = round(float(np.linalg.norm(np.asarray(d.tpl_pos)[on].mean(0)) * 100), 2)
    R["palm_centroid_cm_from_origin_baselink_points"] = palm
    (out / "origin_shift.json").write_text(json.dumps(R, indent=1))
    print("wrote", out / "origin_shift.json")


if __name__ == "__main__":
    main()
