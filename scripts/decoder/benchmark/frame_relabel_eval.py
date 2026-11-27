"""W2-a frame-relabelling test (docs/w2_wrist_frame_proposal.md; evaluation only, existing checkpoints, no training).

The same physical task with every robot's base frame re-expressed by a fixed proper rotation Q:
  base-link template points / normals -> Q^T p, Q^T n; every joint attached to the base link: origin T -> Q^T T;
  start / goal wrist rotations -> R Q (translations, joints and contacts unchanged, so psi's z is unchanged).
The decoder's output R' is mapped back (R = R' Q^T) and scored in the ORIGINAL bank with transfer_benchmark.score and
wrist_fit, so every frame is scored on identical physical quantities. World link poses are identical by construction;
checked numerically per Q (start-pose template points in the world). Q = identity must reproduce the benchmark.
Rotations: the 24 axis-aligned proper rotations (index 0 = identity), applied to all robots at once.
Systems / pools as transfer_v1_followups (G1 and D1, seeds 0/1/2). Results by seed, never pooled.
Predeclared rule: docs/w2_wrist_frame_proposal.md.

    PYTHONPATH=src python scripts/decoder/benchmark/frame_relabel_eval.py --bench configs/decoder/benchmark/transfer_v1_followups.json \
        --systems d0_g1_s0 d0_g1_s1 d0_g1_s2 d1_s0 d1_s1 d1_s2 --out '${GETKITE_RUNS}/w2a_frame_relabel'
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

from getkite.decoder import paths
from getkite.decoder.multifamily import fixed_start
from getkite.decoder.robots import RobotBank
from getkite.decoder.train_scratch import FrozenPsi, build_bank, load_config, load_pool, start_noise

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "diagnostics"))
from crossbody_heldout_eval import ANG_DEG, HD, POS_MM, WUJI, sg_index, sg_to  # noqa: E402
from transfer_benchmark import SHADOW, SHADOW_AD, load_system, score, wrist_fit  # noqa: E402

REPO = Path(paths.REPO)


def rotations24():
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            Q = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                Q[p, i] = s
            if np.linalg.det(Q) > 0:
                out.append(Q)
    out.sort(key=lambda Q: (not np.allclose(Q, np.eye(3)),))  # identity first
    assert len(out) == 24 and np.allclose(out[0], np.eye(3))
    return out


def relabel(desc, Q):
    """Same robot with its base frame rotated by Q (new base axes = old base axes @ Q)."""
    QT4 = np.eye(4); QT4[:3, :3] = Q.T
    joints = [dataclasses.replace(j, T=QT4 @ j.T) if j.parent == desc.base_link else j for j in desc.joints]
    base_i = [i for i, n in enumerate(desc.tpl_link_names) if n == desc.base_link]
    on_base = np.isin(np.asarray(desc.tpl_link), base_i)
    pos, nrm = np.array(desc.tpl_pos, dtype=np.float64), np.array(desc.tpl_normal, dtype=np.float64)
    pos[on_base], nrm[on_base] = pos[on_base] @ Q, nrm[on_base] @ Q  # row form of Q^T p
    return dataclasses.replace(desc, joints=joints, tpl_pos=pos, tpl_normal=nrm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True); ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--n_rot", type=int, default=24)
    a = ap.parse_args()
    B = json.loads(Path(a.bench).read_text())
    out = Path(paths.expand(a.out)); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    bc = Path(paths.expand(B["base_config"]))
    cfg = load_config(str(bc if bc.is_absolute() else REPO / bc), [])
    cfg["surface_sampling"] = B["surface_sampling"]
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
    Qs = rotations24()[: a.n_rot]
    banks, checks = [], {}
    for qi, Q in enumerate(Qs):
        bq = bank if qi == 0 else RobotBank([relabel(d, Q) for d in bank.descs], device=dev, surface_sampling=bank.surface_sampling)
        checks[qi] = {"surf_idx_equal": bool(torch.equal(bq.surf_idx, bank.surf_idx))}
        banks.append(bq)
    R = {"rotations": [Q.tolist() for Q in Qs], "systems": a.systems, "checks": checks, "pools": {}}
    for pn, spec in pools.items():
        rec = load_pool(spec, bank).to(dev)
        sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
        with torch.no_grad():
            z = torch.cat([psi(rec.c_pos[s:s + 256], rec.c_normal[s:s + 256], rec.c_valid[s:s + 256]) for s in range(0, len(rec), 256)])
        fo = fam[rec.r.cpu().numpy().astype(int)]
        res = {f: {n: [] for n in a.systems} for f in sorted(set(fo))}
        for qi, (Q, bq) in enumerate(zip(Qs, banks)):
            Qt = torch.as_tensor(Q, dtype=torch.float32, device=dev)
            sgq = dataclasses.replace(sg, R1=sg.R1 @ Qt, Rg=sg.Rg @ Qt)
            with torch.no_grad():  # world-pose identity check at the start pose
                i = slice(0, min(256, len(rec)))
                q1 = bank.expand(rec.r[i], sg.qc1[i])
                P0, _ = bank.points(rec.r[i], *bank.fk(rec.r[i], q1, sg.R1[i], sg.t1[i]), rec.c_tpl[i].clamp_min(0))
                P1, _ = bq.points(rec.r[i], *bq.fk(rec.r[i], bq.expand(rec.r[i], sg.qc1[i]), sgq.R1[i], sg.t1[i]), rec.c_tpl[i].clamp_min(0))
                checks[qi][f"world_points_max_abs_m/{pn}"] = float((P0 - P1).abs().max())
            for n, S in systems.items():
                outs = []
                with torch.no_grad():
                    for s in range(0, len(rec), 256):
                        i = slice(s, s + 256)
                        o = S.predict(bq, rec.r[i], sg_index(sgq, i), z[i])
                        outs.append({"qc": o["qc"], "R": o["R"] @ Qt.T, "t": o["t"]})
                    o = {k: torch.cat([x[k] for x in outs]) for k in ("qc", "R", "t")}
                    per = score(bank, rec, rec.r, sg, o["qc"], o["R"], o["t"], scfg)
                per = {k: v.float().cpu().numpy() for k, v in per.items()}
                for f in res:
                    sel = fo == f
                    ss = torch.as_tensor(sel, device=dev)
                    wf = wrist_fit(*(x[ss].cpu().numpy() for x in (sg.R1, sg.Rg, o["R"], sg.t1, sg.tg, o["t"])))
                    res[f][n].append({"corr_mm": round(float(per["corr_mm"][sel].mean()), 4),
                                      "compat_success": round(float(per["compat_success"][sel].mean()), 4),
                                      "wrist_mm": round(float(per["wrist_mm"][sel].mean()), 4),
                                      "wrist_deg": round(float(per["wrist_deg"][sel].mean()), 4),
                                      "trans_R2_id": wf["trans_R2_id"], "rot_R2_id": wf["rot_R2_id"], "trans_slope": wf["trans_slope"]})
            print(pn, "rot", qi, json.dumps({f: {n: res[f][n][-1]["trans_R2_id"] for n in a.systems} for f in res}), flush=True)
        R["pools"][pn] = res
        (out / "frame_relabel.json").write_text(json.dumps(R, indent=1))
    print("wrote", out / "frame_relabel.json")


if __name__ == "__main__":
    main()
