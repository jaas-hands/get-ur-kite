"""Diagnostic (evaluation only, never used to choose anything): is CF2's failure on MetaHand a rule-E frame problem?

For each external hand (+ Wuji, Shadow), CF2 s0-s2 are scored with the rule-E frame Q replaced by Q P for each of the 24 proper
axis permutations / sign flips P (P = I is CF2 as deployed). If some P != I brings MetaHand's error down to the D1 level, rule E
picked the wrong axes for it. Also reports the palm points' PCA eigenvalues (axis ambiguity). Diagnostic only; external hands stay
evaluation-only.

    PYTHONPATH=src python scripts/decoder/benchmark/external_frame_probe.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

from getkite.decoder import paths
from getkite.decoder.multifamily import fixed_start
from getkite.decoder.robots import RobotBank, canonical_bank, palm_points, relabel_base
from getkite.decoder.train_scratch import FrozenPsi, build_bank, load_config, load_pool, start_noise

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "diagnostics"))
from crossbody_heldout_eval import ANG_DEG, HD, POS_MM, WUJI, sg_index, sg_to  # noqa: E402
from transfer_benchmark import SHADOW, SHADOW_AD, load_system, score  # noqa: E402
from external_hands_eval import EXT, HANDS, spec  # noqa: E402

REPO = Path(paths.REPO)


def rotations24():
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            P = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                P[p, i] = s
            if np.linalg.det(P) > 0:
                out.append(P)
    return out


def main():
    B = json.loads((REPO / "configs/decoder/benchmark/transfer_v1_leapval105.json").read_text())
    dev = torch.device("cuda")
    cfg = load_config(str(REPO / B["base_config"]), []); cfg["surface_sampling"] = B["surface_sampling"]
    torch.manual_seed(0); noise = start_noise(cfg)
    bank = build_bank({**cfg, "robots": cfg["robots"] + [WUJI, SHADOW] + [spec(h) for h in HANDS]}, dev)
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    S = {s: load_system(f"cf2_s{s}", {"kind": "j1", "ckpt": paths.expand(f"${{GETKITE_RUNS}}/cf2_long_hf{'' if s == 0 else f'_s{s}'}/best.pt")},
                        dev, B["surface_sampling"])[0] for s in range(3)}
    scfg = {"success_pos_mm": POS_MM, "success_normal_deg": ANG_DEG}
    _, Q, infos = canonical_bank(bank); Qn = Q.double().cpu().numpy()
    Ps = rotations24()
    pools = {h: (f"{h}_right", str(EXT / f"{h}_adapter" / f"{h}_eval.npz")) for h in HANDS}
    pools["wuji"] = ("wujihand_right", f"{HD}/wujihand_right_eval.npz")
    pools["shadow"] = ("shadow_hand_right", str(SHADOW_AD / "shadow_eval.npz"))
    res = {}
    for h, (robot, npz) in pools.items():
        ri = bank.names.index(robot); d = bank.descs[ri]
        Pp = palm_points(d); ev = np.linalg.eigvalsh(np.cov((Pp - Pp.mean(0)).T))[::-1]
        geo = {"palm_pca_eigs_mm2": [round(float(x) * 1e6, 2) for x in ev],
               "info": {k: v for k, v in infos[ri].items() if isinstance(v, (int, float, str, bool))}}
        rec = load_pool([{"kind": "kite", "robot": robot, "npz": npz}], bank).to(dev)
        sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
        with torch.no_grad():
            z = torch.cat([psi(rec.c_pos[s:s + 256], rec.c_normal[s:s + 256], rec.c_valid[s:s + 256]) for s in range(0, len(rec), 256)])
        rows = []
        for P in Ps:
            Qc = Qn.copy(); Qc[ri] = Qn[ri] @ P
            bc = RobotBank([relabel_base(dd, q) for dd, q in zip(bank.descs, Qc)], device=dev, surface_sampling=bank.surface_sampling)
            Qt = torch.as_tensor(Qc, dtype=torch.float32, device=dev)
            v = []
            for Sy in S.values():
                Sy.model._cache = {(id(bank), tuple(bank.names), str(bank.lower.device)): (bc, Qt)}
                with torch.no_grad():
                    o = [Sy.predict(bank, rec.r[i], sg_index(sg, i), z[i]) for i in (slice(b, b + 256) for b in range(0, len(rec), 256))]
                    o = {kk: torch.cat([x[kk] for x in o]) for kk in ("qc", "R", "t")}
                    sc = score(bank, rec, rec.r, sg, o["qc"], o["R"], o["t"], scfg)
                v.append(round(float(sc["corr_mm"].mean()), 3))
            rows.append({"P": P.astype(int).tolist(), "identity": bool(np.allclose(P, np.eye(3))), "corr_mm": v})
        rows.sort(key=lambda r: np.mean(r["corr_mm"]))
        ident = next(r for r in rows if r["identity"])
        res[h] = {"geometry": geo, "identity": ident["corr_mm"], "best": rows[0], "rank_of_identity": rows.index(ident) + 1, "all": rows}
        print(h, "identity", ident["corr_mm"], "best", rows[0]["corr_mm"], rows[0]["P"], "rank", rows.index(ident) + 1, geo, flush=True)
    out = Path(paths.expand("${GETKITE_RUNS}/external_frame_probe")); out.mkdir(parents=True, exist_ok=True)
    (out / "frame_probe.json").write_text(json.dumps(res, indent=1))
    print("wrote", out / "frame_probe.json")


if __name__ == "__main__":
    main()
