"""Step 0 of docs/cf_frame_jitter_proposal.md: dose-response of CF to errors in its own canonical frame (evaluation only).

For CF systems, the canonical frame Q_r of every robot is replaced by Q_r J, J = a rotation by a fixed angle about one of
the canonical axes (n, a, d; both signs). Implemented by filling CanonicalFrame's cache with the bank re-expressed by Q_r J
(robots.relabel_base), so the inner decoder sees start rotation R0 Q_r J and its output is mapped back by (Q_r J)^T. Scoring is
in the original bank (transfer_benchmark.score), identical to the benchmark. Angle 0 must reproduce
transfer_benchmark_v1_leapval105 for the same records.
Pool: all 105 held-out LEAP hands (so start poses match the benchmark); the readout uses the hands whose canonical frame is < 3 deg from the mean training LEAP frame
(leap_frame_angles.json), rows [0, 125) as in the benchmark. Results per seed, never pooled.

    PYTHONPATH=src python scripts/decoder/benchmark/frame_perturb_eval.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from getkite.decoder import paths
from getkite.decoder.multifamily import fixed_start
from getkite.decoder.robots import RobotBank, canonical_bank, relabel_base
from getkite.decoder.train_scratch import FrozenPsi, build_bank, load_config, load_pool, start_noise

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "diagnostics"))
from crossbody_heldout_eval import ANG_DEG, POS_MM, WUJI, sg_index, sg_to  # noqa: E402
from transfer_benchmark import load_system, score  # noqa: E402

REPO = Path(paths.REPO)
BENCH = REPO / "configs/decoder/benchmark/transfer_v1_leapval105.json"
SYSTEMS = ["cf2_s0", "cf2_s1", "cf2_s2"]
ANGLES = [0.0, 2.5, 5.0, 10.0, 15.0]
MAX_ANGLE = 3.0


def main():
    R = os.environ["GETKITE_RUNS"]
    out = Path(R) / "cf2_frame_perturb"; out.mkdir(parents=True, exist_ok=True)
    B = json.loads(BENCH.read_text())
    cfg = load_config(str(REPO / B["base_config"]), []); cfg["surface_sampling"] = B["surface_sampling"]
    dev = torch.device("cuda"); torch.manual_seed(0)
    noise = start_noise(cfg)
    bank = build_bank({**cfg, "robots": cfg["robots"] + [WUJI]}, dev)  # as the benchmark (shadow_pool off)
    ang = json.load(open(f"{R}/transfer_benchmark_v1_leapval105/leap_frame_angles.json"))
    keep = sorted(h for h, a in ang.items() if a < MAX_ANGLE)
    spec = cfg["extra_eval"]["leap_heldout"]  # the full 105-hand pool, so start poses match the benchmark exactly
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    systems = {n: load_system(n, {k: (paths.expand(v) if isinstance(v, str) else v) for k, v in B["systems"][n].items()}, dev,
                              B["surface_sampling"])[0] for n in SYSTEMS}
    scfg = {"success_pos_mm": POS_MM, "success_normal_deg": ANG_DEG}
    rec = load_pool(spec, bank).to(dev)
    sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
    with torch.no_grad():
        z = torch.cat([psi(rec.c_pos[s:s + 256], rec.c_normal[s:s + 256], rec.c_valid[s:s + 256]) for s in range(0, len(rec), 256)])
    print("pool", len(spec), "hands", len(rec), "records;", len(keep), "hands < 3 deg", flush=True)
    _, Q, _ = canonical_bank(bank); Qn = Q.double().cpu().numpy()
    conds = [("none", 0, 0.0)] + [(ax, sgn, a) for a in ANGLES[1:] for ax in "nad" for sgn in (1, -1)]
    names = np.array([bank.names[i] for i in rec.r.cpu().numpy().astype(int)])
    allh = sorted(set(names))
    sub = np.isin(names, keep)
    res = {"hands_lt3": keep, "angles_to_mean": {h: ang[h] for h in keep}, "conditions": [], "per_hand": {}, "mean": {}}
    for ax, sgn, a in conds:
        J = Rotation.from_rotvec(np.eye(3)["nad".index(ax) if ax != "none" else 0] * np.radians(sgn * a)).as_matrix()
        QJ = Qn @ J
        bJ = RobotBank([relabel_base(d, q) for d, q in zip(bank.descs, QJ)], device=dev, surface_sampling=bank.surface_sampling)
        QJt = torch.as_tensor(QJ, dtype=torch.float32, device=dev)
        key = f"{ax}{'+' if sgn >= 0 else '-'}{a:g}"; res["conditions"].append(key)
        for n, S in systems.items():
            cf = S.model
            cf._cache = {(id(bank), tuple(bank.names), str(bank.lower.device)): (bJ, QJt)}
            outs = []
            with torch.no_grad():
                for s in range(0, len(rec), 256):
                    i = slice(s, s + 256)
                    outs.append(S.predict(bank, rec.r[i], sg_index(sg, i), z[i]))
                o = {k: torch.cat([x[k] for x in outs]) for k in ("qc", "R", "t")}
                per = score(bank, rec, rec.r, sg, o["qc"], o["R"], o["t"], scfg)
            per = {k: v.float().cpu().numpy() for k, v in per.items()}
            res["mean"].setdefault(n, {})[key] = {m: round(float(per[m][sub].mean()), 4) for m in ("corr_mm", "compat_success", "wrist_mm", "wrist_deg")}
            res["per_hand"].setdefault(n, {})[key] = {h: round(float(per["corr_mm"][names == h].mean()), 4) for h in allh}
        print(key, json.dumps({n: res["mean"][n][key]["corr_mm"] for n in systems}), flush=True)
        (out / "frame_perturb.json").write_text(json.dumps(res, indent=1))
    print("wrote", out / "frame_perturb.json")


if __name__ == "__main__":
    main()
