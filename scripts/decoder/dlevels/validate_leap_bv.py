"""B body-variety LEAP validation (2026-11-25): scripts/decoder/g1/validate_hands.py checks, unchanged, for the LEAP hands new in
configs/decoder/scratch/d1_bv_long_hf.json (configs/decoder/dlevels/bodyvar_manifest.json leap_new_vs_A), sharded:
    PYTHONPATH=src python scripts/decoder/dlevels/validate_leap_bv.py --shard 0 --n_shards 8
Original checks (per hand):
 1. decoder adapter (robots.leap_desc / RobotBank) FK == dataset HandBank FK at random joints/wrist poses (all 4096 points)
 2. native model: floating MJCF (MuJoCo) link frames == HandBank FK (tests/test_leap.py check, 5 random poses)
 3. records used: goal joints within limits; FK at the goal pose reproduces the stored contacts (template points)
 4. surface coverage (hand_frame sampling): every finger subtree >= 5% of template has a token; covering radius < 45 mm
 5. coupling: LEAP has no mimic/coupled joints (asserted from the descriptor)
Writes configs/decoder/g1/leapdiv_validation.json.
    PYTHONPATH=src python scripts/decoder/g1/validate_hands.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests")); sys.path.insert(0, str(REPO / "tests/decoder"))
from getkite.decoder.robots import RobotBank, leap_desc, robot_arrays  # noqa: E402
from getkite.decoder.train_scratch import build_bank, load_config, load_pool  # noqa: E402
from getkite.decoder.handbank import HandBank  # noqa: E402
from getkite.decoder.rotations import quat_to_matrix  # noqa: E402
from getkite.decoder.get_scratch import J_MAX  # noqa: E402
from test_surface_sampling import _finger_of_slot, _hand_points  # noqa: E402
from test_leap import test_floating_mjcf_matches_our_fk  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--n_shards", type=int, default=1)
    args = ap.parse_args()
    man = json.loads((REPO / "configs/decoder/dlevels/bodyvar_manifest.json").read_text())
    todo = sorted(man["leap_new_vs_A"])[args.shard::args.n_shards]
    cfg = load_config(str(REPO / "configs/decoder/scratch/d1_bv_long_hf.json"), [])
    cfg["robots"] = [r for r in cfg["robots"] if r["kind"] == "kite" or Path(r.get("hand_dir", "")).name in todo]
    cfg["train"] = [t for t in cfg["train"] if t["kind"] == "kite" or Path(t["npz"]).stem in todo]
    hands = [Path(r["hand_dir"]) for r in cfg["robots"] if r["kind"] == "leap"]
    bank = build_bank(cfg, "cpu")
    train = load_pool(cfg["train"], bank)
    out, ok_all = {}, True
    for hd in hands:
        name = hd.name; R = {}
        d = leap_desc(hd)
        hb = HandBank([name], hd.parent); rb = RobotBank([d])
        g = torch.Generator().manual_seed(0); n = int(hb.n_dof[0]); err = 0.0
        for _ in range(5):
            q = torch.zeros(1, 16); q[0, :n] = hb.lower[0, :n] + torch.rand(n, generator=g) * hb.range[0, :n]
            R0 = quat_to_matrix(torch.randn(1, 4, generator=g)); t0 = torch.rand(1, 3, generator=g) * 0.8 - 0.4
            h = torch.zeros(1, dtype=torch.long)
            Rh, th = hb.fk(h, q, R0, t0); ph, _ = hb.all_points(h, Rh, th)
            qc = torch.zeros(1, J_MAX); qc[0, :n] = q[0, :n]
            Rr, tr = rb.fk(h, rb.expand(h, qc), R0, t0); pr, _ = rb.points(h, Rr, tr, torch.arange(4096)[None])
            err = max(err, float((ph - pr).abs().max()))
        R["adapter_fk_max_abs_m"] = err
        try:
            with tempfile.TemporaryDirectory() as tmp:
                test_floating_mjcf_matches_our_fk(name, Path(tmp)); R["mjcf_native_fk_1e-6"] = True
        except Exception as e:  # noqa: BLE001
            R["mjcf_native_fk_1e-6"] = f"FAIL {type(e).__name__}: {e}"
        a = robot_arrays(d, "hand_frame")
        P = _hand_points(a); finger = _finger_of_slot(a.parent_slot[: int(a.n_joint)])
        f_of_pt = np.array([finger[int(s)] for s in a.tpl_anchor]); sel = set(f_of_pt[a.surf_idx].tolist())
        R["fingers_uncovered"] = [int(f) for f, c in zip(*np.unique(f_of_pt, return_counts=True)) if f != 0 and c >= 0.05 * len(P) and f not in sel]
        R["cover_radius_mm"] = float(np.sqrt(((P[:, None] - P[a.surf_idx][None]) ** 2).sum(-1)).min(1).max() * 1000)
        R["mimic_joints"] = int(sum(1 for m in getattr(a, "mimic_src", []) if m >= 0)) if hasattr(a, "mimic_src") else 0
        ri = bank.names.index(name) if hasattr(bank, "names") else None
        R["n_dof"] = n
        if ri is not None and (train.r == ri).any():
            s = (train.r == ri).nonzero().squeeze(1); rbk = train.index(s)
            lo, hi = bank.coords_of(rbk.r, bank.lower[rbk.r]), bank.coords_of(rbk.r, bank.upper[rbk.r])
            cv = bank.coord_valid[rbk.r]
            R["goal_out_of_limits"] = int((((rbk.qc2 < lo - 1e-6) | (rbk.qc2 > hi + 1e-6)) & cv).any(1).sum())
            Rs, ts = bank.fk(rbk.r, bank.expand(rbk.r, rbk.qc2), rbk.R2, rbk.t2)
            P2, _ = bank.points(rbk.r, Rs, ts, rbk.c_tpl.clamp_min(0))
            R["goal_contact_reproduction_max_m"] = float(((P2 - rbk.c_pos).norm(dim=-1) * rbk.c_valid).max())
            R["n_train_records"] = int(len(s))
        ok = (R["adapter_fk_max_abs_m"] < 1e-5 and R["mjcf_native_fk_1e-6"] is True and not R["fingers_uncovered"]
              and R["cover_radius_mm"] < 45 and R["mimic_joints"] == 0 and R.get("goal_out_of_limits", 0) == 0
              and R.get("goal_contact_reproduction_max_m", 0) < 1e-5)
        R["pass"] = bool(ok); ok_all &= ok; out[name] = R
        print(name, json.dumps(R), flush=True)
    res = {"all_pass": bool(ok_all), "n_hands": len(out), "hands": out}
    (REPO / f"configs/decoder/dlevels/bodyvar_validation_{args.shard}of{args.n_shards}.json").write_text(json.dumps(res, indent=1))
    print("ALL PASS" if ok_all else "FAILURES", flush=True)


if __name__ == "__main__":
    main()
