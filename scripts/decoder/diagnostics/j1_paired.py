"""Paired per-example comparison: J1 (one jointly trained decoder) vs corrected two-stage (translator -> raw-contact
GET, predicted contacts) vs no movement, on source val, held-out LEAP and near-goal Wuji (2026-11-18). No training.
Same records/starts as crossbody_heldout_eval / wuji_per_example. Also wrist-tracking R^2 (as docs/decoder_audit.md sections 39/40).

    python scripts/decoder/diagnostics/j1_paired.py --out $GETKITE_RUNS/j1_joint50k_hf/paired
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

from getkite.decoder import paths
from getkite.decoder.get_scratch import ScratchConfig
from getkite.decoder.joint_contact_get import JointContactGET
from getkite.decoder.multifamily import fixed_start
from getkite.decoder.spatial_bridge import ContactSetBridge, SetBridgeConfig
from getkite.decoder.train_scratch import FrozenPsi, RawContacts, build_bank, load_config, load_pool, start_noise

sys.path.insert(0, str(Path(__file__).parent))
from crossbody_heldout_eval import HD, WUJI, bridge_contacts, sg_index, sg_to  # noqa: E402
from wuji_per_example import analyse, load_dec, npy, paired_boot  # noqa: E402

RUNS, REPO = paths.get("GETKITE_RUNS"), str(paths.REPO)


def wrist_r2(sg, R, t):
    R1, Rg = npy(sg.R1), npy(sg.Rg)
    need = Rot.from_matrix(np.einsum("nji,njk->nik", R1, Rg)).as_rotvec()
    got = Rot.from_matrix(np.einsum("nji,njk->nik", R1, npy(R))).as_rotvec()
    A, *_ = np.linalg.lstsq(need, got, rcond=None)
    rr = 1 - ((got - need @ A) ** 2).sum() / ((got - got.mean(0)) ** 2).sum()
    tn = np.einsum("nji,nj->ni", R1, npy(sg.tg - sg.t1)); tg = np.einsum("nji,nj->ni", R1, npy(t - sg.t1))
    B, *_ = np.linalg.lstsq(tn, tg, rcond=None)
    tr = 1 - ((tg - tn @ B) ** 2).sum() / ((tg - tg.mean(0)) ** 2).sum()
    return round(float(rr), 3), round(float(tr), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=f"{REPO}/configs/decoder/scratch/c3_psi50k_10x_wrist.json")
    ap.add_argument("--j1", default=f"{RUNS}/j1_joint50k_hf/best.pt")
    ap.add_argument("--ref", default=f"{RUNS}/scratch_c3_raw50k_cf8_10x_wrist_hf/best.pt")
    ap.add_argument("--bridge", default=f"{RUNS}/set_bridge_zworld_c3_20k_b/best.pt")
    ap.add_argument("--out", default=f"{RUNS}/j1_joint50k_hf/paired")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(a.config, [])
    torch.manual_seed(0)
    noise = start_noise(cfg)
    ref, pol, st_r = load_dec(a.ref, dev)
    assert pol == "hand_frame"
    ck = torch.load(a.j1, map_location=dev, weights_only=False)
    assert ck["kind"] == "joint_contact_get"
    j1 = JointContactGET(ScratchConfig.from_dict(ck["model_config"]), SetBridgeConfig(**ck["set_config"])).to(dev)
    j1.load_state_dict(ck["model"]); j1.decoder.amp = False; j1.eval()
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    bs_ = torch.load(a.bridge, map_location=dev)
    bridge = ContactSetBridge(SetBridgeConfig(**bs_["config"])).to(dev); bridge.load_state_dict(bs_["model"]); bridge.eval()
    bank = build_bank({**cfg, "robots": cfg["robots"] + [WUJI], "surface_sampling": pol}, dev)
    pools = {"val": cfg["val"], "leap_heldout": cfg["extra_eval"]["leap_heldout"],
             "wuji": [{"kind": "kite", "robot": "wujihand_right", "npz": f"{HD}/wujihand_right_eval.npz"}]}
    R = {"_meta": {"j1": a.j1, "j1_step": ck.get("step"), "ref": a.ref, "ref_step": st_r, "bridge": a.bridge}}
    raw = RawContacts()
    for pname, spec in pools.items():
        rec = load_pool(spec, bank).to(dev)
        sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
        o = {"two_stage": [], "j1": []}
        with torch.no_grad():
            for s in range(0, len(rec), 256):
                i = slice(s, s + 256); g = sg_index(sg, i)
                z = psi(rec.c_pos[i], rec.c_normal[i], rec.c_valid[i])
                o["two_stage"].append(ref(bank, rec.r[i], g.qc1, g.R1, g.t1, raw(*bridge_contacts(bridge, z))))
                o["j1"].append(j1(bank, rec.r[i], g.qc1, g.R1, g.t1, z))
        o = {k: {m: torch.cat([b[m] for b in v]) for m in ("qc", "R", "t")} for k, v in o.items()}
        res = analyse(bank, rec, sg, o)
        fam = np.array([str(bank.family[i]) for i in rec.r.cpu().numpy()])
        for f in sorted(set(fam)):
            sel = fam == f
            S = {}
            for k in ("no_movement", "two_stage/full", "j1/full", "two_stage/wrist_only", "j1/wrist_only"):
                S[k] = {m: round(float(npy(res[k][m])[sel].mean()), 4) for m in ("corr_mm", "corr_success", "compat_mm", "compat_success")}
            for k in ("two_stage", "j1"):
                ss = torch.as_tensor(sel, device=dev)
                sgs = type(sg)(**{fl: getattr(sg, fl)[ss] for fl in sg.__dataclass_fields__})
                S[k + "/full"]["rot_R2"], S[k + "/full"]["trans_R2"] = wrist_r2(sgs, o[k]["R"][ss], o[k]["t"][ss])
            P = {}
            for m in ("corr_mm", "compat_mm", "corr_success", "compat_success"):
                jj, tt, nn = (npy(res[k][m])[sel] for k in ("j1/full", "two_stage/full", "no_movement"))
                P[f"{m}/j1-two_stage"] = paired_boot(jj, tt)
                P[f"{m}/j1-no_movement"] = paired_boot(jj, nn)
                P[f"{m}/two_stage-no_movement"] = paired_boot(tt, nn)
            S["_paired"] = P; S["_n"] = int(sel.sum())
            R[f"{pname}/{f}"] = S
            print(pname, f, json.dumps(S), flush=True)
    (out / "results.json").write_text(json.dumps(R, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
