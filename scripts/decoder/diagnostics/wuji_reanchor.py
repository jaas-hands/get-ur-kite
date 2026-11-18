"""Coordinate-sensitivity test: re-anchor Wuji's base frame, keep exact physical starts/goals (2026-11-18).

No training. Frozen old (link_local_legacy) and corrected (hand_frame) raw-contact decoders, frozen world-frame
translator, frozen psi. Only Wuji's base-frame ORIGIN moves (rotation unchanged), by a translation o fixed from robot
geometry before any prediction is looked at:
  hand_centroid  centroid of all Wuji template points at the reference configuration q = clip(0, lower, upper)
  palm_centroid  centroid of the base-link (palm) template points (configuration independent)
Description change: a new root link with a fixed joint to the old base (T = translate(-o)); new base world pose
t' = t + R o.  Starts and goals are transformed the same way, so world geometry is identical.  Predictions
(R', t') are mapped back (t = t' - R' o) and scored with the original bank, exactly as in wuji_per_example.py.

Geometry checks run first and abort on failure: template/contact world points at start and goal poses (both
banks), surface-token selection, and every pose-dependent quantity used by scoring.

    python scripts/decoder/diagnostics/wuji_reanchor.py --out $GETKITE_RUNS/wuji_reanchor
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
from getkite.decoder.robots import Joint, RobotBank, make_T
from getkite.decoder.spatial_bridge import ContactSetBridge, SetBridgeConfig
from getkite.decoder.train_scratch import FrozenPsi, RawContacts, build_bank, load_config, load_pool, start_noise

sys.path.insert(0, str(Path(__file__).parent))
from crossbody_heldout_eval import HD, WUJI, bridge_contacts, sg_index, sg_to  # noqa: E402
from wuji_per_example import analyse, load_dec, npy, paired_boot  # noqa: E402

RUNS, REPO = paths.get("GETKITE_RUNS"), str(paths.REPO)


def reanchor_desc(desc, o):
    root = desc.base_link + "__reanchored"
    j = Joint("reanchor_fixed", "fixed", root, desc.base_link, make_T(np.eye(3), -np.asarray(o, float)))
    return dataclasses.replace(desc, base_link=root, joints=[j] + list(desc.joints), name=desc.name)


def offsets(bank, rw):
    n = int(bank.n_points[rw])
    r = torch.tensor([rw], device=bank.lower.device)
    cl, cu = bank.coords_of(r, bank.lower[r]), bank.coords_of(r, bank.upper[r])
    q = torch.minimum(torch.maximum(torch.zeros_like(cl), cl), cu)
    eye = torch.eye(3, device=cl.device)[None]
    Rs, ts = bank.fk(r, bank.expand(r, q), eye, torch.zeros(1, 3, device=cl.device))
    P, _ = bank.points(r, Rs, ts, torch.arange(n, device=cl.device)[None])
    anc = bank.tpl_anchor[rw][:n]
    return {"hand_centroid": P[0].mean(0).cpu().numpy(), "palm_centroid": P[0][anc == 0].mean(0).cpu().numpy()}


def shift(sg, o):
    oo = torch.as_tensor(o, dtype=sg.t1.dtype, device=sg.t1.device)
    return dataclasses.replace(sg, t1=sg.t1 + sg.R1 @ oo, tg=sg.tg + sg.Rg @ oo)


def world_points(bank, r, qc, R, t, idx):
    Rs, ts = bank.fk(r, bank.expand(r, qc), R, t)
    return bank.points(r, Rs, ts, idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=f"{REPO}/configs/decoder/scratch/c3_psi50k_10x_wrist.json")
    ap.add_argument("--old", default=f"{RUNS}/scratch_c3_raw50k_cf8_10x_wrist/best.pt")
    ap.add_argument("--new", default=f"{RUNS}/scratch_c3_raw50k_cf8_10x_wrist_hf/best.pt")
    ap.add_argument("--bridge", default=f"{RUNS}/set_bridge_zworld_c3_20k_b/best.pt")
    ap.add_argument("--out", default=f"{RUNS}/wuji_reanchor")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(a.config, [])
    torch.manual_seed(0)
    noise = start_noise(cfg)
    old, pol_o, st_o = load_dec(a.old, dev)
    new, pol_n, st_n = load_dec(a.new, dev)
    assert (pol_o, pol_n) == ("link_local_legacy", "hand_frame")
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    bs_ = torch.load(a.bridge, map_location=dev)
    bridge = ContactSetBridge(SetBridgeConfig(**bs_["config"])).to(dev); bridge.load_state_dict(bs_["model"]); bridge.eval()
    banks = {p: build_bank({**cfg, "robots": cfg["robots"] + [WUJI], "surface_sampling": p}, dev) for p in (pol_o, pol_n)}
    rw = banks[pol_o].index["wujihand_right"]
    O = offsets(banks[pol_o], rw)
    R = {"_meta": {"old": a.old, "old_step": st_o, "new": a.new, "new_step": st_n, "bridge": a.bridge,
                   "offsets_cm": {k: np.round(v * 100, 3).tolist() for k, v in O.items()},
                   "primary": "hand_centroid", "note": "coordinate-sensitivity test; not a proven explanation"}}
    print("offsets (cm)", R["_meta"]["offsets_cm"], flush=True)
    spec = [{"kind": "kite", "robot": "wujihand_right", "npz": f"{HD}/wujihand_right_eval.npz"}]
    recs = {p: load_pool(spec, b).to(dev) for p, b in banks.items()}
    if a.smoke:
        recs = {p: r_.index(torch.arange(64, device=dev)) for p, r_ in recs.items()}
    sgs = {p: sg_to(fixed_start(banks[p], recs[p], cfg["val_start_seed"], **noise), dev) for p in banks}
    models = {"old": (old, pol_o), "new": (new, pol_n)}
    raw = RawContacts()
    for oname, o in [("none", np.zeros(3))] + list(O.items()):
        Rr = {}
        for tag, (dec, pol) in models.items():
            b0, rec, sg = banks[pol], recs[pol], sgs[pol]
            descs = list(b0.descs); descs[rw] = reanchor_desc(descs[rw], o)
            b1 = RobotBank(descs, dev, surface_sampling=pol)
            sg1 = shift(sg, o)
            # ---- geometry checks (before predictions)
            r = rec.r
            n = int(b0.n_points[rw]); idx = torch.arange(n, device=dev).expand(len(r), -1)
            chk = {}
            for nm_, (q, R0, t0, t1_) in {"start": (sg.qc1, sg.R1, sg.t1, sg1.t1), "goal": (sg.qcg, sg.Rg, sg.tg, sg1.tg)}.items():
                P0, N0 = world_points(b0, r, q, R0, t0, idx); P1, N1 = world_points(b1, r, q, R0, t1_, idx)
                C0, _ = world_points(b0, r, q, R0, t0, rec.c_tpl); C1, _ = world_points(b1, r, q, R0, t1_, rec.c_tpl)
                chk[nm_] = {"template_max_mm": float((P0 - P1).abs().max() * 1000), "normal_max": float((N0 - N1).abs().max()),
                            "contact_pts_max_mm": float(((C0 - C1).abs() * rec.c_valid[..., None]).max() * 1000)}
            chk["surf_idx_equal"] = bool(torch.equal(b0.surf_idx[rw], b1.surf_idx[rw]))
            chk["other_robots_equal"] = all(torch.equal(getattr(b0, f)[:rw], getattr(b1, f)[:rw]) for f in ("A", "tpl_pos", "surf_idx"))
            ok = all(v[k] < 1e-3 for v in chk.values() if isinstance(v, dict) for k in v) and chk["surf_idx_equal"] and chk["other_robots_equal"]
            Rr[f"{tag}/_geometry_check"] = chk | {"pass": ok}
            print(oname, tag, "geometry", json.dumps(chk), "PASS" if ok else "FAIL", flush=True)
            assert ok, "re-anchoring changed geometry"
            # ---- predictions in the re-anchored frame, mapped back to the original base
            outs = {"true": [], "predicted": []}
            with torch.no_grad():
                for s in range(0, len(rec), 256):
                    i = slice(s, s + 256)
                    g = sg_index(sg1, i)
                    pw, nw, pv = bridge_contacts(bridge, psi(rec.c_pos[i], rec.c_normal[i], rec.c_valid[i]))
                    for src, z in (("true", raw(rec.c_pos[i], rec.c_normal[i], rec.c_valid[i])), ("predicted", raw(pw, nw, pv))):
                        outs[src].append(dec(b1, rec.r[i], g.qc1, g.R1, g.t1, z))
            oo = torch.as_tensor(o, dtype=sg.t1.dtype, device=dev)
            conv = {}
            for src, v in outs.items():
                qc, Rp, tp = (torch.cat([b[k] for b in v]) for k in ("qc", "R", "t"))
                conv[f"{tag}/{src}"] = {"qc": qc, "R": Rp, "t": tp - Rp @ oo}
            res = analyse(b0, rec, sg, conv)
            for k, x in res.items():
                if "/" not in k and tag != "old":
                    continue
                Rr[k] = {m: round(float(npy(x[m]).mean()), 4) for m in ("corr_mm", "corr_success", "compat_mm", "compat_success")}
                for m in ("t_progress", "R_err_deg", "q_progress"):
                    if m in x:
                        Rr[k][m] = round(float(np.median(npy(x[m]))), 4)
                if "/full" in k:
                    Rr[k]["_per_example"] = {m: npy(x[m]) for m in ("corr_mm", "compat_mm", "corr_success", "compat_success")}
                    # wrist tracking R^2 (start-wrist frame) as in docs/decoder_audit.md section 39
                    from scipy.spatial.transform import Rotation as Rot
                    R1, Rg = npy(sg.R1), npy(sg.Rg)
                    need = Rot.from_matrix(np.einsum("nji,njk->nik", R1, Rg)).as_rotvec()
                    got = Rot.from_matrix(np.einsum("nji,njk->nik", R1, npy(x["R"]))).as_rotvec()
                    A, *_ = np.linalg.lstsq(need, got, rcond=None)
                    Rr[k]["rot_R2"] = round(float(1 - ((got - need @ A) ** 2).sum() / ((got - got.mean(0)) ** 2).sum()), 3)
                    tn = np.einsum("nji,nj->ni", R1, npy(sg.tg - sg.t1)); tgot = np.einsum("nji,nj->ni", R1, npy(x["t"] - sg.t1))
                    B, *_ = np.linalg.lstsq(tn, tgot, rcond=None)
                    Rr[k]["trans_R2"] = round(float(1 - ((tgot - tn @ B) ** 2).sum() / ((tgot - tgot.mean(0)) ** 2).sum()), 3)
        R[oname] = Rr
        print(oname, json.dumps({k: {m: v for m, v in x.items() if m != "_per_example"} for k, x in Rr.items() if "full" in k or k == "no_movement"}), flush=True)
    # paired: re-anchored vs original anchor, per decoder / source
    P = {}
    for oname in O:
        for k in [k for k in R["none"] if k.endswith("/full")]:
            for m in ("corr_mm", "compat_mm", "compat_success", "corr_success"):
                P[f"{oname}/{k}/{m}"] = paired_boot(R[oname][k]["_per_example"][m], R["none"][k]["_per_example"][m])
    R["_paired_vs_original_anchor"] = P
    for oname in R:
        if isinstance(R[oname], dict):
            for k in R[oname]:
                if isinstance(R[oname][k], dict):
                    R[oname][k].pop("_per_example", None)
    (out / "results.json").write_text(json.dumps(R, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
