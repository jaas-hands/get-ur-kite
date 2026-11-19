"""Per-example Wuji diagnosis: old (link_local_legacy) vs corrected (hand_frame) raw-contact decoder (2026-11-18).

Frozen models only. Same Wuji eval records, same fixed starts (val_start_seed), same frozen world-frame translator
as crossbody_heldout_eval.py. Source val records are run through the identical analysis for contrast.

    python scripts/decoder/diagnostics/wuji_per_example.py --out $GETKITE_RUNS/wuji_per_example [--smoke]

Per example and per decoder x contact source (true / translator-predicted):
  pose variants   full (pred q, pred wrist) | wrist_only (start q, pred wrist) | fingers_only (pred q, start wrist)
  references      no_movement | goal_wrist (start q, goal wrist) | goal_fingers (goal q, start wrist) | goal
  metrics         corr_* (fixed correspondence: record's template point; success = 10 mm AND 15 deg per contact)
                  compat_* (normal-compatible surface score; success = all contacts within 10 mm of a point whose
                  normal is within 15 deg) -- both reported, each with its own baseline
  per contact     corr distance, owning link -> finger
  per joint       pre-clamp limit violation; signed progress  (dq . dq_goal) / |dq_goal|^2  on active joints
  wrist           translation progress along (t_goal - t_start), rotation progress
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from getkite.decoder import paths
from getkite.decoder.get_scratch import GETScratch, ScratchConfig
from getkite.decoder.multifamily import fixed_start
from getkite.decoder.rotations import geodesic_angle
from getkite.decoder.spatial_bridge import ContactSetBridge, SetBridgeConfig
from getkite.decoder.train_scratch import FrozenPsi, RawContacts, build_bank, load_config, load_pool, pose_metrics, start_noise

import sys
sys.path.insert(0, str(Path(__file__).parent))
from crossbody_heldout_eval import HD, WUJI, ANG_DEG, POS_MM, bridge_contacts, clamp_q, sg_index, sg_to, surface_score  # noqa: E402

RUNS, REPO = paths.get("GETKITE_RUNS"), str(paths.REPO)
SCFG = {"success_pos_mm": POS_MM, "success_normal_deg": ANG_DEG}


def finger_of(link: str) -> str:
    m = re.search(r"finger(\d)", link)  # wuji: right_fingerK_*
    if m:
        return f"finger{m.group(1)}"
    m = re.search(r"link_(\d+)\.0", link)  # allegro: 4 links per finger, 0-3 index ... 12-15 thumb
    if m:
        return ["index", "middle", "ring", "thumb"][int(m.group(1)) // 4]
    m = re.search(r"finger_(\d)", link)  # barrett
    if m:
        return f"f{m.group(1)}"
    return "palm"


def load_dec(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = GETScratch(ScratchConfig.from_dict(ck["model_config"])).to(dev)
    m.load_state_dict(ck["model"]); m.amp = False; m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, ck.get("config", {}).get("surface_sampling", "link_local_legacy"), ck.get("step")


def per_contact(bank, rec, r, qc, t, R):
    qcc, _ = clamp_q(bank, r, qc)
    Rs, ts = bank.fk(r, bank.expand(r, qcc), R, t)
    p, n = bank.points(r, Rs, ts, rec.c_tpl)
    d = (p - rec.c_pos).norm(dim=-1) * 1000
    a = torch.rad2deg(torch.acos((n * rec.c_normal).sum(-1).clamp(-1, 1)))
    return d, a


def score(bank, rec, r, qc, t, R, sg):
    pm = pose_metrics(bank, rec, r, qc, t, R, sg, SCFG)
    ss = surface_score(bank, r, qc, R, t, rec.c_pos, rec.c_normal, rec.c_valid)
    d, a = per_contact(bank, rec, r, qc, t, R)
    return {"corr_mm": pm["contact_mm"], "corr_deg": pm["normal_deg"], "corr_success": pm["success"],
            "compat_mm": ss["compat_mm"], "compat_success": ss["success"], "c_mm": d, "c_deg": a}


def analyse(bank, rec, sg, outs):
    r = rec.r
    res = {}
    refs = {"no_movement": (sg.qc1, sg.t1, sg.R1), "goal_wrist": (sg.qc1, sg.tg, sg.Rg),
            "goal_fingers": (sg.qcg, sg.t1, sg.R1), "goal": (sg.qcg, sg.tg, sg.Rg)}
    for k, (q, t, R) in refs.items():
        res[k] = score(bank, rec, r, q, t, R, sg)
    cl, cu = bank.coords_of(r, bank.lower[r]), bank.coords_of(r, bank.upper[r])
    cv = bank.coord_valid[r]
    act = rec.active & cv
    dqg = (sg.qcg - sg.qc1)
    dtg = sg.tg - sg.t1
    ang_g = geodesic_angle(sg.R1, sg.Rg)
    for name, o in outs.items():
        q, t, R = o["qc"], o["t"], o["R"]
        for var, (qq, tt, RR) in {"full": (q, t, R), "wrist_only": (sg.qc1, t, R), "fingers_only": (q, sg.t1, sg.R1)}.items():
            res[f"{name}/{var}"] = score(bank, rec, r, qq, tt, RR, sg)
        dq = q - sg.qc1
        rng = (cu - cl).clamp_min(1e-6)
        x = res[f"{name}/full"]
        x["joint_viol"] = ((q < cl - 1e-6) | (q > cu + 1e-6)) & cv           # (B, J) pre-clamp
        x["joint_over"] = torch.where(cv, (torch.relu(cl - q) + torch.relu(q - cu)) / rng, 0.0)
        num = (dq * dqg * act).sum(1); den = (dqg ** 2 * act).sum(1).clamp_min(1e-12)
        x["q_progress"] = num / den                                            # 1 = exact projection onto goal dq
        x["q_cos"] = num / ((dq ** 2 * act).sum(1).sqrt() * den.sqrt()).clamp_min(1e-12)
        x["q_step_norm"] = (dq / rng * cv).norm(dim=1)
        x["q_goal_norm"] = (dqg / rng * act).norm(dim=1)
        x["t_progress"] = ((t - sg.t1) * dtg).sum(1) / (dtg ** 2).sum(1).clamp_min(1e-12)
        x["t_cos"] = ((t - sg.t1) * dtg).sum(1) / ((t - sg.t1).norm(dim=1) * dtg.norm(dim=1)).clamp_min(1e-12)
        x["R_err_deg"] = torch.rad2deg(geodesic_angle(R, sg.Rg)); x["R_start_err_deg"] = torch.rad2deg(ang_g)
        x["dq"] = dq; x["dqg"] = dqg; x["act"] = act
        x["t"] = t; x["R"] = R; x["q"] = q
    return res


def run(models, bridge, psi, bank_by_policy, rec_by_policy, sg_by_policy, bs=256):
    out = {}
    for tag, (dec, pol) in models.items():
        bank, rec, sg = bank_by_policy[pol], rec_by_policy[pol], sg_by_policy[pol]
        raw = RawContacts()
        o = {"true": [], "predicted": []}
        with torch.no_grad():
            for s in range(0, len(rec), bs):
                i = slice(s, s + bs)
                g = sg_index(sg, i)
                pw, nw, pv = bridge_contacts(bridge, psi(rec.c_pos[i], rec.c_normal[i], rec.c_valid[i]))
                for src, z in (("true", raw(rec.c_pos[i], rec.c_normal[i], rec.c_valid[i])), ("predicted", raw(pw, nw, pv))):
                    o[src].append(dec(bank, rec.r[i], g.qc1, g.R1, g.t1, z))
        o = {src: {k: torch.cat([b[k] for b in v]) for k in ("qc", "R", "t")} for src, v in o.items()}
        res = analyse(bank, rec, sg, {f"{tag}/{src}": o[src] for src in o})
        out.update(res if not out else {k: v for k, v in res.items() if "/" in k})
    return out


def paired_boot(a, b, n=4000, seed=0):
    g = np.random.default_rng(seed)
    d = (a - b)
    idx = g.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return {"mean_diff": round(float(d.mean()), 4), "ci95": [round(float(np.quantile(m, .025)), 4), round(float(np.quantile(m, .975)), 4)],
            "a_better_frac": round(float((a < b).mean()), 3) if a.dtype != bool else None}


def npy(x):
    return x.detach().float().cpu().numpy()


def summarise(res, bank, rec, rname):
    S = {}
    v = npy(rec.c_valid).astype(bool)
    desc = bank.descs[bank.index[rname]] if rname else None
    tl = bank.tpl_link[rec.r[0]].cpu().numpy() if hasattr(bank, "tpl_link") else None
    for k, x in res.items():
        S[k] = {m: round(float(npy(x[m]).mean()), 4) for m in ("corr_mm", "corr_deg", "corr_success", "compat_mm", "compat_success")}
        for m in ("q_progress", "q_cos", "q_step_norm", "q_goal_norm", "t_progress", "t_cos", "R_err_deg", "R_start_err_deg"):
            if m in x:
                S[k][m] = round(float(np.median(npy(x[m]))), 4)
        if "joint_viol" in x:
            S[k]["any_limit_violation"] = round(float(npy(x["joint_viol"]).any(1).mean()), 4)
    if desc is not None:
        links = np.array(desc.tpl_link_names)[np.asarray(desc.tpl_link)]
        fing = np.array([finger_of(l) for l in links])
        ctpl = npy(rec.c_tpl).astype(int)
        cf = np.where(v, fing[np.clip(ctpl, 0, None)], "")
        S["_finger_counts"] = {f: int((cf == f).sum()) for f in sorted(set(cf[v]))}
        for k, x in res.items():
            d = npy(x["c_mm"])
            S[k]["by_finger_mm"] = {f: round(float(d[cf == f].mean()), 2) for f in sorted(set(cf[v]))}
            if "joint_viol" in x:
                jv = npy(x["joint_viol"]).mean(0); jo = npy(x["joint_over"])
                names = list(desc.coords)
                S[k]["joint_viol_frac"] = {names[j]: round(float(jv[j]), 3) for j in range(len(names)) if jv[j] > 0}
                S[k]["joint_over_mean_range"] = {names[j]: round(float(jo[:, j].mean()), 4) for j in range(len(names)) if jv[j] > 0}
                dq, dqg, act = npy(x["dq"]), npy(x["dqg"]), npy(x["act"]).astype(bool)
                S[k]["per_joint_sign_agree"] = {names[j]: round(float((np.sign(dq[act[:, j], j]) == np.sign(dqg[act[:, j], j])).mean()), 3)
                                               for j in range(len(names)) if act[:, j].sum() > 20}
    return S, (cf if desc is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=f"{REPO}/configs/decoder/scratch/c3_psi50k_10x_wrist.json")
    ap.add_argument("--old", default=f"{RUNS}/scratch_c3_raw50k_cf8_10x_wrist/best.pt")
    ap.add_argument("--new", default=f"{RUNS}/scratch_c3_raw50k_cf8_10x_wrist_hf/best.pt")
    ap.add_argument("--bridge", default=f"{RUNS}/set_bridge_zworld_c3_20k_b/best.pt")
    ap.add_argument("--out", default=f"{RUNS}/wuji_per_example")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(a.config, [])
    torch.manual_seed(0)
    noise = start_noise(cfg)
    old, pol_o, st_o = load_dec(a.old, dev)
    new, pol_n, st_n = load_dec(a.new, dev)
    assert (pol_o, pol_n) == ("link_local_legacy", "hand_frame"), (pol_o, pol_n)
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    bs_ = torch.load(a.bridge, map_location=dev)
    bridge = ContactSetBridge(SetBridgeConfig(**bs_["config"])).to(dev); bridge.load_state_dict(bs_["model"]); bridge.eval()
    banks = {p: build_bank({**cfg, "robots": cfg["robots"] + [WUJI], "surface_sampling": p}, dev) for p in (pol_o, pol_n)}
    models = {"old": (old, pol_o), "new": (new, pol_n)}
    R = {"_meta": {"old": a.old, "old_step": st_o, "new": a.new, "new_step": st_n, "bridge": a.bridge,
                   "corr_success": "per contact: fixed-correspondence distance <= 10 mm AND normal <= 15 deg",
                   "compat_success": "per contact: nearest template point with normal within 15 deg is <= 10 mm"}}
    pools = {"wuji": [{"kind": "kite", "robot": "wujihand_right", "npz": f"{HD}/wujihand_right_eval.npz"}], "val": cfg["val"]}
    dump = {}
    for pool, spec in pools.items():
        recs = {p: load_pool(spec, b).to(dev) for p, b in banks.items()}
        if a.smoke:
            recs = {p: r_.index(torch.arange(min(64, len(r_)), device=dev)) for p, r_ in recs.items()}
        # identical records and starts in both banks (records and starts do not depend on surface tokens)
        assert torch.equal(recs[pol_o].c_tpl, recs[pol_n].c_tpl)
        sgs = {p: sg_to(fixed_start(banks[p], recs[p], cfg["val_start_seed"], **noise), dev) for p in banks}
        assert all(torch.equal(getattr(sgs[pol_o], f), getattr(sgs[pol_n], f)) for f in sgs[pol_o].__dataclass_fields__)
        res = run(models, bridge, psi, banks, recs, sgs)
        rec, bank = recs[pol_o], banks[pol_o]
        fam = np.array([str(bank.family[i]) for i in rec.r.cpu().numpy()])
        for f in sorted(set(fam)):
            sel = torch.as_tensor(fam == f, device=dev)
            rsel = {k: {m: (v[sel] if torch.is_tensor(v) and v.shape[:1] == sel.shape else v) for m, v in x.items()} for k, x in res.items()}
            rname = "wujihand_right" if f == "wuji" else None
            S, cf = summarise(rsel, bank, rec.index(sel.nonzero().squeeze(1)), rname)
            # paired comparisons (per example)
            P = {}
            for src in ("true", "predicted"):
                o, n_, nm = rsel[f"old/{src}/full"], rsel[f"new/{src}/full"], rsel["no_movement"]
                for m in ("corr_mm", "compat_mm"):
                    P[f"{src}/{m}/new-old"] = paired_boot(npy(n_[m]), npy(o[m]))
                    P[f"{src}/{m}/old-noop"] = paired_boot(npy(o[m]), npy(nm[m]))
                    P[f"{src}/{m}/new-noop"] = paired_boot(npy(n_[m]), npy(nm[m]))
                for m in ("corr_success", "compat_success"):
                    P[f"{src}/{m}/new-old"] = paired_boot(npy(n_[m]), npy(o[m]))
                    P[f"{src}/{m}/old-noop"] = paired_boot(npy(o[m]), npy(nm[m]))
                    P[f"{src}/{m}/new-noop"] = paired_boot(npy(n_[m]), npy(nm[m]))
            S["_paired"] = P
            S["_n"] = int(sel.sum())
            R[f"{pool}/{f}"] = S
            print(pool, f, json.dumps({k: S[k] for k in ("no_movement", "old/true/full", "new/true/full") if k in S})[:1500], flush=True)
            ii = sel.nonzero().squeeze(1)
            dump = {k: {m: npy(v) for m, v in x.items() if torch.is_tensor(v)} for k, x in rsel.items()}
            dump["_cf"] = cf
            dump["_rec"] = {k: npy(getattr(rec, k)[ii]) for k in ("c_pos", "c_normal", "c_valid", "c_tpl", "r")}
            dump["_sg"] = {k: npy(getattr(sgs[pol_o], k)[ii]) for k in sgs[pol_o].__dataclass_fields__}
            torch.save(dump, out / ("wuji_per_example.pt" if f == "wuji" else f"per_example_{f}.pt"))
    (out / "results.json").write_text(json.dumps(R, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
