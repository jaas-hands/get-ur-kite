"""Frozen two-stage system on (A) the held-out Wuji family and (B) a small cross-body intent benchmark (2026-11-17).

Protocol fixed beforehand: docs/decoder_crossbody_protocol.md. No weights change.

    python scripts/decoder/diagnostics/crossbody_heldout_eval.py --out $GETKITE_RUNS/crossbody_heldout_eval
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from getkite.decoder import paths
from getkite.decoder.get_scratch import GETScratch, ScratchConfig
from getkite.decoder.multifamily import fixed_start
from getkite.decoder.rotations import geodesic_angle
from getkite.decoder.spatial_bridge import ContactSetBridge, SetBridgeConfig
from getkite.decoder.train_scratch import FrozenPsi, RawContacts, build_bank, load_config, load_pool, pose_metrics, start_noise

RUNS, REPO = paths.get("GETKITE_RUNS"), str(paths.REPO)
K = paths.get("KITE_ROOT")
HD = paths.expand("${GETKITE_DATA}/multifamily_heldout")
DS = paths.expand("${GETKITE_DATASET}")
WUJI = {"kind": "kite", "robot": "wujihand_right", "family": "wuji",
        "urdf": f"{K}/assets/spider_robots/wujihand/right_floating.urdf",  # KITE wraps Wuji with this 6-DoF virtual base
        "template": f"{K}/assets/spider_robots/wujihand/surface_template_graph_n4096_998694447d62.pt",
        "coords_npz": f"{HD}/wujihand_right_bank.npz", "ignore_mimic": True}
POS_MM, ANG_DEG = 10.0, 15.0
N_PER_SOURCE, SRC_SEED = 64, 7


def sg_to(sg, dev):
    return type(sg)(**{k: getattr(sg, k).to(dev) for k in sg.__dataclass_fields__})


def sg_index(sg, i):
    return type(sg)(**{k: getattr(sg, k)[i] for k in sg.__dataclass_fields__})


def bridge_contacts(bridge, z):
    pw, nw, lg = bridge(z, torch.eye(3, device=z.device).expand(len(z), 3, 3), torch.zeros(len(z), 3, device=z.device))
    o = lg.argsort(1, descending=True)
    v = lg.gather(1, o).sigmoid() > 0.5
    v[:, 0] = True
    return pw.gather(1, o[..., None].expand_as(pw)), nw.gather(1, o[..., None].expand_as(nw)), v


def clamp_q(bank, r, qc):
    cl, cu = bank.coords_of(r, bank.lower[r]), bank.coords_of(r, bank.upper[r])
    cv = bank.coord_valid[r]
    viol = (((qc < cl - 1e-6) | (qc > cu + 1e-6)) & cv).any(1)
    return torch.where(cv, torch.minimum(torch.maximum(qc, cl), cu), 0.0), viol


def surface_points(bank, r, qc, R, t):
    Rs, ts = bank.fk(r, bank.expand(r, qc), R, t)
    idx = torch.arange(bank.tpl_pos.shape[1], device=qc.device).expand(len(r), -1)  # padded entries repeat a real point
    return bank.points(r, Rs, ts, idx)


def surface_score(bank, r, qc, R, t, c_pos, c_nrm, c_val):
    """Target-surface scoring (joints clamped first). Two assignments per requested contact:
    position-nearest template point (surf_*; lenient, any face) and the protocol revision (compat_*): the nearest
    template point whose normal is within ANG_DEG of the requested normal; success = all compat distances <= POS_MM."""
    qcc, viol = clamp_q(bank, r, qc)
    P, N = surface_points(bank, r, qcc, R, t)
    v = c_val.float()
    out = {"limit_violation": viol.float()}
    D = torch.cdist(c_pos, P)  # (B, K, P)
    cos = torch.einsum("bkc,bpc->bkp", c_nrm, N)
    j = D.argmin(-1)
    d = D.gather(-1, j[..., None]).squeeze(-1) * 1000
    a = torch.rad2deg(torch.acos(cos.gather(-1, j[..., None]).squeeze(-1).clamp(-1, 1)))
    ok_n = cos >= math.cos(math.radians(ANG_DEG))
    dc = torch.where(ok_n, D, torch.full_like(D, float("inf"))).amin(-1) * 1000
    none = torch.isinf(dc) & c_val
    dc = torch.where(torch.isinf(dc), torch.full_like(dc, 1000.0), dc)  # no compatible point: 1 m (counted below)
    out.update({"surf_mm": (d * v).sum(1) / v.sum(1), "surf_rms_mm": ((d ** 2 * v).sum(1) / v.sum(1)).sqrt(),
                "surf_deg": (a * v).sum(1) / v.sum(1),
                "compat_mm": (dc * v).sum(1) / v.sum(1), "compat_rms_mm": ((dc ** 2 * v).sum(1) / v.sum(1)).sqrt(),
                "compat_max_mm": torch.where(c_val, dc, torch.zeros_like(dc)).amax(1),
                "success": (((dc <= POS_MM) | ~c_val).all(1)).float(), "no_compatible_point": none.any(1).float()})
    return out


def motion(R, t, sg):
    return {"move_mm": (t - sg.t1).norm(dim=-1) * 1000, "move_deg": torch.rad2deg(geodesic_angle(R, sg.R1))}


def so3_exp(w):
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    k = w / th
    Kx = torch.zeros(*w.shape[:-1], 3, 3, device=w.device, dtype=w.dtype)
    Kx[..., 0, 1], Kx[..., 0, 2], Kx[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    Kx = Kx - Kx.transpose(-1, -2)
    th = th[..., None]
    return torch.eye(3, device=w.device) + th.sin() * Kx + (1 - th.cos()) * (Kx @ Kx)


def oracle(bank, r, sg, c_pos, c_nrm, c_val, steps=300, restarts=4, seed=0, return_pose=False):
    """Non-learned feasibility reference (protocol): Adam on base + joints from the start (+ perturbed copies)."""
    B = len(r)
    g = torch.Generator(device=c_pos.device).manual_seed(seed)
    rr = r.repeat(restarts)
    q0, R0, t0 = sg.qc1.repeat(restarts, 1), sg.R1.repeat(restarts, 1, 1), sg.t1.repeat(restarts, 1)
    cp, cn, cv = c_pos.repeat(restarts, 1, 1), c_nrm.repeat(restarts, 1, 1), c_val.repeat(restarts, 1)
    pert = torch.ones(B * restarts, 1, device=c_pos.device)
    pert[:B] = 0  # restart 0 = the start itself
    rng = bank.range[rr]
    dt = (torch.randn(B * restarts, 3, generator=g, device=c_pos.device) * 0.01 * pert).requires_grad_()
    dw = (torch.randn(B * restarts, 3, generator=g, device=c_pos.device) * math.radians(10) * pert).requires_grad_()
    dq = (bank.coords_of(rr, torch.randn(B * restarts, rng.shape[1], generator=g, device=c_pos.device) * 0.1 * rng) * pert).requires_grad_()
    opt = torch.optim.Adam([{"params": [dt], "lr": 2e-3}, {"params": [dw], "lr": 1e-2}, {"params": [dq], "lr": 2e-2}])
    v = cv.float()
    for _ in range(steps):
        qc, _ = clamp_q(bank, rr, q0 + dq)
        R, t = so3_exp(dw) @ R0, t0 + dt
        P, N = surface_points(bank, rr, qc, R, t)
        with torch.no_grad():
            D = torch.cdist(cp, P)
            cos = torch.einsum("bkc,bpc->bkp", cn, N)
            Dc = torch.where(cos >= math.cos(math.radians(ANG_DEG)), D, torch.full_like(D, float("inf")))
            j = torch.where(torch.isinf(Dc.amin(-1, keepdim=True)), (D / 0.005) ** 2 + 10 * (1 - cos), Dc).argmin(-1)
        p = P.gather(1, j[..., None].expand(-1, -1, 3))
        n = N.gather(1, j[..., None].expand(-1, -1, 3))
        L = ((((p - cp) / 0.005) ** 2).sum(-1) + 10 * (1 - (n * cn).sum(-1))) * v
        loss = (L.sum(1) / v.sum(1)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        qc, _ = clamp_q(bank, rr, q0 + dq)
        R, t = so3_exp(dw) @ R0, t0 + dt
        s = surface_score(bank, rr, qc, R, t, cp, cn, cv)
        key = s["compat_mm"].view(restarts, B) - 1000 * s["success"].view(restarts, B)  # success first, then distance
        best = key.argmin(0)
        pick = lambda x: x.view(restarts, B, *x.shape[1:])[best, torch.arange(B, device=x.device)]  # noqa: E731
        out = {k: pick(v_) for k, v_ in s.items()}
        out.update(motion(pick(R), pick(t), sg))
        if return_pose:  # opt-in (2026-11-19, wuji_references.py); default output unchanged
            out["_pose"] = {"qc": pick(qc), "R": pick(R), "t": pick(t)}
    return out


def closest_example(req_pos, req_val, ex_pos, ex_val):
    """KITE closest_initial: centred symmetric mean nearest-neighbour distance; returns example index per request."""
    def centre(p, v):
        vf = v.float()[..., None]
        c = (p * vf).sum(-2, keepdim=True) / vf.sum(-2, keepdim=True)
        return p - c, c.squeeze(-2)
    qc_, qcen = centre(req_pos, req_val)
    ec_, ecen = centre(ex_pos, ex_val)
    out = []
    for i in range(len(qc_)):
        D = torch.cdist(qc_[i][None].expand(len(ec_), -1, -1), ec_)  # (N, Kq, Kb)
        big = torch.full_like(D, 1e3)
        D1 = torch.where(ex_val[:, None, :], D, big).amin(2)  # query -> example
        D2 = torch.where(req_val[i][None, :, None], D, big).amin(1)  # example -> query
        s = (D1 * req_val[i]).sum(1) / req_val[i].sum() + (D2 * ex_val).sum(1) / ex_val.sum(1)
        out.append(int(s.argmin()))
    idx = torch.tensor(out, device=req_pos.device)
    return idx, qcen - ecen[idx]


def run_paths(dec, bridge, psi, bank, r, sg, c_pos, c_nrm, c_val, bs=256, direct=None):
    """true / predicted / no movement through the frozen decoder for target robots r at starts sg; optionally the
    frozen direct psi decoder (z = psi(request) + state + robot; its training-only aux head output is ignored)."""
    raw = RawContacts()
    res = {"true": {}, "predicted": {}, "no_movement": {}} | ({"direct": {}} if direct is not None else {})
    with torch.no_grad():
        for s in range(0, len(r), bs):
            i = slice(s, s + bs)
            g = sg_index(sg, i)
            pw, nw, pv = bridge_contacts(bridge, psi(c_pos[i], c_nrm[i], c_val[i]))  # two-stage
            zpsi = psi(c_pos[i], c_nrm[i], c_val[i])
            runs = [("true", dec, raw(c_pos[i], c_nrm[i], c_val[i])), ("predicted", dec, raw(pw, nw, pv))]
            if direct is not None:
                runs.append(("direct", direct, zpsi))
            for name, model, z in runs:
                o = model(bank, r[i], g.qc1, g.R1, g.t1, z)
                m = surface_score(bank, r[i], o["qc"], o["R"], o["t"], c_pos[i], c_nrm[i], c_val[i]) | motion(o["R"], o["t"], g)
                m["_out"] = o
                for k, v in m.items():
                    res[name].setdefault(k, []).append(v)
            m = surface_score(bank, r[i], g.qc1, g.R1, g.t1, c_pos[i], c_nrm[i], c_val[i]) | motion(g.R1, g.t1, g)
            for k, v in m.items():
                res["no_movement"].setdefault(k, []).append(v)
    for p in res.values():
        for k in list(p):
            if k == "_out":
                p[k] = {kk: torch.cat([o[kk] for o in p[k]]) for kk in ("qc", "R", "t")}
            else:
                p[k] = torch.cat(p[k])
    return res


def summ(d, sel):
    return {k: round(float(v[sel].float().mean()), 4) for k, v in d.items() if not k.startswith("_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=f"{REPO}/configs/decoder/scratch/c3_psi50k_10x_wrist.json")
    ap.add_argument("--decoder", default=f"{RUNS}/scratch_c3_raw50k_cf8_10x_wrist/best.pt")
    ap.add_argument("--bridge", default=f"{RUNS}/set_bridge_zworld_c3_20k_b/best.pt")
    ap.add_argument("--out", default=f"{RUNS}/crossbody_heldout_eval")
    ap.add_argument("--direct", default="", help="frozen direct psi decoder checkpoint to add as a path")
    ap.add_argument("--surface_sampling", default="link_local_legacy", choices=["link_local_legacy", "hand_frame"],
                    help="must match the policy the raw-contact decoder was trained with")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    cfg = load_config(a.config, [])
    cfg["surface_sampling"] = a.surface_sampling  # robot bank (surface tokens) matching the evaluated decoder
    dck = torch.load(a.decoder, map_location="cpu", weights_only=False)
    trained_with = dck.get("config", {}).get("surface_sampling", "link_local_legacy")
    assert trained_with == a.surface_sampling, f"decoder trained with {trained_with}, evaluating with {a.surface_sampling}"
    del dck
    torch.manual_seed(0)
    noise = start_noise(cfg)
    bank0 = build_bank(cfg, dev)
    cfg_w = {**cfg, "robots": cfg["robots"] + [WUJI]}  # Wuji appended: existing robot indices unchanged
    bank = build_bank(cfg_w, dev)
    fam = np.array([str(x) for x in bank.family], dtype=object)
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    ck = torch.load(a.decoder, map_location=dev, weights_only=False)
    dec = GETScratch(ScratchConfig.from_dict(ck["model_config"])).to(dev)
    dec.load_state_dict(ck["model"])
    dec.amp = False
    dec.eval()
    bs_ = torch.load(a.bridge, map_location=dev)
    bridge = ContactSetBridge(SetBridgeConfig(**bs_["config"])).to(dev)
    bridge.load_state_dict(bs_["model"])
    bridge.eval()
    for p in list(dec.parameters()) + list(bridge.parameters()):
        p.requires_grad_(False)
    direct = None
    if a.direct:
        ckd = torch.load(a.direct, map_location=dev, weights_only=False)
        if ckd.get("kind") == "joint_contact_get":  # J1: z -> internal world contacts -> GET, one checkpoint
            from getkite.decoder.joint_contact_get import JointContactGET
            assert ckd["config"].get("surface_sampling") == a.surface_sampling, "J1 surface sampling must match"
            direct = JointContactGET(ScratchConfig.from_dict(ckd["model_config"]), SetBridgeConfig(**ckd["set_config"])).to(dev)
            direct.load_state_dict(ckd["model"])
            direct.decoder.amp = False
        else:
            direct = GETScratch(ScratchConfig.from_dict(ckd["model_config"])).to(dev)
            direct.load_state_dict(ckd["model"])
            direct.amp = False
        direct.eval()
        for p in direct.parameters():
            p.requires_grad_(False)
        assert direct.c.intent_source == "psi" or ckd.get("kind") == "joint_contact_get"
    R = {"_meta": {"surface_sampling": a.surface_sampling, "direct": a.direct, "direct_step": ckd.get("step") if a.direct else None, "decoder": a.decoder, "decoder_step": ck.get("step"), "bridge": a.bridge, "bridge_step": bs_.get("step"),
                   "protocol": "docs/decoder_crossbody_protocol.md", "success": [POS_MM, ANG_DEG]}}

    # ---- sanity: adding Wuji to the bank leaves the decoder unchanged on existing records
    val = load_pool(cfg["val"], bank0).to(dev)
    sub = val.index(torch.arange(0, len(val), max(1, len(val) // 64), device=dev)[:64])
    sg0 = sg_to(fixed_start(bank0, sub, cfg["val_start_seed"], **noise), dev)
    with torch.no_grad():
        z = RawContacts()(sub.c_pos, sub.c_normal, sub.c_valid)
        o0 = dec(bank0, sub.r, sg0.qc1, sg0.R1, sg0.t1, z)
        o1 = dec(bank, sub.r, sg0.qc1, sg0.R1, sg0.t1, z)
    R["_sanity_bank_extension_max_abs"] = {k: float((o0[k] - o1[k]).abs().max()) for k in ("qc", "R", "t")}
    if direct is not None:
        with torch.no_grad():
            zp = psi(sub.c_pos, sub.c_normal, sub.c_valid)
            d0, d1 = direct(bank0, sub.r, sg0.qc1, sg0.R1, sg0.t1, zp), direct(bank, sub.r, sg0.qc1, sg0.R1, sg0.t1, zp)
        R["_sanity_bank_extension_max_abs_direct"] = {k: float((d0[k] - d1[k]).abs().max()) for k in ("qc", "R", "t")}
    print("bank sanity", R["_sanity_bank_extension_max_abs"], flush=True)
    rw = bank.index["wujihand_right"]
    R["_meta"]["wuji_dof"] = int(bank.coord_valid[rw].sum())

    # ---- A. same-body: fresh source goals, held-out LEAP, Wuji (identical fixed starts for every path)
    wrec = load_pool([{"kind": "kite", "robot": "wujihand_right", "npz": f"{HD}/wujihand_right_eval.npz"}], bank).to(dev)
    scfg = {"success_pos_mm": POS_MM, "success_normal_deg": ANG_DEG}
    same = {"val": load_pool(cfg["val"], bank).to(dev), "leap_heldout": load_pool(cfg["extra_eval"]["leap_heldout"], bank).to(dev),
            "wuji": wrec}
    for pool, rec in same.items():
        psg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
        res = run_paths(dec, bridge, psi, bank, rec.r, psg, rec.c_pos, rec.c_normal, rec.c_valid, direct=direct)
        fo = fam[rec.r.cpu().numpy()]
        A = {}
        for f in sorted(set(fo)):
            sel = torch.as_tensor(fo == f, device=dev)
            Af = {}
            for p in res:
                if p == "no_movement":
                    pm = pose_metrics(bank, rec, rec.r, psg.qc1, psg.t1, psg.R1, psg, scfg)
                else:
                    o = res[p]["_out"]
                    pm = pose_metrics(bank, rec, rec.r, o["qc"], o["t"], o["R"], psg, scfg)
                Af[p] = {**{"corr_" + k: round(float(pm[k][sel].mean()), 4) for k in ("contact_mm", "normal_deg", "success", "wrist_mm", "wrist_deg")},
                         **summ(res[p], sel)}
            Af["drop_true_to_predicted"] = {k: round(Af["predicted"][k] - Af["true"][k], 4) for k in Af["true"]}
            Af["n"] = int(sel.sum())
            A[f] = Af
        R[f"A_same_body_{pool}"] = A
        if pool == "wuji":
            R["A_wuji_same_body"] = A["wuji"]
        print(f"A {pool}", json.dumps({f: {p: (v[p]["corr_contact_mm"], v[p]["corr_success"]) for p in res} for f, v in A.items()}), flush=True)

    # ---- B. cross-body benchmark
    srcs = {"allegro": None, "barrett": None, "leap": None, "wuji": None}
    valw = load_pool(cfg["val"], bank).to(dev)
    gsel = torch.Generator().manual_seed(SRC_SEED)
    fam_val = fam[valw.r.cpu().numpy()]
    for f in ("allegro", "barrett", "leap"):
        ids = np.flatnonzero(fam_val == f)
        srcs[f] = valw.index(torch.as_tensor(ids[torch.randperm(len(ids), generator=gsel)[:N_PER_SOURCE].numpy()], device=dev))
    srcs["wuji"] = wrec.index(torch.randperm(len(wrec.r), generator=gsel)[:N_PER_SOURCE].to(dev))
    held = json.loads(Path(paths.expand("${GETKITE_DATA}/multifamily_pilot/manifest.json")).read_text())["leap_heldout_hands"]
    held_idx = [next(i for i, n in enumerate(bank.names) if h in n) for h in held]
    banks = {"allegro": load_pool([{"kind": "kite", "robot": "allegro_right", "npz": paths.expand("${GETKITE_DATA}/multifamily_pilot/allegro_right_train.npz")}], bank).to(dev),
             "barrett": load_pool([{"kind": "kite", "robot": "barrett", "npz": paths.expand("${GETKITE_DATA}/multifamily_pilot/barrett_train.npz")}], bank).to(dev),
             "wuji": load_pool([{"kind": "kite", "robot": "wujihand_right", "npz": f"{HD}/wujihand_right_bank.npz"}], bank).to(dev)}
    for h in held:
        banks[h] = load_pool([{"kind": "leap", "npz": f"{DS}/examples/kite_code/val/{h}.npz", "rows": [125, 1000]}], bank).to(dev)
    B = {}
    allrows = []
    for sf, srec in srcs.items():
        for tf in ("allegro", "barrett", "leap_heldout", "wuji"):
            if tf.startswith(sf):
                continue
            n = len(srec.r)
            if tf == "leap_heldout":
                hands = [held[i % len(held)] for i in range(n)]
            else:
                hands = [tf] * n
            starts, rs = [], []
            for h in sorted(set(hands)):
                ii = torch.tensor([i for i in range(n) if hands[i] == h], device=dev)
                ex = banks[h]
                idx, delta = closest_example(srec.c_pos[ii], srec.c_valid[ii], ex.c_pos, ex.c_valid)
                exr = ex.index(idx)
                sg = sg_to(fixed_start(bank, exr, cfg["val_start_seed"] + 11, **noise), dev)
                sg = type(sg)(**{**{k: getattr(sg, k) for k in sg.__dataclass_fields__}, "t1": sg.t1 + delta})
                starts.append((ii, sg, exr.r))
            order = torch.cat([s[0] for s in starts])
            sg = type(starts[0][1])(**{k: torch.cat([getattr(s[1], k) for s in starts]) for k in starts[0][1].__dataclass_fields__})
            r = torch.cat([s[2] for s in starts])
            cp, cn, cv = srec.c_pos[order], srec.c_normal[order], srec.c_valid[order]
            res = run_paths(dec, bridge, psi, bank, r, sg, cp, cn, cv, direct=direct)
            orc = {}
            for s in range(0, len(r), 128):
                i = slice(s, s + 128)
                o = oracle(bank, r[i], sg_index(sg, i), cp[i], cn[i], cv[i], seed=s)
                for k, v in o.items():
                    orc.setdefault(k, []).append(v)
            res["oracle"] = {k: torch.cat(v) for k, v in orc.items()}
            feas = res["oracle"]["success"] > 0.5
            allsel = torch.ones(len(r), dtype=torch.bool, device=dev)
            row = {"n": len(r), "oracle_feasible_frac": round(float(feas.float().mean()), 4)}
            for p in res:
                row[p] = summ(res[p], allsel)
                row[p + "@feasible"] = summ(res[p], feas) if feas.any() else None
            row["drop_true_to_predicted"] = {k: round(row["predicted"][k] - row["true"][k], 4) for k in row["true"]}
            B[f"{sf}->{tf}"] = row
            allrows.append((sf, tf, res, feas))
            print(f"B {sf}->{tf} feasible {row['oracle_feasible_frac']} | compat mm/success true {row['true']['compat_mm']}/{row['true']['success']} "
                  f"pred {row['predicted']['compat_mm']}/{row['predicted']['success']} noop {row['no_movement']['compat_mm']}/"
                  f"{row['no_movement']['success']} oracle {row['oracle']['compat_mm']}/{row['oracle']['success']}", flush=True)
    # per target and overall
    agg = {}
    for key, filt in [("target=" + t, lambda sf, tf, t=t: tf == t) for t in ("allegro", "barrett", "leap_heldout", "wuji")] + [("all", lambda sf, tf: True)]:
        rows = [x for x in allrows if filt(x[0], x[1])]
        cat = {p: {k: torch.cat([x[2][p][k] for x in rows]) for k in rows[0][2][p] if not k.startswith("_")}
               for p in rows[0][2]}
        feas = torch.cat([x[3] for x in rows])
        allsel = torch.ones_like(feas)
        agg[key] = {"n": int(len(feas)), "oracle_feasible_frac": round(float(feas.float().mean()), 4),
                    **{p: summ(cat[p], allsel) for p in cat}, **{p + "@feasible": summ(cat[p], feas) for p in cat}}
        agg[key]["drop_true_to_predicted"] = {k: round(agg[key]["predicted"][k] - agg[key]["true"][k], 4) for k in agg[key]["true"]}
    R["B_crossbody_pairs"] = B
    R["B_crossbody_aggregate"] = agg
    (out / "results.json").write_text(json.dumps(R, indent=1))
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
