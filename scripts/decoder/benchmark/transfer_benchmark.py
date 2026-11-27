"""Frozen transfer benchmark (Task A, 2026-11-18): one protocol for J1, the corrected two-stage reference, no movement
and (optionally) further frozen systems such as direct_worldaux_v1. No weights change; nothing is tuned here.

Pools (records and physical starts identical to the archived evaluations, crossbody_heldout_eval.py):
  1 source fresh goals (val: Allegro / Barrett / training-family LEAP)      positive control
  2 held-out LEAP (4 variants, evaluation-only; inspected repeatedly)       development evidence, within family
  3 Wuji near-goal (unseen by our decoders, seen by KITE's psi)             development evidence, new family
  4 Shadow near-goal (unseen by our decoders AND by psi; reserved final target)  only when the config sets
    "shadow_pool": true (added 2026-11-20); same fixed-start protocol as pool 3, no cross-body pairs
  B cross-body intents (fixed 768-case benchmark, target-hand closest-example starts)  secondary, reported separately

Inference inputs for every learned system: z = psi(request), start configuration/pose, deterministic robot
description/geometry. Goals, contact labels, counts and correspondences are used only to build/score the benchmark.

    python scripts/decoder/benchmark/transfer_benchmark.py --config configs/decoder/benchmark/transfer_v1.json
Outputs (out_dir): manifest.json, metrics.json, per_example.pt, report_tables.md (+ plots via transfer_plots.py)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

from getkite.decoder import paths
from getkite.decoder.get_scratch import GETScratch, ScratchConfig
from getkite.decoder.multifamily import fixed_start
from getkite.decoder.rotations import geodesic_angle
from getkite.decoder.spatial_bridge import ContactSetBridge, SetBridgeConfig
from getkite.decoder.train_joint import contact_eval
from getkite.decoder.train_scratch import FrozenPsi, RawContacts, build_bank, git_state, load_config, load_pool, pose_metrics, start_noise

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "diagnostics"))
from crossbody_heldout_eval import (ANG_DEG, DS, HD, N_PER_SOURCE, POS_MM, SRC_SEED, WUJI, bridge_contacts,  # noqa: E402
                                    closest_example, oracle, sg_index, sg_to, surface_score)
from wuji_per_example import paired_boot  # noqa: E402

RUNS, REPO = paths.get("GETKITE_RUNS"), Path(paths.REPO)
SHADOW_AD = Path.home() / "code/uni-kite/external_assets/shadow_adapter"  # made by scripts/decoder/shadow/
SHADOW = {"kind": "kite", "robot": "shadow_hand_right", "family": "shadow", "urdf": str(SHADOW_AD / "shadow_hand_right_extended.urdf"),
          "template": str(SHADOW_AD / "shadow_surface_template_n4096.pt"), "ignore_mimic": False}  # coupled J1 = J2 via <mimic>


def sha256(path, n=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:n] if n else h.hexdigest()


def npy(x):
    return x.detach().float().cpu().numpy()


# ------------------------------------------------------------------------------------------------ systems
class System:
    """predict(bank, r, sg, z) -> {qc, R, t}; contacts(z) -> world-frame (pos, normal, logit) or None."""

    def __init__(self, name, model, kind, bridge=None):
        self.name, self.model, self.kind, self.bridge = name, model, kind, bridge
        self.raw = RawContacts()

    def predict(self, bank, r, g, z):
        if self.kind == "two_stage":
            return self.model(bank, r, g.qc1, g.R1, g.t1, self.raw(*bridge_contacts(self.bridge, z)))
        if self.kind == "direct_worldaux":
            return self.model(bank, r, g.qc1, g.R1, g.t1, z, aux=False)
        return self.model(bank, r, g.qc1, g.R1, g.t1, z)

    def contact_model(self):
        """An object with predict_contacts(z) and .contacts (set_loss, c.slots), for train_joint.contact_eval."""
        if self.kind == "two_stage":
            br = self.bridge

            class Shim(torch.nn.Module):
                def __init__(s):
                    super().__init__(); s.contacts = br

                def predict_contacts(s, z):
                    B = len(z)
                    return br(z, torch.eye(3, device=z.device).expand(B, 3, 3), torch.zeros(B, 3, device=z.device))
            return Shim()
        return self.model


def load_system(name, spec, dev, surface_sampling):
    if spec["kind"] == "two_stage":
        ck = torch.load(spec["decoder"], map_location=dev, weights_only=False)
        assert ck.get("config", {}).get("surface_sampling", "link_local_legacy") == surface_sampling
        dec = GETScratch(ScratchConfig.from_dict(ck["model_config"])).to(dev); dec.load_state_dict(ck["model"]); dec.amp = False
        bs = torch.load(spec["bridge"], map_location=dev, weights_only=False)
        br = ContactSetBridge(SetBridgeConfig(**bs["config"])).to(dev); br.load_state_dict(bs["model"])
        for m in (dec, br):
            m.eval().requires_grad_(False)
        return System(name, dec, "two_stage", br), {"decoder_step": ck.get("step"), "bridge_step": bs.get("step")}
    ck = torch.load(spec["ckpt"], map_location=dev, weights_only=False)
    assert ck["config"].get("surface_sampling") == surface_sampling, f"{name}: surface sampling mismatch"
    if spec["kind"] == "j1":
        from getkite.decoder.joint_contact_get import JointContactGET
        assert ck["kind"] == "joint_contact_get"
        m = JointContactGET(ScratchConfig.from_dict(ck["model_config"]), SetBridgeConfig(**ck["set_config"])).to(dev)
    elif spec["kind"] == "direct_worldaux":
        from getkite.decoder.direct_worldaux import DirectWorldAux
        assert ck["kind"] in ("direct_worldaux_v1", "direct_frozentrunk_d1")  # same module layout
        m = DirectWorldAux(ScratchConfig.from_dict(ck["model_config"]), SetBridgeConfig(**ck["set_config"])).to(dev)
    else:
        raise ValueError(spec["kind"])
    m.load_state_dict(ck["model"]); m.decoder.amp = False
    m.eval().requires_grad_(False)
    if ck["config"].get("canonical_frame"):  # CF (docs/cf_canonical_palm_frame_proposal.md): canonical palm frame per robot
        from getkite.decoder.canonical import CanonicalFrame
        m = CanonicalFrame(m, origin=bool(ck["config"].get("canonical_origin"))).eval()
    return System(name, m, spec["kind"]), {"step": ck.get("step"), "best": ck.get("best"), "canonical_frame": bool(ck["config"].get("canonical_frame")),
                                           "canonical_origin": bool(ck["config"].get("canonical_origin"))}


# ------------------------------------------------------------------------------------------------ metrics
def wrist_fit(R1, Rg, R, t1, tg, t):
    """Required vs predicted wrist correction in the START-WRIST frame.
    translation need = R1^T (tg - t1), got = R1^T (t - t1); rotation need = rotvec(R1^T Rg), got = rotvec(R1^T R).
    slope    = sum <got, need> / sum |need|^2                 (1 = full-size correction along the required direction)
    R2_lin   = 1 - sum|got - need A|^2 / sum|got - mean got|^2, A = 3x3 least squares (as audit sections 39-42):
               fraction of the PREDICTED correction's variance that is a linear function of the required correction
    R2_id    = 1 - sum|got - need|^2 / sum|need - mean need|^2 (coefficient of determination of got as a predictor
               of need; 1 = exact, 0 = no better than the mean required correction, < 0 = worse)"""
    out = {}
    tn = np.einsum("nji,nj->ni", R1, tg - t1); tp = np.einsum("nji,nj->ni", R1, t - t1)
    rn = Rot.from_matrix(np.einsum("nji,njk->nik", R1, Rg)).as_rotvec()
    rp = Rot.from_matrix(np.einsum("nji,njk->nik", R1, R)).as_rotvec()
    for k, (need, got) in {"trans": (tn, tp), "rot": (rn, rp)}.items():
        A, *_ = np.linalg.lstsq(need, got, rcond=None)
        out[f"{k}_slope"] = round(float((got * need).sum() / max((need * need).sum(), 1e-12)), 4)
        out[f"{k}_R2_lin"] = round(float(1 - ((got - need @ A) ** 2).sum() / max(((got - got.mean(0)) ** 2).sum(), 1e-12)), 4)
        out[f"{k}_R2_id"] = round(float(1 - ((got - need) ** 2).sum() / max(((need - need.mean(0)) ** 2).sum(), 1e-12)), 4)
    return out


def score(bank, rec, r, sg, qc, R, t, scfg):
    pm = pose_metrics(bank, rec, r, qc, t, R, sg, scfg)  # corr_* (joints clamped), wrist errors, pre-clamp violation
    ss = surface_score(bank, r, qc, R, t, rec.c_pos, rec.c_normal, rec.c_valid)  # post-clamp target-surface scores
    cl, cu = bank.coords_of(r, bank.lower[r]), bank.coords_of(r, bank.upper[r])
    cv = bank.coord_valid[r]
    rng = (cu - cl).clamp_min(1e-6)
    over = torch.where(cv, (torch.relu(cl - qc) + torch.relu(qc - cu)) / rng, 0.0)
    dq, dqg = qc - sg.qc1, sg.qcg - sg.qc1
    act = rec.active & cv
    return {"corr_mm": pm["contact_mm"], "corr_deg": pm["normal_deg"], "corr_success": pm["success"],
            "compat_mm": ss["compat_mm"], "compat_success": ss["success"], "surf_mm": ss["surf_mm"],
            "no_compatible_point": ss["no_compatible_point"],
            "wrist_mm": pm["wrist_mm"], "wrist_deg": pm["wrist_deg"],
            "limit_violation_preclamp": pm["limit_violation"], "limit_overshoot_frac_max": over.amax(1),
            "move_mm": (t - sg.t1).norm(dim=-1) * 1000, "move_deg": torch.rad2deg(geodesic_angle(R, sg.R1)),
            "req_mm": (sg.tg - sg.t1).norm(dim=-1) * 1000, "req_deg": torch.rad2deg(geodesic_angle(sg.Rg, sg.R1)),
            "q_progress": (dq * dqg * act).sum(1) / (dqg ** 2 * act).sum(1).clamp_min(1e-12)}


def summarise(m, sel):
    return {k: round(float(np.mean(v[sel])), 4) for k, v in m.items() if not k.startswith("_")}


def start_distribution(bank, rec, sg):
    """Empirical start-vs-goal perturbation of a pool's fixed starts (the code draws N(0, sigma) per Euler angle /
    xyz / joint, then clamps joints; these are NOT bounds)."""
    cl, cu = bank.coords_of(rec.r, bank.lower[rec.r]), bank.coords_of(rec.r, bank.upper[rec.r])
    cv = bank.coord_valid[rec.r]
    dt = npy((sg.t1 - rec.t2).norm(dim=-1) * 1000)
    da = npy(torch.rad2deg(geodesic_angle(sg.R1, rec.R2)))
    dq = npy(((sg.qc1 - rec.qc2).abs() / (cu - cl).clamp_min(1e-6))[cv])
    q = lambda x: {f"q{p}": round(float(np.quantile(x, p / 100)), 3) for p in (50, 90, 99, 100)}  # noqa: E731
    return {"xyz_mm": q(dt), "rot_deg": q(da), "joint_frac_of_range": q(dq)}


def record_keys(bank, rec):
    """Content hash per record (robot name + rounded contacts + goal) for train/eval overlap checks."""
    names = [bank.names[i] for i in npy(rec.r).astype(int)]
    cp, qc, t2 = np.round(npy(rec.c_pos), 5), np.round(npy(rec.qc2), 5), np.round(npy(rec.t2), 5)
    return {hashlib.sha1((n + cp[i].tobytes().hex() + qc[i].tobytes().hex() + t2[i].tobytes().hex()).encode()).hexdigest()
            for i, n in enumerate(names)}


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip_crossbody", action="store_true")
    a = ap.parse_args()
    B = json.loads(Path(a.config).read_text())
    out = Path(paths.expand(B["out_dir"])); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    bc = Path(paths.expand(B["base_config"]))
    cfg = load_config(str(bc if bc.is_absolute() else REPO / bc), [])
    cfg["surface_sampling"] = B["surface_sampling"]
    torch.manual_seed(0)
    noise = start_noise(cfg)
    extra_robots = [WUJI] + ([SHADOW] if B.get("shadow_pool") else [])
    bank = build_bank({**cfg, "robots": cfg["robots"] + extra_robots}, dev)
    fam = np.array([str(x) for x in bank.family], dtype=object)
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    systems, smeta = {}, {}
    for name, spec in B["systems"].items():
        spec = {k: (paths.expand(v) if isinstance(v, str) else v) for k, v in spec.items()}
        systems[name], smeta[name] = load_system(name, spec, dev, B["surface_sampling"])
    scfg = {"success_pos_mm": POS_MM, "success_normal_deg": ANG_DEG}

    # ---------------- manifest (pinned by hash)
    files = {}
    for name, spec in B["systems"].items():
        for k, v in spec.items():
            if k != "kind":
                files[f"system:{name}:{k}"] = paths.expand(v)
    files["psi"] = cfg["kite_ckpt"]
    for spec in cfg["robots"] + extra_robots:
        for k, v in spec.items():
            if isinstance(v, str) and Path(paths.expand(v)).is_file():
                files[f"robot:{spec.get('robot', spec.get('hand_dir', '?'))}:{k}"] = paths.expand(v)
            elif k == "hand_dir":
                for f in sorted(Path(paths.expand(v)).glob("*")):
                    if f.is_file():
                        files[f"robot:{Path(v).name}:{f.name}"] = str(f)
    pools = {"val": cfg["val"], "leap_heldout": cfg["extra_eval"]["leap_heldout"],
             "wuji": [{"kind": "kite", "robot": "wujihand_right", "npz": f"{HD}/wujihand_right_eval.npz"}]}
    if B.get("shadow_pool"):
        pools["shadow"] = [{"kind": "kite", "robot": "shadow_hand_right", "npz": str(SHADOW_AD / "shadow_eval.npz")}]
    for pn, spec in pools.items():
        for s in spec:
            files[f"pool:{pn}:{Path(s['npz']).name}"] = paths.expand(s["npz"])
    for s in cfg["train"]:
        files[f"train:{Path(s['npz']).parent.name}/{Path(s['npz']).name}"] = paths.expand(s["npz"])
    code = [HERE / "transfer_benchmark.py", HERE.parent / "diagnostics/crossbody_heldout_eval.py",
            HERE.parent / "diagnostics/wuji_per_example.py"] + \
           ([REPO / "scripts/decoder/shadow/make_shadow_records.py"] if B.get("shadow_pool") else []) + \
           [REPO / f"src/getkite/decoder/{f}.py" for f in ("train_scratch", "robots", "multifamily", "get_scratch", "spatial_bridge",
                                                           "joint_contact_get", "direct_worldaux", "train_joint", "rotations")]
    for c in code:
        files[f"code:{c.relative_to(REPO)}"] = str(c)
    manifest = {"benchmark": B, "resolved_base_config": cfg, "systems_meta": smeta,
                "sha256": {k: sha256(v) for k, v in sorted(files.items())},
                "surface_sampling": bank.surface_sampling,
                "surf_idx_sha16": hashlib.sha256(bank.surf_idx.cpu().numpy().tobytes()).hexdigest()[:16],
                "surf_idx_sha16_without_wuji": hashlib.sha256(bank.surf_idx[:len(cfg["robots"])].cpu().numpy().tobytes()).hexdigest()[:16],
                "tpl_sha16_per_robot": {n: hashlib.sha256(npy(bank.tpl_pos[i]).tobytes()).hexdigest()[:16] for i, n in enumerate(bank.names)},
                "joint_coords_per_robot": {n: int(bank.coord_valid[i].sum()) for i, n in enumerate(bank.names)},
                "start_noise": noise, "val_start_seed": cfg["val_start_seed"], "success": [POS_MM, ANG_DEG],
                "repo": git_state(REPO), "gpu": torch.cuda.get_device_name(),
                "pool4": B.get("pool4_status")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("manifest written", flush=True)

    train = load_pool(cfg["train"], bank)
    train_keys = record_keys(bank, train)
    del train

    M, PE = {"_meta": {"systems": list(systems), "noop": "no_movement", "R2_definitions": wrist_fit.__doc__}}, {}
    # ---------------- A2. matched near-goal contact realisation
    for pn, spec in pools.items():
        rec = load_pool(spec, bank).to(dev)
        sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
        keys = record_keys(bank, rec)
        P = {"_n": len(rec), "_overlap_with_train_records": len(keys & train_keys),
             "_contact_count_hist": {int(k): int(v) for k, v in zip(*np.unique(npy(rec.c_valid.sum(1)).astype(int), return_counts=True))},
             "_start_distribution": start_distribution(bank, rec, sg)}
        outs = {n: [] for n in systems}
        with torch.no_grad():
            for s in range(0, len(rec), 256):
                i = slice(s, s + 256)
                g = sg_index(sg, i)
                z = psi(rec.c_pos[i], rec.c_normal[i], rec.c_valid[i])
                for n, S in systems.items():
                    outs[n].append(S.predict(bank, rec.r[i], g, z))
        outs = {n: {k: torch.cat([o[k] for o in v]) for k in ("qc", "R", "t")} for n, v in outs.items()}
        with torch.no_grad():
            per = {"no_movement": score(bank, rec, rec.r, sg, sg.qc1, sg.R1, sg.t1, scfg),
                   "goal (sanity)": score(bank, rec, rec.r, sg, sg.qcg, sg.Rg, sg.tg, scfg)}
            for n, o in outs.items():
                per[n] = score(bank, rec, rec.r, sg, o["qc"], o["R"], o["t"], scfg)
        per = {n: {k: npy(v) for k, v in d.items()} for n, d in per.items()}
        fo = fam[npy(rec.r).astype(int)]
        names = np.array([bank.names[i] for i in npy(rec.r).astype(int)])
        for f in sorted(set(fo)):
            sel = fo == f
            F = {"n": int(sel.sum())}
            for n, d in per.items():
                F[n] = summarise(d, sel)
                if n in outs:
                    ss = torch.as_tensor(sel, device=dev)
                    F[n] |= wrist_fit(npy(sg.R1[ss]), npy(sg.Rg[ss]), npy(outs[n]["R"][ss]), npy(sg.t1[ss]), npy(sg.tg[ss]), npy(outs[n]["t"][ss]))
            F["_paired"] = {}
            for n in outs:
                for m in ("corr_mm", "compat_mm", "corr_success", "compat_success", "wrist_mm", "wrist_deg"):
                    F["_paired"][f"{n}-no_movement/{m}"] = paired_boot(per[n][m][sel], per["no_movement"][m][sel])
                    for n2 in outs:
                        if n2 < n:
                            F["_paired"][f"{n}-{n2}/{m}"] = paired_boot(per[n][m][sel], per[n2][m][sel])
            mors = sorted(set(names[sel]))
            if len(mors) > 1:  # variation across morphologies, separately from variation across goals
                F["_per_morphology"] = {mo: {n: {m: round(float(per[n][m][names == mo].mean()), 3) for m in ("corr_mm", "corr_success", "compat_success")}
                                             for n in per if n != "goal (sanity)"} | {"n": int((names == mo).sum())} for mo in mors}
                F["_across_morphology_sd"] = {n: {m: round(float(np.std([F["_per_morphology"][mo][n][m] for mo in mors])), 3)
                                                  for m in ("corr_mm", "corr_success", "compat_success")} for n in per if n != "goal (sanity)"}
            # contact-stage reconstruction per family (systems that have one)
            F["_contact_stage"] = {}
            for n, S in systems.items():
                cm = S.contact_model()
                with torch.no_grad():
                    F["_contact_stage"][n] = contact_eval(cm, psi, rec.index(torch.as_tensor(np.flatnonzero(sel), device=dev)))
                S.model.eval()
                if S.bridge is not None:
                    S.bridge.eval()
            P[f] = F
            print(pn, f, json.dumps({n: (F[n]["corr_mm"], F[n]["compat_success"]) for n in per}), flush=True)
        M[pn] = P
        PE[pn] = {"per": per, "out": {n: {k: npy(v) for k, v in o.items()} for n, o in outs.items()},
                  "sg": {k: npy(getattr(sg, k)) for k in sg.__dataclass_fields__},
                  "rec": {k: npy(getattr(rec, k)) for k in ("r", "c_pos", "c_normal", "c_valid", "c_tpl")}, "family": fo, "names": names}

    # ---------------- A3. cross-body intents (secondary)
    if not a.skip_crossbody:
        valw = load_pool(cfg["val"], bank).to(dev)
        wrec = load_pool(pools["wuji"], bank).to(dev)
        gsel = torch.Generator().manual_seed(SRC_SEED)
        fam_val = fam[npy(valw.r).astype(int)]
        srcs = {}
        for f in ("allegro", "barrett", "leap"):
            ids = np.flatnonzero(fam_val == f)
            srcs[f] = valw.index(torch.as_tensor(ids[torch.randperm(len(ids), generator=gsel)[:N_PER_SOURCE].numpy()], device=dev))
        srcs["wuji"] = wrec.index(torch.randperm(len(wrec.r), generator=gsel)[:N_PER_SOURCE].to(dev))
        held = json.loads(Path(paths.expand("${GETKITE_DATA}/multifamily_pilot/manifest.json")).read_text())["leap_heldout_hands"]
        exb = {"allegro": [{"kind": "kite", "robot": "allegro_right", "npz": paths.expand("${GETKITE_DATA}/multifamily_pilot/allegro_right_train.npz")}],
               "barrett": [{"kind": "kite", "robot": "barrett", "npz": paths.expand("${GETKITE_DATA}/multifamily_pilot/barrett_train.npz")}],
               "wuji": [{"kind": "kite", "robot": "wujihand_right", "npz": f"{HD}/wujihand_right_bank.npz"}]}
        exb |= {h: [{"kind": "leap", "npz": f"{DS}/examples/kite_code/val/{h}.npz", "rows": [125, 1000]}] for h in held}
        banks = {k: load_pool(v, bank).to(dev) for k, v in exb.items()}
        manifest["sha256"] |= {f"crossbody_example_bank:{k}": sha256(paths.expand(v[0]["npz"])) for k, v in exb.items()}
        rows, Bres = [], {}
        for sf, srec in srcs.items():
            for tf in ("allegro", "barrett", "leap_heldout", "wuji"):
                if tf.startswith(sf):
                    continue
                n = len(srec.r)
                hands = [held[i % len(held)] for i in range(n)] if tf == "leap_heldout" else [tf] * n
                starts = []
                for h in sorted(set(hands)):
                    ii = torch.tensor([i for i in range(n) if hands[i] == h], device=dev)
                    idx, delta = closest_example(srec.c_pos[ii], srec.c_valid[ii], banks[h].c_pos, banks[h].c_valid)
                    exr = banks[h].index(idx)
                    sg = sg_to(fixed_start(bank, exr, cfg["val_start_seed"] + 11, **noise), dev)
                    sg = type(sg)(**{**{k: getattr(sg, k) for k in sg.__dataclass_fields__}, "t1": sg.t1 + delta})
                    starts.append((ii, sg, exr.r))
                order = torch.cat([s[0] for s in starts])
                sg = type(starts[0][1])(**{k: torch.cat([getattr(s[1], k) for s in starts]) for k in starts[0][1].__dataclass_fields__})
                r = torch.cat([s[2] for s in starts])
                cp, cn, cv = srec.c_pos[order], srec.c_normal[order], srec.c_valid[order]
                res = {}
                with torch.no_grad():
                    z = psi(cp, cn, cv)
                    for nme, S in systems.items():
                        o = S.predict(bank, r, sg, z)
                        res[nme] = surface_score(bank, r, o["qc"], o["R"], o["t"], cp, cn, cv) | \
                            {"move_mm": (o["t"] - sg.t1).norm(dim=-1) * 1000, "move_deg": torch.rad2deg(geodesic_angle(o["R"], sg.R1))}
                    res["no_movement"] = surface_score(bank, r, sg.qc1, sg.R1, sg.t1, cp, cn, cv)
                orc = {}
                for s in range(0, len(r), 128):
                    i = slice(s, s + 128)
                    for k, v in oracle(bank, r[i], sg_index(sg, i), cp[i], cn[i], cv[i], seed=s).items():
                        orc.setdefault(k, []).append(v)
                res["optimizer"] = {k: torch.cat(v) for k, v in orc.items()}
                res = {p: {k: npy(v) for k, v in d.items()} for p, d in res.items()}
                found = res["optimizer"]["success"] > 0.5
                Bres[f"{sf}->{tf}"] = {"n": n, "optimizer_found_frac": round(float(found.mean()), 4),
                                       **{p: summarise(d, np.ones(n, bool)) for p, d in res.items()},
                                       **{p + "@optimizer_found": summarise(d, found) for p, d in res.items() if found.any()}}
                rows.append((sf, tf, res, found))
                print("B", sf, tf, json.dumps({p: (Bres[f'{sf}->{tf}'][p]['compat_mm'], Bres[f'{sf}->{tf}'][p]['success']) for p in res}), flush=True)
        agg = {}
        for key, filt in [("target=" + t, lambda sf, tf, t=t: tf == t) for t in ("allegro", "barrett", "leap_heldout", "wuji")] + [("all", lambda sf, tf: True)]:
            rs = [x for x in rows if filt(x[0], x[1])]
            cat = {p: {k: np.concatenate([x[2][p][k] for x in rs]) for k in rs[0][2][p]} for p in rs[0][2]}
            found = np.concatenate([x[3] for x in rs])
            A_ = {"n": int(len(found)), "optimizer_found_frac": round(float(found.mean()), 4)}
            A_ |= {p: summarise(d, np.ones(len(found), bool)) for p, d in cat.items()}
            A_ |= {p + "@optimizer_found": summarise(d, found) for p, d in cat.items()}
            A_["_paired"] = {f"{n}-no_movement/{m}": paired_boot(cat[n][m], cat["no_movement"][m]) for n in systems for m in ("compat_mm", "success")}
            agg[key] = A_
        M["crossbody"] = {"pairs": Bres, "aggregate": agg,
                          "_disclosure": "starts: KITE closest-example from TARGET-hand synthetic example banks (target synthetic data used at "
                                         "test time, not for training); one correction step; kinematic only (no object collisions); "
                                         "optimizer = Adam from the start (+3 perturbed restarts), 'found' is not a feasibility classification"}
        PE["crossbody"] = {k: {"res": v[2], "found": v[3]} for k, v in zip(Bres, rows)}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "metrics.json").write_text(json.dumps(M, indent=1))
    torch.save(PE, out / "per_example.pt")
    print("wrote", out)


if __name__ == "__main__":
    main()
