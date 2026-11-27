"""Phase 0 wrist diagnostics (docs/coupling_and_wrist_proposal.md; evaluation only, no model is re-run or trained).

Reads the saved per-example outputs of a transfer benchmark (per_example.pt: predicted qc / R / t, fixed starts and goals,
records) and reports, per pool and system, by training seed (never pooled across runs):
  W-a  translation correction per start-wrist-frame axis: need = R1^T (tg - t1), got = R1^T (t - t1); per axis slope,
       R2_id and RMS error (no movement = RMS of need); plus the component of got along need (parallel, / |need|) and the
       perpendicular RMS.
  W-b  Wuji / Shadow records split by where the contacts are: >= 1 contact on a long-finger distal link (Shadow: the links
       moved by the coupled J1 = J2 joints; Wuji: the same links on an uncoupled hand, as the control) vs none. Per
       stratum: n, wrist mm (system and no movement), paired difference with bootstrap CI, translation R2_id.
  W-c  oracle composition, re-scored with the benchmark's own score(): (1) decoder wrist + goal fingers, (2) goal wrist +
       decoder fingers, (3) full decoder, (4) no movement. Requires rebuilding the bank and the pool records (identical
       fixed starts are verified against the saved ones).
Predeclared decision rule: docs/coupling_and_wrist_proposal.md, Phase 0.

    PYTHONPATH=src python scripts/decoder/benchmark/wrist_diag.py --bench configs/decoder/benchmark/transfer_v1_followups.json \
        --systems d0_g1 d1 d1b n5 r4 --out $GETKITE_RUNS/wrist_diag_v1 [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from getkite.decoder import paths
from getkite.decoder.multifamily import fixed_start
from getkite.decoder.robots import urdf_desc
from getkite.decoder.train_scratch import build_bank, load_config, load_pool, start_noise

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "diagnostics"))
from crossbody_heldout_eval import ANG_DEG, HD, POS_MM, WUJI, sg_to  # noqa: E402
from transfer_benchmark import SHADOW, SHADOW_AD, score  # noqa: E402
from wuji_per_example import paired_boot  # noqa: E402

REPO = Path(paths.REPO)
SEEDS = (0, 1, 2)
POOLS = [("val", "allegro"), ("val", "barrett"), ("val", "leap"), ("leap_heldout", "leap"), ("wuji", "wuji"), ("shadow", "shadow")]


def r2id(got, need):
    return float(1 - ((got - need) ** 2).sum() / max(((need - need.mean(0)) ** 2).sum(), 1e-12))


def wa(sg, out, sel):
    R1, t1, tg, t = sg["R1"][sel], sg["t1"][sel], sg["tg"][sel], out["t"][sel]
    need = np.einsum("nji,nj->ni", R1, tg - t1) * 1000; got = np.einsum("nji,nj->ni", R1, t - t1) * 1000
    res = {}
    for k, ax in enumerate("xyz"):
        n, g = need[:, k], got[:, k]
        res[ax] = {"slope": round(float((g * n).sum() / max((n * n).sum(), 1e-12)), 3), "R2_id": round(r2id(g[:, None], n[:, None]), 3),
                   "rms_err_mm": round(float(np.sqrt(((g - n) ** 2).mean())), 3), "rms_need_mm": round(float(np.sqrt((n ** 2).mean())), 3)}
    u = need / np.linalg.norm(need, axis=1, keepdims=True).clip(1e-9)
    par = (got * u).sum(1)
    perp = got - par[:, None] * u
    res["parallel_over_need"] = round(float(np.mean(par / np.linalg.norm(need, axis=1).clip(1e-9))), 3)
    res["perp_rms_mm"] = round(float(np.sqrt((perp ** 2).sum(1).mean())), 3)
    res["got_rms_mm"] = round(float(np.sqrt((got ** 2).sum(1).mean())), 3)
    return res


def distal_links(names, robot):
    if robot == "shadow":  # J1 (coupled to J2) moves the distal link of ff / mf / rf / lf
        return {i for i, n in enumerate(names) if any(f"rh_{f}distal" in n or f"rh_{f}tip" in n for f in ("ff", "mf", "rf", "lf"))}
    # wuji: finger1 is the thumb; link4 + tip of fingers 2-5 are the corresponding distal links (no coupling)
    return {i for i, n in enumerate(names) if any(f"finger{f}_link4" in n or f"finger{f}_tip" in n for f in (2, 3, 4, 5))}


def wb(PE, pool, robot, desc, systems):
    P = PE[pool]
    tl = np.asarray(desc.tpl_link)
    dset = distal_links(list(desc.tpl_link_names), robot)
    ct, cv = P["rec"]["c_tpl"].astype(int), P["rec"]["c_valid"] > 0.5
    on_distal = np.array([any(tl[ct[i, k]] in dset for k in range(ct.shape[1]) if cv[i, k]) for i in range(len(ct))])
    out = {"distal_links": sorted(desc.tpl_link_names[i] for i in dset), "n_with_distal": int(on_distal.sum()), "n_without": int((~on_distal).sum())}
    nm = P["per"]["no_movement"]["wrist_mm"]
    for sname in systems:
        for s in SEEDS:
            n = f"{sname}_s{s}"
            w = P["per"][n]["wrist_mm"]
            row = {}
            for lab, sel in (("distal", on_distal), ("no_distal", ~on_distal)):
                R1, t1, tg, t = P["sg"]["R1"][sel], P["sg"]["t1"][sel], P["sg"]["tg"][sel], P["out"][n]["t"][sel]
                need = np.einsum("nji,nj->ni", R1, tg - t1); got = np.einsum("nji,nj->ni", R1, t - t1)
                pb = paired_boot(w[sel], nm[sel])
                row[lab] = {"n": int(sel.sum()), "wrist_mm": round(float(w[sel].mean()), 3), "nomove_mm": round(float(nm[sel].mean()), 3),
                            "diff": round(pb["mean_diff"], 3), "ci95": [round(x, 3) for x in pb["ci95"]], "trans_R2_id": round(r2id(got, need), 3)}
            out[n] = row
    return out


def wc(B, PE, systems, dev):
    bc = Path(paths.expand(B["base_config"]))
    cfg = load_config(str(bc if bc.is_absolute() else REPO / bc), [])
    cfg["surface_sampling"] = B["surface_sampling"]
    torch.manual_seed(0)
    noise = start_noise(cfg)
    bank = build_bank({**cfg, "robots": cfg["robots"] + [WUJI, SHADOW]}, dev)
    pools = {"val": cfg["val"], "leap_heldout": cfg["extra_eval"]["leap_heldout"],
             "wuji": [{"kind": "kite", "robot": "wujihand_right", "npz": f"{HD}/wujihand_right_eval.npz"}],
             "shadow": [{"kind": "kite", "robot": "shadow_hand_right", "npz": str(SHADOW_AD / "shadow_eval.npz")}]}
    scfg = {"success_pos_mm": POS_MM, "success_normal_deg": ANG_DEG}
    res = {}
    for pn, spec in pools.items():
        rec = load_pool(spec, bank).to(dev)
        sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
        saved = PE[pn]["sg"]
        dmax = max(float(np.abs(getattr(sg, k).cpu().numpy() - saved[k]).max()) for k in ("qc1", "t1", "R1", "qcg", "tg", "Rg"))
        assert dmax < 1e-5, f"{pn}: fixed starts differ from the benchmark's ({dmax})"
        fo = PE[pn]["family"]
        T = lambda x: torch.as_tensor(x, device=dev)  # noqa: E731
        comps = {}
        with torch.no_grad():
            comps["no_movement"] = score(bank, rec, rec.r, sg, sg.qc1, sg.R1, sg.t1, scfg)
            for sname in systems:
                for s in SEEDS:
                    n = f"{sname}_s{s}"; o = PE[pn]["out"][n]
                    comps[f"{n}/full"] = score(bank, rec, rec.r, sg, T(o["qc"]), T(o["R"]), T(o["t"]), scfg)
                    comps[f"{n}/dec_wrist+goal_fingers"] = score(bank, rec, rec.r, sg, sg.qcg, T(o["R"]), T(o["t"]), scfg)
                    comps[f"{n}/goal_wrist+dec_fingers"] = score(bank, rec, rec.r, sg, T(o["qc"]), sg.Rg, sg.tg, scfg)
        comps = {k: {m: v[m].float().cpu().numpy() for m in ("corr_mm", "compat_success", "corr_success", "wrist_mm")} for k, v in comps.items()}
        res[pn] = {}
        for f in sorted(set(fo)):
            sel = fo == f
            res[pn][f] = {k: {m: round(float(v[m][sel].mean()), 3) for m in v} for k, v in comps.items()}
            # paired: dec_wrist+goal_fingers vs no movement on corr_mm (does the decoder's wrist alone help?)
            for sname in systems:
                for s in SEEDS:
                    n = f"{sname}_s{s}"
                    pb = paired_boot(comps[f"{n}/dec_wrist+goal_fingers"]["corr_mm"][sel], comps["no_movement"]["corr_mm"][sel])
                    res[pn][f][f"{n}/dec_wrist+goal_fingers"]["corr_mm_vs_nomove"] = [round(pb["mean_diff"], 3)] + [round(x, 3) for x in pb["ci95"]]
        print("W-c", pn, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True); ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cpu"); ap.add_argument("--skip_wc", action="store_true")
    a = ap.parse_args()
    B = json.loads(Path(a.bench).read_text())
    out = Path(paths.expand(a.out)); out.mkdir(parents=True, exist_ok=True)
    PE = torch.load(Path(paths.expand(B["out_dir"])) / "per_example.pt", weights_only=False)
    R = {"bench": a.bench, "systems": a.systems, "W-a": {}, "W-b": {}}
    for pn, f in POOLS:
        P = PE[pn]; sel = P["family"] == f
        R["W-a"][f"{pn}/{f}"] = {}
        for sname in a.systems:
            for s in SEEDS:
                R["W-a"][f"{pn}/{f}"][f"{sname}_s{s}"] = wa(P["sg"], P["out"][f"{sname}_s{s}"], sel)
    dW = urdf_desc(WUJI["urdf"], WUJI["template"], "wujihand_right", "wuji", ignore_mimic=WUJI["ignore_mimic"])
    dS = urdf_desc(SHADOW["urdf"], SHADOW["template"], "shadow_hand_right", "shadow", ignore_mimic=False)
    R["W-b"]["wuji"] = wb(PE, "wuji", "wuji", dW, a.systems)
    R["W-b"]["shadow"] = wb(PE, "shadow", "shadow", dS, a.systems)
    (out / "wrist_diag.json").write_text(json.dumps(R, indent=1))
    print("W-a / W-b written", flush=True)
    if not a.skip_wc:
        R["W-c"] = wc(B, PE, a.systems, torch.device(a.device))
        (out / "wrist_diag.json").write_text(json.dumps(R, indent=1))
    print("wrote", out / "wrist_diag.json")


if __name__ == "__main__":
    main()
