"""Reference systems on the frozen v1 near-goal Wuji pool (2026-11-19). No training, nothing tuned.

Same records, same fixed starts (val_start_seed + KITE start noise) and same `score` as transfer_benchmark.py. Two
references, each with access our decoders do not have (labelled in the output):

  kite_released  KITE's released decoder bank (checkpoints/decoder/release.pt, EMA weights), its wujihand_right head,
                 run through KITE's own predict_action (getkite.decoder.kite_reference.kite_predict).
                 ADVANTAGE: trained on Wuji (in-family: Wuji has its own per-embodiment head and was in KITE's decoder
                 training set), jointly with psi, for far more samples than our decoders.
  optimizer      the cross-body protocol's non-learned optimizer (crossbody_heldout_eval.oracle: Adam on wrist + joints
                 from the start, 300 steps, restart 0 = the start + 3 perturbed restarts, best restart by compat score).
                 ADVANTAGE: test-time access to the raw target contacts (diagnostic reference; not a deployment-time
                 input for our decoders), Wuji's exact FK and surface template, and iterative optimisation of the
                 compatible-contact objective that compat_success measures. No correspondence labels.

Also runs KITE's released Allegro/Barrett heads on the source val pool as a positive control (also in-family).
Paired comparisons against our six seed runs use per-record values from transfer_benchmark_v1_seeds/per_example.pt
(identical records and order, checked via the no-movement scores).

    python scripts/decoder/benchmark/wuji_references.py --config configs/decoder/benchmark/transfer_v1_seeds.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from getkite.decoder import paths
from getkite.decoder.kite_reference import build_kite_bank, kite_predict
from getkite.decoder.multifamily import fixed_start
from getkite.decoder.train_scratch import build_bank, load_config, load_pool, start_noise

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "diagnostics"))
from crossbody_heldout_eval import ANG_DEG, HD, POS_MM, WUJI, oracle, sg_index, sg_to  # noqa: E402
from transfer_benchmark import REPO, npy, score, summarise, wrist_fit  # noqa: E402
from wuji_per_example import paired_boot  # noqa: E402

PAIR_M = ("corr_mm", "corr_success", "compat_mm", "compat_success", "wrist_mm")
OURS = ("j1", "j1_s1", "j1_s2", "g1_s0", "g1_s1", "g1_s2")


def run_kite(model, meta, bank, rec, sg, dev):
    outs = {"qc": [], "R": [], "t": []}
    for s in range(0, len(rec), 256):
        idx = torch.arange(s, min(s + 256, len(rec)), device=dev)
        rb, g = rec.index(idx), sg_index(sg, slice(s, s + 256))
        ri = int(rb.r[0])
        assert bool((rb.r == ri).all())
        qc, t, R = kite_predict(model, meta, bank, ri, rb, g, dev)
        for k, v in (("qc", qc), ("R", R), ("t", t)):
            outs[k].append(v)
    return {k: torch.cat(v) for k, v in outs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", default="${GETKITE_RUNS}/wuji_references_v1")
    a = ap.parse_args()
    B = json.loads(Path(a.config).read_text())
    out = Path(paths.expand(a.out_dir)); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    bc = Path(paths.expand(B["base_config"]))
    cfg = load_config(str(bc if bc.is_absolute() else REPO / bc), [])
    cfg["surface_sampling"] = B["surface_sampling"]
    torch.manual_seed(0)
    noise = start_noise(cfg)
    bank = build_bank({**cfg, "robots": cfg["robots"] + [WUJI]}, dev)
    scfg = {"success_pos_mm": POS_MM, "success_normal_deg": ANG_DEG}
    model, meta = build_kite_bank(cfg["kite_ckpt"], cfg["kite_root"], dev)
    PE = torch.load(Path(paths.expand(B["out_dir"])) / "per_example.pt", weights_only=False)
    M = {"_meta": {"advantages": __doc__, "kite_ckpt": cfg["kite_ckpt"], "kite_heads": meta["manipulators"],
                   "seed_benchmark": paths.expand(B["out_dir"])}}

    pools = {"wuji": [{"kind": "kite", "robot": "wujihand_right", "npz": f"{HD}/wujihand_right_eval.npz"}], "val": cfg["val"]}
    for pn, spec in pools.items():
        rec = load_pool(spec, bank).to(dev)
        sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
        fam = np.array([str(bank.family[i]) for i in npy(rec.r).astype(int)])
        with torch.no_grad():
            per = {"no_movement": {k: npy(v) for k, v in score(bank, rec, rec.r, sg, sg.qc1, sg.R1, sg.t1, scfg).items()}}
        # identity check against the seed benchmark's stored per-record no-movement scores
        ref = PE[pn]["per"]["no_movement"]["corr_mm"]
        assert ref.shape == per["no_movement"]["corr_mm"].shape and np.allclose(ref, per["no_movement"]["corr_mm"], atol=1e-4), pn
        kite_fams = {str(bank.family[i]) for i, n in enumerate(bank.names) if n in meta["manipulators"]}
        sel_k = np.isin(fam, sorted(kite_fams))
        outs = {}
        with torch.no_grad():
            if sel_k.any():
                ik = torch.as_tensor(np.flatnonzero(sel_k), device=dev)
                # kite_predict needs one robot per batch: order records by robot, keep original indices
                o = {"qc": torch.zeros_like(sg.qc1), "R": sg.R1.clone(), "t": sg.t1.clone()}
                for ri in sorted(set(npy(rec.r[ik]).astype(int))):
                    ii = (rec.r == ri).nonzero().squeeze(1)
                    sub = sg_index(sg, ii)
                    p = run_kite(model, meta, bank, rec.index(ii), sub, dev)
                    for k in o:
                        o[k][ii] = p[k]
                outs["kite_released"] = o
        if pn == "wuji":
            r = oracle(bank, rec.r, sg, rec.c_pos, rec.c_normal, rec.c_valid, seed=0, return_pose=True)
            outs["optimizer"] = {k: v.detach() for k, v in r["_pose"].items()}
        for n, o in outs.items():
            with torch.no_grad():
                per[n] = {k: npy(v) for k, v in score(bank, rec, rec.r, sg, o["qc"], o["R"], o["t"], scfg).items()}
        P = {"_n": len(rec)}
        for f in sorted(set(fam)):
            sel = fam == f
            F = {"n": int(sel.sum())}
            for n, d in per.items():
                if n != "no_movement" and n == "kite_released" and f not in kite_fams:
                    continue
                F[n] = summarise(d, sel)
                if n in outs:
                    F[n] |= wrist_fit(npy(sg.R1)[sel], npy(sg.Rg)[sel], npy(outs[n]["R"])[sel], npy(sg.t1)[sel], npy(sg.tg)[sel], npy(outs[n]["t"])[sel])
            F["_paired"] = {}
            for n in outs:
                for m in PAIR_M:
                    F["_paired"][f"{n}-no_movement/{m}"] = paired_boot(per[n][m][sel], per["no_movement"][m][sel])
                    for o in OURS:
                        F["_paired"][f"{n}-{o}/{m}"] = paired_boot(per[n][m][sel], PE[pn]["per"][o][m][sel])
            P[f] = F
        M[pn] = P
        torch.save({k: v for k, v in per.items()}, out / f"per_example_{pn}.pt")
    (out / "metrics.json").write_text(json.dumps(M, indent=1))
    for pn in pools:
        for f, F in M[pn].items():
            if f.startswith("_"):
                continue
            for n in ("no_movement", "kite_released", "optimizer"):
                if n in F:
                    d = F[n]
                    print(pn, f, n, {k: d[k] for k in ("corr_mm", "corr_success", "compat_mm", "compat_success", "wrist_mm", "move_mm") if k in d},
                          {k: d[k] for k in ("trans_slope", "trans_R2_lin", "rot_slope", "rot_R2_lin") if k in d})


if __name__ == "__main__":
    main()
