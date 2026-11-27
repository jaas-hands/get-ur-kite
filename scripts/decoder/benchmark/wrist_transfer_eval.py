"""W1 wrist-transfer diagnostic evaluation (2026-11-19). Evaluation only; nothing is tuned here.

Protocol: the frozen v1 records (source val, held-out LEAP, training goals, Wuji near-goal) with WRIST-ONLY starts:
fixed_start(val_start_seed, xyz 5 mm, rot 5 deg, joint_frac=0), i.e. the frozen benchmark's wrist draws with the
fingers exactly at the goal. Inputs to the decoders are the TRUE request contacts (raw contacts = diagnostic
reference, not a deployment-time input).

Systems:
  no_movement     start pose
  raw_ref         scratch_c3_raw50k_cf8_10x_wrist_hf (true-contact GET trained with KITE's standard starts: fingers
                  perturbed 5% of range). Control: same architecture/data/loss, different start distribution in training.
  w1              w1_wristonly_true50k_hf (same, trained with fingers at goal) -- scored only if its best.pt exists
  kite_released   KITE's released decoder (its own heads; in-family for Allegro/Barrett/Wuji; never LEAP).
                  ADVANTAGE: trained on Wuji, jointly with psi, far more samples. Input: request contacts via psi.

Source learning (val / train goals / held-out LEAP) and Wuji transfer (unseen by our decoders, seen by KITE's psi)
are reported separately.

    python scripts/decoder/benchmark/wrist_transfer_eval.py --config configs/decoder/scratch/w1_wristonly_true50k_hf.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from getkite.decoder import paths
from getkite.decoder.get_scratch import GETScratch, ScratchConfig
from getkite.decoder.kite_reference import build_kite_bank, kite_predict
from getkite.decoder.multifamily import fixed_start
from getkite.decoder.train_scratch import RawContacts, build_bank, load_config, load_pool, start_noise

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "diagnostics"))
from crossbody_heldout_eval import ANG_DEG, HD, POS_MM, WUJI, sg_index, sg_to  # noqa: E402
from transfer_benchmark import npy, score, summarise, wrist_fit  # noqa: E402
from wuji_per_example import paired_boot  # noqa: E402

PAIR_M = ("corr_mm", "corr_success", "compat_success", "wrist_mm", "wrist_deg")


def load_get(path, dev, surface_sampling):
    ck = torch.load(path, map_location=dev, weights_only=False)
    assert ck["config"].get("surface_sampling") == surface_sampling
    assert ck["model_config"]["intent_source"] == "raw_contacts"
    m = GETScratch(ScratchConfig.from_dict(ck["model_config"])).to(dev)
    m.load_state_dict(ck["model"]); m.amp = False
    return m.eval().requires_grad_(False), {"step": ck.get("step"), "best": ck.get("best"),
                                            "start_joint_frac": ck["config"].get("start_joint_frac", 0.05)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. out_dir=... for a seed run")
    ap.add_argument("--config", required=True, help="W1 config (records, noise)")
    ap.add_argument("--raw_ref", default="${GETKITE_RUNS}/scratch_c3_raw50k_cf8_10x_wrist_hf/best.pt")
    ap.add_argument("--out_dir", default="${GETKITE_RUNS}/w1_wrist_transfer_eval")
    ap.add_argument("--no_kite", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    out = Path(paths.expand(a.out_dir)); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    noise = start_noise(cfg)
    assert noise["joint_frac"] == 0.0, "W1 protocol: fingers at goal"
    bank = build_bank({**cfg, "robots": cfg["robots"] + [WUJI]}, dev)
    raw = RawContacts()
    scfg = {"success_pos_mm": POS_MM, "success_normal_deg": ANG_DEG}
    systems, meta = {}, {}
    systems["raw_ref"], meta["raw_ref"] = load_get(paths.expand(a.raw_ref), dev, cfg["surface_sampling"])
    w1 = Path(paths.expand(cfg["out_dir"])) / "best.pt"
    if w1.exists():
        systems["w1"], meta["w1"] = load_get(str(w1), dev, cfg["surface_sampling"])
    kite = None if a.no_kite else build_kite_bank(cfg["kite_ckpt"], cfg["kite_root"], dev)  # KITE runs its psi internally
    M = {"_meta": {"doc": __doc__, "noise": noise, "val_start_seed": cfg["val_start_seed"], "systems": meta}}
    pools = {"val": cfg["val"], "train_goals": cfg["extra_eval"]["train_goals"], "leap_heldout": cfg["extra_eval"]["leap_heldout"],
             "wuji": [{"kind": "kite", "robot": "wujihand_right", "npz": f"{HD}/wujihand_right_eval.npz"}]}
    PE = {}
    for pn, spec in pools.items():
        rec = load_pool(spec, bank).to(dev)
        sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
        assert torch.allclose(sg.qc1, sg.qcg), "fingers must start at the goal"
        fam = np.array([str(bank.family[i]) for i in npy(rec.r).astype(int)])
        outs = {}
        with torch.no_grad():
            for n, m in systems.items():
                o = {"qc": [], "R": [], "t": []}
                for s in range(0, len(rec), 256):
                    i = slice(s, s + 256)
                    g = sg_index(sg, i)
                    p = m(bank, rec.r[i], g.qc1, g.R1, g.t1, raw(rec.c_pos[i], rec.c_normal[i], rec.c_valid[i]))
                    for k in o:
                        o[k].append(p[k])
                outs[n] = {k: torch.cat(v) for k, v in o.items()}
            if kite is not None:
                model, kmeta = kite
                o = {"qc": sg.qc1.clone(), "R": sg.R1.clone(), "t": sg.t1.clone()}
                has = torch.zeros(len(rec), dtype=torch.bool, device=dev)
                for ri in sorted(set(npy(rec.r).astype(int))):
                    if bank.names[ri] not in kmeta["manipulators"]:
                        continue
                    ii = (rec.r == ri).nonzero().squeeze(1)
                    for s in range(0, len(ii), 256):
                        jj = ii[s:s + 256]
                        qc, t, R = kite_predict(model, kmeta, bank, ri, rec.index(jj), sg_index(sg, jj), dev)
                        o["qc"][jj], o["t"][jj], o["R"][jj] = qc, t, R
                    has[ii] = True
                if has.any():
                    outs["kite_released"] = o
                    kite_has = npy(has).astype(bool)
            per = {"no_movement": {k: npy(v) for k, v in score(bank, rec, rec.r, sg, sg.qc1, sg.R1, sg.t1, scfg).items()}}
            for n, o in outs.items():
                per[n] = {k: npy(v) for k, v in score(bank, rec, rec.r, sg, o["qc"], o["R"], o["t"], scfg).items()}
        P = {"_n": len(rec)}
        for f in sorted(set(fam)):
            sel = fam == f
            F = {"n": int(sel.sum())}
            for n, d in per.items():
                if n == "kite_released" and not kite_has[sel].all():
                    continue
                F[n] = summarise(d, sel)
                if n in outs:
                    F[n] |= wrist_fit(npy(sg.R1)[sel], npy(sg.Rg)[sel], npy(outs[n]["R"])[sel], npy(sg.t1)[sel], npy(sg.tg)[sel], npy(outs[n]["t"])[sel])
            F["_paired"] = {}
            for n in F:
                if n in outs:
                    for m in PAIR_M:
                        F["_paired"][f"{n}-no_movement/{m}"] = paired_boot(per[n][m][sel], per["no_movement"][m][sel])
            if "w1" in outs:
                for m in PAIR_M:
                    F["_paired"][f"w1-raw_ref/{m}"] = paired_boot(per["w1"][m][sel], per["raw_ref"][m][sel])
            P[f] = F
        M[pn] = P
        PE[pn] = per
    (out / "metrics.json").write_text(json.dumps(M, indent=1))
    torch.save(PE, out / "per_example.pt")
    for pn in pools:
        for f, F in M[pn].items():
            if f.startswith("_"):
                continue
            for n in ("no_movement", "raw_ref", "w1", "kite_released"):
                if n in F:
                    d = F[n]
                    print(pn, f, n, {k: d[k] for k in ("corr_mm", "corr_success", "compat_success", "wrist_mm", "wrist_deg", "move_mm") if k in d},
                          {k: d[k] for k in ("trans_slope", "trans_R2_id", "rot_slope", "rot_R2_id") if k in d}, flush=True)


if __name__ == "__main__":
    main()
