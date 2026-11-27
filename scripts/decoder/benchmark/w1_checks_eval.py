"""Two evaluation-only checks on the existing W1 checkpoint (2026-11-19). No training, nothing tuned.

Protocol identical to wrist_transfer_eval.py: frozen v1 records (source val, training goals, held-out LEAP, Wuji
near-goal), wrist-only starts fixed_start(val_start_seed, joint_frac=0) -> the fingers start exactly at the goal.

Conditions (same W1 checkpoint, same records, same starts):
  full / true          W1's full prediction (wrist + fingers), true request contacts as input       (= wrist_transfer_eval)
  wrist_only / true    W1's predicted wrist pose, fingers kept EXACTLY at their starting values
  full / translator    full prediction, input = contacts predicted from psi(request) by the frozen world-frame
                       translator set_bridge_zworld_c3_20k_b (trained on the source training records only; never saw Wuji)
  wrist_only / translator
  no_movement
Context (evaluation-only, same records/starts; added after W1's joint head was found to be ~0 by construction):
  raw_ref full / true, raw_ref wrist_only / true   the fingers-at-goal control checkpoint (trained with finger noise)
Reported: corr / compat contact error and both success measures, wrist error, wrist and finger movement (mean |dq| over
valid coordinates, rad and fraction of range), paired bootstrap differences per record.

    python scripts/decoder/benchmark/w1_checks_eval.py --config configs/decoder/scratch/w1_wristonly_true50k_hf.json
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
from getkite.decoder.spatial_bridge import ContactSetBridge, SetBridgeConfig
from getkite.decoder.train_scratch import FrozenPsi, RawContacts, build_bank, load_config, load_pool, start_noise

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "diagnostics"))
from crossbody_heldout_eval import ANG_DEG, HD, POS_MM, WUJI, bridge_contacts, sg_index, sg_to  # noqa: E402
from transfer_benchmark import npy, score, summarise, wrist_fit  # noqa: E402
from wrist_transfer_eval import load_get  # noqa: E402
from wuji_per_example import paired_boot  # noqa: E402

PAIR_M = ("corr_mm", "corr_success", "compat_success", "wrist_mm", "wrist_deg")
PAIRS = [("wrist_only/true", "full/true"), ("full/translator", "full/true"), ("wrist_only/translator", "wrist_only/true"),
         ("full/true", "no_movement"), ("wrist_only/true", "no_movement"), ("full/translator", "no_movement"),
         ("wrist_only/translator", "no_movement"), ("raw_ref:wrist_only/true", "raw_ref:full/true"),
         ("raw_ref:wrist_only/true", "no_movement"), ("full/true", "raw_ref:wrist_only/true")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. out_dir=... for a seed run")
    ap.add_argument("--config", required=True)
    ap.add_argument("--translator", default="${GETKITE_RUNS}/set_bridge_zworld_c3_20k_b/best.pt")
    ap.add_argument("--out_dir", default="${GETKITE_RUNS}/w1_checks_eval")
    ap.add_argument("--raw_ref", default="${GETKITE_RUNS}/scratch_c3_raw50k_cf8_10x_wrist_hf/best.pt")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    out = Path(paths.expand(a.out_dir)); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    noise = start_noise(cfg)
    assert noise["joint_frac"] == 0.0
    bank = build_bank({**cfg, "robots": cfg["robots"] + [WUJI]}, dev)
    raw = RawContacts()
    scfg = {"success_pos_mm": POS_MM, "success_normal_deg": ANG_DEG}
    w1, meta = load_get(str(Path(paths.expand(cfg["out_dir"])) / "best.pt"), dev, cfg["surface_sampling"])
    rref, rmeta = load_get(paths.expand(a.raw_ref), dev, cfg["surface_sampling"])
    jh = w1.joint_head[-1]
    meta["joint_head_last_layer_max_abs"] = max(float(jh.weight.abs().max()), float(jh.bias.abs().max()))
    bs = torch.load(paths.expand(a.translator), map_location=dev, weights_only=False)
    br = ContactSetBridge(SetBridgeConfig(**bs["config"])).to(dev); br.load_state_dict(bs["model"]); br.eval().requires_grad_(False)
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    M = {"_meta": {"doc": __doc__, "w1": meta, "raw_ref": rmeta, "translator": paths.expand(a.translator), "translator_step": bs.get("step"),
                   "translator_train_config": bs.get("args", {}).get("config"), "noise": noise, "val_start_seed": cfg["val_start_seed"]}}
    pools = {"val": cfg["val"], "train_goals": cfg["extra_eval"]["train_goals"], "leap_heldout": cfg["extra_eval"]["leap_heldout"],
             "wuji": [{"kind": "kite", "robot": "wujihand_right", "npz": f"{HD}/wujihand_right_eval.npz"}]}
    PE = {}
    for pn, spec in pools.items():
        rec = load_pool(spec, bank).to(dev)
        sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
        assert torch.allclose(sg.qc1, sg.qcg)
        fam = np.array([str(bank.family[i]) for i in npy(rec.r).astype(int)])
        preds = {"true": {"qc": [], "R": [], "t": []}, "translator": {"qc": [], "R": [], "t": []}, "raw_ref": {"qc": [], "R": [], "t": []}}
        with torch.no_grad():
            for s in range(0, len(rec), 256):
                i = slice(s, s + 256)
                g = sg_index(sg, i)
                ins = {"true": raw(rec.c_pos[i], rec.c_normal[i], rec.c_valid[i]),
                       "translator": raw(*bridge_contacts(br, psi(rec.c_pos[i], rec.c_normal[i], rec.c_valid[i])))}
                for k, zin in list(ins.items()) + [("raw_ref", ins["true"])]:
                    p = (rref if k == "raw_ref" else w1)(bank, rec.r[i], g.qc1, g.R1, g.t1, zin)
                    for f in ("qc", "R", "t"):
                        preds[k][f].append(p[f])
        preds = {k: {f: torch.cat(v) for f, v in d.items()} for k, d in preds.items()}
        outs = {}
        for k, p in preds.items():
            pre = "raw_ref:" if k == "raw_ref" else ""
            src = "true" if k == "raw_ref" else k
            outs[f"{pre}full/{src}"] = p
            outs[f"{pre}wrist_only/{src}"] = {"qc": sg.qc1.clone(), "R": p["R"], "t": p["t"]}
        cv = bank.coord_valid[rec.r]
        rng = bank.coords_of(rec.r, bank.range[rec.r]).clamp_min(1e-6)
        per = {}
        with torch.no_grad():
            per["no_movement"] = {k: npy(v) for k, v in score(bank, rec, rec.r, sg, sg.qc1, sg.R1, sg.t1, scfg).items()}
            for n, o in outs.items():
                d = {k: npy(v) for k, v in score(bank, rec, rec.r, sg, o["qc"], o["R"], o["t"], scfg).items()}
                dq = (o["qc"] - sg.qc1).abs() * cv
                d["finger_move_rad"] = npy(dq.sum(1) / cv.sum(1))
                d["finger_move_frac_range"] = npy((dq / rng).sum(1) / cv.sum(1))
                d["finger_move_max_rad"] = npy(dq.amax(1))
                per[n] = d
        P = {"_n": len(rec)}
        for f in sorted(set(fam)):
            sel = fam == f
            F = {"n": int(sel.sum())}
            for n, d in per.items():
                F[n] = summarise(d, sel)
                if n in outs:
                    F[n] |= wrist_fit(npy(sg.R1)[sel], npy(sg.Rg)[sel], npy(outs[n]["R"])[sel], npy(sg.t1)[sel], npy(sg.tg)[sel], npy(outs[n]["t"])[sel])
            F["_paired"] = {f"{x}-{y}/{m}": paired_boot(per[x][m][sel], per[y][m][sel]) for x, y in PAIRS for m in PAIR_M}
            P[f] = F
        M[pn] = P
        PE[pn] = per
    (out / "metrics.json").write_text(json.dumps(M, indent=1))
    torch.save(PE, out / "per_example.pt")
    for pn in pools:
        for f, F in M[pn].items():
            if f.startswith("_"):
                continue
            for n in ("no_movement", "full/true", "wrist_only/true", "full/translator", "wrist_only/translator", "raw_ref:full/true", "raw_ref:wrist_only/true"):
                d = F[n]
                print(pn, f, n, {k: d[k] for k in ("corr_mm", "corr_success", "compat_success", "wrist_mm", "move_mm", "finger_move_rad", "finger_move_frac_range") if k in d}, flush=True)


if __name__ == "__main__":
    main()
