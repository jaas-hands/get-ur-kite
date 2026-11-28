"""Decoders on the external evaluation hands (evaluation only; scripts/decoder/external_hands/make_hand_adapter.py).

Hands: Ability, Inspire, XHand, Schunk SVH, MetaHand (SPIDER files), Sharpa Wave, Tetheria Aero (Menagerie), unseen by KITE's encoder
and by every decoder, plus Wuji and Shadow for reference. Never used for training, selection or tuning.
Per hand: no movement; every seed of each arm; each arm's 3-seed ensemble; the mixed D1-bv + CF2 ensemble; paired bootstrap vs CF2
(same seed) and between ensembles. Start poses: the benchmark recipe (fixed_start, val_start_seed). Per seed, never pooled.

    PYTHONPATH=src python scripts/decoder/benchmark/external_hands_eval.py --out '${GETKITE_RUNS}/external_hands_v1' [--arms d1long cf2 d1bv cf3d]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import os
from pathlib import Path

import numpy as np
import torch

from getkite.decoder import paths
from getkite.decoder.multifamily import fixed_start
from getkite.decoder.train_scratch import FrozenPsi, build_bank, load_config, load_pool, start_noise
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
EXT_ROOT = Path(os.environ.get("GETKITE_EXTERNAL", _REPO.parent / "external_assets"))  # third-party hand assets (docs/external_hands.md)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "diagnostics"))
from crossbody_heldout_eval import ANG_DEG, HD, POS_MM, WUJI, sg_index, sg_to  # noqa: E402
from transfer_benchmark import SHADOW, SHADOW_AD, load_system, score, wrist_fit  # noqa: E402
from wuji_per_example import paired_boot  # noqa: E402
from seed_ensemble_eval import chordal_mean  # noqa: E402

REPO = Path(paths.REPO)
EXT = EXT_ROOT
HANDS = ["ability", "inspire", "xhand", "schunk", "metahand", "sharpa", "aero"]
RUNS = {"d1long": "d1_long_hf", "cf2": "cf2_long_hf", "d1bv": "d1_bv_long_hf", "cf3d": "cf3d_long_hf", "cf3e": "cf3e_long_hf",
        **{f"cf3f{a}": f"cf3f_{a}_hf" for a in ("leap", "long", "teach", "big", "rec")},  # CF-3f arms (docs/cf3f_proposal.md)
        **{f"cf3flong{k}k": f"cf3f_long_snap{k}k_hf" for k in (175, 225, 275)},
        "cf3fbigext": "cf3f_bigext_hf", "cf3fbigextlast": "cf3f_bigextlast_hf",
        "cf3fmatch": "cf3f_match_hf",  # CF-3f matched teachers  # big continued to 200k: selected / final (copy of last.pt)  # long-arm mid-training snapshots (copies of last.pt)
        "cf3fbig300k": "cf3f_big300k_hf", "cf3fcontact": "cf3f_contact_hf"}  # extra CF-3f arms run on Aryan's compute (docs/cf3f_extra_arms.md)
METRICS = ("corr_mm", "compat_success", "wrist_mm", "wrist_deg")


def spec(h):
    ad = EXT / f"{h}_adapter"
    return {"kind": "kite", "robot": f"{h}_right", "family": h, "urdf": str(ad / f"{h}_right_extended.urdf"),
            "template": str(ad / f"{h}_surface_template_n4096.pt"), "ignore_mimic": False}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", default=["d1long", "cf2", "d1bv"])
    a = ap.parse_args()
    B = json.loads((REPO / "configs/decoder/benchmark/transfer_v1_leapval105.json").read_text())
    out = Path(paths.expand(a.out)); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    cfg = load_config(str(REPO / B["base_config"]), []); cfg["surface_sampling"] = B["surface_sampling"]
    torch.manual_seed(0)
    noise = start_noise(cfg)
    hands = [h for h in HANDS if (EXT / f"{h}_adapter" / f"{h}_eval.npz").exists()]
    bank = build_bank({**cfg, "robots": cfg["robots"] + [WUJI, SHADOW] + [spec(h) for h in hands]}, dev)
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    sysm = {(arm, s): load_system(f"{arm}_s{s}", {"kind": "j1", "ckpt": paths.expand(f"${{GETKITE_RUNS}}/{RUNS[arm]}{'' if s == 0 else f'_s{s}'}/best.pt")},
                                  dev, B["surface_sampling"])[0] for arm in a.arms for s in range(3)}
    scfg = {"success_pos_mm": POS_MM, "success_normal_deg": ANG_DEG}
    pools = {h: [{"kind": "kite", "robot": f"{h}_right", "npz": str(EXT / f"{h}_adapter" / f"{h}_eval.npz")}] for h in hands}
    pools["wuji"] = [{"kind": "kite", "robot": "wujihand_right", "npz": f"{HD}/wujihand_right_eval.npz"}]
    pools["shadow"] = [{"kind": "kite", "robot": "shadow_hand_right", "npz": str(SHADOW_AD / "shadow_eval.npz")}]
    mixes = [m for m in ("d1bv+cf2", "d1bv+cf3d", "d1bv+cf3e") if all(x in a.arms for x in m.split("+"))]
    R = {"arms": a.arms, "hands": list(pools), "pools": {}}
    for pn, sp in pools.items():
        rec = load_pool(sp, bank).to(dev)
        sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
        with torch.no_grad():
            z = torch.cat([psi(rec.c_pos[s:s + 256], rec.c_normal[s:s + 256], rec.c_valid[s:s + 256]) for s in range(0, len(rec), 256)])
            nm = {k: v.float().cpu().numpy() for k, v in score(bank, rec, rec.r, sg, sg.qc1, sg.R1, sg.t1, scfg).items()}
        outs, per = {}, {}
        for (arm, s), S in sysm.items():
            o = []
            with torch.no_grad():
                for b in range(0, len(rec), 256):
                    i = slice(b, b + 256)
                    o.append(S.predict(bank, rec.r[i], sg_index(sg, i), z[i]))
                o = {k: torch.cat([x[k] for x in o]) for k in ("qc", "R", "t")}
                outs[(arm, s)] = o
                per[(arm, s)] = {k: v.float().cpu().numpy() for k, v in score(bank, rec, rec.r, sg, o["qc"], o["R"], o["t"], scfg).items()}
        ens = {}
        for name, members in [(arm, [(arm, s) for s in range(3)]) for arm in a.arms] + [(m, [(x, s) for x in m.split("+") for s in range(3)]) for m in mixes]:
            os_ = [outs[k] for k in members]
            with torch.no_grad():
                e = {"qc": sum(o["qc"] for o in os_) / len(os_), "t": sum(o["t"] for o in os_) / len(os_), "R": chordal_mean([o["R"] for o in os_])}
                ens[name] = {k: v.float().cpu().numpy() for k, v in score(bank, rec, rec.r, sg, e["qc"], e["R"], e["t"], scfg).items()}
        F = {"n": len(rec), "no_movement": {m: round(float(nm[m].mean()), 4) for m in METRICS}}
        for arm in a.arms:
            F[arm] = {"seeds": {m: [round(float(per[(arm, s)][m].mean()), 4) for s in range(3)] for m in METRICS},
                      "ens": {m: round(float(ens[arm][m].mean()), 4) for m in METRICS}}
            if arm != "cf2" and "cf2" in a.arms:
                F[arm]["minus_cf2"] = {m: [paired_boot(per[(arm, s)][m], per[("cf2", s)][m]) for s in range(3)] for m in ("corr_mm", "wrist_mm")}
        for m_ in mixes:
            F[m_] = {"ens": {m: round(float(ens[m_][m].mean()), 4) for m in METRICS}}
        F["_ens_pairs"] = {f"{x}-{y}": {m: paired_boot(ens[x][m], ens[y][m]) for m in ("corr_mm", "wrist_mm")}
                           for x, y in itertools.combinations(list(a.arms) + mixes, 2)}
        R["pools"][pn] = F
        print(pn, "none", F["no_movement"]["corr_mm"], {arm: (F[arm]["seeds"]["corr_mm"], F[arm]["ens"]["corr_mm"]) for arm in a.arms},
              {m_: F[m_]["ens"]["corr_mm"] for m_ in mixes}, flush=True)
        (out / "external_hands.json").write_text(json.dumps(R, indent=1))
    print("wrote", out / "external_hands.json")


if __name__ == "__main__":
    main()
