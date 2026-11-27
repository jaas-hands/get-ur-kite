"""CF-3 step 0: how much of CF2's held-out-LEAP gap would a perfect per-hand frame remove? (evaluation only, existing checkpoints)

For the 105 held-out LEAP hands, CF2's rule-E canonical frame Q_r is replaced by the reference frame = chordal mean of the
training LEAP hands' rule-E frames (the same reference as leap_frame_angles.json). LEAP variants share the base-link convention, so
this is the frame CF2 saw for typical training LEAP hands. Conditions:
  ruleE        : CF2 as deployed (must reproduce transfer_benchmark_v1_leapval105);
  oracle_tilt  : reference frame only for hands whose rule-E frame is > 6 deg from it, rule E otherwise;
  oracle_all   : reference frame for every held-out LEAP hand.
Native systems (D1-long, D1-bv) are scored once on the same records. Per seed, paired bootstrap over records, never pooled.
The oracle uses information not available at deployment (which hands are LEAP variants); it bounds what a learned frame
correction (CF-3 candidate 2) could gain, nothing more.

    PYTHONPATH=src python scripts/decoder/benchmark/oracle_frame_eval.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from getkite.decoder import paths
from getkite.decoder.multifamily import fixed_start
from getkite.decoder.robots import RobotBank, canonical_bank, relabel_base
from getkite.decoder.train_scratch import FrozenPsi, build_bank, load_config, load_pool, start_noise

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "diagnostics"))
from crossbody_heldout_eval import ANG_DEG, POS_MM, WUJI, sg_index, sg_to  # noqa: E402
from transfer_benchmark import load_system, score  # noqa: E402
from wuji_per_example import paired_boot  # noqa: E402

REPO = Path(paths.REPO)
BENCH = REPO / "configs/decoder/benchmark/transfer_v1_leapval105.json"
SEEDS = (0, 1, 2)
TILT = 6.0
BINS = [(0, 3), (3, 6), (6, 10), (10, 15), (15, 180)]
METRICS = ("corr_mm", "compat_success", "wrist_mm", "wrist_deg")


def main():
    R = os.environ["GETKITE_RUNS"]
    out = Path(R) / "cf2_oracle_frame"; out.mkdir(parents=True, exist_ok=True)
    B = json.loads(BENCH.read_text())
    cfg = load_config(str(REPO / B["base_config"]), []); cfg["surface_sampling"] = B["surface_sampling"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.manual_seed(0)
    noise = start_noise(cfg)
    bank = build_bank({**cfg, "robots": cfg["robots"] + [WUJI]}, dev)  # as the benchmark (shadow_pool off)
    ang = json.load(open(f"{R}/transfer_benchmark_v1_leapval105/leap_frame_angles.json"))
    fr = json.load(open(f"{R}/cf_bv_long_hf/canonical_frames.json"))
    A = json.load(open(REPO / "configs/decoder/scratch/cf2_long_hf.json"))
    a_tr = [os.path.basename(t["npz"])[:-4] for t in A["train"] if t["kind"] == "leap" and "/prim/" not in t["npz"]]
    U, _, Vt = np.linalg.svd(sum(np.array(fr[h]["Q"]) for h in a_tr)); Mean = U @ Vt
    spec = cfg["extra_eval"]["leap_heldout"]
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    names_cf = [f"cf2_s{s}" for s in SEEDS]; names_nat = [f"{a}_s{s}" for a in ("d1long", "d1bv") for s in SEEDS]
    systems = {n: load_system(n, {k: (paths.expand(v) if isinstance(v, str) else v) for k, v in B["systems"][n].items()}, dev,
                              B["surface_sampling"])[0] for n in names_cf + names_nat}
    scfg = {"success_pos_mm": POS_MM, "success_normal_deg": ANG_DEG}
    rec = load_pool(spec, bank).to(dev)
    sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
    with torch.no_grad():
        z = torch.cat([psi(rec.c_pos[s:s + 256], rec.c_normal[s:s + 256], rec.c_valid[s:s + 256]) for s in range(0, len(rec), 256)])
    names = np.array([bank.names[i] for i in rec.r.cpu().numpy().astype(int)])
    hands = sorted(set(names))
    assert set(hands) <= set(ang), "held-out hands missing from leap_frame_angles.json"
    _, Q, _ = canonical_bank(bank); Qn = Q.double().cpu().numpy()
    # check: recomputed angles match the cache
    for i, n in enumerate(bank.names):
        if n in ang:
            a = float(np.degrees(np.arccos(np.clip((np.trace(Mean.T @ Qn[i]) - 1) / 2, -1, 1))))
            assert abs(a - ang[n]) < 0.05, (n, a, ang[n])
    conds = {"ruleE": lambda n: False, "oracle_tilt": lambda n: n in ang and ang[n] > TILT, "oracle_all": lambda n: n in ang}

    def run(S):
        outs = []
        with torch.no_grad():
            for s in range(0, len(rec), 256):
                i = slice(s, s + 256)
                outs.append(S.predict(bank, rec.r[i], sg_index(sg, i), z[i]))
            o = {k: torch.cat([x[k] for x in outs]) for k in ("qc", "R", "t")}
            per = score(bank, rec, rec.r, sg, o["qc"], o["R"], o["t"], scfg)
        return {k: v.float().cpu().numpy() for k, v in per.items()}

    per = {}
    for n in names_nat:
        per[(n, "native")] = run(systems[n])
    for cn, use in conds.items():
        Qc = np.stack([Mean if use(n) else Qn[i] for i, n in enumerate(bank.names)])
        bc = RobotBank([relabel_base(d, q) for d, q in zip(bank.descs, Qc)], device=dev, surface_sampling=bank.surface_sampling)
        Qt = torch.as_tensor(Qc, dtype=torch.float32, device=dev)
        for n in names_cf:
            systems[n].model._cache = {(id(bank), tuple(bank.names), str(bank.lower.device)): (bc, Qt)}
            per[(n, cn)] = run(systems[n])
        print(cn, {n: round(float(per[(n, cn)]["corr_mm"].mean()), 4) for n in names_cf}, flush=True)
    a_rec = np.array([ang[h] for h in names])
    res = {"tilt_deg": TILT, "n_hands": len(hands), "n_tilted": int(sum(ang[h] > TILT for h in hands)), "seeds": {}}
    for s in SEEDS:
        c = f"cf2_s{s}"; S = {}
        S["means"] = {k: {m: round(float(per[(c if k in conds else f"{k}_s{s}", k if k in conds else "native")][m].mean()), 4) for m in METRICS}
                      for k in list(conds) + ["d1long", "d1bv"]}
        S["paired_corr"] = {}
        for k in conds:
            for ref in ("d1long", "d1bv"):
                S["paired_corr"][f"{k}-{ref}"] = paired_boot(per[(c, k)]["corr_mm"], per[(f"{ref}_s{s}", "native")]["corr_mm"])
            if k != "ruleE":
                for m in ("corr_mm", "wrist_mm", "wrist_deg"):
                    S["paired_corr"][f"{k}-ruleE/{m}"] = paired_boot(per[(c, k)][m], per[(c, "ruleE")][m])
        S["bins_vs_d1long"] = {f"[{lo},{hi})": {k: round(float((per[(c, k)]["corr_mm"] - per[(f"d1long_s{s}", "native")]["corr_mm"])[(a_rec >= lo) & (a_rec < hi)].mean()), 4)
                                                for k in conds} | {"n_hands": int(sum(lo <= ang[h] < hi for h in hands))} for lo, hi in BINS}
        res["seeds"][s] = S
        print(s, json.dumps({k: (round(v["mean_diff"], 3), [round(x, 3) for x in v["ci95"]]) for k, v in S["paired_corr"].items()}), flush=True)
    (out / "oracle_frame.json").write_text(json.dumps(res, indent=1))
    print("wrote", out / "oracle_frame.json")


if __name__ == "__main__":
    main()
