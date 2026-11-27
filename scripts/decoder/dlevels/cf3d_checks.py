"""CF-3d implementation checks 2 and 3 (docs/cf3_distillation_proposal.md). Check 1 (alpha = 0 reproduces CF2) is run by
slurm/cf3d_checks.sbatch as two short training runs.

2. teacher_pose (the training-side teacher) on the benchmark's source val pool reproduces the scored mixed D1-bv+CF2 ensemble
   (seed_ensemble_mix_v1) to within 0.01 mm corr_mm per family.
3. Frame mapping end to end: with the single teacher CF2 s0, distill_targets (canonical training frame) equals the residual of the pose
   CF2 s0's inner model itself outputs on the canonical bank for the same start (max abs diff small).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from getkite.decoder import paths
from getkite.decoder.multifamily import fixed_start
from getkite.decoder.robots import canonical_bank
from getkite.decoder.train_joint import distill_targets, load_teacher, teacher_pose
from getkite.decoder.train_scratch import FrozenPsi, build_bank, load_config, load_pool, start_noise

REPO = Path(paths.REPO)
HERE = REPO / "scripts/decoder/benchmark"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "diagnostics"))
from crossbody_heldout_eval import ANG_DEG, POS_MM, WUJI, sg_index, sg_to  # noqa: E402
from transfer_benchmark import SHADOW, score  # noqa: E402

B = json.loads((REPO / "configs/decoder/benchmark/transfer_v1_leapval105.json").read_text())
dev = torch.device("cuda")
cfg = load_config(str(REPO / B["base_config"]), []); cfg["surface_sampling"] = B["surface_sampling"]
torch.manual_seed(0)
noise = start_noise(cfg)
bank = build_bank({**cfg, "robots": cfg["robots"] + [WUJI, SHADOW]}, dev)
psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
T = [load_teacher(paths.expand(f"${{GETKITE_RUNS}}/{a}{s}/best.pt"), dev) for a in ("d1_bv_long_hf", "cf2_long_hf") for s in ("", "_s1", "_s2")]
rec = load_pool(cfg["val"], bank).to(dev)
sg = sg_to(fixed_start(bank, rec, cfg["val_start_seed"], **noise), dev)
with torch.no_grad():
    z = torch.cat([psi(rec.c_pos[s:s + 256], rec.c_normal[s:s + 256], rec.c_valid[s:s + 256]) for s in range(0, len(rec), 256)])
    o = [teacher_pose(T, bank, rec.r[i], sg_index(sg, i), z[i]) for i in (slice(b, b + 256) for b in range(0, len(rec), 256))]
    qc, R, t = (torch.cat([x[k] for x in o]) for k in range(3))
    sc = score(bank, rec, rec.r, sg, qc, R, t, {"success_pos_mm": POS_MM, "success_normal_deg": ANG_DEG})
fam = np.array([str(bank.family[i]) for i in rec.r.cpu().numpy().astype(int)])
ref = json.load(open(paths.expand("${GETKITE_RUNS}/seed_ensemble_mix_v1/seed_ensemble.json")))["pools"]["val"]
c2 = {}
for f in sorted(set(fam)):
    got = float(sc["corr_mm"].float().cpu().numpy()[fam == f].mean()); want = ref[f]["d1bv+cf2"]["ens"]["corr_mm"]
    c2[f] = {"got": round(got, 4), "want": want, "ok": abs(got - want) <= 0.01}
print("check2", json.dumps(c2), flush=True)

# check 3: single teacher = CF2 s0; its inner model on the canonical bank must output exactly the distillation residual target
cf = T[3]; inner = cf.inner
bc, Q, _ = canonical_bank(bank)
i = slice(0, 256); r = rec.r[i]; g = sg_index(sg, i); Qb = Q[r]
gc = type(g)(**{**{k: getattr(g, k) for k in g.__dataclass_fields__}, "R1": g.R1 @ Qb, "Rg": g.Rg @ Qb})
with torch.no_grad():
    tr_root, tr_j = distill_targets([cf], bank, bc, r, r, g, gc, Qb, z[i], inner.decoder.c)
    out = inner(bc, r, gc.qc1, gc.R1, gc.t1, z[i])
    from getkite.decoder.get_scratch import target_residuals
    s_root, s_j = target_residuals(bc, r, gc.qc1, gc.R1, gc.t1, out["qc"], out["R"], out["t"], inner.decoder.c)  # its own pose, re-encoded
m = bank.coord_valid[r]
d_root = float((s_root - tr_root).abs().max()); d_j = float(((s_j - tr_j).abs() * m).max())
c3 = {"max_root": d_root, "max_joint": d_j, "ok": d_root < 1e-3 and d_j < 1e-3}
print("check3", json.dumps(c3), flush=True)
json.dump({"check2": c2, "check3": c3, "pass": all(v["ok"] for v in c2.values()) and c3["ok"]},
          open(paths.expand("${GETKITE_RUNS}/cf3d_checks.json"), "w"), indent=1)
print("PASS" if all(v["ok"] for v in c2.values()) and c3["ok"] else "FAIL")
