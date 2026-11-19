"""Independent check of the Wuji robot description against KITE's own sampler output (2026-11-17, diagnostics only).
For every generated record (KITE ContactPolicyProbeDataset, KITE's robot model) the contacts were produced by KITE's
FK at the goal q2; our RobotBank FK + template at the same q2 must reproduce position, normal and template index."""
import importlib.util, json
from pathlib import Path
import numpy as np, torch
spec = importlib.util.spec_from_file_location("x", Path(__file__).with_name("crossbody_heldout_eval.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from getkite.decoder.rotations import geodesic_angle
cfg = m.load_config(m.REPO + "/configs/decoder/scratch/c3_psi50k_10x_wrist.json", [])
bank = m.build_bank({**cfg, "robots": cfg["robots"] + [m.WUJI]}, "cpu")
r = bank.index["wujihand_right"]; d = bank.descs[r]
out = {"n_coords": len(d.coords), "n_template_points": int(bank.n_points[r]), "template_links": len(d.tpl_link_names)}
for split in ("eval", "bank"):
    npz = f"{m.HD}/wujihand_right_{split}.npz"
    meta = json.loads(str(np.load(npz)["meta"]))
    out[f"{split}_coord_order_matches_kite"] = meta["coords"] == list(d.coords)
    rec = m.load_pool([{"kind": "kite", "robot": "wujihand_right", "npz": npz}], bank)
    sg = m.fixed_start(bank, rec, 1, xyz_std=0.0, rot_std_deg=0.0, joint_frac=0.0)
    Rs, ts = bank.fk(rec.r, bank.expand(rec.r, sg.qcg), sg.Rg, sg.tg)
    p, n = bank.points(rec.r, Rs, ts, rec.c_tpl)
    v = rec.c_valid
    dmm = ((p - rec.c_pos).norm(dim=-1) * 1000)[v]
    ang = torch.rad2deg(torch.acos((n * rec.c_normal).sum(-1).clamp(-1, 1)))[v]
    cl, cu = bank.coords_of(rec.r, bank.lower[rec.r]), bank.coords_of(rec.r, bank.upper[rec.r]); cv = bank.coord_valid[rec.r]
    outside = (((sg.qcg < cl - 1e-6) | (sg.qcg > cu + 1e-6)) & cv)
    out[split] = {"n_records": len(rec.r), "n_contacts": int(v.sum()), "pos_err_mm_max": round(float(dmm.max()), 5),
                  "pos_err_mm_mean": round(float(dmm.mean()), 6), "normal_err_deg_max": round(float(ang.max()), 4),
                  "goal_joints_outside_limits_frac": round(float(outside.sum() / cv.sum()), 5),
                  "contacts_per_record": {int(k): int(c) for k, c in zip(*np.unique(v.sum(1).numpy(), return_counts=True))}}
lim = list(zip(d.coords, [round(float(x), 4) for x in bank.coords_of(torch.tensor([r]), bank.lower[[r]])[0][:len(d.coords)]],
               [round(float(x), 4) for x in bank.coords_of(torch.tensor([r]), bank.upper[[r]])[0][:len(d.coords)]]))
out["joint_limits"] = lim
print(json.dumps(out, indent=1))
Path(m.paths.expand("${GETKITE_RUNS}/surface_coverage/wuji_geometry_check.json")).write_text(json.dumps(out, indent=1))
