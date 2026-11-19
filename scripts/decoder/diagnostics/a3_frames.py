"""Preflight audit, part 3: numerical round trips for the frames and conventions used by the two runs."""
import torch
from getkite.decoder.config import Config
from getkite.decoder.data import SplitData
from getkite.decoder.model import build_model
from getkite.decoder.losses import kite_residual, XYZ_STD, ROT_STD, JOINT_STD_FRAC
from getkite.decoder.rotations import sixd_to_matrix_symmetric, geodesic_angle, quat_to_matrix
from kite_data.dataset import load_split
from pathlib import Path
K = paths.get("KITE_ROOT"); D = paths.get("GETKITE_DATASET")
SET = [f"kite_root={K}", f"kite_ckpt={K}/checkpoints/decoder/release.pt", f"data_dir={D}", "amp=none"]
torch.set_grad_enabled(False)
cfg = Config.load("configs/kite_trunk_get_ft.json", SET)
N = 4096
dw = SplitData(cfg, "val", "cpu", N)
raw = load_split("val", regime="kite_code", root=Path(D) / "examples" / "kite_code")
sel = torch.as_tensor(dw.sel if hasattr(dw, "sel") else [])
b = dw.batch(torch.arange(dw.n))
print(f"[frame] runs use frame={cfg.frame!r}: every model input and output is world-frame (KITE's floating-base frame)")
# 1. psi input == stored world contacts (eval path; training adds 0.1 rad normal noise to psi's normal input only)
m = build_model(cfg).eval(); cap = {}
m.psi.register_forward_hook(lambda mod, inp, out: cap.update(pos=inp[0], nrm=inp[1], valid=inp[2]))
m(b, dw.bank)
import numpy as np
from getkite.decoder import paths
idx = np.flatnonzero(np.isin(np.arange(len(raw["hand"])), getattr(dw, "sel_idx", np.arange(0))))
# compare against stored arrays matched by (hand, contact) content
print(f"[psi in] pos == batch c_pos: {torch.equal(cap['pos'], b['c_pos'])}, normals == batch c_normal: {torch.equal(cap['nrm'], b['c_normal'])}")
sp = torch.as_tensor(raw["contact_pos"]).float(); key = lambda x: x.reshape(len(x), -1)
st = {tuple(np.round(r, 7)) for r in key(sp).numpy()}
print(f"[psi in] psi's contacts are stored world-frame contacts verbatim: {sum(tuple(np.round(r, 7)) in st for r in key(cap['pos']).numpy())}/{len(cap['pos'])}")
# 2. FK of the stored goal pose reproduces the stored contacts
Rs, ts = dw.bank.fk(b["h"], b["q_goal"], b["R_goal"], b["t_goal"])
x, n = dw.bank.points(b["h"], Rs, ts, b["tidx"]); cv = b["c_valid"]
print(f"[FK] goal pose -> contacts: max pos err {((x - b['c_pos']).norm(dim=-1)[cv]).max() * 1e3:.2e} mm, "
      f"max normal err {torch.rad2deg(torch.acos((n * b['c_normal']).sum(-1)[cv].clamp(-1, 1))).max():.2e} deg")
# 3. residual target <-> decoder update: encoding the goal with KITE's normaliser and decoding it with the
#    trunk's update rule (t0 + r*5mm, sym6D(r*5deg + I6) @ R0, q0 + r*0.05*range) returns the goal
r = kite_residual(b["t_goal"], b["R_goal"], b["q_goal"], b, dw.bank)
I6 = torch.tensor([1., 0, 0, 0, 1, 0])
t = b["t_init"] + r[:, :3] * XYZ_STD
R = sixd_to_matrix_symmetric(r[:, 3:9] * ROT_STD + I6) @ b["R_init"]
v = b["dof_valid"]; q = torch.where(v, b["q_init"] + r[:, 9:9 + v.shape[1]] * JOINT_STD_FRAC * dw.bank.range[b["h"]], b["q_init"])
print(f"[residual] round trip: t {(t - b['t_goal']).abs().max() * 1e3:.2e} mm, R {torch.rad2deg(geodesic_angle(R, b['R_goal'])).max():.2e} deg, "
      f"q {(q - b['q_goal'])[v].abs().max():.2e} rad | residual magnitude median |xyz| {r[:, :3].norm(dim=-1).median():.2f}, |rot| {r[:, 3:9].norm(dim=-1).median():.2f}")
# 4. wrist-relative transform (used by raw_wrist / psi-decoder paths, not by these two runs): wrist_init batch -> world
dl = SplitData(Config.load("configs/kite_trunk_get_ft.json", SET + ["frame=wrist_init"]), "val", "cpu", N)
bl = dl.batch(torch.arange(dl.n)); R0, t0 = bl["R_init_world"], bl["t_init_world"]
cw = bl["c_pos"] @ R0.transpose(1, 2) + t0[:, None]; nw = bl["c_normal"] @ R0.transpose(1, 2)
Rg, tg = R0 @ bl["R_goal"], (R0 @ bl["t_goal"][..., None]).squeeze(-1) + t0
print(f"[wrist_init] round trip to world: contacts {(cw - b['c_pos']).abs().max() * 1e3:.2e} mm, normals {(nw - b['c_normal']).abs().max():.2e}, "
      f"goal t {(tg - b['t_goal']).abs().max() * 1e3:.2e} mm, goal R {torch.rad2deg(geodesic_angle(Rg, b['R_goal'])).max():.2e} deg, "
      f"start wrist is identity: {torch.allclose(bl['R_init'], torch.eye(3).expand_as(bl['R_init'])) and bool((bl['t_init'] == 0).all())}")
