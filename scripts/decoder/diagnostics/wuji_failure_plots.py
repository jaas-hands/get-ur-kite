"""Representative Wuji failures from wuji_per_example.pt: hand surface at start (grey), corrected-decoder prediction
(red), old-decoder prediction (blue), goal (green); requested contacts (black x). Two orthogonal views per example."""
import sys
from pathlib import Path
import numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from crossbody_heldout_eval import WUJI, clamp_q
from getkite.decoder.train_scratch import build_bank, load_config
from getkite.decoder import paths
d = torch.load(sys.argv[1], weights_only=False); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
cfg = load_config(f"{paths.REPO}/configs/decoder/scratch/c3_psi50k_10x_wrist.json", [])
b = build_bank({**cfg, "robots": cfg["robots"] + [WUJI]}, "cpu"); rw = b.index["wujihand_right"]
sg, rec = d["_sg"], d["_rec"]
nm, old, new = d["no_movement"]["corr_mm"], d["old/true/full"]["corr_mm"], d["new/true/full"]["corr_mm"]
gw = d["goal_wrist"]["corr_mm"]
pick = {"new worse than no movement": int(np.argmax(new - nm)), "both decoders worse than no movement": int(np.argmax(np.minimum(new, old) - nm)),
        "typical (median no-movement error)": int(np.argsort(nm)[len(nm) // 2]), "old decoder helps most": int(np.argmax(nm - old))}
def pts(q, R, t):
    r = torch.tensor([rw]); q = torch.as_tensor(q)[None].float(); q, _ = clamp_q(b, r, q)
    Rs, ts = b.fk(r, b.expand(r, q), torch.as_tensor(R)[None].float(), torch.as_tensor(t)[None].float())
    P, _ = b.points(r, Rs, ts, torch.arange(int(b.n_points[rw]))[None]); return P[0].numpy()[::4] * 100
fig, ax = plt.subplots(len(pick), 2, figsize=(10, 4.4 * len(pick)))
for row, (title, i) in enumerate(pick.items()):
    poses = {"start": (sg["qc1"][i], sg["R1"][i], sg["t1"][i], "0.75", 1), "goal": (sg["qcg"][i], sg["Rg"][i], sg["tg"][i], "tab:green", 1),
             "old": (d["old/true/full"]["q"][i], d["old/true/full"]["R"][i], d["old/true/full"]["t"][i], "tab:blue", 1),
             "corrected": (d["new/true/full"]["q"][i], d["new/true/full"]["R"][i], d["new/true/full"]["t"][i], "tab:red", 1)}
    P = {k: pts(*v[:3]) for k, v in poses.items()}
    v = rec["c_valid"][i].astype(bool); C = rec["c_pos"][i][v] * 100
    # view axes: principal axes of the goal hand surface
    c0 = P["goal"].mean(0); U = np.linalg.svd(P["goal"] - c0)[2]
    for col, (a1, a2) in enumerate([(0, 1), (0, 2)]):
        A = ax[row, col]
        for k, (_, _, _, colr, _) in poses.items():
            X = (P[k] - c0) @ U.T; A.scatter(X[:, a1], X[:, a2], s=1.5, c=colr, alpha=.35, label=k)
        Xc = (C - c0) @ U.T; A.scatter(Xc[:, a1], Xc[:, a2], marker="x", c="k", s=40, label="requested contacts")
        A.set_aspect("equal"); A.set_xlabel(f"PC{a1+1} (cm)"); A.set_ylabel(f"PC{a2+1} (cm)")
    ax[row, 0].set_title(f"#{i} {title}\ncorr mm: no-move {nm[i]:.1f} | old {old[i]:.1f} | corrected {new[i]:.1f} | goal wrist only {gw[i]:.1f}", fontsize=9)
    ax[row, 1].set_title(f"rot err to goal: start {np.degrees(np.arccos(np.clip((np.trace(sg['R1'][i].T@sg['Rg'][i])-1)/2,-1,1))):.1f}°, "
                         f"old {d['old/true/full']['R_err_deg'][i]:.1f}°, corrected {d['new/true/full']['R_err_deg'][i]:.1f}°", fontsize=9)
ax[0, 0].legend(markerscale=6, fontsize=7, loc="upper right")
fig.tight_layout(); fig.savefig(out / "wuji_failures.png", dpi=110); print(pick)
