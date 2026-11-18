"""Which links/fingers receive the 32 surface tokens? Current selection (fallback FPS on link-LOCAL template points for
URDF robots; dataset-provided for LEAP) vs FPS in a common hand frame at the zero pose. Diagnostics only (2026-11-17)."""
import importlib.util, json, re
from collections import Counter
from pathlib import Path
import numpy as np, torch
import sys
if "--plot" in sys.argv:
    OUT = Path(__file__).resolve().parents[4] / "runs/surface_coverage"
spec = importlib.util.spec_from_file_location("x", Path(__file__).with_name("crossbody_heldout_eval.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from getkite.decoder.robots import _fps, N_SURF
OUT = Path(m.paths.expand("${GETKITE_RUNS}/surface_coverage")); OUT.mkdir(parents=True, exist_ok=True)
cfg = m.load_config(m.REPO + "/configs/decoder/scratch/c3_psi50k_10x_wrist.json", [])
bank = m.build_bank({**cfg, "robots": cfg["robots"] + [m.WUJI]}, "cpu")

def finger_of(link):
    l = link.lower()
    for pat, f in [(r"finger(\d)", "finger{}"), (r"thumb|_th_|link_1[2-5]|fingertip_3|th", "thumb"), (r"link_([0-3])\.0", "allegro_f0"),
                   (r"link_([4-7])\.0", "allegro_f1"), (r"link_([89]|1[01])\.0", "allegro_f2"), (r"finger_(\d)", "finger_{}"), (r"f(\d)", "f{}")]:
        mm = re.search(pat, l)
        if mm:
            return f.format(*mm.groups()) if "{}" in f else f
    return "palm/base:" + link
report = {}
picks = {"allegro_right": "allegro_right", "barrett": "barrett", "wujihand_right": "wujihand_right"}
leap_name = next(n for n in bank.names if n not in picks)
for name in list(picks) + [leap_name]:
    r = bank.index[name]; d = bank.descs[r]
    rt = torch.tensor([r])
    qp = torch.zeros(1, bank.parent_slot.shape[1])
    Rs, ts = bank.fk(rt, qp, torch.eye(3)[None], torch.zeros(1, 3))
    P = bank.points(rt, Rs, ts, torch.arange(int(bank.n_points[r]))[None])[0][0].numpy()  # common hand frame, zero pose
    links = np.array([d.tpl_link_names[i] for i in d.tpl_link])
    cur = bank.surf_idx[r].numpy()
    com = _fps(P, N_SURF)
    def stats(idx):
        D = np.sqrt(((P[:, None] - P[idx][None]) ** 2).sum(-1)).min(1) * 1000
        return {"per_finger": dict(Counter(finger_of(l) for l in links[idx])), "n_links_hit": int(len(set(links[idx]))),
                "n_links_total": int(len(set(links))), "covering_radius_mm": round(float(D.max()), 1), "mean_gap_mm": round(float(D.mean()), 2),
                "n_distinct_positions": int(len(np.unique(np.round(P[idx], 5), axis=0)))}
    report[name] = {"source": "dataset fps" if d.surf_idx is not None else "fallback FPS on link-local template",
                    "current": stats(cur), "common_frame_fps": stats(com)}
    np.savez(OUT / f"{name}.npz", P=P, cur=cur, com=com)
    print(name, json.dumps(report[name]), flush=True)
(OUT / "coverage.json").write_text(json.dumps(report, indent=1))


def plot():  # run with any python that has matplotlib:  python3 surface_coverage.py --plot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for f in sorted(OUT.glob("*.npz")):
        z = np.load(f); P = z["P"]
        fig = plt.figure(figsize=(12, 6))
        for k, (idx, title) in enumerate([(z["cur"], "current selection"), (z["com"], "FPS in common frame")]):
            for v, (a, b) in enumerate([(0, 1), (0, 2)]):
                ax = fig.add_subplot(2, 2, 2 * v + k + 1)
                ax.scatter(P[:, a], P[:, b], s=1, c="lightgrey"); ax.scatter(P[idx, a], P[idx, b], s=18, c="crimson")
                ax.set_title(f"{f.stem}: {title} ({'x-y' if v == 0 else 'x-z'} view, zero pose)", fontsize=8)
                ax.set_aspect("equal"); ax.tick_params(labelsize=6)
        fig.tight_layout(); fig.savefig(OUT / f"{f.stem}.png", dpi=110); plt.close(fig)
