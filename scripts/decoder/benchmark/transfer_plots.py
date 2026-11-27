"""Fixed visual examples for the transfer benchmark: 10 per pool/family, chosen by rank of the reference system's
(--by, default j1) correspondence contact error at deciles 0, 1/9, ..., 1 (best -> worst; deterministic, covers
successes and failures). Start (grey), goal (green), each system's prediction (colours); requested contacts (x).
Two orthogonal views along the goal hand surface's principal axes.

    python scripts/decoder/benchmark/transfer_plots.py $GETKITE_RUNS/transfer_benchmark_v1
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diagnostics"))
from crossbody_heldout_eval import WUJI, clamp_q  # noqa: E402
from getkite.decoder import paths  # noqa: E402
from getkite.decoder.train_scratch import build_bank, load_config  # noqa: E402

COL = {"j1": "tab:red", "two_stage": "tab:blue", "direct_worldaux": "tab:purple", "g1_leapdiv": "tab:olive"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--by", default="j1")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    run = Path(a.run)
    man = json.loads((run / "manifest.json").read_text())
    cfg = man["resolved_base_config"]
    b = build_bank({**cfg, "robots": cfg["robots"] + [WUJI]}, "cpu")
    PE = torch.load(run / "per_example.pt", weights_only=False)
    vis = run / "visuals"; vis.mkdir(exist_ok=True)
    picks = {}

    def pts(r, q, R, t):
        rr = torch.tensor([int(r)]); q = torch.as_tensor(q)[None].float(); q, _ = clamp_q(b, rr, q)
        Rs, ts = b.fk(rr, b.expand(rr, q), torch.as_tensor(R)[None].float(), torch.as_tensor(t)[None].float())
        P, _ = b.points(rr, Rs, ts, torch.arange(int(b.n_points[int(r)]))[None])
        return P[0].numpy()[::4] * 100

    for pool in ("val", "leap_heldout", "wuji"):
        D = PE[pool]
        systems = list(D["out"])
        for f in sorted(set(D["family"])):
            idx = np.flatnonzero(D["family"] == f)
            e = D["per"][a.by]["corr_mm"][idx]
            order = idx[np.argsort(e, kind="stable")]
            sel = [int(order[int(round(k * (len(order) - 1) / (a.n - 1)))]) for k in range(a.n)]
            picks[f"{pool}/{f}"] = sel
            fig, ax = plt.subplots(a.n, 2, figsize=(10, 4.2 * a.n))
            for row, i in enumerate(sel):
                sg, rec, r = D["sg"], D["rec"], D["rec"]["r"][i]
                poses = {"start (= no movement)": (sg["qc1"][i], sg["R1"][i], sg["t1"][i], "0.7"),
                         "goal": (sg["qcg"][i], sg["Rg"][i], sg["tg"][i], "tab:green")}
                poses |= {s: (D["out"][s]["qc"][i], D["out"][s]["R"][i], D["out"][s]["t"][i], COL.get(s, "tab:orange")) for s in systems}
                P = {k: pts(r, *v[:3]) for k, v in poses.items()}
                v = rec["c_valid"][i].astype(bool); C = rec["c_pos"][i][v] * 100
                c0 = P["goal"].mean(0); U = np.linalg.svd(P["goal"] - c0)[2]
                for col, (a1, a2) in enumerate([(0, 1), (0, 2)]):
                    A = ax[row, col]
                    for k, (_, _, _, c) in poses.items():
                        X = (P[k] - c0) @ U.T; A.scatter(X[:, a1], X[:, a2], s=1.2, c=c, alpha=.35, label=k)
                    Xc = (C - c0) @ U.T; A.scatter(Xc[:, a1], Xc[:, a2], marker="x", c="k", s=40, label="requested contacts")
                    A.set_aspect("equal"); A.set_xlabel(f"PC{a1 + 1} (cm)"); A.set_ylabel(f"PC{a2 + 1} (cm)")
                per = D["per"]
                txt = " | ".join(f"{s} {per[s]['corr_mm'][i]:.1f} ({'ok' if per[s]['corr_success'][i] > .5 else 'fail'})"
                                 for s in ["no_movement"] + systems)
                ax[row, 0].set_title(f"{pool}/{f} #{i} {D['names'][i]} | corr mm: {txt}", fontsize=8)
                ax[row, 1].set_title(f"required: {per['no_movement']['req_mm'][i]:.1f} mm / {per['no_movement']['req_deg'][i]:.1f} deg; "
                                     + " | ".join(f"{s} moved {per[s]['move_mm'][i]:.1f} mm / {per[s]['move_deg'][i]:.1f} deg" for s in systems), fontsize=8)
            ax[0, 0].legend(markerscale=6, fontsize=7, loc="upper right")
            fig.tight_layout(); fig.savefig(vis / f"{pool}_{f}.png", dpi=90); plt.close(fig)
            print("wrote", vis / f"{pool}_{f}.png", flush=True)
    (vis / "picks.json").write_text(json.dumps({"rule": f"rank of {a.by} corr_mm at deciles (best->worst)", "picks": picks}, indent=1))


if __name__ == "__main__":
    main()
