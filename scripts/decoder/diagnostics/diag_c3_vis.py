"""Step 2 of the C3 generalisation diagnosis: visual inspection + raw-contact encoder checks.

    python scripts/decoder/diagnostics/diag_c3_vis.py --outdir $GETKITE_RUNS/c3_diag_vis

Figures (one per family; columns = 3 familiar training goals then 3 validation goals, evaluation start pose):
  grey  hand surface at q_init          red   target contacts (+ normal ticks)
  blue  contact template points at q_init (where those hand points start)
  green the same points at the raw model's prediction, black line = predicted motion of each contact point
  orange the same points at the psi model's prediction
Each panel is centred on the contacts; text gives contact error (mm) for noop / raw / psi.

Encoder checks (printed and saved to checks.json):
  - masks: invalid contact slots are zero in z, and changing them leaves the output unchanged
  - Fourier wavelengths of the contact features in metres
  - translation sensitivity: shift the whole scene (contacts, start pose and goal) by d mm in a random
    direction. A representation that encodes geometry relative to the hand would give the same residual;
    we report the mean change in predicted residual (normalised units) vs d, for training and validation goals.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from getkite.decoder.get_scratch import GETScratch, ScratchConfig  # noqa: E402
from getkite.decoder.multifamily import fixed_start  # noqa: E402
from getkite.decoder.train_scratch import FrozenPsi, RawContacts, build_bank, load_config, load_pool, pose_metrics  # noqa: E402

from getkite.decoder import paths  # noqa: E402
RUNS_DIR = paths.get("GETKITE_RUNS")
REPO = str(paths.REPO)
RUNS = {"psi": f"{RUNS_DIR}/scratch_c3_long50k", "raw": f"{RUNS_DIR}/scratch_c3_raw50k"}
import os  # noqa: E402
if os.environ.get("DIAG_RAW_RUN"):  # compare another raw-contact run against the same psi run
    RUNS["raw"] = os.environ["DIAG_RAW_RUN"]
SC = {"success_pos_mm": 10, "success_normal_deg": 15}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=f"{REPO}/configs/decoder/scratch/c3_raw50k.json")
    ap.add_argument("--outdir", default=f"{RUNS_DIR}/c3_diag_vis")
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(a.config, [])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bank = build_bank(cfg, dev)
    fam = np.array([str(x) for x in bank.family], dtype=object)
    train = load_pool(cfg["train"], bank).to(dev)
    val = load_pool(cfg["val"], bank).to(dev)
    enc = {"psi": FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev), "raw": RawContacts()}
    models = {}
    for k, run in RUNS.items():
        ck = torch.load(f"{run}/last.pt", map_location=dev, weights_only=False)
        m = GETScratch(ScratchConfig.from_dict(ck["model_config"])).to(dev)
        m.load_state_dict(ck["model"])
        m.amp = False
        models[k] = m.eval()

    def predict(k, rec, sg, shift=None):
        pos = rec.c_pos if shift is None else rec.c_pos + shift[:, None] * rec.c_valid[..., None]
        t1 = sg.t1 if shift is None else sg.t1 + shift
        with torch.no_grad():
            return models[k](bank, rec.r, sg.qc1, sg.R1, t1, enc[k](pos, rec.c_normal, rec.c_valid))

    checks = {}
    # ---- masks
    rec = val.index(torch.arange(0, len(val), 7, device=dev))
    z = RawContacts()(rec.c_pos, rec.c_normal, rec.c_valid).view(len(rec), -1, 7)
    checks["mask_invalid_slots_zero"] = bool((z[~rec.c_valid] == 0).all())
    sg = fixed_start(bank, rec, 999)
    sg = type(sg)(**{f: getattr(sg, f).to(dev) for f in sg.__dataclass_fields__})
    junk = rec.index(torch.arange(len(rec), device=dev))
    junk.c_pos = torch.where(rec.c_valid[..., None], rec.c_pos, torch.randn_like(rec.c_pos))
    junk.c_normal = torch.where(rec.c_valid[..., None], rec.c_normal, torch.randn_like(rec.c_normal))
    a0, a1 = predict("raw", rec, sg), predict("raw", junk, sg)
    checks["mask_invalid_slots_no_effect_max_abs"] = float((a0["res_root"] - a1["res_root"]).abs().max())
    c = models["raw"].c
    checks["contact_fourier_wavelengths_m"] = [round(2 / (c.fourier_base * 2 ** i) / c.pos_scale, 5) for i in range(models["raw"]._cf())]
    checks["contact_token_input"] = "[xyz * %.1f (world), normal, sin/cos(2^i pi xyz*%.1f), i<%d]; no hand-relative feature" % (
        c.pos_scale, c.pos_scale, c.pos_fourier)

    # ---- translation sensitivity
    g = torch.Generator().manual_seed(3)
    sens = {}
    for name, pool in (("train_goals", train), ("val_goals", val)):
        idx = torch.randperm(len(pool), generator=g)[:1500].to(dev)
        rec = pool.index(idx)
        sg = fixed_start(bank, rec, 4321)
        sg = type(sg)(**{f: getattr(sg, f).to(dev) for f in sg.__dataclass_fields__})
        dirs = torch.nn.functional.normalize(torch.randn(len(rec), 3, generator=g), dim=-1).to(dev)
        for k in models:
            base = predict(k, rec, sg)
            row = {}
            for d_mm in (1, 3, 10, 30):
                o = predict(k, rec, sg, dirs * d_mm / 1000)
                row[f"{d_mm}mm"] = {"root": round(float((o["res_root"] - base["res_root"]).norm(dim=-1).mean()), 3),
                                    "joint": round(float((o["res_coord"] - base["res_coord"]).norm(dim=-1).mean()), 3)}
            row["base_norm"] = {"root": round(float(base["res_root"].norm(dim=-1).mean()), 3),
                                "joint": round(float(base["res_coord"].norm(dim=-1).mean()), 3)}
            sens[f"{name}/{k}"] = row
    checks["translation_sensitivity"] = sens

    # ---- figures
    for f in sorted(set(fam)):
        fig = plt.figure(figsize=(18, 12))
        for col, (pool, label) in enumerate([(train, "familiar")] * 3 + [(val, "validation")] * 3):
            sel = (torch.as_tensor(fam[pool.r.cpu().numpy()] == f, device=dev)).nonzero().squeeze(1)
            i = sel[[3, 11, 29][col % 3] % len(sel)][None]
            rec = pool.index(i)
            sg = fixed_start(bank, rec, cfg["val_start_seed"] + int(i))
            sg = type(sg)(**{k: getattr(sg, k).to(dev) for k in sg.__dataclass_fields__})
            r = rec.r
            d = bank.descs[int(r)]
            Rs, ts = bank.fk(r, bank.expand(r, sg.qc1), sg.R1, sg.t1)
            P = len(d.tpl_pos)
            hp, _ = bank.points(r, Rs, ts, torch.arange(0, P, 4, device=dev)[None])
            v = rec.c_valid[0]
            cp, cn = rec.c_pos[0, v], rec.c_normal[0, v]
            sp, _ = bank.points(r, Rs, ts, rec.c_tpl)
            txt = [f"noop {float(pose_metrics(bank, rec, r, sg.qc1, sg.t1, sg.R1, sg, SC)['contact_mm']):.1f}"]
            preds = {}
            for k in models:
                o = predict(k, rec, sg)
                Rp, tp = bank.fk(r, bank.expand(r, o["qc"]), o["R"], o["t"])
                preds[k], _ = bank.points(r, Rp, tp, rec.c_tpl)
                txt.append(f"{k} {float(pose_metrics(bank, rec, r, o['qc'], o['t'], o['R'], sg, SC)['contact_mm']):.1f}")
            ax = fig.add_subplot(2, 3, col + 1, projection="3d")
            cen = cp.mean(0).cpu().numpy()
            H = hp[0].cpu().numpy() - cen
            near = np.linalg.norm(H, axis=1) < 0.07
            ax.scatter(*H[near].T, s=4, c="0.55", alpha=0.45, label="hand surface at q_init")
            C, N = cp.cpu().numpy() - cen, cn.cpu().numpy()
            ax.scatter(*C.T, s=40, c="red", label="target contact")
            for p_, n_ in zip(C, N):
                ax.plot(*np.stack([p_, p_ + 0.01 * n_]).T, c="red", lw=1)
            S = sp[0, v].cpu().numpy() - cen
            ax.scatter(*S.T, s=25, c="blue", marker="^", label="contact point at q_init")
            for k, colr in (("raw", "green"), ("psi", "orange")):
                Q = preds[k][0, v].cpu().numpy() - cen
                ax.scatter(*Q.T, s=25, c=colr, marker="s", label=f"{k} prediction")
                for s_, q_ in zip(S, Q):
                    ax.plot(*np.stack([s_, q_]).T, c=colr, lw=1)
            lim = 0.045
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
            ax.set_title(f"{f} {label} #{int(i)}\ncontact mm: " + ", ".join(txt), fontsize=12)
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        fig.savefig(out / f"{f}.png", dpi=110)
        plt.close(fig)
    json.dump(checks, open(out / "checks.json", "w"), indent=1)
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
