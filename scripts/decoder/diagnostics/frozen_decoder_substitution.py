"""Frozen raw-contact GET decoder fed (a) the true contact request, (b) contacts reconstructed from frozen psi(z) by the
frozen world-frame contact-set bridge, (c) no movement -- on identical records and identical start poses (2026-11-17).

    python scripts/decoder/diagnostics/frozen_decoder_substitution.py --out $GETKITE_RUNS/frozen_decoder_substitution

Both networks frozen (eval mode, no_grad), existing selected checkpoints (best.pt), nothing tuned here.
Predicted path: z = psi(request) -> bridge(z) -> world-frame slots (position, normal, existence); validity = predicted
existence > 0.5 (>= 1 slot), predicted count = its sum; packed exactly like RawContacts packs the true request
(pos*valid, normal*valid, valid) in the WORLD frame; the decoder's own frame='wrist' preprocessing (R0^T(c - t0),
R0^T n, contact Fourier) is then applied once, inside the decoder, as for the true path.
Scoring: pose_metrics against the ORIGINAL target contacts (positions, normals, template correspondences) and the goal
wrist pose, for every path; success = all contacts within 10 mm / 15 deg (training config).
Sanity: the true path with a random slot permutation (decoder must not depend on slot order, since bridge slots are
unordered).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from getkite.decoder import paths
from getkite.decoder.get_scratch import GETScratch, ScratchConfig
from getkite.decoder.multifamily import fixed_start
from getkite.decoder.spatial_bridge import ContactSetBridge, SetBridgeConfig
from getkite.decoder.train_scratch import FrozenPsi, RawContacts, build_bank, load_config, load_pool, pose_metrics, start_noise

RUNS = paths.get("GETKITE_RUNS")
REPO = str(paths.REPO)
METRICS = ("contact_mm", "normal_deg", "success", "wrist_mm", "wrist_deg")


def bridge_contacts(bridge, z):
    """Frozen bridge: world-frame slots sorted by existence; validity = existence > 0.5 (at least one slot)."""
    pw, nw, lg = bridge(z, torch.eye(3, device=z.device).expand(len(z), 3, 3), torch.zeros(len(z), 3, device=z.device))
    order = lg.argsort(1, descending=True)
    pw = pw.gather(1, order[..., None].expand_as(pw))
    nw = nw.gather(1, order[..., None].expand_as(nw))
    v = lg.gather(1, order).sigmoid() > 0.5
    v[:, 0] = True
    return pw, nw, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=f"{REPO}/configs/decoder/scratch/c3_psi50k_10x_wrist.json")
    ap.add_argument("--decoder", default=f"{RUNS}/scratch_c3_raw50k_cf8_10x_wrist/best.pt")
    ap.add_argument("--bridge", default=f"{RUNS}/set_bridge_zworld_c3_20k_b/best.pt")
    ap.add_argument("--out", default=f"{RUNS}/frozen_decoder_substitution")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(a.config, [])
    torch.manual_seed(0)
    bank = build_bank(cfg, dev)
    fam = np.array([str(x) for x in bank.family], dtype=object)
    families = sorted(set(fam))
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    raw = RawContacts()

    ck = torch.load(a.decoder, map_location=dev, weights_only=False)
    dec = GETScratch(ScratchConfig.from_dict(ck["model_config"])).to(dev)
    dec.load_state_dict(ck["model"])
    dec.amp = False
    dec.eval()
    assert dec.c.intent_source == "raw_contacts" and dec.c.frame == "wrist", (dec.c.intent_source, dec.c.frame)
    bs = torch.load(a.bridge, map_location=dev)
    bridge = ContactSetBridge(SetBridgeConfig(**bs["config"])).to(dev)
    bridge.load_state_dict(bs["model"])
    bridge.eval()
    for p in list(dec.parameters()) + list(bridge.parameters()):
        p.requires_grad_(False)
    meta = {"decoder": a.decoder, "decoder_step": ck.get("step"), "bridge": a.bridge, "bridge_step": bs.get("step"),
            "success": {"pos_mm": cfg.get("success_pos_mm", 10.0), "normal_deg": cfg.get("success_normal_deg", 15.0)}}
    scfg = {"success_pos_mm": meta["success"]["pos_mm"], "success_normal_deg": meta["success"]["normal_deg"]}
    print(meta, flush=True)

    pools = {"val": cfg["val"], "leap_heldout": cfg["extra_eval"]["leap_heldout"]}
    results, gperm = {"_meta": meta}, torch.Generator().manual_seed(3)
    for pname, spec in pools.items():
        rec = load_pool(spec, bank).to(dev)
        sg = fixed_start(bank, rec, cfg["val_start_seed"], **start_noise(cfg))  # identical starts for every path
        sg = type(sg)(**{k: getattr(sg, k).to(dev) for k in sg.__dataclass_fields__})
        acc = {}
        with torch.no_grad():
            for s in range(0, len(rec), 512):
                i = torch.arange(s, min(s + 512, len(rec)), device=dev)
                rb = rec.index(i)
                g = type(sg)(**{k: getattr(sg, k)[i] for k in sg.__dataclass_fields__})
                r = rb.r
                pw, nw, pv = bridge_contacts(bridge, psi(rb.c_pos, rb.c_normal, rb.c_valid))
                perm = torch.stack([torch.randperm(rb.c_pos.shape[1], generator=gperm) for _ in range(len(r))]).to(dev)
                take = lambda x: x.gather(1, perm[..., None].expand_as(x)) if x.dim() == 3 else x.gather(1, perm)  # noqa: E731
                zs = {"true": raw(rb.c_pos, rb.c_normal, rb.c_valid),
                      "true_slot_shuffled": raw(take(rb.c_pos), take(rb.c_normal), take(rb.c_valid)),
                      "predicted": raw(pw, nw, pv)}
                rows = {}
                for k, z in zs.items():
                    o = dec(bank, r, g.qc1, g.R1, g.t1, z)
                    rows[k] = pose_metrics(bank, rb, r, o["qc"], o["t"], o["R"], g, scfg)  # vs ORIGINAL targets
                rows["no_movement"] = pose_metrics(bank, rb, r, g.qc1, g.t1, g.R1, g, scfg)
                rows["_count"] = {"pred_count": pv.sum(1).float(), "true_count": rb.c_valid.sum(1).float()}
                for who, mets in rows.items():
                    for m, v in mets.items():
                        acc.setdefault(who, {}).setdefault(m, []).append(v.float())
                acc.setdefault("_r", []).append(r)
        r_all = torch.cat(acc.pop("_r")).cpu().numpy()
        cat = {w: {m: torch.cat(v) for m, v in d.items()} for w, d in acc.items()}
        res = {}
        for f in families:
            sel = torch.as_tensor(fam[r_all] == f, device=dev)
            if not sel.any():
                continue
            row = {w: {m: round(float(cat[w][m][sel].mean()), 4) for m in (METRICS if not w.startswith("_") else cat[w])}
                   for w in cat}
            row["_count"]["count_acc"] = round(float((cat["_count"]["pred_count"][sel] == cat["_count"]["true_count"][sel]).float().mean()), 4)
            row["drop_true_to_predicted"] = {m: round(row["predicted"][m] - row["true"][m], 4) for m in METRICS}
            row["max_abs_true_vs_shuffled"] = {m: round(float((cat["true"][m][sel] - cat["true_slot_shuffled"][m][sel]).abs().max()), 6)
                                               for m in METRICS}
            row["n"] = int(sel.sum())
            res[f] = row
        results[pname] = res
    (out / "results.json").write_text(json.dumps(results, indent=1))
    for pname in pools:
        print(f"\n== {pname}")
        for f, row in results[pname].items():
            print(f"  {f} (n={row['n']}, pred count acc {row['_count']['count_acc']})")
            for w in ("true", "predicted", "no_movement", "drop_true_to_predicted"):
                print(f"    {w:24s} " + "  ".join(f"{m}={row[w][m]}" for m in METRICS))
            print(f"    {'|true - shuffled| max':24s} " + "  ".join(f"{m}={row['max_abs_true_vs_shuffled'][m]}" for m in METRICS))


if __name__ == "__main__":
    main()
