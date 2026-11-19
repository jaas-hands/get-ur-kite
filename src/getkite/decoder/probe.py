"""Does KITE's frozen z still describe LEAP contacts? (MODEL_PLAN D3)

Encode our contacts with KITE's psi, then train a small set decoder z -> up to 6 (x, n) points and
measure how well the contact set can be recovered. If a probe cannot get contacts back to well
under the 10 mm success tolerance, no decoder conditioned on z can either.

    python -m getkite.decoder.probe --set data_dir=... kite_ckpt=.../release.pt kite_root=.../KITE

Prints held-out recovery error (mm, deg) and the count accuracy, on val and every test split.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .data import SplitData
from .kite_release import KitePsi, load_release


class SetProbe(nn.Module):
    def __init__(self, z_dim, k=6, width=512):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(nn.LayerNorm(z_dim), nn.Linear(z_dim, width), nn.GELU(), nn.Linear(width, width),
                                 nn.GELU(), nn.Linear(width, k * 7))

    def forward(self, z):
        o = self.net(z).view(len(z), self.k, 7)
        return o[..., :3] * 0.4, F.normalize(o[..., 3:6], dim=-1), o[..., 6]


def chamfer(pp, pn, plogit, tp, tn, tv):
    """Symmetric Chamfer between predicted (soft-valid) and true contact sets; metres and 1-cos."""
    d = torch.cdist(tp, pp)  # (B, K_true, K_pred)
    pw = torch.sigmoid(plogit)
    big = d.detach().max() + 1
    # every true contact must be explained by some predicted point (validity is learned separately)
    d_t, j = d.min(-1)
    ncos = 1 - (tn * torch.gather(pn, 1, j[..., None].expand(-1, -1, 3))).sum(-1)
    t_term = ((d_t ** 2 + 0.01 * ncos) * tv).sum(1) / tv.sum(1)
    d_p = torch.where(tv[:, :, None], d, d + big).amin(1)  # each predicted point near a true one
    p_term = (d_p ** 2 * pw).sum(1) / pw.sum(1).clamp_min(1e-3)
    count = F.binary_cross_entropy_with_logits(plogit, _count_target(tv, plogit))
    return (t_term + p_term).mean() / 1e-4 + count


def _count_target(tv, plogit):
    k = tv.sum(1, keepdim=True)
    return (torch.arange(plogit.shape[1], device=plogit.device)[None] < k).float()


@torch.no_grad()
def evaluate(probe, z, b):
    pp, pn, pl = probe(z)
    valid_pred = torch.sigmoid(pl) > 0.5
    d = torch.cdist(b["c_pos"], pp).masked_fill(~valid_pred[:, None], 1e3)
    j = d.argmin(-1)
    err_mm = d.amin(-1) * 1000
    cos = (b["c_normal"] * torch.gather(pn, 1, j[..., None].expand(-1, -1, 3))).sum(-1).clamp(-1, 1)
    v = b["c_valid"]
    return {"pos_mm_mean": float(err_mm[v].mean()), "pos_mm_p90": float(err_mm[v].quantile(0.9)),
            "normal_deg_mean": float(torch.rad2deg(torch.acos(cos))[v].mean()),
            "count_acc": float((valid_pred.sum(1) == v.sum(1)).float().mean())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--train-examples", type=int, default=200_000)
    ap.add_argument("--steps", type=int, default=20_000)
    args = ap.parse_args(argv)
    cfg = Config.load(args.config, ["frame=world"] + args.set)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rel = load_release(cfg.kite_ckpt, cfg.kite_root or None)
    psi = KitePsi(rel.modules["contact_only_encoder"]).to(dev).eval()

    def encode(d):
        zs = [psi(b["c_pos"], b["c_normal"], b["c_valid"]) for b in d.iterate(4096)]
        return torch.cat(zs), d.batch(torch.arange(d.n, device=dev))

    with torch.no_grad():
        ztr, btr = encode(SplitData(cfg, cfg.train_split, dev, args.train_examples))
    print(f"z: dim {ztr.shape[1]}, mean |z| {ztr.norm(dim=1).mean():.3g}, per-dim std {ztr.std(0).mean():.3g} "
          f"(min {ztr.std(0).min():.3g})")
    s = torch.linalg.svdvals(ztr[:20000] - ztr[:20000].mean(0)).pow(2)
    p = s / s.sum()
    print(f"z participation ratio {float(1 / (p ** 2).sum()):.1f} of {ztr.shape[1]} dims")

    probe = SetProbe(ztr.shape[1]).to(dev)
    opt = torch.optim.AdamW(probe.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    g = torch.Generator(device="cpu").manual_seed(0)
    for step in range(args.steps):
        i = torch.randint(0, len(ztr), (1024,), generator=g).to(dev)
        loss = chamfer(*probe(ztr[i]), btr["c_pos"][i], btr["c_normal"][i], btr["c_valid"][i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % 2000 == 0:
            print(f"step {step + 1} loss {loss.item():.4g}", flush=True)
    probe.eval()
    res = {}
    for split in cfg.eval_splits:
        with torch.no_grad():
            z, b = encode(SplitData(cfg, split, dev, cfg.eval_max_examples))
        res[split] = evaluate(probe, z, b)
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
