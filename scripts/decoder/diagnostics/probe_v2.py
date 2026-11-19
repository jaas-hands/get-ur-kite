"""G2 probe v2: set probe with exact (permutation) matching, calibrated on an oracle input.

The repo probe (getkite.decoder.probe: Chamfer + validity head) cannot recover contacts even from the exact
contacts (oracle p90 ~70 mm), so it cannot measure psi. Here:
  - loss: min over all 720 slot permutations of sum_i [valid_i * (|dp|^2/s^2 + (1-cos n))] + BCE(validity)
  - metric: matched per-contact position error (mm) and normal error (deg) over true contacts, plus count accuracy
  - inputs: psi (KITE z) and oracle (contacts flattened, shuffled per example so it is a set, not an identity map)
The probe is trustworthy only if the oracle arm reaches ~1 mm.
"""
import argparse
import itertools
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from getkite.decoder.config import Config
from getkite.decoder.data import SplitData
from getkite.decoder.kite_release import KitePsi, load_release

K = 6
PERMS = torch.tensor(list(itertools.permutations(range(K))))  # (720, 6)
S = 0.01  # 10 mm position scale in the loss


class Probe(nn.Module):
    def __init__(self, d_in, width=1024, depth=4):
        super().__init__()
        layers, d = [nn.LayerNorm(d_in)], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.GELU()]
            d = width
        layers.append(nn.Linear(d, K * 7))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        o = self.net(x).view(len(x), K, 7)
        return o[..., :3] * 0.4, F.normalize(o[..., 3:6], dim=-1), o[..., 6]


def match(pp, pn, tp, tn, tv):
    """Best permutation per example. cost[b, i, j]: true i -> pred slot j."""
    cost = (torch.cdist(tp, pp) / S) ** 2 + 1.0 * (1 - torch.einsum("bik,bjk->bij", tn, pn))
    cost = cost * tv[:, :, None]
    perms = PERMS.to(pp.device)
    tot = cost[:, torch.arange(K, device=pp.device)[None, :], perms].sum(-1)  # (B, 720)
    best = perms[tot.argmin(1)]  # (B, 6): true i -> slot best[b, i]
    return best


def loss_fn(out, tp, tn, tv):
    pp, pn, pl = out
    with torch.no_grad():
        best = match(pp, pn, tp, tn, tv)
    mp = torch.gather(pp, 1, best[..., None].expand(-1, -1, 3))
    mn = torch.gather(pn, 1, best[..., None].expand(-1, -1, 3))
    ml = torch.gather(pl, 1, best)
    pos = (((mp - tp) / S) ** 2).sum(-1)
    nrm = 1 - (mn * tn).sum(-1)
    reg = ((pos + 1.0 * nrm) * tv).sum(1) / tv.sum(1)
    return reg.mean() + F.binary_cross_entropy_with_logits(ml, tv.float())


@torch.no_grad()
def metrics(probe, x, b, bs=8192):
    errs, angs, cnt = [], [], []
    for i in range(0, len(x), bs):
        pp, pn, pl = probe(x[i:i + bs])
        tp, tn, tv = b["c_pos"][i:i + bs], b["c_normal"][i:i + bs], b["c_valid"][i:i + bs]
        best = match(pp, pn, tp, tn, tv)
        mp = torch.gather(pp, 1, best[..., None].expand(-1, -1, 3))
        mn = torch.gather(pn, 1, best[..., None].expand(-1, -1, 3))
        errs.append(((mp - tp).norm(dim=-1) * 1000)[tv])
        angs.append(torch.rad2deg(torch.acos((mn * tn).sum(-1).clamp(-1, 1)))[tv])
        cnt.append(((torch.sigmoid(pl) > 0.5).sum(1) == tv.sum(1)).float())
    e, a, c = torch.cat(errs), torch.cat(angs), torch.cat(cnt)
    q = lambda t: {f"p{int(p * 100)}": round(float(t.quantile(p)), 2) for p in (0.5, 0.75, 0.9, 0.99)}
    return {"n": len(x), "pos_mm": {**q(e), "mean": round(float(e.mean()), 2)},
            "normal_deg": {"p50": round(float(a.median()), 2), "p90": round(float(a.quantile(0.9)), 2)},
            "count_acc": round(float(c.mean()), 4)}


def oracle_input(b, gen):
    v = b["c_valid"]
    perm = torch.argsort(torch.rand(v.shape, generator=gen).to(v.device), dim=1)  # shuffle slot order
    g = lambda t: torch.gather(t, 1, perm[..., None].expand(-1, -1, t.shape[-1]))
    pos, nrm, val = g(b["c_pos"]), g(b["c_normal"]), torch.gather(v, 1, perm)
    return torch.cat([pos * val[..., None], nrm * val[..., None], val[..., None].float()], -1).flatten(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--train-examples", type=int, default=200_000)
    ap.add_argument("--steps", type=int, default=40_000)
    ap.add_argument("--arms", nargs="*", default=["oracle", "psi"])
    ap.add_argument("--eval-splits", nargs="*",
                    default=["val", "test_geometry", "test_joint_count", "test_finger_subset", "test_combined"])
    a = ap.parse_args()
    cfg = Config.load(None, ["frame=world"] + a.set)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rel = load_release(cfg.kite_ckpt, cfg.kite_root or None)
    psi = KitePsi(rel.modules["contact_only_encoder"]).to(dev).eval()
    gen = torch.Generator().manual_seed(1)

    def load(split, n):
        d = SplitData(cfg, split, dev, n)
        b = d.batch(torch.arange(d.n, device=dev))
        with torch.no_grad():
            z = torch.cat([psi(x["c_pos"], x["c_normal"], x["c_valid"]) for x in d.iterate(4096)])
        return {"psi": z, "oracle": oracle_input(b, gen)}, b

    xtr, btr = load(cfg.train_split, a.train_examples)
    evals = {s: load(s, cfg.eval_max_examples) for s in a.eval_splits}
    evals["train(seen)"] = ({k: v[:20000] for k, v in xtr.items()}, {k: v[:20000] for k, v in btr.items()})
    for kind in a.arms:
        torch.manual_seed(0)
        x = xtr[kind]
        probe = Probe(x.shape[1]).to(dev)
        opt = torch.optim.AdamW(probe.parameters(), 3e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=a.steps, pct_start=0.05)
        g = torch.Generator(device="cpu").manual_seed(0)
        for step in range(a.steps):
            i = torch.randint(0, len(x), (1024,), generator=g).to(dev)
            loss = loss_fn(probe(x[i]), btr["c_pos"][i], btr["c_normal"][i], btr["c_valid"][i])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if (step + 1) % max(1, a.steps // 8) == 0:
                print(f"[{kind}] step {step + 1} loss {loss.item():.4g}", flush=True)
        probe.eval()
        res = {s: metrics(probe, xs[kind], bs) for s, (xs, bs) in evals.items()}
        print(json.dumps({kind: res}, indent=1), flush=True)


if __name__ == "__main__":
    main()
