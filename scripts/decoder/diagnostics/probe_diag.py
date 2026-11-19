"""G2 diagnostics for getkite.decoder.probe (does not modify the repo).

Runs the same SetProbe/chamfer/training recipe twice:
  psi    : input = KITE psi's z (what getkite.decoder.probe does)
  oracle : input = the contacts themselves, flattened in stored order (pos, normal, valid) -> probe capacity floor
and reports robust metrics: sentinel rate (examples with no predicted-valid point), median/p75/p90 error
over predicted-valid points, and error ignoring the validity head (nearest of all K predicted points).
"""
import argparse
import json

import torch

from getkite.decoder.config import Config
from getkite.decoder.data import SplitData
from getkite.decoder.kite_release import KitePsi, load_release
from getkite.decoder.probe import SetProbe, chamfer


def metrics(probe, x, b):
    with torch.no_grad():
        pp, pn, pl = probe(x)
    v = b["c_valid"]
    valid_pred = torch.sigmoid(pl) > 0.5
    d_all = torch.cdist(b["c_pos"], pp)                      # (B, Kt, Kp) metres
    no_pred = ~valid_pred.any(1)                             # sentinel cases in probe.evaluate
    d_masked = d_all.masked_fill(~valid_pred[:, None], float("inf")).amin(-1) * 1000
    d_any = d_all.amin(-1) * 1000                            # ignore validity head
    ok = v & ~no_pred[:, None]
    q = lambda t: {f"p{int(p*100)}": round(float(t.quantile(p)), 2) for p in (0.5, 0.75, 0.9)}
    j = d_all.argmin(-1)
    cos = (b["c_normal"] * torch.gather(pn, 1, j[..., None].expand(-1, -1, 3))).sum(-1).clamp(-1, 1)
    return {"n": int(len(x)), "no_valid_pred_frac": round(float(no_pred.float().mean()), 4),
            "pos_mm_validhead": q(d_masked[ok]), "pos_mm_validhead_mean": round(float(d_masked[ok].mean()), 2),
            "pos_mm_anypred": q(d_any[v]), "pos_mm_anypred_mean": round(float(d_any[v].mean()), 2),
            "normal_deg_median": round(float(torch.rad2deg(torch.acos(cos))[v].median()), 2),
            "count_acc": round(float((valid_pred.sum(1) == v.sum(1)).float().mean()), 4),
            "k_mean": round(float(v.sum(1).float().mean()), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--train-examples", type=int, default=200_000)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--eval-splits", nargs="*", default=["val", "test_combined"])
    a = ap.parse_args()
    cfg = Config.load(None, ["frame=world"] + a.set)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rel = load_release(cfg.kite_ckpt, cfg.kite_root or None)
    psi = KitePsi(rel.modules["contact_only_encoder"]).to(dev).eval()

    def load(split, n):
        d = SplitData(cfg, split, dev, n)
        b = d.batch(torch.arange(d.n, device=dev))
        with torch.no_grad():
            z = torch.cat([psi(x["c_pos"], x["c_normal"], x["c_valid"]) for x in d.iterate(4096)])
        flat = torch.cat([b["c_pos"] * b["c_valid"][..., None], b["c_normal"] * b["c_valid"][..., None],
                          b["c_valid"][..., None].float()], -1).flatten(1)
        return {"psi": z, "oracle": flat}, b

    xtr, btr = load(cfg.train_split, a.train_examples)
    evals = {s: load(s, cfg.eval_max_examples) for s in a.eval_splits}
    evals["train(seen)"] = ({k: v[:20000] for k, v in xtr.items()}, {k: v[:20000] for k, v in btr.items()})
    out = {}
    for kind in ("psi", "oracle"):
        torch.manual_seed(0)
        x = xtr[kind]
        probe = SetProbe(x.shape[1]).to(dev)
        opt = torch.optim.AdamW(probe.parameters(), 1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        g = torch.Generator(device="cpu").manual_seed(0)
        for step in range(a.steps):
            i = torch.randint(0, len(x), (1024,), generator=g).to(dev)
            loss = chamfer(*probe(x[i]), btr["c_pos"][i], btr["c_normal"][i], btr["c_valid"][i])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if (step + 1) % 5000 == 0:
                print(f"[{kind}] step {step + 1} loss {loss.item():.4g}", flush=True)
        probe.eval()
        out[kind] = {s: metrics(probe, xs[kind], bs) for s, (xs, bs) in evals.items()}
        print(json.dumps({kind: out[kind]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
