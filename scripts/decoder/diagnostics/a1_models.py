"""Preflight audit, part 1: trainability, shared checkpoint, GET/slot parity (CPU)."""
import json, torch
from getkite.decoder.config import Config
from getkite.decoder.model import build_model
from getkite.decoder.train import param_groups, KITE_PRETRAINED
from getkite.decoder import paths
K = paths.get("KITE_ROOT")
SET = [f"kite_root={K}", f"kite_ckpt={K}/checkpoints/decoder/release.pt",
       "data_dir=" + paths.get("GETKITE_DATASET")]
cfgs, models = {}, {}
for n in ["kite_trunk_get_ft", "kite_trunk_slot_ft"]:
    c = Config.load(f"configs/{n}.json", SET); torch.manual_seed(c.seed)
    cfgs[n], models[n] = c, build_model(c)
dc = {k: (cfgs["kite_trunk_get_ft"].to_dict()[k], cfgs["kite_trunk_slot_ft"].to_dict()[k])
      for k in cfgs["kite_trunk_get_ft"].to_dict() if cfgs["kite_trunk_get_ft"].to_dict()[k] != cfgs["kite_trunk_slot_ft"].to_dict()[k]}
print("RESOLVED CONFIG DIFFERENCES:", dc)
c = cfgs["kite_trunk_get_ft"]
print("shared settings:", {k: c.to_dict()[k] for k in ["regime", "train_split", "train_part", "holdout_frac", "batch_size", "total_steps",
      "lr", "warmup_steps", "kite_trunk_lr_mult", "weight_decay", "grad_clip", "seed", "kite_freeze", "kite_init_head", "frame",
      "hand_points", "resample_q_init", "normal_noise_rad", "eval_every", "eval_max_examples", "amp"] if k in c.to_dict()})
for n, m in models.items():
    m.train()
    psi = list(m.psi.parameters())
    print(f"\n== {n}")
    print(f"  psi: {sum(p.numel() for p in psi):,} params, trainable {sum(p.numel() for p in psi if p.requires_grad)}, "
          f"training-mode modules after model.train(): {sum(x.training for x in m.psi.modules())}")
    rows = {}
    for k, p in m.named_parameters():
        g = k.split(".")[0]; a, b = rows.get(g, (0, 0)); rows[g] = (a + p.numel() * p.requires_grad, b + p.numel())
    for g, (a, b) in rows.items():
        print(f"  {g:14s} trainable {a:>10,} / {b:>10,}  lr x{c.kite_trunk_lr_mult if (g + '.').startswith(KITE_PRETRAINED) else 1.0}")
    print(f"  total trainable {sum(p.numel() for p in m.parameters() if p.requires_grad):,}; non-trainable outside psi: "
          f"{[k for k, p in m.named_parameters() if not p.requires_grad and not k.startswith('psi.')]}")
ga, sa = dict(models["kite_trunk_get_ft"].named_parameters()), dict(models["kite_trunk_slot_ft"].named_parameters())
shared = sorted(set(ga) & set(sa))
diff = [k for k in shared if not torch.equal(ga[k], sa[k])]
print(f"\nshared parameter tensors: {len(shared)}, identical at init: {len(shared) - len(diff)}, differing: {diff}")
print("GET-only params:", sorted({k.rsplit('.', 2)[0] if k.count('.') > 1 else k for k in set(ga) - set(sa)}),
      f"({sum(ga[k].numel() for k in set(ga) - set(sa)):,})")
print("slot-only params:", sorted(set(sa) - set(ga)), f"({sum(sa[k].numel() for k in set(sa) - set(ga)):,})")
gb, sb = dict(models["kite_trunk_get_ft"].named_buffers()), dict(models["kite_trunk_slot_ft"].named_buffers())
print("buffers identical:", all(torch.equal(gb[k], sb[k]) for k in set(gb) & set(sb)), "| only-in-one:", sorted(set(gb) ^ set(sb)))
