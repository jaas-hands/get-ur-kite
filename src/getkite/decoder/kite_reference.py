"""Published KITE reference: the released decoder bank, run through KITE's own predict_action, on our
fixed evaluation records (same contacts, same q1, same metrics as get_universal_scratch).

    python -m getkite.decoder.kite_reference --config configs/scratch/c3_small.json [--pool val] [--out FILE]

Only robots that have a released head are scored (allegro_right, barrett here). Inputs follow KITE's
released pipeline exactly:
  contact_cloud (B, 6, 6) world [xyz, normal] + contact_valid_mask; no input noise at evaluation
  q1_padded     [t, intrinsic-XYZ Euler, joints in KITE order] (our records keep KITE's coordinate order)
  hand_points_world / hand_normals_world: KITE's 32 hand tokens = greedy FPS from index 0 over the
                template's link-local points (robot_model._farthest_point_sample_indices_local), posed at q1
The prediction ("action": absolute q) is scored with the same pose_metrics as our model. KITE's decoder has
prior training on these embodiments and a very different training history: a reference, not an
equal-compute comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .handbank import kite_local_fps
from .multifamily import fixed_start
from .rotations import euler_xyz_to_matrix, matrix_to_euler_xyz
from .train_scratch import build_bank, load_config, load_pool, pose_metrics


def build_kite_bank(ckpt: str, kite_root: str, device):
    from .kite_release import kite_networks, load_release
    rel = load_release(ckpt, kite_root)
    kite_networks(kite_root)
    from networks.contact_probe_policy_paired_cotrain import ContactProbePolicyPairedCoTrain
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = torch.load(Path(ckpt).parent / "dataset_meta.pt", map_location="cpu", weights_only=False)
    m = rel.meta
    shared = dict(variant="mlp", d_model=m["d_model"], num_heads=m["num_heads"], num_scene_layers=m["scene_layers"],
                  num_action_layers=m["action_layers"], mlp_ratio=m["mlp_ratio"], hand_points=32,
                  use_contact_tokens_in_decoder=False, action_tokenization="per_joint", use_qref_action_tokens=True,
                  action_token_layer_norm=True, contact_bottleneck_enabled=True,
                  contact_bottleneck_num_encoder_layers=m["enc_layers"],
                  contact_bottleneck_unpack_tokens=m["unpack_tokens"],
                  contact_bottleneck_unpack_mlp_ratio=m["unpack_ratio"])
    specs = []
    for name in meta["manipulators"]:
        e = meta["per_manipulator"][name]
        specs.append({"name": name, "action_dim": int(e["action_dim"]),
                      **{k: torch.as_tensor(e[k], dtype=torch.float32).view(-1) for k in
                         ("action_normalizer_scale", "action_normalizer_offset", "residual_normalizer_scale",
                          "residual_normalizer_offset")},
                      "base_translation_indices": list(e["base_translation_indices"]),
                      "base_rotation_indices": list(e["base_rotation_indices"]),
                      "pose_groups": e.get("pose_groups")})
    model = ContactProbePolicyPairedCoTrain(manipulators=specs, shared_kwargs=shared, per_manipulator_action_io=True)
    model.load_state_dict(ck["ema_model"] if "ema_model" in ck else ck["model"], strict=True)  # EMA, as KITE evaluates
    return model.eval().to(device), meta


@torch.no_grad()
def kite_predict(model, meta, bank, rob_idx, rec, sg, device):
    name = bank.names[rob_idx]
    head = model.head_for(name)
    e = meta["per_manipulator"][name]
    assert list(e["base_translation_indices"]) == [0, 1, 2] and list(e["base_rotation_indices"]) == [3, 4, 5]
    desc = bank.descs[rob_idx]
    nc = len(desc.coords)
    assert int(e["action_dim"]) == 6 + nc, (name, e["action_dim"], nc)
    fps = torch.as_tensor(kite_local_fps(desc.tpl_pos, 32), device=device)
    B = len(rec)
    r = rec.r
    Rs, ts = bank.fk(r, bank.expand(r, sg.qc1), sg.R1, sg.t1)
    hp, hn = bank.points(r, Rs, ts, fps[None].expand(B, -1))
    q1 = torch.cat([sg.t1, matrix_to_euler_xyz(sg.R1), sg.qc1[:, :nc]], -1)
    v = rec.c_valid
    batch = {"contact_cloud": torch.cat([rec.c_pos, rec.c_normal], -1) * v[..., None], "contact_valid_mask": v,
             "q1_padded": q1, "hand_points_world": hp, "hand_normals_world": hn,
             "local_dof": torch.full((B,), 6 + nc, device=device)}
    a = head.predict_action(batch)["action"].float()
    qc = torch.zeros_like(sg.qc1)
    qc[:, :nc] = a[:, 6:6 + nc]
    return qc, a[:, :3], euler_xyz_to_matrix(a[:, 3:6])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pool", default="val")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    cfg = load_config(a.config, [])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bank = build_bank(cfg, dev)
    rec = load_pool(cfg[a.pool] if a.pool in ("train", "val") else cfg["extra_eval"][a.pool], bank).to(dev)
    seed = cfg["val_start_seed"] if a.pool != "train" or cfg["start"] != "fixed" else cfg["train_start_seed"]
    sg_all = fixed_start(bank, rec, seed)  # identical q1 to the scratch model's evaluation of this pool
    model, meta = build_kite_bank(cfg["kite_ckpt"], cfg["kite_root"], dev)
    res = {}
    for i, name in enumerate(bank.names):
        if name not in meta["manipulators"]:
            continue
        sel = (rec.r == i).nonzero().squeeze(1)
        if len(sel) == 0:
            continue
        acc = {}
        for s in range(0, len(sel), 512):
            idx = sel[s:s + 512]
            rb = rec.index(idx)
            g = type(sg_all)(**{k: getattr(sg_all, k)[idx] for k in sg_all.__dataclass_fields__})
            qc, t, R = kite_predict(model, meta, bank, i, rb, g, dev)
            for k, v in pose_metrics(bank, rb, rb.r, qc, t, R, g, cfg).items():
                acc.setdefault(k, []).append(v.cpu())
        fam = bank.family[i]
        for k, v in acc.items():
            res[f"{fam}/kite_released/{k}"] = float(torch.cat(v).mean())
        res[f"{fam}/kite_released/n"] = int(len(sel))
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res | {"_pool": a.pool, "_start_seed": seed, "_config": a.config}, indent=1))


if __name__ == "__main__":
    main()
