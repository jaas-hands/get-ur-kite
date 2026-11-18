"""Check KITE's real release checkpoint against our loader, then look at z on our LEAP data.

    python scripts/check_kite_release.py --kite-root ~/KITE --data-dir ../get-ur-kite-dataset/data

1. Builds KITE's own ContactProbePolicyPairedCoTrain from release.pt + dataset_meta.pt (KITE's
   loading path) and checks our KitePsi / trunk modules give identical outputs on random inputs.
2. Prints which heads exist and their joint counts (pick kite_init_head from these).
3. Encodes val contacts from our dataset and prints z statistics. Run getkite.decoder.probe next.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kite-root", required=True)
    ap.add_argument("--ckpt", default=None, help="default: <kite-root>/checkpoints/decoder/release.pt")
    ap.add_argument("--data-dir", default="")
    ap.add_argument("--regime", default="kite")
    a = ap.parse_args()
    ckpt = a.ckpt or str(Path(a.kite_root) / "checkpoints/decoder/release.pt")

    from getkite.decoder.kite_release import KitePsi, kite_networks, load_release
    rel = load_release(ckpt, a.kite_root)
    print("meta:", rel.meta)
    for h in rel.heads:
        e = rel.per_head[h].get("joint_identity_embed")
        print(f"  head {h}: {0 if e is None else e.shape[0]} joint tokens")

    # 1. against KITE's own loader
    kite_networks(a.kite_root)
    from networks.contact_probe_policy_paired_cotrain import ContactProbePolicyPairedCoTrain
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = torch.load(Path(ckpt).parent / "dataset_meta.pt", map_location="cpu", weights_only=False)
    m = rel.meta  # conf/kite.yaml's model section, sizes read from the weights
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
                      # robotiq_gripper_stick has two pose groups (gripper + stick); KITE's loader passes these
                      # (experiment/pipeline_high_low_policy.py), else its residual-dim check fails (19 vs 16).
                      "pose_groups": e.get("pose_groups")})
        print(f"  {name}: action_dim {e['action_dim']}, residual scale (xyz, rot, first joint) "
              f"{[round(float(x), 3) for x in torch.as_tensor(e['residual_normalizer_scale']).view(-1)[[0, 3, 9]]]}")
    model = ContactProbePolicyPairedCoTrain(manipulators=specs, shared_kwargs=shared, per_manipulator_action_io=True)
    model.load_state_dict(ck["ema_model"] if "ema_model" in ck else ck["model"], strict=True)  # strict: config matches
    model.eval()
    head = model.head_for(rel.heads[0])
    g = torch.Generator().manual_seed(0)
    pos = (torch.rand(64, 6, 3, generator=g) - 0.5) * 0.8
    nrm = torch.nn.functional.normalize(torch.randn(64, 6, 3, generator=g), dim=-1)
    valid = torch.arange(6)[None] < torch.randint(2, 7, (64, 1), generator=g)
    with torch.no_grad():
        z_kite = head.encode_contact_latent({"contact_cloud": torch.cat([pos, nrm], -1) * valid[..., None],
                                             "contact_valid_mask": valid})
        z_ours = KitePsi(rel.modules["contact_only_encoder"]).eval()(pos, nrm, valid)
    err = (z_kite - z_ours).abs().max().item()
    print(f"psi: max |z_kite - z_ours| = {err:.2e}  {'OK' if err < 1e-4 else 'MISMATCH'}")
    print("expected per-head residual scale for KITE's closed-form normaliser: xyz 200, rot 11.459, "
          "joint 1/(0.05 range) - if these differ, update XYZ_STD/ROT_STD/JOINT_STD_FRAC in kite_trunk.py")

    # 3. z on our data
    if a.data_dir:
        from getkite.decoder.config import Config
        from getkite.decoder.data import SplitData
        d = SplitData(Config(data_dir=a.data_dir, regime=a.regime, frame="world"), "val", max_examples=20000)
        with torch.no_grad():
            z = torch.cat([KitePsi(rel.modules["contact_only_encoder"]).eval()(b["c_pos"], b["c_normal"], b["c_valid"])
                           for b in d.iterate(4096)])
        s = torch.linalg.svdvals(z - z.mean(0)).pow(2)
        p = s / s.sum()
        print(f"z on LEAP val ({len(z)}): |z| {z.norm(dim=1).mean():.3g}, per-dim std {z.std(0).mean():.3g}, "
              f"participation ratio {float(1 / (p ** 2).sum()):.1f}/{z.shape[1]}")
        print("next: python -m getkite.decoder.probe --set kite_ckpt=... kite_root=... data_dir=...")


if __name__ == "__main__":
    main()
