"""Our KITE loader against KITE's own code. Needs a KITE checkout: set KITE_ROOT.

No released weights are needed: we build KITE's ContactProbePolicyPairedCoTrain with the released
config's shapes (conf/kite.yaml), random weights, save it in release.pt's format, then check that
our loader reproduces KITE's encode_contact_latent and its latent -> scene path exactly.
For the real checkpoint, run scripts/check_kite_release.py.
"""

import os
from pathlib import Path

import pytest
import torch

from getkite.decoder import paths

KITE_ROOT = paths.get("KITE_ROOT") if (Path(paths.get("KITE_ROOT")) / "networks").is_dir() else ""
pytestmark = pytest.mark.skipif(not KITE_ROOT, reason="no KITE checkout (git submodule update --init, or set KITE_ROOT)")

RELEASE_SHARED = dict(variant="mlp", d_model=256, num_heads=8, num_scene_layers=3, num_action_layers=2,
                      mlp_ratio=4.0, hand_points=32, use_contact_tokens_in_decoder=False, step_embed_dim=256,
                      num_inference_steps=1, action_tokenization="per_joint", use_qref_action_tokens=True,
                      action_token_layer_norm=True, contact_bottleneck_enabled=True,
                      contact_bottleneck_num_encoder_layers=2, contact_bottleneck_unpack_tokens=4,
                      contact_bottleneck_unpack_mlp_ratio=2.0)


def kite_normalizers(lower, upper):
    """KITE's absolute normaliser (contact_policy_common) and closed-form residual normaliser for a
    floating hand with joint limits lower/upper (base box +-0.4 m, Euler angles +-pi)."""
    import math
    from networks.residual_action_utils import build_closed_form_residual_normalizer, build_residual_slot_indices
    lo = torch.cat([torch.full((3,), -0.4), torch.full((3,), -math.pi), lower])
    hi = torch.cat([torch.full((3,), 0.4), torch.full((3,), math.pi), upper])
    a_scale = 2 / (hi - lo)
    slot = build_residual_slot_indices(dof=len(lo), base_translation_indices=[0, 1, 2], base_rotation_indices=[3, 4, 5])
    r_scale, r_off = build_closed_form_residual_normalizer(num_robots=1, slot=slot, xyz_std=0.005,
                                                           rot_std_rad=math.radians(5), joint_std=0.05 * (upper - lower))
    return dict(action_normalizer_scale=a_scale, action_normalizer_offset=-1 - a_scale * lo,
                residual_normalizer_scale=r_scale.view(-1), residual_normalizer_offset=r_off.view(-1))


def make_fake_release(path, joints=(16, 4), names=None, limits=None):
    """limits: optional {name: (lower, upper)} for heads whose normalisers should be KITE's exact ones."""
    from getkite.decoder.kite_release import kite_networks
    kite_networks(KITE_ROOT)
    from networks.contact_probe_policy_paired_cotrain import ContactProbePolicyPairedCoTrain
    manips = []
    names = names or [f"robot{i}" for i in range(len(joints))]
    for name, nj in zip(names, joints):
        dof = 6 + nj
        lim = (limits or {}).get(name)
        norm = kite_normalizers(*lim) if lim else dict(action_normalizer_scale=torch.rand(dof) + 0.5,
                                                       action_normalizer_offset=torch.randn(dof) * 0.1)
        manips.append(dict(name=name, action_dim=dof, base_translation_indices=[0, 1, 2],
                           base_rotation_indices=[3, 4, 5], **norm))
    torch.manual_seed(0)
    model = ContactProbePolicyPairedCoTrain(manipulators=manips, shared_kwargs=dict(RELEASE_SHARED),
                                           per_manipulator_action_io=True)
    for p in model.parameters():  # no zero-initialised layers left
        p.data.add_(0.05 * torch.randn_like(p))
    torch.save({"model": model.state_dict(), "ema_model": model.state_dict(), "config": {"model": {"num_heads": 8}}},
               path)
    return model.eval()


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    path = tmp_path_factory.mktemp("kite") / "release.pt"
    return path, make_fake_release(path)


def contacts(B=8, K=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = (torch.rand(B, K, 3, generator=g) - 0.5) * 0.8
    n = torch.nn.functional.normalize(torch.randn(B, K, 3, generator=g), dim=-1)
    k = torch.randint(2, K + 1, (B,), generator=g)
    valid = torch.arange(K)[None] < k[:, None]
    return pos, n, valid


def test_loader_matches_kite_encode_contact_latent(fake):
    from getkite.decoder.kite_release import KitePsi, load_release
    path, model = fake
    rel = load_release(str(path), KITE_ROOT)
    assert rel.heads == ["robot0", "robot1"]
    assert rel.meta == dict(d_model=256, num_heads=8, mlp_ratio=4.0, enc_layers=2, scene_layers=3, action_layers=2,
                            unpack_tokens=4, unpack_ratio=2.0)
    pos, n, valid = contacts()
    head = model.head_for("robot1")
    cloud = torch.cat([pos, n], -1) * valid[..., None]
    with torch.no_grad():
        z_kite = head.encode_contact_latent({"contact_cloud": cloud, "contact_valid_mask": valid})
        z_ours = KitePsi(rel.modules["contact_only_encoder"]).eval()(pos, n, valid)
    assert torch.allclose(z_kite, z_ours, atol=1e-5)
    # padding values must not matter
    junk = torch.where(valid[..., None], pos, torch.randn_like(pos))
    with torch.no_grad():
        assert torch.allclose(KitePsi(rel.modules["contact_only_encoder"]).eval()(junk, n, valid), z_ours, atol=1e-5)


def test_loader_matches_kite_scene_memory(fake):
    from getkite.decoder.kite_release import load_release
    path, model = fake
    rel = load_release(str(path), KITE_ROOT)
    head = model.head_for("robot0")
    B = 4
    hx, hn = torch.randn(B, 32, 3) * 0.1, torch.nn.functional.normalize(torch.randn(B, 32, 3), dim=-1)
    z = torch.randn(B, 256)
    with torch.no_grad():
        mem = head._build_scene_memory_from_latent(z, batch={}, hand_xyz=hx, hand_normals=hn)["memory_feat"]
        m = rel.modules
        feat, _ = m["bottleneck_scene_encoder"](hand_feat=m["hand_proj"](torch.cat([hx, hn], -1)), hand_xyz=hx,
                                                z_tokens=m["cls_unpacker"](z))
    assert torch.allclose(mem, feat, atol=1e-5)


def test_per_head_parts_are_exported(fake):
    from getkite.decoder.kite_release import load_release
    path, _ = fake
    rel = load_release(str(path), KITE_ROOT)
    assert rel.per_head["robot0"]["joint_identity_embed"].shape == (16, 256)
    assert rel.per_head["robot1"]["joint_identity_embed"].shape == (4, 256)
    assert rel.per_head["robot0"]["action_queries"].shape == (2, 256)
    assert "net.0.weight" not in rel.per_head["robot0"]["head_joints"]  # Sequential keys: '0.weight', ...
    assert rel.per_head["robot0"]["head_joints"]["0.weight"].shape == (256, 256)


def test_get_arch_with_kite_psi(fake):
    from getkite.decoder.config import Config
    from getkite.decoder.kite_release import load_release
    from getkite.decoder.model import build_model
    path, _ = fake
    rel = load_release(str(path), KITE_ROOT)
    cfg = Config(arch="get", intent="kite_psi", frame="world", z_dim=256, d_model=64, n_layers=1, n_heads=4, amp="none")
    m = build_model(cfg, rel)
    assert not any(p.requires_grad for p in m.intent.psi.parameters())
    assert all(p.requires_grad for p in m.intent.unpacker.parameters())
    pos, n, valid = contacts(4)
    it = m.intent(pos, n, valid)
    assert it.tokens.shape == (4, 4, 64) and it.z.shape == (4, 256)
