"""KiteTrunkDecoder against KITE's own decoder head. Needs the KITE checkout and generated hands (GETKITE_DATASET).

The central check: with joint_encoder='slot', no tree bias and the slot table filled from a KITE
head's joint_identity_embed, our decoder must reproduce that head's predict_action exactly. That
pins every convention we rely on: psi input, hand tokens, q_init normalisation (Euler, boxes),
token order, the residual normaliser and the rotation update (world-frame left delta).
"""

import math
import os
from pathlib import Path

import pytest
import torch

from test_kite_release import KITE_ROOT, make_fake_release

from getkite.decoder import paths

DATA = paths.get("GETKITE_DATASET") if (Path(paths.get("GETKITE_DATASET")) / "hands").is_dir() else ""
pytestmark = pytest.mark.skipif(not (KITE_ROOT and DATA), reason="needs the KITE checkout and the dataset's hands "
                                "(GETKITE_DATASET)")
HAND = "I4-M4-R0-T4_g0"  # 12 movable joints


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    from pathlib import Path

    from getkite.decoder.handbank import HandBank
    from getkite.decoder.kite_release import load_release
    bank = HandBank([HAND], Path(DATA) / "hands")
    nd = int(bank.n_dof[0])
    lim = {"leapvar": (bank.lower[0, :nd], bank.upper[0, :nd])}
    path = tmp_path_factory.mktemp("kite") / "release.pt"
    kite = make_fake_release(path, joints=(nd, 4), names=["leapvar", "other"], limits=lim)
    return bank, kite, load_release(str(path), KITE_ROOT)


def random_batch(bank, B=16, seed=0):
    from getkite.decoder.rotations import quat_to_matrix
    g = torch.Generator().manual_seed(seed)
    h = torch.zeros(B, dtype=torch.long)
    u = torch.rand(B, 16, generator=g)
    q0 = torch.where(bank.dof_valid[h], bank.lower[h] + u * bank.range[h], torch.zeros(()))
    R0 = quat_to_matrix(torch.randn(B, 4, generator=g))
    t0 = (torch.rand(B, 3, generator=g) - 0.5) * 0.8
    Rs, ts = bank.fk(h, q0, R0, t0)
    k = torch.randint(2, 7, (B,), generator=g)
    tidx = torch.randint(0, bank.tpl_pos.shape[1], (B, 6), generator=g)
    tidx = torch.where(torch.arange(6)[None] < k[:, None], tidx, torch.full_like(tidx, -1))
    x, n = bank.points(h, Rs, ts, tidx)
    return {"h": h, "q_init": q0, "R_init": R0, "t_init": t0, "c_pos": x, "c_normal": n, "c_valid": tidx >= 0}


def kite_predict(kite, bank, b, cfg):
    """Run KITE's own head on the same inputs, in its batch format."""
    from getkite.decoder.rotations import euler_xyz_to_matrix, matrix_to_euler_xyz
    head = kite.head_for("leapvar")
    nd = int(bank.n_dof[0])
    Rs, ts = bank.fk(b["h"], b["q_init"], b["R_init"], b["t_init"])
    idx = bank.kite_hand_idx[b["h"]] if cfg.hand_points == "kite" else bank.surf_idx[b["h"]]
    hx, hn = bank.points(b["h"], Rs, ts, idx)
    q1 = torch.cat([b["t_init"], matrix_to_euler_xyz(b["R_init"]), b["q_init"][:, :nd]], -1)
    cloud = torch.cat([b["c_pos"], b["c_normal"]], -1) * b["c_valid"][..., None]
    with torch.no_grad():
        a = head.predict_action({"q1_padded": q1, "hand_points_world": hx, "hand_normals_world": hn,
                                 "contact_cloud": cloud, "contact_valid_mask": b["c_valid"]})["action"]
    return a[:, :3], euler_xyz_to_matrix(a[:, 3:6]), a[:, 6:]


def make_cfg(**kw):
    from getkite.decoder.config import Config
    return Config(**({"arch": "kite_trunk", "frame": "world", "kite_init_head": "leapvar", "amp": "none"} | kw))


def test_slot_trunk_reproduces_kite_head(setup):
    from getkite.decoder.kite_trunk import KiteTrunkDecoder
    bank, kite, rel = setup
    cfg = make_cfg(joint_encoder="slot", tree_bias=False)
    model = KiteTrunkDecoder(cfg, rel).eval()
    nd = int(bank.n_dof[0])
    with torch.no_grad():  # slot table := KITE head's per-joint embedding, by canonical joint id
        model.slot_embed[bank.slot[0, :nd]] = rel.per_head["leapvar"]["joint_identity_embed"]
    b = random_batch(bank)
    with torch.no_grad():
        out = model(b, bank)
    t, R, q = kite_predict(kite, bank, b, cfg)
    assert (out["t"] - t).abs().max() < 1e-5
    assert (out["R"] - R).abs().max() < 1e-4
    assert (out["q"][:, :nd] - q).abs().max() < 1e-4


def test_get_trunk_runs_masks_padding_and_trains(setup):
    from getkite.decoder.kite_trunk import KiteTrunkDecoder
    from getkite.decoder.losses import compute_losses
    bank, _, rel = setup
    torch.manual_seed(0)
    model = KiteTrunkDecoder(make_cfg(joint_encoder="get"), rel)
    b = random_batch(bank, B=8)
    b |= {"q_goal": b["q_init"], "R_goal": b["R_init"], "t_goal": b["t_init"], "dof_valid": bank.dof_valid[b["h"]],
          "tidx": torch.zeros(8, 6, dtype=torch.long), "mask": bank.dof_valid[b["h"]]}
    model.eval()
    with torch.no_grad():
        o1 = model(b, bank)
        b2 = dict(b, q_init=torch.where(b["dof_valid"], b["q_init"], torch.full_like(b["q_init"], 3.0)))
        o2 = model(b2, bank)  # garbage in padded joint slots must not matter
    assert torch.allclose(o1["q"], o2["q"].where(b["dof_valid"], o1["q"]), atol=1e-5)
    assert torch.allclose(o1["t"], o2["t"], atol=1e-5)
    model.train()
    losses = compute_losses(model(b, bank), b, bank, model.cfg)
    losses["total"].backward()
    assert model.psi.enc.input_proj.weight.grad is None  # psi stays frozen
    assert model.joint_in[0].weight.grad.abs().sum() > 0
    assert model.tree_bias[0].spd.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------- T5: joint-order permutation
SWAP = list(range(4, 8)) + list(range(0, 4)) + list(range(8, 16))  # new joint i = old joint SWAP[i]: index <-> middle


def permute_bank(bank, perm, remap_slot=True):
    """The same physical hand with its movable joints listed in a different order. perm must keep parents
    before children (HandBank.fk walks joints in order). Every per-joint array, slot reference (parent_slot,
    tpl_anchor) and graph node (spd/anc/node_valid) is remapped. With remap_slot=False the canonical LEAP
    joint ids stay positional, i.e. the fingers keep their joint geometry but swap names."""
    import copy

    from getkite.decoder.handbank import MAX_DOF, N_BODY
    p = torch.as_tensor(perm)
    inv = torch.empty_like(p)
    inv[p] = torch.arange(MAX_DOF)
    slot_map = torch.cat([torch.zeros(1, dtype=torch.long), inv + 1])  # old slot -> new slot
    node = torch.cat([torch.zeros(1, dtype=torch.long), p + 1, torch.arange(MAX_DOF + 1, N_BODY)])  # new -> old
    nb = copy.copy(bank)
    for k in ("A", "axis", "lower", "upper", "dof_valid", "joint_feat", "finger") + (("slot",) if remap_slot else ()):
        setattr(nb, k, getattr(bank, k)[:, p])
    par = bank.parent_slot[:, p]
    nb.parent_slot = torch.where(nb.dof_valid, slot_map[par], par)
    nb.tpl_anchor = slot_map[bank.tpl_anchor]
    nb.spd, nb.anc = bank.spd[:, node][:, :, node], bank.anc[:, node][:, :, node]
    nb.node_valid = bank.node_valid[:, node]
    assert all(int(nb.parent_slot[0, j]) <= j for j in range(int(bank.n_dof[0])))  # parents first
    return nb


def jitter_params(model, seed=0):
    """Zero-initialised biases/heads would make the check trivial; perturb every parameter."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.02)


def run_perm(model, bank, nb, perm):
    b = random_batch(bank, B=8, seed=3)
    bp = dict(b, q_init=b["q_init"][:, perm])
    with torch.no_grad():
        o, op = model(b, bank), model(bp, nb)
    return o, op


@pytest.mark.parametrize("arch,tree_bias", [("kite_trunk", True), ("kite_trunk", False), ("get", True), ("get", False)])
def test_get_joint_tokens_are_permutation_equivariant(setup, arch, tree_bias):
    from getkite.decoder.model import build_model
    bank, _, rel = setup
    torch.manual_seed(0)
    kw = {"joint_encoder": "get"} if arch == "kite_trunk" else {"intent": "raw", "d_model": 64, "n_layers": 2, "n_heads": 4}
    model = build_model(make_cfg(arch=arch, tree_bias=tree_bias, **kw), rel).eval()
    jitter_params(model)
    nb = permute_bank(bank, SWAP, remap_slot=False)  # GET never reads the canonical ids
    o, op = run_perm(model, bank, nb, SWAP)
    nd = int(bank.n_dof[0])
    assert (o["q"][:, :nd] - random_batch(bank, 8, 3)["q_init"][:, :nd]).abs().max() > 1e-4  # not a no-op
    assert torch.allclose(op["q"], o["q"][:, SWAP], atol=1e-5)
    assert torch.allclose(op["t"], o["t"], atol=1e-5)
    assert torch.allclose(op["R"], o["R"], atol=1e-5)


def test_slot_depends_on_joint_identity(setup):
    """Documents the baseline: slot reads which LEAP joint id sits where. Relabelling the index and middle
    fingers (same geometry, swapped ids) changes its prediction; a full consistent remap does not."""
    from getkite.decoder.kite_trunk import KiteTrunkDecoder
    bank, _, rel = setup
    torch.manual_seed(0)
    model = KiteTrunkDecoder(make_cfg(joint_encoder="slot", tree_bias=False), rel).eval()
    jitter_params(model)
    o, op = run_perm(model, bank, permute_bank(bank, SWAP, remap_slot=True), SWAP)
    assert torch.allclose(op["q"], o["q"][:, SWAP], atol=1e-5) and torch.allclose(op["t"], o["t"], atol=1e-5)
    o, op = run_perm(model, bank, permute_bank(bank, SWAP, remap_slot=False), SWAP)
    assert (op["q"] - o["q"][:, SWAP]).abs().max() > 1e-4
