"""Core checks. Needs generated data: set KITE_DATA_DIR (kite_data's data/) and KITE_REGIME
(default 'kite'; any regime with a val split works, e.g. a small one for CI)."""

import os

import numpy as np
import pytest
import torch

from getkite.decoder.config import Config
from getkite.decoder.data import SplitData
from getkite.decoder.losses import compute_losses
from getkite.decoder.model import GETDecoder

DATA = os.environ.get("KITE_DATA_DIR", "")
REGIME = os.environ.get("KITE_REGIME", "kite")
SPLIT = os.environ.get("KITE_SPLIT", "val")


def cfg(**kw):
    base = dict(data_dir=DATA, regime=REGIME, d_model=64, n_layers=2, n_heads=4, z_dim=32, psi_width=64,
                psi_layers=1, amp="none")
    return Config(**(base | kw))


@pytest.fixture(scope="module")
def world():
    return SplitData(cfg(frame="world"), SPLIT, max_examples=256)


@pytest.fixture(scope="module")
def local():
    return SplitData(cfg(frame="wrist_init"), SPLIT, max_examples=256)


def _goal_contacts(d, b):
    R, t = d.bank.fk(b["h"], b["q_goal"], b["R_goal"], b["t_goal"])
    return d.bank.points(b["h"], R, t, b["tidx"])


@pytest.mark.parametrize("which", ["world", "local"])
def test_torch_fk_reproduces_stored_contacts(which, request):
    d = request.getfixturevalue(which)
    b = d.batch(torch.arange(d.n))
    x, n = _goal_contacts(d, b)
    v = b["c_valid"]
    assert (x - b["c_pos"]).abs()[v].max() < 2e-5  # metres; stored as float32
    assert (n - b["c_normal"]).abs()[v].max() < 2e-5


def test_slot_fk_matches_kite_data(world):
    hand_mod = pytest.importorskip("kite_data.build_hands")
    import json
    hid = world.bank.hand_ids[0]
    from getkite.decoder.data import data_root
    desc = json.loads((data_root(cfg()) / "hands" / hid / "hand.json").read_text())
    tid, draw = hid.rsplit("_g", 1)
    try:
        hand = hand_mod.make_hand(tid, int(draw), desc["build"]["seed"], desc["build"]["geometry_attempt"])
    except Exception as e:  # third_party assets not fetched
        pytest.skip(str(e))
    rng = np.random.default_rng(0)
    q = hand.lower + rng.random((8, hand.n_dof)) * (hand.upper - hand.lower)
    ref = hand.dof_link_poses(q)  # palm at identity
    h = torch.zeros(8, dtype=torch.long)
    qt = torch.zeros(8, 16)
    qt[:, :hand.n_dof] = torch.tensor(q, dtype=torch.float32)
    R, t = world.bank.fk(h, qt, torch.eye(3).expand(8, 3, 3), torch.zeros(8, 3))
    assert np.abs(t[:, 1:1 + hand.n_dof].numpy() - ref[..., :3, 3]).max() < 1e-6
    assert np.abs(R[:, 1:1 + hand.n_dof].numpy() - ref[..., :3, :3]).max() < 1e-5
    g = hand.graph()
    n = hand.n_dof + 1
    assert np.array_equal(world.bank.spd[0, :n, :n].numpy(), g["spd"])
    assert np.array_equal(world.bank.anc[0, :n, :n].numpy(), g["anc"])


@pytest.mark.parametrize("intent", ["raw", "own_psi"])
def test_untrained_model_predicts_q_init(intent, local):
    torch.manual_seed(0)
    m = GETDecoder(cfg(intent=intent)).eval()
    b = local.batch(torch.arange(64))
    out = m(b, local.bank)
    assert torch.allclose(out["q"], b["q_init"], atol=1e-6)
    assert torch.allclose(out["t"], b["t_init"], atol=1e-6)
    assert torch.allclose(out["R"], b["R_init"], atol=1e-5)


@pytest.mark.parametrize("frame", ["world", "wrist_init"])
def test_batch_composition_and_padding_do_not_matter(frame, world, local):
    d = world if frame == "world" else local
    torch.manual_seed(0)
    m = GETDecoder(cfg(frame=frame))
    for p in m.parameters():  # make the zero-initialised heads non-trivial
        p.data.add_(0.02 * torch.randn_like(p))
    m.eval()
    idx = torch.randperm(d.n)[:32]
    full = m(d.batch(idx), d.bank)
    for i in (0, 7, 31):
        one = m(d.batch(idx[i:i + 1]), d.bank)
        for k in ("q", "t", "R"):
            assert torch.allclose(full[k][i], one[k][0], atol=1e-5), k


@pytest.mark.parametrize("intent", ["raw", "own_psi"])
def test_contact_order_does_not_matter(intent, local):
    torch.manual_seed(0)
    m = GETDecoder(cfg(intent=intent))
    for p in m.parameters():
        p.data.add_(0.02 * torch.randn_like(p))
    m.eval()
    b = local.batch(torch.arange(32))
    perm = dict(b)
    order = torch.tensor([5, 3, 1, 0, 2, 4])
    for k in ("c_pos", "c_normal", "c_valid", "tidx"):
        perm[k] = b[k][:, order]
    a, c = m(b, local.bank), m(perm, local.bank)
    assert torch.allclose(a["q"], c["q"], atol=1e-5) and torch.allclose(a["t"], c["t"], atol=1e-5)


def test_losses_finite_and_differentiable(local):
    m = GETDecoder(cfg())
    b = local.batch(torch.arange(64))
    losses = compute_losses(m(b, local.bank), b, local.bank, m.cfg)
    assert all(torch.isfinite(v) for v in losses.values())
    losses["total"].backward()
    assert m.joint_head[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["euler_xyz", "rotvec"])
def test_resampled_start_pose_follows_kite_noise_model(mode):
    from getkite.decoder.losses import kite_residual
    c = cfg(frame="world", wrist_rot_noise=mode)
    d = SplitData(c, SPLIT, max_examples=4000)
    g = torch.Generator().manual_seed(0)
    b = d.batch(torch.arange(d.n), g)
    stored = d.batch(torch.arange(d.n))
    # goal and contacts are untouched; start pose is new
    assert torch.equal(b["t_goal"], stored["t_goal"]) and not torch.equal(b["t_init"], stored["t_init"])
    assert torch.equal(b["q_goal"][b["mask"]], d.t["joints_contact"][b["mask"]])
    assert torch.equal(b["q_goal"][~b["mask"]], b["q_init"][~b["mask"]])
    r = kite_residual(b["t_goal"], b["R_goal"], b["q_goal"], b, d.bank)  # in units of the noise std
    assert 0.9 < float(r[:, :3].std()) < 1.1  # translation
    from getkite.decoder.rotations import geodesic_angle
    ang = geodesic_angle(b["R_init"], b["R_goal"]) / (5 * torch.pi / 180)  # Maxwell for rotvec: mean 1.60
    assert 1.4 < float(ang.mean()) < 1.8
    jm = b["mask"] & b["dof_valid"]
    lo = d.bank.lower[b["h"]] + 0.15 * d.bank.range[b["h"]]
    hi = d.bank.upper[b["h"]] - 0.15 * d.bank.range[b["h"]]
    inner = jm & (d.t["joints_contact"] > lo) & (d.t["joints_contact"] < hi)  # away from clamping
    assert 0.9 < float(r[:, 9:][inner].std()) < 1.1


def test_holdout_parts():
    import numpy as np

    from getkite.decoder.data import parse_split, part_mask
    hand = np.repeat([0, 1, 2], [10, 7, 1])
    hold, fit = part_mask(hand, 0.1, "holdout"), part_mask(hand, 0.1, "fit")
    assert not (hold & fit).any() and (hold | fit).all()
    assert hold.tolist() == [False] * 9 + [True] + [False] * 6 + [True] + [True]  # last ceil(0.1 n) per hand
    assert part_mask(hand, 0.0, "fit").all() and part_mask(hand, 0.1, "all").all()
    assert parse_split("train@holdout") == ("train", "holdout") and parse_split("val") == ("val", "all")


def test_world_frame_intent_sees_world_contacts_under_wrist_init(world, local):
    """KITE's psi needs world-frame contacts. A wrist_init batch keeps them, and intent_inputs() hands a
    world-frame backend exactly what the world-frame batch would (normals rotated back, incl. jittered ones)."""
    idx = torch.arange(64)
    bw, bl = world.batch(idx), local.batch(idx)
    m = GETDecoder(cfg(frame="wrist_init", intent="raw")).eval()
    pos, nrm, valid = m.intent_inputs(bl)
    assert torch.equal(pos, bl["c_pos"])  # contact-reading backends keep the model frame
    m.intent.needs_world = True
    pos, nrm, valid = m.intent_inputs(bl)
    assert torch.allclose(pos, bw["c_pos"]) and torch.equal(valid, bw["c_valid"])
    assert torch.allclose(nrm, bw["c_normal"], atol=1e-5)
    bl2 = dict(bl, c_normal_in=bl["c_normal"].roll(1, 1))
    bw2 = dict(bw, c_normal_in=bw["c_normal"].roll(1, 1))
    assert torch.allclose(m.intent_inputs(bl2)[1], bw2["c_normal_in"], atol=1e-5)
    mw = GETDecoder(cfg(frame="world", intent="raw")).eval()
    mw.intent.needs_world = True
    assert torch.equal(mw.intent_inputs(bw)[0], bw["c_pos"])


def test_slot_joint_encoder_runs_and_starts_at_q_init(local):
    torch.manual_seed(0)
    m = GETDecoder(cfg(intent="raw", joint_encoder="slot", tree_bias=False)).eval()
    b = local.batch(torch.arange(64))
    out = m(b, local.bank)
    assert torch.allclose(out["q"], b["q_init"], atol=1e-6)
    assert m.slot_embed.shape == (16, 64)
    m.train()
    b |= {"mask": b["mask"]}
    compute_losses(m(b, local.bank), b, local.bank, m.cfg)["total"].backward()
    assert m.slot_embed.grad.abs().sum() > 0


def test_world_intent_decoder_gets_world_start_pose(local, monkeypatch):
    """With a world-frame intent under wrist_init, the wrist token must see the world start pose:
    changing only t_init_world changes the prediction. A contact-reading backend has no such input."""
    from getkite.decoder import intent as I
    b = local.batch(torch.arange(16))
    b2 = dict(b, t_init_world=b["t_init_world"] + 0.1)
    torch.manual_seed(0)
    plain = GETDecoder(cfg(frame="wrist_init", intent="raw")).eval()
    monkeypatch.setattr(I.RawContactIntent, "needs_world", True, raising=False)
    torch.manual_seed(0)
    worldz = GETDecoder(cfg(frame="wrist_init", intent="raw")).eval()
    assert worldz.world_pose_in and not plain.world_pose_in
    with torch.no_grad():
        for m in (plain, worldz):
            for p in m.parameters():
                p.normal_(std=0.05)
        assert torch.equal(plain(b, local.bank)["t"], plain(b2, local.bank)["t"])
        assert (worldz(b, local.bank)["t"] - worldz(b2, local.bank)["t"]).abs().max() > 1e-8
