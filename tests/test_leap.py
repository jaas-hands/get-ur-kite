"""Tests of the LEAP dataloader, KITE's intent encoder and the floating LEAP MJCF.

They need the getyourkite environment (torch), the generated dataset
(third_party/get-ur-kite-dataset/data) and, for psi, KITE's downloaded assets; each test skips
when its inputs are missing.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from getkite.leap import paths  # noqa: E402

NOMINAL = "I4-M4-R4-T4_g0"
LESIONED = "I3-M0-R2-T4_g2"  # 3-joint index, no middle, 2-joint ring, random geometry
needs_data = pytest.mark.skipif(not (paths.hands_dir() / NOMINAL / "hand.json").exists(), reason="dataset not generated")
needs_kite = pytest.mark.skipif(not paths.kite_checkpoint().exists(), reason="KITE assets not downloaded")


def val_hands(n=3):
    from getkite.leap.data import split_hand_ids

    return split_hand_ids("val")[:n]


@needs_data
def test_fk_at_the_label_reproduces_the_stored_contacts():
    """The template points that make each contact, placed by our FK at q_goal, are the stored contacts."""
    from getkite.leap.data import LeapSplit

    split = LeapSplit.load("val", hands=val_hands(), full_template=True, max_examples=600)
    b = split.batch(torch.arange(len(split)), dtype=torch.float64)
    T = split.bank.fk(b["hand"], b["joints_goal"], b["wrist_goal"])
    pos, nrm = split.bank.template_points(b["hand"], T, b["template_idx"].clamp_min(0))
    v = b["contact_valid"]
    assert (pos - b["contact_pos"].double()).norm(dim=-1)[v].max() < 1e-5
    assert (nrm - b["contact_normal"].double()).norm(dim=-1)[v].max() < 1e-4


@needs_data
def test_graph_and_features_match_the_descriptor():
    from getkite.leap.hands import HandBank, graph_matrices

    bank = HandBank.load([NOMINAL, LESIONED])
    for h, hid in enumerate(bank.hand_ids):
        d = json.loads((paths.hands_dir() / hid / "hand.json").read_text())
        dof = [j for j in d["joints"] if not j["locked"]]
        n = d["n_dof"]
        assert int(bank.n_dof[h]) == n == len(dof) and int(bank.dof_valid[h].sum()) == n
        g = graph_matrices([j["parent_dof"] for j in sorted(dof, key=lambda j: j["dof"])])
        assert np.array_equal(bank.spd[h, : n + 1, : n + 1].numpy(), g["spd"])
        assert (bank.spd[h, n + 1 :] == -1).all()
        # finger roots hang off the wrist (node 0) at distance 1
        roots = [j["dof"] + 1 for j in dof if j["parent_dof"] < 0]
        assert all(int(bank.spd[h, 0, r]) == 1 for r in roots)
        assert np.allclose(bank.lower[h, :n].numpy(), [j["lower"] for j in sorted(dof, key=lambda j: j["dof"])])
    assert int(bank.n_dof[1]) == 3 + 0 + 2 + 4


@needs_data
def test_graph_matches_kite_data():
    """Our graph matrices equal kite_data's own Hand.graph() (nominal hand: no geometry draw)."""
    kd = pytest.importorskip("kite_data.build_hands")
    from getkite.leap.hands import HandBank

    d = json.loads((paths.hands_dir() / NOMINAL / "hand.json").read_text())
    build = d.get("build", {})
    hand = kd.make_hand(NOMINAL.split("_g")[0], draw=0, seed=build["seed"], attempt=build["geometry_attempt"])
    g = hand.graph()
    bank = HandBank.load([NOMINAL])
    for key in ("spd", "anc", "desc"):
        assert np.array_equal(getattr(bank, key)[0].numpy(), g[key]), key


@needs_data
def test_batch_contents():
    from getkite.leap.data import LeapSplit
    from getkite.leap.hands import JOINT_FEATURES, MAX_DOF, MAX_NODES

    split = LeapSplit.load("val", hands=val_hands(2), max_examples=64)
    b = split.batch(torch.arange(16))
    B = 16
    assert b["joints_init"].shape == (B, MAX_DOF) and b["joint_features"].shape == (B, MAX_DOF, len(JOINT_FEATURES))
    assert b["hand_points"].shape == (B, 32, 3) and b["spd"].shape == (B, MAX_NODES, MAX_NODES)
    assert not b["mask"][~b["dof_valid"]].any()
    assert torch.allclose(b["hand_normals"].norm(dim=-1), torch.ones(B, 32), atol=1e-4)
    # joint axes stay unit length; padding is zero
    n = b["joint_axis_init"].norm(dim=-1)
    assert torch.allclose(n[b["dof_valid"]], torch.ones_like(n[b["dof_valid"]]), atol=1e-5)
    assert (n[~b["dof_valid"]] == 0).all()
    # the label only moves masked joints
    moved = (b["joints_goal"] - b["joints_init"]).abs() > 1e-7
    assert not (moved & ~b["mask"]).any()


@needs_kite
def test_psi_reproduces_kites_own_latents():
    """Our IntentEncoder gives the latents KITE stored for its Objects Placing test episodes."""
    from getkite.leap.intent import IntentEncoder

    payload = paths.KITE_ROOT / "experiment" / "data" / "high_level_policy" / "object_suite_main_paper_contacts_v1_test.pt"
    if not payload.exists():
        pytest.skip("KITE example data not downloaded")
    d = torch.load(payload, map_location="cpu", weights_only=False)["data"]
    idx = torch.arange(0, 12000, 97)
    contacts, valid, z_kite = d["contact_action"][idx].float(), d["contact_action_valid_mask"][idx].bool(), d["latent_action"][idx].float()
    psi = IntentEncoder.load()
    z = psi(contacts[..., :3], contacts[..., 3:6], valid=valid)
    rel = ((z - z_kite).norm(dim=-1) / z_kite.norm(dim=-1)).max()
    assert rel < 1e-3, float(rel)


@needs_kite
def test_psi_is_permutation_and_padding_invariant():
    from getkite.leap.intent import IntentEncoder

    psi = IntentEncoder.load()
    g = torch.Generator().manual_seed(0)
    pos = torch.rand(8, 6, 3, generator=g) * 0.4
    nrm = torch.nn.functional.normalize(torch.randn(8, 6, 3, generator=g), dim=-1)
    k = torch.full((8,), 4)
    z = psi(pos, nrm, k)
    perm = torch.tensor([2, 0, 3, 1, 4, 5])
    assert torch.allclose(z, psi(pos[:, perm], nrm[:, perm], k), atol=1e-4)
    pos2, nrm2 = pos.clone(), nrm.clone()
    pos2[:, 4:], nrm2[:, 4:] = 9.0, 1.0  # garbage in the padded slots
    assert torch.allclose(z, psi(pos2, nrm2, k), atol=1e-5)


@needs_data
@pytest.mark.parametrize("hand_id", [NOMINAL, LESIONED])
def test_floating_mjcf_matches_our_fk(hand_id, tmp_path):
    """Link frames in the floating MJCF equal the dataset FK for the same wrist pose and joints."""
    mujoco = pytest.importorskip("mujoco")
    from scipy.spatial.transform import Rotation

    from getkite.leap.hands import HandBank
    from getkite.leap.mjcf import VIRTUAL_JOINTS, wrist_to_virtual, write_floating_leap

    path = write_floating_leap(hand_id, tmp_path)
    m = mujoco.MjModel.from_xml_path(str(path))
    bank = HandBank.load([hand_id])
    n = int(bank.n_dof[0])
    assert m.nu == 6 + n
    d = json.loads((paths.hands_dir() / hand_id / "hand.json").read_text())
    names = d["dof_names"]
    rng = np.random.default_rng(0)
    lo, hi = bank.lower[0, :n].numpy(), bank.upper[0, :n].numpy()
    for _ in range(5):
        q = rng.uniform(lo, hi)
        R = Rotation.random(random_state=rng.integers(1 << 30)).as_matrix()
        p = rng.uniform(-0.3, 0.3, 3)
        data = mujoco.MjData(m)
        for name, value in zip(VIRTUAL_JOINTS, wrist_to_virtual(p, R)):
            data.qpos[m.jnt_qposadr[m.joint(name).id]] = value
        for name, value in zip(names, q):
            data.qpos[m.jnt_qposadr[m.joint(name).id]] = value
        mujoco.mj_kinematics(m, data)
        W = np.eye(4); W[:3, :3], W[:3, 3] = R, p
        qq = torch.zeros(1, 16, dtype=torch.float64); qq[0, :n] = torch.as_tensor(q)
        T = bank.fk(torch.tensor([0]), qq, torch.as_tensor(W)[None])[0].numpy()
        links = [lk["name"] for lk in d["links"]]
        checked = 0
        for i, link in enumerate(links):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link)
            if bid < 0:
                continue  # rigidly merged into its parent body by the dataset's MJCF writer
            assert np.allclose(data.xpos[bid], T[i, :3, 3], atol=1e-6), link
            assert np.allclose(data.xmat[bid].reshape(3, 3), T[i, :3, :3], atol=1e-6), link
            checked += 1
        assert checked >= 1 + n
