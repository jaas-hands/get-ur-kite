"""KITE's LEAP (DRO-Grasp URDF, KITE's frames) against the team dataset's nominal hand."""

import json

import numpy as np
import pytest
import torch

from getkite.kite_head import LEAP_URDF, euler_xyz_to_matrix, from_kite_q, matrix_to_euler_xyz, palm_offset, to_kite_q

pytestmark = pytest.mark.skipif(not LEAP_URDF.is_file(), reason="LEAP not installed in KITE (scripts/install_kite_leap.py)")

HID = "I4-M4-R4-T4_g0"


def _random_poses(n, seed=0):
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(seed)
    W = np.tile(np.eye(4), (n, 1, 1))
    W[:, :3, :3] = Rotation.random(n, random_state=seed).as_matrix()
    W[:, :3, 3] = rng.uniform(-0.3, 0.3, (n, 3))
    return torch.tensor(W), rng


def test_euler_round_trip():
    W, _ = _random_poses(64)
    R = W[:, :3, :3]
    assert torch.allclose(euler_xyz_to_matrix(matrix_to_euler_xyz(R)), R, atol=1e-10)


def test_palm_offset_is_dro_base_joint():
    T = palm_offset()
    assert torch.allclose(T[:3, 3], torch.tensor([0.0, 0.038, 0.098], dtype=T.dtype))


def test_kite_chain_matches_dataset_fk():
    pk = pytest.importorskip("pytorch_kinematics")
    from getkite.leap.hands import HandBank
    from getkite.leap.paths import hands_dir

    if not (hands_dir() / HID / "hand.json").is_file():
        pytest.skip("dataset not generated")
    bank = HandBank.load([HID])
    desc = json.loads((hands_dir() / HID / "hand.json").read_text())
    links = [l["name"] if isinstance(l, dict) else l for l in desc["links"]]
    chain = pk.build_chain_from_urdf(LEAP_URDF.read_bytes()).to(dtype=torch.float64)
    assert chain.get_joint_parameter_names()[6:] == desc["dof_names"]

    n = 32
    W, rng = _random_poses(n, seed=1)
    q = torch.tensor(rng.uniform(bank.lower[0].numpy(), bank.upper[0].numpy(), (n, 16)))
    T_palm = palm_offset()
    qk = to_kite_q(W, q, T_palm)
    W2, q2 = from_kite_q(qk, T_palm)
    assert torch.allclose(W2, W, atol=1e-9) and torch.allclose(q2, q)

    frames = chain.forward_kinematics(qk)
    T = bank.fk(torch.zeros(n, dtype=torch.long), q, W)
    for i, link in enumerate(links):
        A = frames[{"pip_4": "thumb_temp_base"}.get(link, link)].get_matrix()
        assert (A[:, :3, 3] - T[:, i, :3, 3]).abs().max() < 2e-4  # DRO rounds origins to 0.1 mm
        assert (A[:, :3, :3] - T[:, i, :3, :3]).abs().max() < 2e-3


def test_kite_frame_mjcf_matches_kite_chain(tmp_path):
    """MuJoCo with qpos = KITE's configuration puts every LEAP link where KITE's robot model does."""
    mujoco = pytest.importorskip("mujoco")
    pk = pytest.importorskip("pytorch_kinematics")
    from getkite.leap.mjcf import write_floating_leap
    from getkite.leap.paths import hands_dir

    if not (hands_dir() / HID / "hand.xml").is_file():
        pytest.skip("dataset not generated")
    m = mujoco.MjModel.from_xml_path(str(write_floating_leap(HID, tmp_path, kite_frame=True)))
    d = mujoco.MjData(m)
    chain = pk.build_chain_from_urdf(LEAP_URDF.read_bytes()).to(dtype=torch.float64)
    names = chain.get_joint_parameter_names()
    rng = np.random.default_rng(2)
    for _ in range(8):
        q = np.zeros(len(names))
        q[:3], q[3:6] = rng.uniform(-0.3, 0.3, 3), rng.uniform(-np.pi, np.pi, 3)
        for i, n in enumerate(names[6:], start=6):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
            lo, hi = m.jnt_range[jid]
            q[i] = rng.uniform(lo, hi)
        for i, n in enumerate(names):
            d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]] = q[i]
        mujoco.mj_kinematics(m, d)
        frames = chain.forward_kinematics(torch.tensor(q)[None])
        for b in range(m.nbody):
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
            link = {"pip_4": "thumb_temp_base"}.get(body, body)
            if link not in frames or body.startswith(("virtual", "base")) or body == "world":
                continue
            A = frames[link].get_matrix()[0].numpy()
            assert np.abs(A[:3, 3] - d.xpos[b]).max() < 2e-4, body
            assert np.abs(A[:3, :3] - d.xmat[b].reshape(3, 3)).max() < 2e-3, body
