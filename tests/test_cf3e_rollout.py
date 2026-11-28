"""Rotation and input guards for the Oracle-Intent adapter; no checkpoints needed."""
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from getkite.cf3e_rollout import CF3eAdapter, nearby_euler, wrist_matrix


def test_serial_urdf_rotation_order():
    q = np.array([[0., 0., 0., .4, -.7, 1.2]])
    rx = Rotation.from_rotvec([.4, 0, 0]).as_matrix()
    ry = Rotation.from_rotvec([0, -.7, 0]).as_matrix()
    rz = Rotation.from_rotvec([0, 0, 1.2]).as_matrix()
    np.testing.assert_allclose(wrist_matrix(q)[0], rx @ ry @ rz, atol=1e-12)


def test_angles_across_pi_and_alternative_branch():
    previous = np.array([[-3.15, .2, .8], [.4, 2., -1.], [7., -.3, -7.]])
    target = previous + .001
    R = Rotation.from_euler("XYZ", target).as_matrix()
    result = nearby_euler(R, previous)
    np.testing.assert_allclose(result, target, atol=1e-10)
    np.testing.assert_allclose(Rotation.from_euler("XYZ", result).as_matrix(), R, atol=1e-10)


@pytest.mark.parametrize("q", [np.zeros((1, 13)), np.full((1, 14), np.nan)])
def test_bad_state_fails_before_model_call(q):
    adapter = CF3eAdapter.__new__(CF3eAdapter)
    adapter.nj, adapter.width, adapter.device = 8, 24, torch.device("cpu")
    with pytest.raises(ValueError, match="Invalid robot state"):
        adapter.inputs(q)
