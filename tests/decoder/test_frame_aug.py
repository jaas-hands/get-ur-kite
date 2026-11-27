"""W2-b base-frame relabelling (robots.relabel_base, train_joint.frame_aug_bank; 2026-11-21)."""
import json

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from getkite.decoder.get_scratch import J_MAX, ScratchConfig, target_residuals
from getkite.decoder.robots import RobotBank, leap_desc, relabel_base
from kite_data.prim_hand import PrimHand, sample_coupling, sample_spec

SEED = 20261119


def _descs(tmp_path, n=4):
    out = []
    for i in range(n):
        h = PrimHand(sample_coupling(sample_spec(i, SEED), SEED))
        d = tmp_path / h.id; d.mkdir()
        (d / "hand.json").write_text(json.dumps(h.descriptor()))
        np.savez(d / "template.npz", **h.build_template(SEED))
        out.append(leap_desc(d))
    return out


def test_relabel_keeps_world_geometry(tmp_path):
    descs = _descs(tmp_path)
    Q = Rotation.random(random_state=3).as_matrix()
    b0 = RobotBank(descs, surface_sampling="hand_frame")
    b1 = RobotBank([relabel_base(d, Q) for d in descs], surface_sampling="hand_frame")
    assert torch.equal(b0.surf_idx, b1.surf_idx)
    g = torch.Generator().manual_seed(0)
    r = torch.arange(len(descs))
    qc = (torch.rand(len(r), J_MAX, generator=g) - 0.5) * b0.coord_valid
    R = torch.as_tensor(Rotation.random(len(r), random_state=1).as_matrix(), dtype=torch.float32)
    t = torch.rand(len(r), 3, generator=g)
    Qt = torch.as_tensor(Q, dtype=torch.float32)
    idx = torch.arange(4000)[None].expand(len(r), -1)
    P0, N0 = b0.points(r, *b0.fk(r, b0.expand(r, qc), R, t), idx)
    P1, N1 = b1.points(r, *b1.fk(r, b1.expand(r, qc), R @ Qt, t), idx)
    assert (P0 - P1).abs().max() < 1e-5 and (N0 - N1).abs().max() < 1e-5


def test_relabel_rotates_wrist_targets(tmp_path):
    """The wrist-frame translation target of the same physical example is Q^T times the original one."""
    descs = _descs(tmp_path, 2)
    Q = Rotation.random(random_state=5).as_matrix(); Qt = torch.as_tensor(Q, dtype=torch.float32)
    b0 = RobotBank(descs, surface_sampling="hand_frame")
    b1 = RobotBank([relabel_base(d, Q) for d in descs], surface_sampling="hand_frame")
    mc = ScratchConfig(frame="wrist", intent_source="raw_contacts")
    r = torch.arange(2)
    qc = torch.zeros(2, J_MAX)
    R0 = torch.as_tensor(Rotation.random(2, random_state=7).as_matrix(), dtype=torch.float32); t0 = torch.rand(2, 3)
    Rg = torch.as_tensor(Rotation.random(2, random_state=8).as_matrix(), dtype=torch.float32); tg = t0 + 0.01 * torch.randn(2, 3)
    a, _ = target_residuals(b0, r, qc, R0, t0, qc, Rg, tg, mc)
    b, _ = target_residuals(b1, r, qc, R0 @ Qt, t0, qc, Rg @ Qt, tg, mc)
    assert torch.allclose(b[:, :3], a[:, :3] @ Qt, atol=1e-5)  # row form of Q^T x
