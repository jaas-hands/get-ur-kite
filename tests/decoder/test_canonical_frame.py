"""CF canonical palm frame (robots.canonical_frame / canonical_bank, canonical.CanonicalFrame; 2026-11-22)."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from getkite.decoder.canonical import CanonicalFrame
from getkite.decoder.get_scratch import J_MAX, ScratchConfig
from getkite.decoder.robots import RobotBank, canonical_bank, canonical_frame, relabel_base

from test_frame_aug import _descs


def test_frame_is_proper_and_invariant_to_relabelling(tmp_path):
    for d in _descs(tmp_path, 3):
        Q0, info = canonical_frame(d)
        assert np.allclose(Q0.T @ Q0, np.eye(3), atol=1e-9) and np.linalg.det(Q0) > 0
        for s in range(2):
            Rr = Rotation.random(random_state=s).as_matrix()
            Q1, _ = canonical_frame(relabel_base(d, Rr))
            assert np.abs(Rr @ Q1 - Q0).max() < 1e-8  # same physical frame


def test_generated_bodies_close_to_native_convention(tmp_path):
    """Generated bodies are built with x = palmar, y = across, z = distal; the rule should roughly recover it."""
    angs = [np.degrees(Rotation.from_matrix(canonical_frame(d)[0]).magnitude()) for d in _descs(tmp_path, 4)]
    assert np.median(angs) < 30, angs


class _Probe(torch.nn.Module):
    """Records what it was called with; returns a fixed wrist update in the start-wrist frame."""
    def __init__(self):
        super().__init__()
        self.c = ScratchConfig(frame="wrist", intent_source="raw_contacts")

    def forward(self, bank, r, qc0, R0, t0, z):
        self.seen = (bank, R0)
        dR = torch.as_tensor(Rotation.from_rotvec([0.1, -0.2, 0.05]).as_matrix(), dtype=R0.dtype)
        return {"R": R0 @ dR, "t": t0 + (R0 @ torch.tensor([0.01, 0.0, 0.0])[:, None]).squeeze(-1), "qc": qc0, "res_root": torch.zeros(len(r), 9), "res_coord": torch.zeros_like(qc0)}


def test_wrapper_maps_frames_back(tmp_path):
    descs = _descs(tmp_path, 3)
    bank = RobotBank(descs, surface_sampling="hand_frame")
    bc, Q, _ = canonical_bank(bank)
    m = CanonicalFrame(_Probe())
    r = torch.arange(3)
    R0 = torch.as_tensor(Rotation.random(3, random_state=2).as_matrix(), dtype=torch.float32); t0 = torch.zeros(3, 3)
    qc = torch.zeros(3, J_MAX)
    out = m(bank, r, qc, R0, t0, None)
    assert torch.allclose(m.inner.seen[1], R0 @ Q[r], atol=1e-6)
    # the wrist update is applied in the canonical start frame; mapped back, the world pose is R0 Q dR Q^T
    dR = torch.as_tensor(Rotation.from_rotvec([0.1, -0.2, 0.05]).as_matrix(), dtype=torch.float32)
    assert torch.allclose(out["R"], R0 @ Q[r] @ dR @ Q[r].transpose(1, 2), atol=1e-5)
    # world geometry of the canonical bank at R0 Q equals the original at R0
    idx = torch.arange(2000)[None].expand(3, -1)
    P0, _ = bank.points(r, *bank.fk(r, bank.expand(r, qc), R0, t0), idx)
    P1, _ = bc.points(r, *bc.fk(r, bc.expand(r, qc), R0 @ Q[r], t0), idx)
    assert (P0 - P1).abs().max() < 1e-5
