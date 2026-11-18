"""Substitution eval: predicted contacts reach the decoder in the WORLD frame (the decoder's own wrist preprocessing is
the only transform), with predicted validity (>=1 slot), and no ground-truth count/validity."""
import importlib.util
import inspect
from pathlib import Path

import torch

from getkite.decoder.spatial_bridge import ContactSetBridge, SetBridgeConfig

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/decoder/diagnostics/frozen_decoder_substitution.py"


def _script():
    spec = importlib.util.spec_from_file_location("fds", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_bridge_contacts_world_frame_predicted_validity():
    S = _script()
    torch.manual_seed(0)
    b = ContactSetBridge(SetBridgeConfig(z_dim=16)).eval()
    z = torch.randn(32, 16)
    with torch.no_grad():
        pw, nw, v = S.bridge_contacts(b, z)
        p0, n0, lg = b(z, torch.eye(3).expand(32, 3, 3), torch.zeros(32, 3))
    o = lg.argsort(1, descending=True)
    torch.testing.assert_close(pw, p0.gather(1, o[..., None].expand_as(p0)))  # untransformed (world) slots
    assert v[:, 0].all() and torch.equal(v[:, 1:], lg.gather(1, o)[:, 1:].sigmoid() > 0.5)
    assert list(inspect.signature(S.bridge_contacts).parameters) == ["bridge", "z"]


def test_no_extra_transform_in_script():
    src = SCRIPT.read_text()
    assert "to_wrist" not in src and "request_to_wrist" not in src and "contact_set_label" not in src
