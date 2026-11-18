"""Spatial bridge: label definition and input isolation (target contacts / goal never reach the prediction)."""
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import torch

from getkite.decoder.rotations import euler_xyz_to_matrix
from getkite.decoder.spatial_bridge import BridgeConfig, SpatialBridge, centroid_wrist_label

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/decoder/diagnostics/spatial_bridge.py"


def _script():
    spec = importlib.util.spec_from_file_location("bridge_script", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _batch(n=8, k=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    rec = SimpleNamespace(c_pos=torch.randn(n, k, 3, generator=g) * 0.05, c_normal=torch.nn.functional.normalize(torch.randn(n, k, 3, generator=g), dim=-1),
                          c_valid=torch.rand(n, k, generator=g) > 0.3, c_tpl=torch.randint(0, 100, (n, k), generator=g),
                          t2=torch.randn(n, 3, generator=g), R2=euler_xyz_to_matrix(torch.randn(n, 3, generator=g)),
                          qc2=torch.randn(n, 22, generator=g), r=torch.zeros(n, dtype=torch.long))
    rec.c_valid[:, 0] = True
    sg = SimpleNamespace(R1=euler_xyz_to_matrix(torch.randn(n, 3, generator=g)), t1=torch.randn(n, 3, generator=g) * 0.1,
                         Rg=rec.R2, tg=rec.t2, qcg=rec.qc2)
    return rec, sg


def test_forward_takes_only_z_and_start_pose():
    assert list(inspect.signature(SpatialBridge.forward).parameters) == ["self", "z", "R0", "t0"]


def test_label_is_centroid_in_start_wrist_frame():
    rec, sg = _batch()
    y = centroid_wrist_label(rec.c_pos, rec.c_valid, sg.R1, sg.t1)
    for i in range(len(y)):
        c = rec.c_pos[i][rec.c_valid[i]].mean(0)
        torch.testing.assert_close(sg.R1[i] @ y[i] + sg.t1[i], c, rtol=1e-5, atol=1e-6)


def test_prediction_inputs_exclude_goal_and_correspondences():
    """Changing goal pose/configuration, contact template correspondences, invalid contact slots, or the target
    contacts themselves (with z held fixed) leaves the bridge prediction unchanged."""
    S = _script()
    bridge = SpatialBridge(BridgeConfig(z_dim=16)).eval()
    rec, sg = _batch()
    psi = lambda p, n, v: torch.cat([(p * v[..., None]).sum(1), (n * v[..., None]).sum(1)], -1).repeat(1, 3)[:, :16]  # noqa: E731
    with torch.no_grad():
        base = bridge(*S.predict_inputs(psi, rec, sg))
        rec2, sg2 = _batch(seed=1)  # different goals / correspondences / everything not deployment-visible
        alt = SimpleNamespace(**{**vars(rec2), "c_pos": rec.c_pos, "c_normal": rec.c_normal, "c_valid": rec.c_valid})
        alt_sg = SimpleNamespace(**{**vars(sg2), "R1": sg.R1, "t1": sg.t1})
        torch.testing.assert_close(bridge(*S.predict_inputs(psi, alt, alt_sg)), base)
        junk = SimpleNamespace(**vars(rec))
        junk.c_pos = torch.where(rec.c_valid[..., None], rec.c_pos, torch.randn_like(rec.c_pos))
        torch.testing.assert_close(bridge(*S.predict_inputs(psi, junk, sg)), base)
        const = lambda p, n, v: torch.ones(len(p), 16)  # noqa: E731  contacts reach the bridge only through z
        a = bridge(*S.predict_inputs(const, rec, sg))
        b = bridge(*S.predict_inputs(const, rec2, sg))
        torch.testing.assert_close(a, b)
        # but the start pose does reach it
        assert not torch.allclose(bridge(*S.predict_inputs(psi, rec, sg2)), base)


def test_random_init_no_loaded_weights():
    torch.manual_seed(0)
    a = SpatialBridge()
    torch.manual_seed(1)
    b = SpatialBridge()
    assert not torch.equal(a.head.weight, b.head.weight)
    assert all(p.requires_grad for p in a.parameters())
