"""Surface-token sampling regression checks (2026-11-17, docs/decoder_audit.md section 37A).

hand_frame: FPS on template points in ONE hand frame at the zero pose (joints clamped to limits) must give every
finger (subtree under a first-level movable joint holding >= 5% of the template) at least one token and a covering
radius < 45 mm. link_local_legacy must reproduce the original selection exactly (existing checkpoints). Dataset FPS
(LEAP) is unaffected by the policy."""
from pathlib import Path

import numpy as np
import pytest
import torch

from getkite.decoder import paths
from getkite.decoder.robots import N_SURF, RobotBank, _fps, _hand_frame_template, leap_desc, robot_arrays, urdf_desc

KA = Path(paths.get("KITE_ROOT")) / "assets"
HANDS = Path(paths.get("GETKITE_DATASET")) / "hands"
URDFS = {
    "allegro": ("ohra_urdf_robot/allegro/allegro_hand_right_extended.urdf", "ohra_urdf_robot/allegro/surface_template_graph_n4096_ddcc74fe8fec.pt"),
    "barrett": ("dro_urdf_robot/barrett/model_extended.urdf", "dro_urdf_robot/barrett/surface_template_graph_n4096_3e0ba32e061c.pt"),
    "wuji": ("spider_robots/wujihand/right_floating.urdf", "spider_robots/wujihand/surface_template_graph_n4096_998694447d62.pt"),
}


def _desc(name):
    u, t = URDFS[name]
    if not (KA / u).exists():
        pytest.skip("KITE runtime assets not available")
    return urdf_desc(KA / u, KA / t, name, name)


def _hand_points(a):
    return _hand_frame_template(a.parent_slot, a.A, a.axis, a.jtype, a.lower, a.upper, int(a.n_joint), a.tpl_anchor, a.tpl_pos)


def _finger_of_slot(parent_slot):
    root = {0: 0}
    for i, p in enumerate(parent_slot):
        s = i + 1
        root[s] = s if p == 0 else root.get(int(p), int(p))
    return root


@pytest.mark.parametrize("name", list(URDFS))
def test_legacy_policy_unchanged(name):
    d = _desc(name)
    assert np.array_equal(robot_arrays(d).surf_idx, _fps(d.tpl_pos, N_SURF))
    assert np.array_equal(robot_arrays(d, "link_local_legacy").surf_idx, _fps(d.tpl_pos, N_SURF))


@pytest.mark.parametrize("name", list(URDFS))
def test_hand_frame_points_match_torch_fk(name):
    d = _desc(name)
    a = robot_arrays(d, "hand_frame")
    P = _hand_points(a)
    bank = RobotBank([d], surface_sampling="hand_frame")
    q0 = torch.clamp(torch.zeros(1, bank.lower.shape[1], dtype=torch.float64), bank.lower.double(), bank.upper.double()).float()
    Rs, ts = bank.fk(torch.tensor([0]), q0, torch.eye(3)[None], torch.zeros(1, 3))
    Pt = bank.points(torch.tensor([0]), Rs, ts, torch.arange(len(d.tpl_pos))[None])[0][0].numpy()
    assert np.abs(Pt - P).max() < 1e-5


@pytest.mark.parametrize("name", list(URDFS))
def test_hand_frame_covers_every_finger(name):
    d = _desc(name)
    a = robot_arrays(d, "hand_frame")
    P = _hand_points(a)
    finger = _finger_of_slot(a.parent_slot[: int(a.n_joint)])
    f_of_pt = np.array([finger[int(s)] for s in a.tpl_anchor])
    sel = set(f_of_pt[a.surf_idx].tolist())
    for f, cnt in zip(*np.unique(f_of_pt, return_counts=True)):
        if f != 0 and cnt >= 0.05 * len(P):
            assert f in sel, f"{name}: finger subtree at slot {f} ({cnt} template points) has no surface token"
    radius = np.sqrt(((P[:, None] - P[a.surf_idx][None]) ** 2).sum(-1)).min(1).max() * 1000
    assert radius < 45.0, f"{name}: covering radius {radius:.1f} mm"
    legacy = robot_arrays(d).surf_idx
    radius_legacy = np.sqrt(((P[:, None] - P[legacy][None]) ** 2).sum(-1)).min(1).max() * 1000
    assert radius < radius_legacy


def test_dataset_fps_unaffected_by_policy():
    hands = sorted(HANDS.glob("*")) if HANDS.exists() else []
    if not hands:
        pytest.skip("no LEAP hands")
    d = leap_desc(hands[0])
    assert np.array_equal(robot_arrays(d, "hand_frame").surf_idx, robot_arrays(d).surf_idx)
