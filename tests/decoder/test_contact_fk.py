"""CF-3f-contact: train_joint.contact_fk_loss / fk_weight_calibrate (2026-11-28). CPU only, generated bodies."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from getkite.decoder.get_scratch import J_MAX
from getkite.decoder.robots import RobotBank, leap_desc, relabel_base
from getkite.decoder.train_joint import J1_DEFAULTS, contact_fk_loss, fk_weight_calibrate

pytest.importorskip("kite_data", reason="needs get-ur-kite-dataset installed (pip install -e ../get-ur-kite-dataset)")
from kite_data.prim_hand import PrimHand, sample_coupling, sample_spec  # noqa: E402

SEED = 20261119
REPO = Path(__file__).resolve().parents[2]
UNIT = J1_DEFAULTS["contact_fk_unit_m"]


def _descs(tmp_path, n=4):
    out = []
    for i in range(n):
        h = PrimHand(sample_coupling(sample_spec(i, SEED), SEED))
        d = tmp_path / h.id; d.mkdir()
        (d / "hand.json").write_text(json.dumps(h.descriptor()))
        np.savez(d / "template.npz", **h.build_template(SEED))
        out.append(leap_desc(d))
    return out


def _example(bank, n_rob, seed=0, B=8, K=6):
    """A batch with goal poses and contacts that the goal pose realises exactly (like a dataset record)."""
    g = torch.Generator().manual_seed(seed)
    r = torch.arange(B) % n_rob
    qc = (torch.rand(B, J_MAX, generator=g) - 0.5) * bank.coord_valid[r]
    R = torch.as_tensor(Rotation.random(B, random_state=seed + 1).as_matrix(), dtype=torch.float32)
    t = (torch.rand(B, 3, generator=g) - 0.5) * 0.8
    n_tpl = bank.points(r[:1], *bank.fk(r[:1], bank.expand(r[:1], qc[:1]), R[:1], t[:1]),
                        torch.zeros(1, 1, dtype=torch.long))[0].shape  # touch points() once (shape check only)
    assert n_tpl[-1] == 3
    c_tpl = torch.randint(0, 4000, (B, K), generator=g)
    k = torch.randint(2, K + 1, (B,), generator=g)
    c_valid = torch.arange(K)[None] < k[:, None]
    c_pos, _ = bank.points(r, *bank.fk(r, bank.expand(r, qc), R, t), c_tpl)
    return r, {"qc": qc, "R": R, "t": t}, c_tpl, c_pos * c_valid[..., None], c_valid


def test_zero_at_goal_pose(tmp_path):
    bank = RobotBank(_descs(tmp_path), surface_sampling="hand_frame")
    r, goal, c_tpl, c_pos, c_valid = _example(bank, 4)
    loss, mm = contact_fk_loss(bank, r, goal, c_tpl, c_pos, c_valid, torch.zeros_like(r), 1, UNIT)
    assert mm.max() < 2e-3 and float(loss) < 1e-3  # 1e-12 inside the sqrt gives a 1e-6 m (0.001 mm) floor


def test_rigid_shift_scale_and_gradient(tmp_path):
    """Shifting the predicted wrist by 3 mm moves every contact by 3 mm: per-example error 3 mm, loss 3/5, and the gradient
    on t points along the shift."""
    bank = RobotBank(_descs(tmp_path), surface_sampling="hand_frame")
    r, goal, c_tpl, c_pos, c_valid = _example(bank, 4, seed=1)
    shift = torch.tensor([0.003, 0.0, 0.0])
    t = (goal["t"] + shift).requires_grad_(True)
    loss, mm = contact_fk_loss(bank, r, {**goal, "t": t}, c_tpl, c_pos, c_valid, torch.zeros_like(r), 1, UNIT)
    assert torch.allclose(mm, torch.full_like(mm, 3.0), atol=1e-3)
    assert abs(loss.item() - 3.0 / 5.0) < 1e-3
    loss.backward()
    gx = t.grad[:, 0]
    assert (gx > 0).all() and t.grad[:, 1:].abs().max() < 1e-3 * gx.abs().min()


def test_gradient_reaches_finger_joints(tmp_path):
    bank = RobotBank(_descs(tmp_path), surface_sampling="hand_frame")
    r, goal, c_tpl, c_pos, c_valid = _example(bank, 4, seed=2)
    qc = (goal["qc"] + 0.05 * bank.coord_valid[r]).requires_grad_(True)
    loss, _ = contact_fk_loss(bank, r, {**goal, "qc": qc}, c_tpl, c_pos, c_valid, torch.zeros_like(r), 1, UNIT)
    loss.backward()
    assert float(qc.grad.abs().sum()) > 0
    assert float((qc.grad * (~bank.coord_valid[r])).abs().sum()) == 0  # padded coordinates get no gradient


def test_canonical_bank_gives_identical_world_loss(tmp_path):
    """A relabelled (canonical) bank with R -> R Q gives the same contact error: the loss is computed in world geometry."""
    descs = _descs(tmp_path)
    Q = torch.as_tensor(Rotation.random(random_state=11).as_matrix(), dtype=torch.float32)
    b0 = RobotBank(descs, surface_sampling="hand_frame")
    b1 = RobotBank([relabel_base(d, Q.numpy()) for d in descs], surface_sampling="hand_frame")
    r, goal, c_tpl, c_pos, c_valid = _example(b0, 4, seed=3)
    off = {"qc": goal["qc"] + 0.03 * b0.coord_valid[r], "R": goal["R"], "t": goal["t"] + 0.002}
    l0, m0 = contact_fk_loss(b0, r, off, c_tpl, c_pos, c_valid, torch.zeros_like(r), 1, UNIT)
    l1, m1 = contact_fk_loss(b1, r, {**off, "R": off["R"] @ Q}, c_tpl, c_pos, c_valid, torch.zeros_like(r), 1, UNIT)
    assert float(m0.mean()) > 0.5  # the perturbation is not trivial
    assert torch.allclose(m0, m1, atol=1e-3) and abs(float(l0 - l1)) < 1e-4


def test_family_average(tmp_path):
    """Loss = mean over families of the per-family mean (as residual_loss), not the batch mean."""
    bank = RobotBank(_descs(tmp_path), surface_sampling="hand_frame")
    r, goal, c_tpl, c_pos, c_valid = _example(bank, 4, seed=4, B=8)
    t = goal["t"].clone(); t[:2, 0] += 0.004; t[2:, 0] += 0.001  # family 0 (2 ex.) 4 mm, family 1 (6 ex.) 1 mm
    fam = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1])
    loss, _ = contact_fk_loss(bank, r, {**goal, "t": t}, c_tpl, c_pos, c_valid, fam, 2, UNIT)
    assert abs(float(loss) - ((4.0 + 1.0) / 2) / 5.0) < 1e-3


def test_auto_weight_calibration_reaches_decoder(tmp_path):
    bank = RobotBank(_descs(tmp_path), surface_sampling="hand_frame")
    r, goal, c_tpl, c_pos, c_valid = _example(bank, 4, seed=5)

    class M(torch.nn.Module):  # stand-in with the attribute the rule uses (model.decoder)
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Linear(3, 3)
            torch.nn.init.normal_(self.decoder.weight, std=0.01)

    m = M()
    dt = m.decoder(torch.randn(len(r), 3, generator=torch.Generator().manual_seed(0))) * 0.01
    out = {**goal, "t": goal["t"] + dt}
    lm = (dt ** 2).sum(1).mean() / UNIT ** 2
    lfk, _ = contact_fk_loss(bank, r, out, c_tpl, c_pos, c_valid, torch.zeros_like(r), 1, UNIT)
    w, nm, nf = fk_weight_calibrate(m, lm, lfk)
    assert nf > 0 and nm > 0 and np.isfinite(w) and w > 0
    assert abs(w - nm / nf) < 1e-9 * max(1.0, w)


def test_weight_zero_is_default_and_cf3e_unchanged():
    """CF-3e runs are untouched: the option defaults to 0 (code path skipped); the contact configs differ from CF-3e only by
    contact_fk_weight (and _note / out_dir)."""
    assert J1_DEFAULTS["contact_fk_weight"] == 0.0
    sc = REPO / "configs/decoder/scratch"
    for sfx in ("", "_s1", "_s2"):
        e = json.loads((sc / f"cf3e_long_hf{sfx}.json").read_text())
        c = json.loads((sc / f"cf3f_contact_hf{sfx}.json").read_text())
        assert not e.get("contact_fk_weight")
        diff = {k for k in set(e) | set(c) if e.get(k) != c.get(k)} - {"_note", "out_dir"}
        assert diff == {"contact_fk_weight"} and c["contact_fk_weight"] == "auto"
