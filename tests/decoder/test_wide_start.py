"""Wide-start arm: train_joint.draw_start_mix (2026-11-28). CPU only, generated bodies."""
import json

import numpy as np
import pytest
import torch

from getkite.decoder.multifamily import Records, _records, draw_start
from getkite.decoder.robots import RobotBank, leap_desc
from getkite.decoder.train_joint import J1_DEFAULTS, draw_start_mix

pytest.importorskip("kite_data", reason="needs get-ur-kite-dataset installed (pip install -e ../get-ur-kite-dataset)")
from kite_data.prim_hand import PrimHand, sample_coupling, sample_spec  # noqa: E402

SEED = 20261119
NOISE = {"xyz_std": 0.005, "rot_std_deg": 5.0, "joint_frac": 0.05}


def _bank(tmp_path, n=3):
    out = []
    for i in range(n):
        h = PrimHand(sample_coupling(sample_spec(i, SEED), SEED))
        d = tmp_path / h.id; d.mkdir()
        (d / "hand.json").write_text(json.dumps(h.descriptor()))
        np.savez(d / "template.npz", **h.build_template(SEED))
        out.append(leap_desc(d))
    return RobotBank(out, surface_sampling="hand_frame")


def _rec(bank, n_per=1500):
    """n_per records per robot with goal joints inside the limits (contacts unused here)."""
    g = torch.Generator().manual_seed(0)
    out = []
    for i in range(len(bank.names)):
        r = torch.full((n_per,), i)
        cv = bank.coord_valid[i]
        D = int(cv.sum())
        cl, cu = bank.coords_of(r, bank.lower[r])[:, :D], bank.coords_of(r, bank.upper[r])[:, :D]
        qc = cl + (0.2 + 0.6 * torch.rand(cl.shape, generator=g)) * (cu - cl)
        K = 3
        out.append(_records(i, (torch.rand(n_per, 3, generator=g) * 0.1).numpy(), (torch.rand(n_per, 3, generator=g) - 0.5).numpy(),
                            qc.numpy(), np.ones((n_per, D), bool), np.zeros((n_per, K, 3), np.float32),
                            np.zeros((n_per, K, 3), np.float32), np.ones((n_per, K), bool), np.zeros((n_per, K), np.int64)))
    return Records.cat(out)


def test_defaults_off():
    assert J1_DEFAULTS["start_wide_mult"] is None  # CF-3e and every existing config: KITE noise only


def test_mix_fraction_scale_and_goal(tmp_path):
    bank = _bank(tmp_path)
    rec = _rec(bank)
    sg, wide = draw_start_mix(bank, rec, torch.Generator().manual_seed(1), NOISE, 3.0, 0.5)
    assert 0.46 < float(wide.float().mean()) < 0.54
    dt = (sg.t1 - rec.t2).norm(dim=-1)
    ratio = float(dt[wide].mean() / dt[~wide].mean())
    assert 2.7 < ratio < 3.3, ratio  # wide starts are ~3x further from the goal (translation)
    assert torch.equal(sg.tg, rec.t2) and torch.equal(sg.Rg, rec.R2)  # goals untouched
    r = rec.r
    cl, cu, cv = bank.coords_of(r, bank.lower[r]), bank.coords_of(r, bank.upper[r]), bank.coord_valid[r]
    assert bool(((sg.qc1 >= cl - 1e-6) & (sg.qc1 <= cu + 1e-6) | ~cv).all())  # joints clamped to limits
    a = draw_start(bank, rec, torch.Generator().manual_seed(1), **NOISE)  # the KITE-noise rows are an ordinary KITE draw
    assert torch.allclose(sg.t1[~wide], a.t1[~wide])
