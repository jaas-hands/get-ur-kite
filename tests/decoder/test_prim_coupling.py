"""Decoder side of coupled primitive bodies (robots.leap_desc reads hand.json "mimic"; 2026-11-21)."""
import json

import numpy as np
import torch

from getkite.decoder.get_scratch import J_MAX
from getkite.decoder.robots import RobotBank, leap_desc
from kite_data.prim_hand import PrimHand, sample_coupling, sample_spec

SEED = 20261119


def _write(tmp_path, hand):
    d = tmp_path / hand.id
    d.mkdir()
    (d / "hand.json").write_text(json.dumps(hand.descriptor()))
    np.savez(d / "template.npz", **hand.build_template(SEED))
    return d


def test_coupled_body_adapter_matches_generator(tmp_path):
    n_checked = 0
    for i in range(30):
        c = sample_coupling(sample_spec(i, SEED), SEED)
        if not c.n_coupled:
            continue
        h = PrimHand(c)
        d = leap_desc(_write(tmp_path, h))
        assert d.coords == [j.name for j in h.dof_joints]
        assert sum(j.mimic is not None for j in d.joints) == c.n_coupled
        bank = RobotBank([d])
        tpl = np.load(tmp_path / h.id / "template.npz")
        g = torch.Generator().manual_seed(i)
        q = h.lower + torch.rand(h.n_dof, generator=g).double().numpy() * (h.upper - h.lower)
        T = h.fk(q)
        Pg = np.stack([T[h.links[l]][:3, :3] @ p + T[h.links[l]][:3, 3] for l, p in zip(tpl["link"], tpl["pos"])])
        qc = torch.zeros(1, J_MAX); qc[0, : h.n_dof] = torch.as_tensor(q)
        r = torch.zeros(1, dtype=torch.long)
        Rr, tr = bank.fk(r, bank.expand(r, qc), torch.eye(3)[None], torch.zeros(1, 3))
        Pr, _ = bank.points(r, Rr, tr, torch.arange(len(Pg))[None])
        assert np.abs(Pr[0].double().numpy() - Pg).max() < 1e-5
        n_checked += 1
    assert n_checked >= 5
