"""Robot adapters, generic FK and the scratch GET decoder (get_universal_scratch)."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from getkite.decoder.get_scratch import GETScratch, ScratchConfig, target_residuals
from getkite.decoder.robots import J_MAX, RobotBank, leap_desc, urdf_desc
from getkite.decoder.rotations import euler_xyz_to_matrix, quat_to_matrix

from getkite.decoder import paths

KITE = Path(paths.get("KITE_ROOT")) / "assets"  # KITE's runtime assets (download_assets.py --only runtime)
HANDS = Path(paths.get("GETKITE_DATASET")) / "hands"  # get-ur-kite-dataset's generated LEAP hands
ALLEGRO = ("ohra_urdf_robot/allegro/allegro_hand_right_extended.urdf",
           "ohra_urdf_robot/allegro/surface_template_graph_n4096_ddcc74fe8fec.pt", "allegro_right", "allegro", np.pi)
BARRETT = ("dro_urdf_robot/barrett/model_extended.urdf",
           "dro_urdf_robot/barrett/surface_template_graph_n4096_3e0ba32e061c.pt", "barrett", "barrett", 2 * np.pi)
need_kite = pytest.mark.skipif(not (KITE / ALLEGRO[0]).exists(), reason="KITE runtime assets not downloaded")
need_leap = pytest.mark.skipif(not HANDS.exists(), reason="LEAP hands not built")


def kite_robot(spec):
    return urdf_desc(KITE / spec[0], KITE / spec[1], spec[2], spec[3])


def leap(name="I4-M4-R4-T4_g0"):
    return leap_desc(HANDS / name)


@need_kite
@pytest.mark.parametrize("spec", [ALLEGRO, BARRETT])
def test_fk_matches_kite_canonical_template(spec):
    """KITE stores its template posed at q = 0.75 lower + 0.25 upper (virtual base translation 0)."""
    bank = RobotBank([kite_robot(spec)], dtype=torch.float64)
    r = torch.zeros(1, dtype=torch.long)
    qp = (0.75 * bank.lower + 0.25 * bank.upper) * bank.joint_valid
    R0 = euler_xyz_to_matrix(torch.full((1, 3), -0.5 * spec[4], dtype=torch.float64))
    R, t = bank.fk(r, qp, R0, torch.zeros(1, 3, dtype=torch.float64))
    p, n = bank.points(r, R, t, torch.arange(4096)[None])
    tpl = torch.load(KITE / spec[1], weights_only=False)
    assert (p[0] - tpl["surface_template_points_canonical_world"].double()).abs().max() < 2e-4
    assert (n[0] - tpl["surface_template_normals_canonical_world"].double()).abs().max() < 2e-3


@need_kite
def test_barrett_mimic_expansion():
    d = kite_robot(BARRETT)
    bank = RobotBank([d])
    assert int(bank.n_joint[0]) == 8 and int(bank.n_coord[0]) == 4
    r = torch.zeros(1, dtype=torch.long)
    qc = torch.zeros(1, J_MAX)
    qc[0, :4] = torch.tensor([1.0, 0.5, 2.0, 1.5])
    qp = bank.expand(r, qc)
    phys = {n: i for i, n in enumerate(_phys_order(d))}
    v = lambda n: qp[0, phys[n]].item()  # noqa: E731
    assert v("bh_j13_joint") == pytest.approx(0.344262295082 * v("bh_j12_joint"))
    assert v("bh_j21_joint") == pytest.approx(v("bh_j11_joint"))
    assert torch.allclose(bank.coords_of(r, qp), qc)


def _phys_order(d):
    """Physical joint order used by robot_arrays (stable sort by movable depth)."""
    by_child = {j.child: j for j in d.joints}

    def depth(j):
        k, link = 0, j.parent
        while link != d.base_link:
            p = by_child[link]
            k, link = k + (p.type != "fixed"), p.parent
        return k
    return [j.name for j in sorted([j for j in d.joints if j.type != "fixed"], key=depth)]


@need_leap
def test_leap_adapter_matches_handbank_fk():
    from getkite.decoder.handbank import HandBank
    name = "I3-M0-R2-T2_g1"
    hb = HandBank([name], HANDS)
    rb = RobotBank([leap(name)])
    g = torch.Generator().manual_seed(0)
    n = int(hb.n_dof[0])
    q = torch.zeros(1, 16)
    q[0, :n] = hb.lower[0, :n] + torch.rand(n, generator=g) * hb.range[0, :n]
    R0 = quat_to_matrix(torch.randn(1, 4, generator=g))
    t0 = torch.rand(1, 3, generator=g) * 0.8 - 0.4
    h = torch.zeros(1, dtype=torch.long)
    Rh, th = hb.fk(h, q, R0, t0)
    ph, _ = hb.all_points(h, Rh, th)
    qc = torch.zeros(1, J_MAX)
    qc[0, :n] = q[0, :n]
    Rr, tr = rb.fk(h, rb.expand(h, qc), R0, t0)
    pr, _ = rb.points(h, Rr, tr, torch.arange(4096)[None])
    assert (ph - pr).abs().max() < 1e-5


def _bank_and_batch(seed=0):
    descs = []
    if (KITE / ALLEGRO[0]).exists():
        descs += [kite_robot(ALLEGRO), kite_robot(BARRETT)]
    if HANDS.exists():
        descs += [leap("I3-M0-R2-T2_g1"), leap("I4-M4-R4-T4_g0")]
    if not descs:
        pytest.skip("no robot assets")
    bank = RobotBank(descs)
    g = torch.Generator().manual_seed(seed)
    B = len(descs)
    r = torch.arange(B)
    u = torch.rand(B, J_MAX, generator=g)
    qp = bank.lower[r] + u * bank.range[r]
    qc = torch.where(bank.coord_valid[r], bank.coords_of(r, qp), 0.0)
    R0 = quat_to_matrix(torch.randn(B, 4, generator=g))
    t0 = torch.rand(B, 3, generator=g) * 0.8 - 0.4
    z = torch.randn(B, 256, generator=g) * 3
    return bank, r, qc, R0, t0, z


def _randomise_heads(m):
    torch.manual_seed(1)
    for p in list(m.root_head.parameters()) + list(m.joint_head.parameters()):
        torch.nn.init.normal_(p, std=0.1)
    for blk in m.blocks:  # non-zero graph biases so the structure matters
        for e in (blk.bias.spd, blk.bias.anc, blk.bias.desc, blk.bias.pair):
            torch.nn.init.normal_(e.weight, std=0.5)


def test_untrained_predicts_q_init_and_residual_round_trip():
    bank, r, qc, R0, t0, z = _bank_and_batch()
    m = GETScratch(ScratchConfig()).eval()
    out = m(bank, r, qc, R0, t0, z)
    assert torch.allclose(out["qc"], qc, atol=1e-6)
    assert torch.allclose(out["t"], t0, atol=1e-7) and torch.allclose(out["R"], R0, atol=1e-6)
    # decode(target_residuals(goal)) == goal
    g = torch.Generator().manual_seed(3)
    qg = torch.where(bank.coord_valid[r], qc + 0.05 * torch.randn(qc.shape, generator=g), 0.0)
    dR = euler_xyz_to_matrix(torch.randn(len(r), 3, generator=g) * 0.05)
    Rg, tg = dR @ R0, t0 + 0.004 * torch.randn(len(r), 3, generator=g)
    rr, rj = target_residuals(bank, r, qc, R0, t0, qg, Rg, tg, m.c)
    dec = m.decode(bank, r, qc, R0, t0, rr, rj)
    assert torch.allclose(dec["qc"], qg, atol=1e-5)
    assert torch.allclose(dec["t"], tg, atol=1e-6) and torch.allclose(dec["R"], Rg, atol=1e-5)


def test_padding_has_no_effect_and_batching_is_independent():
    bank, r, qc, R0, t0, z = _bank_and_batch()
    m = GETScratch(ScratchConfig()).eval()
    _randomise_heads(m)
    out = m(bank, r, qc, R0, t0, z)
    qc2 = torch.where(bank.coord_valid[r], qc, torch.randn_like(qc) * 5)  # garbage in padded coordinates
    out2 = m(bank, r, qc2, R0, t0, z)
    assert torch.allclose(out["res_coord"], out2["res_coord"], atol=1e-5)
    assert torch.allclose(out["res_root"], out2["res_root"], atol=1e-5)
    assert (out["res_coord"][~bank.coord_valid[r]] == 0).all()
    for i in range(len(r)):  # one robot alone == the same robot inside a mixed batch
        o1 = m(bank, r[i:i + 1], qc[i:i + 1], R0[i:i + 1], t0[i:i + 1], z[i:i + 1])
        assert torch.allclose(o1["res_coord"], out["res_coord"][i:i + 1], atol=1e-4)


@need_leap
def test_joint_permutation_equivariance():
    """Reordering a robot's joints and coordinates (with all graph/geometry data) permutes the outputs."""
    d = leap("I4-M4-R4-T4_g0")
    d2 = copy.deepcopy(d)
    rng = np.random.default_rng(0)
    d2.joints = [d2.joints[i] for i in rng.permutation(len(d2.joints))]
    perm = rng.permutation(len(d.coords))
    d2.coords = [d.coords[i] for i in perm]
    bank = RobotBank([d, d2])
    m = GETScratch(ScratchConfig()).eval()
    _randomise_heads(m)
    g = torch.Generator().manual_seed(0)
    n = len(d.coords)
    qc = torch.zeros(2, J_MAX)
    qc[0, :n] = torch.rand(n, generator=g)
    qc[1, :n] = qc[0, :n][perm]
    R0 = quat_to_matrix(torch.randn(1, 4, generator=g)).expand(2, 3, 3)
    t0 = (torch.rand(1, 3, generator=g) * 0.8 - 0.4).expand(2, 3)
    z = torch.randn(1, 256, generator=g).expand(2, 256)
    out = m(bank, torch.arange(2), qc, R0, t0, z)
    assert torch.allclose(out["res_coord"][1, :n], out["res_coord"][0, :n][perm], atol=1e-4)
    assert torch.allclose(out["res_root"][1], out["res_root"][0], atol=1e-4)


def test_no_learned_robot_or_joint_identities():
    m = GETScratch(ScratchConfig())
    for name, mod in m.named_modules():
        if isinstance(mod, torch.nn.Embedding):  # only graph-structure buckets and the 4 token types
            assert name == "type_emb" or name.split(".")[-1] in ("spd", "anc", "desc", "pair"), name
    for name, p in m.named_parameters():
        assert J_MAX not in p.shape, name  # no per-joint-slot table
    n = sum(p.numel() for p in m.parameters())
    assert 0.5e6 < n < 3e6, n


def test_raw_contacts_intent_masks_invalid():
    """raw_contacts diagnostic: untrained = q_init; invalid contact slots are ignored; valid ones matter."""
    from getkite.decoder.train_scratch import RawContacts
    bank, r, qc, R0, t0, _ = _bank_and_batch()
    m = GETScratch(ScratchConfig(intent_source="raw_contacts", z_dim=42, pos_fourier=8)).eval()
    B = len(r)
    g = torch.Generator().manual_seed(5)
    pos = torch.randn(B, 6, 3, generator=g) * 0.05
    nrm = torch.nn.functional.normalize(torch.randn(B, 6, 3, generator=g), dim=-1)
    valid = torch.zeros(B, 6, dtype=torch.bool)
    valid[:, :3] = True
    z1 = RawContacts()(pos, nrm, valid)
    out = m(bank, r, qc, R0, t0, z1)
    assert torch.allclose(out["qc"], qc, atol=1e-6) and torch.allclose(out["t"], t0, atol=1e-7)
    _randomise_heads(m)
    pos2 = pos.clone()
    pos2[:, 3:] += 1.0
    a, b = m(bank, r, qc, R0, t0, z1), m(bank, r, qc, R0, t0, RawContacts()(pos2, nrm, valid))
    assert torch.allclose(a["res_root"], b["res_root"], atol=1e-5)
    pos3 = pos.clone()
    pos3[:, 0] += 0.01
    c = m(bank, r, qc, R0, t0, RawContacts()(pos3, nrm, valid))
    assert not torch.allclose(a["res_root"], c["res_root"], atol=1e-5)


def test_backbone_init_identical_across_intent_sources():
    """psi vs raw_contacts variants differ only in the intent encoder: same seed -> same backbone weights."""
    sds = []
    for src in ("psi", "raw_contacts"):
        torch.manual_seed(0)
        sds.append(GETScratch(ScratchConfig(intent_source=src, z_dim=256 if src == "psi" else 42,
                                            pos_fourier=8, rot_fourier=8)).state_dict())
    shared = [k for k in sds[0] if k in sds[1] and not k.startswith(("intent_proj", "z_"))]
    assert set(sds[0]) - set(shared) <= {k for k in sds[0] if k.startswith(("intent_proj", "z_"))}
    assert set(sds[1]) - set(shared) == {k for k in sds[1] if k.startswith(("contact_in", "z_"))}
    for k in shared:
        assert torch.equal(sds[0][k], sds[1][k]), k
