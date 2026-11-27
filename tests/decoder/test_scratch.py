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


def _rigid(g, B):
    Q = quat_to_matrix(torch.randn(B, 4, generator=g))
    d = torch.randn(B, 3, generator=g) * 0.2
    return Q, d


def _scene(bank, r, qc, R0, t0, g):
    B = len(r)
    pos = t0[:, None] + torch.randn(B, 6, 3, generator=g) * 0.06
    nrm = torch.nn.functional.normalize(torch.randn(B, 6, 3, generator=g), dim=-1)
    valid = torch.rand(B, 6, generator=g) < 0.7
    valid[:, :2] = True
    return pos, nrm, valid


def test_wrist_frame_round_trip_and_untrained_identity():
    """frame=wrist: decode(target_residuals(goal)) == goal; an untrained model returns q_init exactly."""
    from getkite.decoder.train_scratch import RawContacts
    bank, r, qc, R0, t0, _ = _bank_and_batch()
    c = ScratchConfig(intent_source="raw_contacts", z_dim=42, frame="wrist", pos_fourier=8, rot_fourier=8, contact_fourier=0)
    m = GETScratch(c).eval()
    g = torch.Generator().manual_seed(11)
    pos, nrm, valid = _scene(bank, r, qc, R0, t0, g)
    out = m(bank, r, qc, R0, t0, RawContacts()(pos, nrm, valid))
    assert torch.allclose(out["qc"], qc, atol=1e-6) and torch.allclose(out["t"], t0, atol=1e-7)
    assert torch.allclose(out["R"], R0, atol=1e-6)
    qg = torch.where(bank.coord_valid[r], qc + 0.05 * torch.randn(qc.shape, generator=g), 0.0)
    Rg = euler_xyz_to_matrix(torch.randn(len(r), 3, generator=g) * 0.08) @ R0
    tg = t0 + 0.006 * torch.randn(len(r), 3, generator=g)
    rr, rj = target_residuals(bank, r, qc, R0, t0, qg, Rg, tg, c)
    dec = m.decode(bank, r, qc, R0, t0, rr, rj)
    assert torch.allclose(dec["qc"], qg, atol=1e-5)
    assert torch.allclose(dec["t"], tg, atol=1e-6) and torch.allclose(dec["R"], Rg, atol=1e-5)
    # translation residual norm is frame-independent; rotation residual has the same geodesic angle
    cw = ScratchConfig()
    rr_w, _ = target_residuals(bank, r, qc, R0, t0, qg, Rg, tg, cw)
    assert torch.allclose(rr[:, :3].norm(dim=-1), rr_w[:, :3].norm(dim=-1), atol=1e-4)


@pytest.mark.parametrize("cf", [0, 8])
def test_wrist_frame_rigid_transform_consistency(cf):
    """frame=wrist: a rigid transform of the whole scene (contacts, normals, start pose, goal) leaves targets and
    predicted residuals unchanged, and the decoded world goal moves with the scene. The world-frame model does not."""
    from getkite.decoder.train_scratch import RawContacts
    bank, r, qc, R0, t0, _ = _bank_and_batch()
    B = len(r)
    g = torch.Generator().manual_seed(12)
    pos, nrm, valid = _scene(bank, r, qc, R0, t0, g)
    Q, d = _rigid(g, B)
    pos2 = (pos @ Q.transpose(1, 2)) + d[:, None]
    nrm2 = nrm @ Q.transpose(1, 2)
    R02, t02 = Q @ R0, (Q @ t0[..., None]).squeeze(-1) + d
    qg = torch.where(bank.coord_valid[r], qc + 0.05 * torch.randn(qc.shape, generator=g), 0.0)
    Rg = euler_xyz_to_matrix(torch.randn(B, 3, generator=g) * 0.08) @ R0
    tg = t0 + 0.006 * torch.randn(B, 3, generator=g)
    Rg2, tg2 = Q @ Rg, (Q @ tg[..., None]).squeeze(-1) + d
    outs = {}
    for frame in ("wrist", "world"):
        c = ScratchConfig(intent_source="raw_contacts", z_dim=42, frame=frame, pos_fourier=8, rot_fourier=8, contact_fourier=cf)
        torch.manual_seed(0)
        m = GETScratch(c).eval()
        m.amp = False
        _randomise_heads(m)
        a = m(bank, r, qc, R0, t0, RawContacts()(pos, nrm, valid))
        b = m(bank, r, qc, R02, t02, RawContacts()(pos2, nrm2, valid))
        outs[frame] = (a, b)
        if frame == "wrist":
            t1, j1 = target_residuals(bank, r, qc, R0, t0, qg, Rg, tg, c)
            t2, j2 = target_residuals(bank, r, qc, R02, t02, qg, Rg2, tg2, c)
            assert torch.allclose(t1, t2, atol=1e-3) and torch.allclose(j1, j2)
            assert torch.allclose(a["res_root"], b["res_root"], atol=1e-3), (a["res_root"] - b["res_root"]).abs().max()
            assert torch.allclose(a["res_coord"], b["res_coord"], atol=1e-3)
            assert torch.allclose((Q @ a["t"][..., None]).squeeze(-1) + d, b["t"], atol=1e-5)
            assert torch.allclose(Q @ a["R"], b["R"], atol=1e-4)
            assert torch.allclose(a["qc"], b["qc"], atol=1e-4)
    a, b = outs["world"]
    assert not torch.allclose(a["res_root"], b["res_root"], atol=1e-2)  # sanity: the test can detect a frame dependence


def _psi_wrist_cfg(**kw):
    return ScratchConfig(intent_source="psi", z_dim=256, frame="wrist", world_pose_in=True, pos_fourier=8, rot_fourier=8, **kw)


def test_psi_wrist_requires_world_pose_and_no_contact_fourier():
    with pytest.raises(AssertionError):
        GETScratch(ScratchConfig(intent_source="psi", frame="wrist"))
    with pytest.raises(AssertionError):
        GETScratch(_psi_wrist_cfg(contact_fourier=8))
    with pytest.raises(AssertionError):
        GETScratch(ScratchConfig(world_pose_in=True))  # world frame: redundant input, refused


def test_psi_wrist_round_trip_untrained_identity_and_backbone_init():
    """psi + wrist frame: untrained model returns q_init; decode(target) == goal; backbone init equals the raw-contact
    wrist model's (only intent_proj / world_pose_proj / z stats differ)."""
    from getkite.decoder.train_scratch import RawContacts
    bank, r, qc, R0, t0, _ = _bank_and_batch()
    g = torch.Generator().manual_seed(13)
    torch.manual_seed(0)
    m = GETScratch(_psi_wrist_cfg()).eval()
    z = torch.randn(len(r), 256, generator=g)
    out = m(bank, r, qc, R0, t0, z)
    assert torch.allclose(out["qc"], qc, atol=1e-6) and torch.allclose(out["t"], t0, atol=1e-7)
    assert torch.allclose(out["R"], R0, atol=1e-6)
    qg = torch.where(bank.coord_valid[r], qc + 0.05 * torch.randn(qc.shape, generator=g), 0.0)
    Rg = euler_xyz_to_matrix(torch.randn(len(r), 3, generator=g) * 0.08) @ R0
    tg = t0 + 0.006 * torch.randn(len(r), 3, generator=g)
    rr, rj = target_residuals(bank, r, qc, R0, t0, qg, Rg, tg, m.c)
    dec = m.decode(bank, r, qc, R0, t0, rr, rj)
    assert torch.allclose(dec["t"], tg, atol=1e-6) and torch.allclose(dec["R"], Rg, atol=1e-5)
    assert torch.allclose(dec["qc"], qg, atol=1e-5)
    torch.manual_seed(0)
    raw = GETScratch(ScratchConfig(intent_source="raw_contacts", z_dim=42, frame="wrist", pos_fourier=8, rot_fourier=8,
                                   contact_fourier=8)).state_dict()
    sd = m.state_dict()
    only = ("intent_proj", "world_pose_proj", "contact_in", "z_")
    shared = [k for k in sd if k in raw and not k.startswith(only)]
    assert set(sd) - set(shared) == {k for k in sd if k.startswith(only)}
    for k in shared:
        assert torch.equal(sd[k], raw[k]), k


def test_psi_wrist_geometry_is_wrist_relative_and_world_pose_reaches_decoder():
    """With z fixed: a rigid transform of the start pose leaves every body/surface token unchanged (geometry is
    wrist-relative) but changes the root token via world_pose_proj (so z can be related to the hand); with
    world_pose_proj zeroed, residuals are exactly invariant and the decoded goal moves rigidly with the start."""
    bank, r, qc, R0, t0, _ = _bank_and_batch()
    B = len(r)
    g = torch.Generator().manual_seed(14)
    Q, d = _rigid(g, B)
    R02, t02 = Q @ R0, (Q @ t0[..., None]).squeeze(-1) + d
    torch.manual_seed(0)
    m = GETScratch(_psi_wrist_cfg()).eval()
    m.amp = False
    _randomise_heads(m)
    z = torch.randn(B, 256, generator=g)
    seen = {}
    hooks = [m.joint_in.register_forward_hook(lambda mod, i, o: seen.setdefault("joint", []).append(o)),
             m.surf_in.register_forward_hook(lambda mod, i, o: seen.setdefault("surf", []).append(o)),
             m.world_pose_proj.register_forward_hook(lambda mod, i, o: seen.setdefault("wp", []).append(o))]
    a = m(bank, r, qc, R0, t0, z)
    b = m(bank, r, qc, R02, t02, z)
    for h in hooks:
        h.remove()
    assert torch.allclose(seen["joint"][0], seen["joint"][1], atol=1e-4)
    assert torch.allclose(seen["surf"][0], seen["surf"][1], atol=1e-4)
    assert not torch.allclose(seen["wp"][0], seen["wp"][1], atol=1e-3)
    assert not torch.allclose(a["res_root"], b["res_root"], atol=1e-3)
    for p in m.world_pose_proj[-1].parameters():
        torch.nn.init.zeros_(p)
    a = m(bank, r, qc, R0, t0, z)
    b = m(bank, r, qc, R02, t02, z)
    assert torch.allclose(a["res_root"], b["res_root"], atol=1e-4) and torch.allclose(a["res_coord"], b["res_coord"], atol=1e-4)
    assert torch.allclose((Q @ a["t"][..., None]).squeeze(-1) + d, b["t"], atol=1e-5)
    assert torch.allclose(Q @ a["R"], b["R"], atol=1e-4)


def test_decoder_and_psi_inputs_are_deployment_only():
    """Only deployment-time information reaches the decoder: robot description (bank, r), start pose, z.
    psi sees only contact positions/normals/validity; goal poses and contact template correspondences are
    supervision/evaluation data and are not arguments of either."""
    import inspect
    from getkite.decoder.train_scratch import FrozenPsi, RawContacts
    # intent_tokens (optional): contact-encoder tokens computed from the requested contacts (J1), deployment-time as well
    assert list(inspect.signature(GETScratch.forward).parameters) == ["self", "bank", "r", "qc0", "R0", "t0", "z", "intent_tokens"]
    for enc in (FrozenPsi, RawContacts):
        assert list(inspect.signature(enc.forward).parameters) == ["self", "pos", "normal", "valid"]


KITE_CKPT = Path(paths.get("KITE_ROOT")) / "checkpoints" / "decoder" / "release.pt"


def _psi_scene_cfg(**kw):
    if not KITE_CKPT.exists():
        pytest.skip("no KITE release checkpoint")
    return ScratchConfig(intent_source="psi_scene", z_dim=256, frame="wrist", world_pose_in=True, pos_fourier=8,
                         rot_fourier=8, kite_ckpt=str(KITE_CKPT), kite_root=str(paths.get("KITE_ROOT")), **kw)


def test_psi_scene_matches_kite_scene_encoding():
    """Our call of KITE's frozen hand_proj/cls_unpacker/bottleneck_scene_encoder reproduces KITE's own _encode_scene
    (released head, same contacts, same q1, KITE's 32 link-local FPS hand tokens posed in the world)."""
    from getkite.decoder.kite_reference import build_kite_bank
    from getkite.decoder.handbank import kite_local_fps
    from getkite.decoder.train_scratch import FrozenPsi
    bank, r, qc, R0, t0, _ = _bank_and_batch()
    if bank.names[0] != "allegro_right":
        pytest.skip("needs KITE robots")
    g = torch.Generator().manual_seed(21)
    pos, nrm, valid = _scene(bank, r, qc, R0, t0, g)
    psi = FrozenPsi(str(KITE_CKPT), str(paths.get("KITE_ROOT")))
    z = psi(pos, nrm, valid)
    m = GETScratch(_psi_scene_cfg()).eval()
    I = torch.eye(3).expand(len(r), 3, 3)
    Rs, ts = bank.fk(r, bank.expand(r, qc), I, torch.zeros_like(t0))  # wrist-frame FK, as in the model
    hf, zt, _, _ = m.kite_scene(bank, r, Rs, ts, R0, t0, z)
    kite, meta = build_kite_bank(str(KITE_CKPT), str(paths.get("KITE_ROOT")), torch.device("cpu"))
    for i in (0, 1):  # allegro_right, barrett
        head = kite.head_for(bank.names[i])
        fps = torch.as_tensor(kite_local_fps(bank.descs[i].tpl_pos, 32))
        Rw, tw = bank.fk(r[i:i + 1], bank.expand(r[i:i + 1], qc[i:i + 1]), R0[i:i + 1], t0[i:i + 1])
        hp, hn = bank.points(r[i:i + 1], Rw, tw, fps[None])
        v = valid[i:i + 1]
        batch = {"contact_cloud": torch.cat([pos[i:i + 1], nrm[i:i + 1]], -1) * v[..., None], "contact_valid_mask": v,
                 "hand_points_world": hp, "hand_normals_world": hn}
        with torch.no_grad():
            khf, kxyz, kzt, _, _ = head._encode_scene(batch)
        assert torch.allclose(khf, hf[i:i + 1], rtol=1e-4, atol=1e-3), (khf - hf[i:i + 1]).abs().max()
        assert torch.allclose(kzt, zt[i:i + 1], rtol=1e-4, atol=1e-3), (kzt - zt[i:i + 1]).abs().max()


def test_psi_scene_frozen_identity_and_backbone_init():
    """KITE modules stay frozen and in eval mode; untrained model predicts q_init; backbone init equals the raw wrist
    model's (only kite.*, scene_in, world_pose_proj, contact_in and z stats differ)."""
    bank, r, qc, R0, t0, z = _bank_and_batch()
    torch.manual_seed(0)
    m = GETScratch(_psi_scene_cfg())
    m.train()
    assert not m.kite.training and not any(p.requires_grad for p in m.kite.parameters())
    before = {k: v.clone() for k, v in m.kite.state_dict().items()}
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=0.1)
    out = m(bank, r, qc, R0, t0, z)
    assert torch.allclose(out["qc"], qc, atol=1e-6) and torch.allclose(out["t"], t0, atol=1e-7)
    (out["res_root"].sum() + out["res_coord"].sum() + 1.0 * sum(p.sum() for p in m.scene_in.parameters())).backward()
    opt.step()
    for k, v in m.kite.state_dict().items():
        assert torch.equal(v, before[k]), k
    torch.manual_seed(0)
    raw = GETScratch(ScratchConfig(intent_source="raw_contacts", z_dim=42, frame="wrist", pos_fourier=8, rot_fourier=8,
                                   contact_fourier=8)).state_dict()
    torch.manual_seed(0)
    sd = GETScratch(_psi_scene_cfg()).state_dict()
    only = ("kite.", "scene_in", "world_pose_proj", "contact_in", "z_")
    shared = [k for k in sd if k in raw and not k.startswith(only)]
    assert set(sd) - set(shared) == {k for k in sd if k.startswith(only)}
    for k in shared:
        assert torch.equal(sd[k], raw[k]), k


# ---------------------------------------------------------------------------------------------- joint spatial bridge
def test_bridge_requires_psi_wrist_world_pose():
    with pytest.raises(AssertionError):
        GETScratch(ScratchConfig(intent_source="psi", frame="world", bridge=True))


def test_bridge_backbone_init_identical_and_untrained_identity():
    """Bridge built last: every parameter shared with the plain psi-wrist model has the same initial value; the
    untrained joint model still returns q_init (zero heads)."""
    bank, r, qc, R0, t0, _ = _bank_and_batch()
    torch.manual_seed(0)
    plain = GETScratch(_psi_wrist_cfg()).state_dict()
    torch.manual_seed(0)
    m = GETScratch(_psi_wrist_cfg(bridge=True)).eval()
    sd = m.state_dict()
    assert set(plain) <= set(sd) and all(k.startswith(("bridge.", "bridge_tok.")) for k in set(sd) - set(plain))
    for k in plain:
        assert torch.equal(sd[k], plain[k]), k
    z = torch.randn(len(r), 256, generator=torch.Generator().manual_seed(3))
    out = m(bank, r, qc, R0, t0, z)
    assert torch.allclose(out["qc"], qc, atol=1e-6) and torch.allclose(out["t"], t0, atol=1e-7)


def test_bridge_token_uses_predicted_centroid_from_z_and_world_pose_only():
    """out['bridge_c'] = bridge(z, R0w, t0w) (world pose, not the wrist-frame identity); the forward signature has no
    label/contact input; changing the bridge head (hence the predicted centroid) changes the decoder output; and the
    decoder loss back-propagates into the bridge."""
    bank, r, qc, R0, t0, _ = _bank_and_batch()
    torch.manual_seed(0)
    m = GETScratch(_psi_wrist_cfg(bridge=True)).eval()
    _randomise_heads(m)
    z = torch.randn(len(r), 256, generator=torch.Generator().manual_seed(4))
    out = m(bank, r, qc, R0, t0, z)
    torch.testing.assert_close(out["bridge_c"], m.bridge(z, R0, t0))
    with torch.no_grad():
        m.bridge.head.bias.add_(torch.tensor([0.3, -0.2, 0.1]))
    out2 = m(bank, r, qc, R0, t0, z)
    assert not torch.allclose(out2["bridge_c"], out["bridge_c"]) and not torch.allclose(out2["res_root"], out["res_root"])
    m.train()
    m.zero_grad()
    m(bank, r, qc, R0, t0, z)["res_root"].pow(2).sum().backward()
    assert m.bridge.trunk[0].weight.grad is not None and m.bridge.trunk[0].weight.grad.abs().sum() > 0


def test_aux_set_head_is_training_only_readout():
    """Direct decoder aux head: built last (backbone init identical to the plain psi-wrist model); changing it never
    changes the motion outputs (res_root/res_coord/qc/t/R); it returns 6 wrist-frame slots; and the aux loss reaches the
    shared trunk (so it is auxiliary supervision, not an input)."""
    from getkite.decoder.spatial_bridge import SetMatcher, contact_set_label
    bank, r, qc, R0, t0, _ = _bank_and_batch()
    torch.manual_seed(0)
    plain = GETScratch(_psi_wrist_cfg()).state_dict()
    torch.manual_seed(0)
    m = GETScratch(_psi_wrist_cfg(aux_set=True)).eval()
    sd = m.state_dict()
    assert set(sd) - set(plain) and all(k.startswith("aux_head.") for k in set(sd) - set(plain))
    for k in plain:
        assert torch.equal(sd[k], plain[k]), k
    _randomise_heads(m)
    z = torch.randn(len(r), 256, generator=torch.Generator().manual_seed(4))
    out = m(bank, r, qc, R0, t0, z)
    p, n, lg = out["aux_set"]
    assert p.shape == (len(r), 6, 3) and n.shape == (len(r), 6, 3) and lg.shape == (len(r), 6)
    with torch.no_grad():
        for prm in m.aux_head.parameters():
            prm.add_(torch.randn_like(prm))
    out2 = m(bank, r, qc, R0, t0, z)
    for k in ("res_root", "res_coord", "qc", "t", "R"):
        torch.testing.assert_close(out2[k], out[k])
    assert not torch.allclose(out2["aux_set"][0], p)
    m.train()
    m.zero_grad()
    g = torch.Generator().manual_seed(5)
    cp = t0[:, None] + torch.randn(len(r), 6, 3, generator=g) * 0.05
    cn = torch.nn.functional.normalize(torch.randn(len(r), 6, 3, generator=g), dim=-1)
    cv = torch.ones(len(r), 6, dtype=torch.bool)
    L, _ = SetMatcher().set_loss(*m(bank, r, qc, R0, t0, z)["aux_set"], *contact_set_label(cp, cn, cv, R0, t0))
    L["total"].backward()
    assert any(q.grad is not None and q.grad.abs().sum() > 0 for q in m.intent_proj.parameters())
    assert all(p_.grad is None or p_.grad.abs().sum() == 0 for p_ in m.root_head.parameters())
