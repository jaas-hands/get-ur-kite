"""Contact-set bridge: inputs (z, start pose only), label frame, exact set matching (permutation invariance, consistent
position/normal pairing), no ground-truth count at inference, random init."""
import importlib.util
import inspect
import itertools
from pathlib import Path
from types import SimpleNamespace

import torch

from getkite.decoder.rotations import euler_xyz_to_matrix
from getkite.decoder.spatial_bridge import ContactSetBridge, SetBridgeConfig, contact_set_label

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/decoder/diagnostics/contact_set_bridge.py"


def _script():
    spec = importlib.util.spec_from_file_location("set_bridge_script", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _batch(n=8, k=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    rec = SimpleNamespace(c_pos=torch.randn(n, k, 3, generator=g) * 0.05,
                          c_normal=torch.nn.functional.normalize(torch.randn(n, k, 3, generator=g), dim=-1),
                          c_valid=torch.rand(n, k, generator=g) > 0.3, c_tpl=torch.randint(0, 100, (n, k), generator=g),
                          t2=torch.randn(n, 3, generator=g), R2=euler_xyz_to_matrix(torch.randn(n, 3, generator=g)),
                          qc2=torch.randn(n, 22, generator=g), r=torch.zeros(n, dtype=torch.long))
    rec.c_valid[:, :2] = True
    sg = SimpleNamespace(R1=euler_xyz_to_matrix(torch.randn(n, 3, generator=g)), t1=torch.randn(n, 3, generator=g) * 0.1,
                         Rg=rec.R2, tg=rec.t2, qcg=rec.qc2)
    return rec, sg


def test_forward_takes_only_z_and_start_pose():
    assert list(inspect.signature(ContactSetBridge.forward).parameters) == ["self", "z", "R0", "t0"]


def test_label_in_start_wrist_frame():
    rec, sg = _batch()
    p, n, v = contact_set_label(rec.c_pos, rec.c_normal, rec.c_valid, sg.R1, sg.t1)
    for i in range(len(p)):
        for k in range(p.shape[1]):
            if v[i, k]:
                torch.testing.assert_close(sg.R1[i] @ p[i, k] + sg.t1[i], rec.c_pos[i, k], rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(sg.R1[i] @ n[i, k], rec.c_normal[i, k], rtol=1e-5, atol=1e-6)
            else:
                assert p[i, k].abs().sum() == 0 and n[i, k].abs().sum() == 0


def test_loss_zero_for_permuted_exact_prediction_and_matching_is_exact():
    torch.manual_seed(0)
    b = ContactSetBridge(SetBridgeConfig(z_dim=8))
    rec, sg = _batch()
    gp, gn, gv = contact_set_label(rec.c_pos, rec.c_normal, rec.c_valid, sg.R1, sg.t1)
    perm = torch.stack([torch.randperm(6) for _ in range(len(gp))])
    take = lambda x: x.gather(1, perm[..., None].expand_as(x)) if x.dim() == 3 else x.gather(1, perm)  # noqa: E731
    big = 30.0
    logit = torch.where(take(gv), big, -big)
    loss, a = b.set_loss(take(gp), take(gn).clone(), logit, gp, gn, gv)
    assert loss["pos"] < 1e-8 and loss["normal"] < 1e-5 and loss["exist"] < 1e-8
    # slot i matched to target perm[i] for every valid target -> positions AND normals paired with the same contact
    assert (a.gather(1, torch.arange(6).expand(len(a), 6)) [take(gv)] == perm[take(gv)]).all()
    # exactness vs brute force on random predictions
    pos, nrm, lg = torch.randn(len(gp), 6, 3) * 0.05, torch.nn.functional.normalize(torch.randn(len(gp), 6, 3), dim=-1), torch.randn(len(gp), 6)
    cost = b.pair_costs(pos, nrm, lg, gp, gn, gv)[0]
    a = b.match(cost)
    for i in range(len(gp)):
        best = min(sum(float(cost[i, s, p[s]]) for s in range(6)) for p in itertools.permutations(range(6)))
        assert abs(sum(float(cost[i, s, a[i, s]]) for s in range(6)) - best) < 1e-4


def test_loss_invariant_to_target_slot_order():
    torch.manual_seed(0)
    b = ContactSetBridge(SetBridgeConfig(z_dim=8))
    rec, sg = _batch()
    gp, gn, gv = contact_set_label(rec.c_pos, rec.c_normal, rec.c_valid, sg.R1, sg.t1)
    pos, nrm, lg = torch.randn(len(gp), 6, 3) * 0.05, torch.nn.functional.normalize(torch.randn(len(gp), 6, 3), dim=-1), torch.randn(len(gp), 6)
    l1 = b.set_loss(pos, nrm, lg, gp, gn, gv)[0]["total"]
    p = torch.randperm(6)
    l2 = b.set_loss(pos, nrm, lg, gp[:, p], gn[:, p], gv[:, p])[0]["total"]
    torch.testing.assert_close(l1, l2)


def test_prediction_inputs_exclude_goal_correspondences_and_count():
    """With z held fixed, goal pose/config, template correspondences, padding contents and the contact count never
    change the prediction; the start pose does."""
    S = _script()
    bridge = ContactSetBridge(SetBridgeConfig(z_dim=16)).eval()
    rec, sg = _batch()
    psi = lambda p, n, v: torch.cat([(p * v[..., None]).sum(1), (n * v[..., None]).sum(1)], -1).repeat(1, 3)[:, :16]  # noqa: E731
    with torch.no_grad():
        base = bridge(*S.predict_inputs(psi, rec, sg))
        rec2, sg2 = _batch(seed=1)
        alt = SimpleNamespace(**{**vars(rec2), "c_pos": rec.c_pos, "c_normal": rec.c_normal, "c_valid": rec.c_valid})
        alt_sg = SimpleNamespace(**{**vars(sg2), "R1": sg.R1, "t1": sg.t1})
        for x, y in zip(bridge(*S.predict_inputs(psi, alt, alt_sg)), base):
            torch.testing.assert_close(x, y)
        const = lambda p, n, v: torch.ones(len(p), 16)  # noqa: E731  contacts (and their count) reach it only via z
        for x, y in zip(bridge(*S.predict_inputs(const, rec, sg)), bridge(*S.predict_inputs(const, rec2, sg))):
            torch.testing.assert_close(x, y)
        assert not torch.allclose(bridge(*S.predict_inputs(psi, rec, sg2))[0], base[0])


def test_random_init():
    torch.manual_seed(0)
    a = ContactSetBridge()
    torch.manual_seed(1)
    b = ContactSetBridge()
    assert not torch.equal(a.head.weight, b.head.weight)
    assert all(p.requires_grad for p in a.parameters())


def test_raw_control_inputs_are_request_and_pose_only():
    from getkite.decoder.spatial_bridge import RawContactSetBridge
    assert list(inspect.signature(RawContactSetBridge.forward).parameters) == ["self", "c_pos", "c_normal", "c_valid", "R0", "t0"]
    src = inspect.getsource(_script().bridge_args)
    assert '"gp"' not in src and '"gv"' not in src and '"gn"' not in src  # labels never passed to either bridge


def test_raw_encoder_permutation_invariant_ignores_padding_and_sees_every_contact():
    from getkite.decoder.spatial_bridge import RawSetEncoder
    torch.manual_seed(0)
    enc = RawSetEncoder(SetBridgeConfig(z_dim=32)).eval()
    rec, _ = _batch()
    with torch.no_grad():
        base = enc(rec.c_pos, rec.c_normal, rec.c_valid)
        p = torch.randperm(6)
        torch.testing.assert_close(enc(rec.c_pos[:, p], rec.c_normal[:, p], rec.c_valid[:, p]), base, rtol=1e-4, atol=1e-5)
        junk_p = torch.where(rec.c_valid[..., None], rec.c_pos, torch.randn_like(rec.c_pos))
        junk_n = torch.where(rec.c_valid[..., None], rec.c_normal, torch.randn_like(rec.c_normal))
        torch.testing.assert_close(enc(junk_p, junk_n, rec.c_valid), base, rtol=1e-4, atol=1e-5)
        for k in range(6):  # moving any valid contact by 2 mm, or flipping its normal, changes the encoding
            sel = rec.c_valid[:, k]
            q = rec.c_pos.clone(); q[:, k, 0] += 0.002
            n = rec.c_normal.clone(); n[:, k] = -n[:, k]
            assert ((enc(q, rec.c_normal, rec.c_valid) - base).abs().amax(-1)[sel] > 1e-5).all()
            assert ((enc(rec.c_pos, n, rec.c_valid) - base).abs().amax(-1)[sel] > 1e-5).all()


def test_copy_reference_equals_label_and_wrist_raw_bridge_uses_it():
    from getkite.decoder.spatial_bridge import RawContactSetBridge, request_to_wrist
    rec, sg = _batch()
    cp, cn = request_to_wrist(rec.c_pos, rec.c_normal, rec.c_valid, sg.R1, sg.t1)
    gp, gn, _ = contact_set_label(rec.c_pos, rec.c_normal, rec.c_valid, sg.R1, sg.t1)
    torch.testing.assert_close(cp, gp); torch.testing.assert_close(cn, gn)
    torch.manual_seed(0)
    b = RawContactSetBridge(SetBridgeConfig(z_dim=32, raw_frame="wrist", raw_fourier=0)).eval()
    with torch.no_grad():  # a rigid change of world frame applied to request AND start pose leaves the output unchanged
        Rw = euler_xyz_to_matrix(torch.tensor([[0.3, -0.5, 1.1]]))[0]; tw = torch.tensor([0.1, -0.2, 0.05])
        out1 = b(rec.c_pos, rec.c_normal, rec.c_valid, sg.R1, sg.t1)[0]
        out2 = b(rec.c_pos @ Rw.T + tw, rec.c_normal @ Rw.T, rec.c_valid, Rw @ sg.R1, sg.t1 @ Rw.T + tw)[0]
    assert b.enc.inp[0].in_features == 6
    # encoder input is world-invariant; the head still also sees the pose, so only check the encoding
    with torch.no_grad():
        e1 = b.enc(*request_to_wrist(rec.c_pos, rec.c_normal, rec.c_valid, sg.R1, sg.t1), rec.c_valid)
        e2 = b.enc(*request_to_wrist(rec.c_pos @ Rw.T + tw, rec.c_normal @ Rw.T, rec.c_valid, Rw @ sg.R1, sg.t1 @ Rw.T + tw), rec.c_valid)
    torch.testing.assert_close(e1, e2, rtol=1e-4, atol=1e-5)


def test_world_target_mode_uses_z_only_and_maps_exactly():
    S = _script()
    torch.manual_seed(0)
    b = ContactSetBridge(SetBridgeConfig(z_dim=16)).eval()
    b.world_target = True
    rec, sg = _batch()
    d = {"z": torch.randn(len(rec.c_pos), 16), "R0": sg.R1, "t0": sg.t1}
    z_, R_, t_ = S.bridge_args(b, d, slice(None))
    assert torch.equal(R_, torch.eye(3).expand_as(sg.R1)) and not t_.any()  # start pose never reaches the network
    with torch.no_grad():
        pw, nw, lg = b(z_, R_, t_)
        pp, pn, pe = S.predict(b, d)
    order = lg.argsort(1, descending=True)
    pw, nw = pw.gather(1, order[..., None].expand_as(pw)), nw.gather(1, order[..., None].expand_as(nw))
    for i in range(len(pp)):  # exact rigid map world -> start-wrist frame
        torch.testing.assert_close(pp[i] @ sg.R1[i].T + sg.t1[i], pw[i], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(pn[i] @ sg.R1[i].T, nw[i], rtol=1e-5, atol=1e-6)


def test_aux_set_metrics_are_per_example():
    """Exact prediction for some examples and a shifted one for others: per-example errors differ (0 for exact)."""
    from getkite.decoder.train_scratch import aux_set_metrics
    rec, sg = _batch()
    gp, gn, gv = contact_set_label(rec.c_pos, rec.c_normal, rec.c_valid, sg.R1, sg.t1)
    perm = torch.randperm(6)
    p, n = gp[:, perm].clone(), gn[:, perm].clone()
    lg = torch.where(gv[:, perm], 30.0, -30.0)
    p[4:] += torch.tensor([0.01, 0.0, 0.0])  # 10 mm shift on examples 4..7
    m = aux_set_metrics((p, n, lg), rec, sg)
    assert m["aux_pos_mm"][:4].abs().max() < 1e-3 and (m["aux_pos_mm"][4:] - 10).abs().max() < 1e-2
    assert m["aux_count_acc"].min() == 1.0
