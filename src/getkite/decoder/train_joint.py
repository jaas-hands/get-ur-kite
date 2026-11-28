"""Train J1 (JointContactGET): frozen psi -> world contact-set predictor -> exact wrist transform -> GET motion decoder.

    python -m getkite.decoder.train_joint --config configs/decoder/scratch/j1_joint50k_hf.json [--set k=v ...] [--checks_only]

Same data, seed, start sampling, motion loss, budget and decoder optimiser settings as the corrected raw-contact
decoder (c3_raw50k_cf8_10x_wrist_hf). Contact predictor: architecture/config of the reference world-frame translator
(set_bridge_zworld_c3_20k_b), random init, its optimiser settings (lr 1e-3, clip 1.0) in a separate param group.
Loss = motion residual loss + contact_weight * contact set loss (world-frame labels from the training contacts;
supervision only, never an input). Normal jitter is applied to psi's input only (deployment: psi sees the request).
Before training, gradient/packing checks run and are written to out_dir/checks.json (training aborts on failure).
Checkpoint selection: val macro success, then macro contact error (as the reference decoder).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch

from .get_scratch import ScratchConfig, target_residuals
from .joint_contact_get import JointContactGET, pack_contacts
from .multifamily import BalancedSampler, draw_start, fixed_start
from .spatial_bridge import SetBridgeConfig, contact_set_label
from .train_scratch import (FrozenPsi, RawContacts, build_bank, evaluate, git_state, jitter_normals, load_config,
                            load_pool, residual_loss, sha16, start_noise)

J1_DEFAULTS = dict(contact_weight=1.0, contacts_lr=1e-3, contacts_grad_clip=1.0, contacts_weight_decay=1e-4,
                   base_frame_aug=None, canonical_frame=False, canonical_jitter=None, canonical_origin=False,
                   distill_teachers=None, distill_alpha=0.5, distill_matched=False,
                   contact_fk_weight=0.0, contact_fk_unit_m=0.005,
                   start_wide_mult=None, start_wide_frac=0.5, start_wide_teacher="label")


def draw_start_mix(bank, rb, gen, noise, mult, frac):
    """Wide-start arm (docs/wide_start_proposal.md): per TRAINING example, the start is drawn with KITE's noise (prob 1 - frac)
    or with every noise scale multiplied by mult (prob frac). Both draws use the same generator; the goal fields are identical
    in both (qcg differs only on inactive joints, which follow the start, so it is selected with the rest). Validation,
    checkpoint selection and all evaluation keep KITE's noise. -> (StartGoal, wide mask (B,) bool)."""
    a = draw_start(bank, rb, gen, **noise)
    b = draw_start(bank, rb, gen, **{k: v * mult for k, v in noise.items()})
    wide = (torch.rand(len(rb.r), generator=gen) < frac).to(rb.r.device)
    pick = lambda x, y: torch.where(wide.view(-1, *([1] * (x.dim() - 1))), y, x)  # noqa: E731
    return type(a)(**{k: pick(getattr(a, k), getattr(b, k)) for k in a.__dataclass_fields__}), wide


FK_GRAD_NOTE = ("The contact-FK loss acts on the decoder's predicted pose; its gradient also reaches the contact predictor through "
                "the decoder input (positions / normals), the same path the motion loss already uses. One change, not two.")


def contact_fk_loss(bank, r, out, c_tpl, c_pos, c_valid, family_idx, n_fam, unit_m=0.005):
    """CF-3f-contact: distance between each requested contact and the template point meant to touch it, on the hand posed at
    the predicted goal (FK of out["qc"], out["R"], out["t"] in `bank`). The template correspondences are supervision only, as
    in the benchmark's corr_mm. Joints are clamped to their limits exactly as the benchmark does (train_scratch.pose_metrics),
    so training matches evaluation; out-of-limit joints are corrected by the residual pose loss. Per example: mean over valid
    contacts of |p - c| / unit_m (unit_m = 5 mm, the residual scale); then mean per family, then over families (as
    residual_loss). World geometry is identical in a relabelled / canonical bank, so `bank` may be the training bank.
    Returns (loss, per-example mean distance in mm, fraction of valid coordinates outside limits before clamping), the last
    two detached."""
    qc = out["qc"]
    cl, cu = bank.coords_of(r, bank.lower[r]), bank.coords_of(r, bank.upper[r])
    cv = bank.coord_valid[r]
    viol = ((qc < cl - 1e-6) | (qc > cu + 1e-6)) & cv
    qcc = torch.where(cv, torch.minimum(torch.maximum(qc, cl), cu), 0.0)
    Rs, ts = bank.fk(r, bank.expand(r, qcc), out["R"], out["t"])
    p, _ = bank.points(r, Rs, ts, c_tpl)
    d = torch.sqrt(((p - c_pos.to(p)) ** 2).sum(-1) + 1e-12)
    v = c_valid.to(p.dtype)
    per = (d * v).sum(1) / v.sum(1).clamp_min(1)
    fams = [per[family_idx == f].mean() for f in range(n_fam) if (family_idx == f).any()]
    viol_frac = viol.sum().float() / cv.sum().clamp_min(1)
    return torch.stack(fams).mean() / unit_m, per.detach() * 1000, viol_frac.detach()


def fk_grad_ratio(model, lm, lfk):
    """Motion-loss / contact-FK-loss gradient-norm ratio on the decoder parameters for one batch (no parameter .grad is
    touched: torch.autograd.grad with retain_graph)."""
    params = [p for p in model.decoder.parameters() if p.requires_grad]
    gm = torch.autograd.grad(lm, params, retain_graph=True, allow_unused=True)
    gf = torch.autograd.grad(lfk, params, retain_graph=True, allow_unused=True)
    nm = float(torch.cat([g.flatten() for g in gm if g is not None]).norm())
    nf = float(torch.cat([g.flatten() for g in gf if g is not None]).norm())
    assert nf > 0, "contact-FK loss gives no gradient to the decoder"
    return nm / nf, nm, nf


def fk_weight_calibrate(model, batch_losses, n_batches=200):
    """Predeclared rule for contact_fk_weight = "auto" (CF-3f-contact, revised 2026-11-28 after review): before training on
    a fresh run, at initialisation and with no optimizer step, compute the motion / contact-FK gradient-norm ratio on the
    decoder parameters for `n_batches` batches; weight = median of the per-batch ratios. `batch_losses(i)` returns (lm, lfk)
    for calibration batch i, drawn from its own sampler / generator so the training stream is untouched.
    Returns (weight, list of per-batch ratios, list of (|g_motion|, |g_fk|))."""
    ratios, norms = [], []
    for i in range(n_batches):
        lm, lfk = batch_losses(i)
        q, nm, nf = fk_grad_ratio(model, lm, lfk)
        ratios.append(q); norms.append((nm, nf))
    return float(np.median(ratios)), ratios, norms


def load_teacher(path, dev):
    """A frozen J1 checkpoint as a distillation teacher (as transfer_benchmark.load_system: fp32, eval; CF checkpoints wrapped in
    CanonicalFrame so the teacher predicts in the native frame)."""
    ck = torch.load(path, map_location=dev, weights_only=False)
    assert ck["kind"] == "joint_contact_get" and ck["config"].get("surface_sampling") == "hand_frame"
    m = JointContactGET(ScratchConfig.from_dict(ck["model_config"]), SetBridgeConfig(**ck["set_config"])).to(dev)
    m.load_state_dict(ck["model"]); m.decoder.amp = False
    m.eval().requires_grad_(False)
    if ck["config"].get("canonical_frame"):
        from .canonical import CanonicalFrame
        m = CanonicalFrame(m, origin=bool(ck["config"].get("canonical_origin"))).eval()
    return m


def chordal_mean(Rs, w=None):
    """Chordal L2 mean of rotations; w (optional, len(Rs) x B) = per-example weights (CF-3f matched teachers)."""
    S = sum(Rs) if w is None else sum(wi[:, None, None] * Ri for wi, Ri in zip(w, Rs))
    U, _, Vt = torch.linalg.svd(S)
    d = torch.sign(torch.linalg.det(U @ Vt))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    return U @ D @ Vt


@torch.no_grad()
def teacher_pose(teachers, bank, r, sg, z, match=None):
    """CF-3d (docs/cf3_distillation_proposal.md): ensemble teacher pose in the NATIVE frame for native start sg: mean joint
    coordinates, mean translation, chordal-mean rotation (as scripts/decoder/benchmark/seed_ensemble_eval.py).
    match (n_teachers x n_robots bool, CF-3f matched teachers): per example, average only the teachers whose training set
    contained that example's robot (every robot has >= 1; see teacher_match)."""
    outs = [T(bank, r, sg.qc1, sg.R1, sg.t1, z) for T in teachers]
    if match is not None:
        w = match[:, r].to(outs[0]["t"].dtype)  # T x B
        w = w / w.sum(0, keepdim=True)
        return (sum(wi[:, None] * o["qc"] for wi, o in zip(w, outs)), chordal_mean([o["R"] for o in outs], w),
                sum(wi[:, None] * o["t"] for wi, o in zip(w, outs)))
    return (sum(o["qc"] for o in outs) / len(outs), chordal_mean([o["R"] for o in outs]), sum(o["t"] for o in outs) / len(outs))


def teacher_match(teacher_paths, bank, dev):
    """CF-3f matched teachers: (n_teachers x n_robots) bool, True where the robot is in that teacher checkpoint's training set
    (kite entries by robot name, LEAP / generated-body entries by the npz's hand_id). Robots no teacher trained on (e.g.
    evaluation-only robots) get all teachers, as before."""
    from . import paths as _paths
    M = torch.zeros(len(teacher_paths), len(bank.names), dtype=torch.bool)
    for i, p in enumerate(teacher_paths):
        ck = torch.load(_paths.expand(p), map_location="cpu", weights_only=False)
        for t in ck["config"]["train"]:
            name = t["robot"] if t["kind"] == "kite" else str(np.load(_paths.expand(t["npz"]))["hand_id"])
            if name in bank.index:
                M[i, bank.index[name]] = True
    M[:, ~M.any(0)] = True
    return M.to(dev)


def distill_targets(teachers, bank, bank_tr, r, r_tr, sg_nat, sg_tr, Qb, z, mc, match=None):
    """Normalised residual targets (root, joints) toward the teacher pose, in the training frame: the teacher's goal rotation gets
    the same relabelling as the ground-truth goal (R -> R Q_b; translation unchanged, native origin)."""
    qT, RT, tT = teacher_pose(teachers, bank, r, sg_nat, z, match)
    if Qb is not None:
        RT = RT @ Qb
    return target_residuals(bank_tr, r_tr, sg_tr.qc1, sg_tr.R1, sg_tr.t1, qT, RT, tT, mc)


def frame_aug_bank(bank, k, seed, dev):
    """W2-b base-frame augmentation (docs/w2_wrist_frame_proposal.md). A bank holding every robot under k fixed base-frame
    rotations: block 0 = the identity (the original robots, same indices), blocks 1..k-1 = uniform SO(3) draws (seeded).
    Training remaps an example of robot r to r + b * n_robots and its start / goal wrist rotations R -> R Q_b."""
    from scipy.spatial.transform import Rotation
    from .robots import RobotBank, relabel_base
    Qs = [np.eye(3)] + list(Rotation.random(k - 1, random_state=seed).as_matrix())
    descs = [relabel_base(d, Q) if b else d for b, Q in enumerate(Qs) for d in bank.descs]
    big = RobotBank(descs, device=dev, surface_sampling=bank.surface_sampling)
    assert torch.equal(big.surf_idx[: len(bank.descs)], bank.surf_idx)
    return big, torch.as_tensor(np.stack(Qs), dtype=torch.float32, device=dev)


def canonical_jitter_bank(bank_c, k, max_deg, seed, dev):
    """CF-J (docs/cf_frame_jitter_proposal.md). The canonical bank under k fixed small frame rotations J_b: block 0 = the
    identity (the canonical robots, same indices), blocks 1..k-1 = a rotation about a uniformly random axis by an angle
    uniform in [0, max_deg] (seeded). A robot in block b is in frame Q_r J_b. Training only: training remaps an example of
    robot r to r + b * n_robots and its start / goal wrist rotations R -> R Q_r J_b; evaluation and selection are unjittered."""
    from scipy.spatial.transform import Rotation
    from .robots import RobotBank, relabel_base
    rng = np.random.default_rng(seed)
    Js = [np.eye(3)]
    for _ in range(k - 1):
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        Js.append(Rotation.from_rotvec(ax * np.radians(rng.uniform(0.0, max_deg))).as_matrix())
    descs = [relabel_base(d, J) if b else d for b, J in enumerate(Js) for d in bank_c.descs]
    big = RobotBank(descs, device=dev, surface_sampling=bank_c.surface_sampling)
    assert torch.equal(big.surf_idx[: len(bank_c.descs)], bank_c.surf_idx)
    return big, torch.as_tensor(np.stack(Js), dtype=torch.float32, device=dev)


def world_labels(rb):
    B = len(rb.r)
    return contact_set_label(rb.c_pos, rb.c_normal, rb.c_valid, torch.eye(3, device=rb.c_pos.device).expand(B, 3, 3),
                             torch.zeros(B, 3, device=rb.c_pos.device))


@torch.no_grad()
def contact_eval(model, psi, rec, bs=1024):
    """Internal contact stage on a pool (world frame): matched position/normal error over true contacts, count acc."""
    model.eval()
    pe, ne, ca, cnt = [], [], [], []
    for s in range(0, len(rec), bs):
        rb = rec.index(torch.arange(s, min(s + bs, len(rec)), device=rec.r.device))
        pos, nrm, lg = model.predict_contacts(psi(rb.c_pos, rb.c_normal, rb.c_valid))
        gp, gn, gv = world_labels(rb)
        _, a = model.contacts.set_loss(pos, nrm, lg, gp, gn, gv)
        S = model.contacts.c.slots
        K = gp.shape[1]
        if K < S:
            gp = torch.nn.functional.pad(gp, (0, 0, 0, S - K)); gn = torch.nn.functional.pad(gn, (0, 0, 0, S - K))
            gv = torch.nn.functional.pad(gv, (0, S - K), value=False)
        tp, tn, tv = (x.gather(1, a[..., None].expand_as(x)) if x.dim() == 3 else x.gather(1, a) for x in (gp, gn, gv))
        d = (pos - tp).norm(dim=-1) * 1000
        ang = torch.rad2deg(torch.acos((nrm * tn).sum(-1).clamp(-1, 1)))
        m = tv.float()
        pe.append((d * m).sum(1) / m.sum(1)); ne.append((ang * m).sum(1) / m.sum(1))
        _, v, _ = pack_contacts(pos, nrm, lg)
        cnt.append(v.sum(1).float()); ca.append((v.sum(1) == gv.sum(1)).float())
    model.train()
    c = lambda x: float(torch.cat(x).mean())  # noqa: E731
    return {"contact_pos_mm": c(pe), "contact_normal_deg": c(ne), "count_acc": c(ca), "pred_count_mean": c(cnt),
            "true_count_mean": float(rec.c_valid.sum(1).float().mean())}


def grad_norm(params):
    g = [p.grad.detach().flatten() for p in params if p.grad is not None]
    return float(torch.cat(g).norm()) if g else 0.0


def run_checks(model, psi, bank, train, cfg, mc, noise, dev):
    """Gradient path / packing / frozen-psi checks on one real batch. Returns dict with 'pass'.
    The decoder's root/joint output heads are zero-initialised (as in the reference decoder), so at step 0 the motion
    loss is input-independent and its gradient into the contact predictor is exactly zero by construction. The
    path checks therefore run on a deep COPY whose two final head layers get small random weights; the model that
    is trained is untouched (its motion->contact gradient is logged from the first steps instead)."""
    import copy
    with torch.no_grad():
        zero_heads = all(float(m[-1].weight.abs().max()) == 0.0 for m in (model.decoder.root_head, model.decoder.joint_head))
    model = copy.deepcopy(model)
    g0 = torch.Generator(device="cpu").manual_seed(7)
    with torch.no_grad():
        for m in (model.decoder.root_head, model.decoder.joint_head):
            m[-1].weight.copy_(torch.randn(m[-1].weight.shape, generator=g0) * 1e-2)
    amp0 = getattr(model.decoder, "amp", True)
    model.decoder.amp = False  # fp32 for the finite-difference comparison
    gen = torch.Generator().manual_seed(123)
    sampler = BalancedSampler(train.to("cpu"), bank.family, seed=123)
    rb = train.index(sampler.sample(64).to(dev))
    sg = draw_start(bank, rb, gen, **noise)
    z = psi(rb.c_pos, jitter_normals(rb.c_normal, cfg["normal_noise_rad"], gen), rb.c_valid)
    C = {}
    C["psi_frozen"] = (not any(p.requires_grad for p in psi.parameters())) and not z.requires_grad
    # packing == the two-stage evaluation's packing (bridge_contacts + RawContacts) on the same predictions
    pos, nrm, lg = model.predict_contacts(z)
    zc, v, _ = pack_contacts(pos, nrm, lg)
    o = lg.argsort(1, descending=True)
    p2 = pos.gather(1, o[..., None].expand_as(pos)); n2 = nrm.gather(1, o[..., None].expand_as(nrm))
    v2 = lg.gather(1, o).sigmoid() > 0.5; v2[:, 0] = True
    C["packing_matches_two_stage_max_abs"] = float((zc - RawContacts()(p2, n2, v2)).detach().abs().max())
    # motion-loss gradient reaches the contact predictor through positions/normals; not through existence
    model.zero_grad(set_to_none=True)
    pos.retain_grad(); nrm.retain_grad(); lg.retain_grad()
    zc, v, _ = pack_contacts(pos, nrm, lg)
    out = model.decoder(bank, rb.r, sg.qc1, sg.R1, sg.t1, zc)
    tr_root, tr_j = target_residuals(bank, rb.r, sg.qc1, sg.R1, sg.t1, sg.qcg, sg.Rg, sg.tg, mc)
    fam = sorted(set(bank.family)); fi = torch.tensor([fam.index(f) for f in bank.family], device=dev)
    lm, _, _ = residual_loss(out, tr_root, tr_j, bank.coord_valid[rb.r], fi[rb.r], len(fam))
    lm.backward()
    C["motion_grad_pos_norm"] = float(pos.grad.norm()); C["motion_grad_normal_norm"] = float(nrm.grad.norm())
    C["motion_grad_logit_norm"] = float(lg.grad.norm()) if lg.grad is not None else 0.0
    C["motion_grad_contact_params_norm"] = grad_norm(model.contacts.parameters())
    C["motion_grad_decoder_params_norm"] = grad_norm(model.decoder.parameters())
    # gradient only on valid slots (invalid slots are zeroed in packing)
    vs = v.gather(1, torch.argsort(o, 1))  # validity back in slot order
    C["motion_grad_on_invalid_slots"] = float(pos.grad[~vs].abs().max()) if (~vs).any() else 0.0
    # central directional derivative of the motion loss along the autograd gradient of ALL predicted positions
    # (through transform + packing); a single-coordinate fp32 difference is below the loss's rounding resolution.
    with torch.no_grad():
        gpos = pos.grad.detach().clone()
        d = gpos / gpos.norm()
        eps = 2e-3  # metres along the unit direction
        def L(sgn):
            zz, _, _ = pack_contacts(pos.detach() + sgn * eps * d, nrm.detach(), lg.detach())
            oo = model.decoder(bank, rb.r, sg.qc1, sg.R1, sg.t1, zz)
            return float(residual_loss(oo, tr_root, tr_j, bank.coord_valid[rb.r], fi[rb.r], len(fam))[0].double())
        fd = (L(1) - L(-1)) / (2 * eps)
    C["fd_grad_pos"] = fd; C["autograd_grad_pos"] = float(gpos.norm())  # directional derivative along g = |g|
    # contact loss: gradient to the contact predictor only
    model.zero_grad(set_to_none=True)
    pos, nrm, lg = model.predict_contacts(z)
    L_, _ = model.contacts.set_loss(pos, nrm, lg, *world_labels(rb))
    L_["total"].backward()
    C["contact_grad_decoder_params_norm"] = grad_norm(model.decoder.parameters())
    C["contact_grad_contact_params_norm"] = grad_norm(model.contacts.parameters())
    C["init_losses"] = {"motion": float(lm), **{k: float(v_) for k, v_ in L_.items()}}
    model.zero_grad(set_to_none=True)
    model.decoder.amp = amp0
    rel = abs(C["fd_grad_pos"] - C["autograd_grad_pos"]) / max(1e-8, abs(C["fd_grad_pos"]) + abs(C["autograd_grad_pos"]))
    C["fd_rel_err"] = rel
    C["trained_model_heads_zero_init"] = zero_heads
    C["note"] = "path checks on a copy with small random output heads; trained model unchanged (zero-init heads)"
    C["pass"] = bool(C["psi_frozen"] and C["packing_matches_two_stage_max_abs"] < 1e-6 and C["motion_grad_pos_norm"] > 0
                     and C["motion_grad_contact_params_norm"] > 0 and C["motion_grad_logit_norm"] == 0.0
                     and C["motion_grad_on_invalid_slots"] == 0.0 and C["contact_grad_decoder_params_norm"] == 0.0
                     and rel < 0.1)
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--checks_only", action="store_true")
    a = ap.parse_args()
    cfg = J1_DEFAULTS | load_config(a.config, a.set)
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    bank = build_bank(cfg, dev)
    assert bank.surface_sampling == "hand_frame", "J1 uses the corrected surface sampling"
    families = sorted(set(bank.family))
    fam_index = torch.tensor([families.index(f) for f in bank.family], device=dev)
    train = load_pool(cfg["train"], bank).to(dev)
    val = load_pool(cfg["val"], bank).to(dev)
    extra = {k: load_pool(v, bank).to(dev) for k, v in cfg.get("extra_eval", {}).items()}
    mc = ScratchConfig.from_dict(cfg["model"])
    ref = torch.load(cfg["set_config_from"], map_location="cpu", weights_only=False)  # config only; weights NOT loaded
    scfg = SetBridgeConfig(**ref["config"])
    model = JointContactGET(mc, scfg).to(dev)
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev)
    noise = start_noise(cfg)
    # contact-predictor z standardisation from the training pool (clean normals, as the reference translator)
    with torch.no_grad():
        zs = torch.cat([psi(train.c_pos[s:s + 4096], train.c_normal[s:s + 4096], train.c_valid[s:s + 4096]) for s in range(0, len(train), 4096)])
        model.contacts.set_z_stats(zs)
    n_params = {"contacts": sum(p.numel() for p in model.contacts.parameters()), "decoder": sum(p.numel() for p in model.decoder.parameters())}
    checks = run_checks(model, psi, bank, train, cfg, mc, noise, dev)
    (out / "checks.json").write_text(json.dumps(checks, indent=1))
    print("checks", json.dumps(checks), flush=True)
    assert checks["pass"], "J1 gradient/packing checks failed"
    if a.checks_only:
        return

    val_sg = fixed_start(bank, val, cfg["val_start_seed"], **noise)
    val_sg = type(val_sg)(**{k: getattr(val_sg, k).to(dev) for k in val_sg.__dataclass_fields__})
    extra_sg = {}
    for k, v in extra.items():
        g = fixed_start(bank, v, cfg["val_start_seed"], **noise)
        extra_sg[k] = type(g)(**{f: getattr(g, f).to(dev) for f in g.__dataclass_fields__})
    opt = torch.optim.AdamW([{"params": model.decoder.parameters(), "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]},
                             {"params": model.contacts.parameters(), "lr": cfg["contacts_lr"], "weight_decay": cfg["contacts_weight_decay"]}])
    W, Tt = cfg["warmup_steps"], cfg["total_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / W) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / Tt))))
    step, best = 0, None
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step, best = ck["step"], ck.get("best")
        print(f"resumed from step {step}")
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    data_files = sorted({s["npz"] for s in cfg["train"] + cfg.get("val", []) + sum(cfg.get("extra_eval", {}).values(), [])})
    prov = {"kind": "J1 joint contact->GET", "surface_sampling": bank.surface_sampling,
            "surf_idx_sha16": hashlib.sha256(bank.surf_idx.cpu().numpy().tobytes()).hexdigest()[:16],
            "n_params": n_params, "psi_ckpt_sha16": psi.ckpt_sha, "set_config_from": cfg["set_config_from"],
            "contact_weight": cfg["contact_weight"], "model_repo": git_state(Path(__file__).parents[3]),
            "data_sha16": {f: sha16(f) for f in data_files}, "n_train": len(train), "n_val": len(val),
            "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else None}
    (out / "provenance.json").write_text(json.dumps(prov, indent=1))
    print(f"params {n_params} on {dev}; train {len(train)} / val {len(val)}", flush=True)

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    sampler = BalancedSampler(train.to("cpu"), bank.family, seed=cfg["seed"])
    aug = cfg.get("base_frame_aug")
    if aug:  # training batches use the augmented bank; checks, evaluation and selection use the original (identity) bank
        bank_tr, Qaug = frame_aug_bank(bank, int(aug["k"]), int(aug["seed"]), dev)
        fam_tr = fam_index.repeat(int(aug["k"]))
        n_rob = len(bank.descs)
        print(f"base_frame_aug: {int(aug['k'])} base frames x {n_rob} robots", flush=True)
    else:
        bank_tr, fam_tr = bank, fam_index
    canon = bool(cfg.get("canonical_frame"))
    evm = model  # the model as evaluated / selected
    if canon:  # CF: every robot in its canonical palm frame for training, evaluation and selection
        assert not aug, "canonical_frame and base_frame_aug are separate arms"
        from .canonical import CanonicalFrame
        from .robots import canonical_bank
        corig = bool(cfg.get("canonical_origin"))  # CF-O (docs/canonical_origin_proposal.md)
        bank_tr, Qc, cinfo = canonical_bank(bank, origin=corig)
        Oc = torch.as_tensor([i["o"] for i in cinfo], dtype=Qc.dtype, device=Qc.device) if corig else None
        evm = CanonicalFrame(model, origin=corig)
        evm._cache[(id(bank), tuple(bank.names), str(bank.lower.device))] = (bank_tr, Qc, Oc)
        (out / "canonical_frames.json").write_text(json.dumps({n: {"Q": q.tolist(), **i} for n, q, i in zip(bank.names, Qc.cpu().numpy(), cinfo)}, indent=1))
        print(f"canonical_frame: {len(bank.names)} robots", flush=True)
    jit = cfg.get("canonical_jitter")
    if jit:  # CF-J: training batches see the canonical frame tilted by a small J_b; evaluation / selection use evm (unjittered)
        assert canon and not cfg.get("canonical_origin"), "canonical_jitter needs canonical_frame (and is a separate arm from canonical_origin)"
        n_rob = len(bank.descs)
        bank_tr, Jaug = canonical_jitter_bank(bank_tr, int(jit["k"]), float(jit["max_deg"]), int(jit["seed"]), dev)
        fam_tr = fam_index.repeat(int(jit["k"]))
        ang = np.degrees(np.arccos(np.clip((np.einsum("bii->b", Jaug.cpu().numpy()) - 1) / 2, -1, 1)))
        (out / "canonical_jitter.json").write_text(json.dumps({"J": Jaug.cpu().numpy().tolist(), "angle_deg": ang.tolist(), **jit}, indent=1))
        print(f"canonical_jitter: {int(jit['k'])} frames (angles {ang.min():.1f}-{ang.max():.1f} deg) x {n_rob} robots", flush=True)
    teachers, tmatch = None, None
    if cfg.get("distill_teachers"):  # CF-3d: ensemble distillation (docs/cf3_distillation_proposal.md)
        assert not aug and not jit and not cfg.get("canonical_origin"), "distillation is a separate arm"
        from . import paths as _paths
        teachers = [load_teacher(_paths.expand(t), dev) for t in cfg["distill_teachers"]]
        alpha = float(cfg["distill_alpha"])
        (out / "distill.json").write_text(json.dumps({"teachers": cfg["distill_teachers"], "alpha": alpha}, indent=1))
        print(f"distill: {len(teachers)} teachers, alpha {alpha}", flush=True)
        if cfg.get("distill_matched"):  # CF-3f matched teachers (docs/cf3f_proposal.md)
            tmatch = teacher_match(cfg["distill_teachers"], bank, dev)
            pat = {}
            for j in range(tmatch.shape[1]):
                pat[tuple(tmatch[:, j].tolist())] = pat.get(tuple(tmatch[:, j].tolist()), 0) + 1
            (out / "distill_matched.json").write_text(json.dumps({"patterns": {"".join("1" if b else "0" for b in k): v for k, v in pat.items()}}, indent=1))
            print("distill_matched: teacher patterns over robots", {"".join("1" if b else "0" for b in k): v for k, v in pat.items()}, flush=True)
    wide_mult = cfg.get("start_wide_mult")  # wide-start arm (docs/wide_start_proposal.md); None = KITE's noise only, as CF-3e
    if wide_mult:
        wide_mult = float(wide_mult)
        assert wide_mult > 1 and 0 < float(cfg["start_wide_frac"]) < 1 and cfg["start_wide_teacher"] in ("label", "teacher"), \
            (wide_mult, cfg["start_wide_frac"], cfg["start_wide_teacher"])
        assert not aug and not cfg.get("canonical_jitter"), "the wide-start arm is one change vs CF-3e"
        print(f"wide starts: frac {cfg['start_wide_frac']} at {wide_mult}x KITE noise {noise} (training only; validation and "
              f"selection at KITE noise); teacher term on wide examples: {cfg['start_wide_teacher']}", flush=True)
    fk_cfg = cfg.get("contact_fk_weight") or 0.0
    fk_w = None if fk_cfg == "auto" else float(fk_cfg)
    fk_info = {}
    fk_json = out / "contact_fk.json"
    if fk_cfg == "auto" and fk_json.exists():  # resumed: keep the weight fixed at the value calibrated before step 0
        fk_info = json.loads(fk_json.read_text())
        fk_w = float(fk_info["weight"])
    if fk_cfg:  # CF-3f-contact (everything below is skipped when contact_fk_weight is 0, so CF-3e runs are unchanged)
        assert not aug and not jit, "contact_fk_weight is a separate arm (canonical frame or native frame only)"
        unit_m = float(cfg["contact_fk_unit_m"])
        gbank = bank_tr if canon else bank

        def to_train_frame(sg, r):  # the same transform the training loop applies (canonical frame; CF-O origin)
            if not canon:
                return sg
            Qb = Qc[r]; upd = {"R1": sg.R1 @ Qb, "Rg": sg.Rg @ Qb}
            if Oc is not None:
                Ob = Oc[r][..., None]
                upd |= {"t1": sg.t1 + (sg.R1 @ Ob).squeeze(-1), "tg": sg.tg + (sg.Rg @ Ob).squeeze(-1)}
            return type(sg)(**{**{k: getattr(sg, k) for k in sg.__dataclass_fields__}, **upd})

        if not fk_info:  # fresh run: goal-pose self-check per family, then the weight calibration, both before step 0
            # (1) FK at the GOAL pose must reproduce the requested contacts, in the bank and frame the loss uses. Up to 2 records of
            # EVERY training robot (so every family and every generated body, including those repaired by repair_prim_fps).
            g0 = torch.Generator().manual_seed(0)
            rr = train.r.cpu()
            gi = torch.cat([torch.nonzero(rr == i).flatten()[:2] for i in torch.unique(rr)]).to(dev)
            per_fam = {}
            with torch.no_grad():
                for s in range(0, len(gi), 1024):
                    grb = train.index(gi[s:s + 1024])
                    gsg = to_train_frame(draw_start(bank, grb, g0, **noise), grb.r)
                    _, gmm, _ = contact_fk_loss(gbank, grb.r, {"qc": gsg.qcg, "R": gsg.Rg, "t": gsg.tg}, grb.c_tpl, grb.c_pos,
                                                grb.c_valid, fam_index[grb.r], len(families), unit_m)
                    for f in torch.unique(fam_index[grb.r]).tolist():
                        per_fam.setdefault(families[f], []).append(gmm[fam_index[grb.r] == f].cpu())
            fk_check = {f: {"n": int(torch.cat(v).numel()), "max_mm": float(torch.cat(v).max()), "mean_mm": float(torch.cat(v).mean())}
                        for f, v in per_fam.items()}
            print("contact_fk goal check per family", json.dumps(fk_check), flush=True)
            bad = {f: c for f, c in fk_check.items() if c["max_mm"] >= 0.1}
            assert not bad and set(fk_check) == set(families), f"FK at the goal pose does not reproduce the contacts: {fk_check}"
            fk_info = {"goal_check_per_family": fk_check, "unit_m": unit_m, "clamp": "joints clamped to limits as in pose_metrics",
                       "gradient_path": FK_GRAD_NOTE}

        def fk_calib(seed_off, n):  # (2) median motion / contact-FK gradient-norm ratio over n batches, own RNG streams
            cs = BalancedSampler(train.to("cpu"), bank.family, seed=cfg["seed"] + seed_off)
            cg = torch.Generator().manual_seed(cfg["seed"] + seed_off)

            def batch_losses(_):
                rb = train.index(cs.sample(cfg["batch_size"]).to(dev))
                sg_nat = draw_start(bank, rb, cg, **noise)
                z = psi(rb.c_pos, jitter_normals(rb.c_normal, cfg["normal_noise_rad"], cg), rb.c_valid)
                sg = to_train_frame(sg_nat, rb.r)
                o = model(gbank, rb.r, sg.qc1, sg.R1, sg.t1, z)
                tr_root, tr_j = target_residuals(gbank, rb.r, sg.qc1, sg.R1, sg.t1, sg.qcg, sg.Rg, sg.tg, mc)
                lm_, _, _ = residual_loss(o, tr_root, tr_j, gbank.coord_valid[rb.r], fam_index[rb.r], len(families))
                if teachers is not None:
                    td_root, td_j = distill_targets(teachers, bank, gbank, rb.r, rb.r, sg_nat, sg, Qc[rb.r] if canon else None, z, mc, tmatch)
                    lt_, _, _ = residual_loss(o, td_root, td_j, gbank.coord_valid[rb.r], fam_index[rb.r], len(families))
                    lm_ = (1 - alpha) * lm_ + alpha * lt_
                lf_, _, _ = contact_fk_loss(gbank, rb.r, o, rb.c_tpl, rb.c_pos, rb.c_valid, fam_index[rb.r], len(families), unit_m)
                return lm_, lf_

            with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):  # leave the global RNG streams untouched
                return fk_weight_calibrate(model, batch_losses, n)

        if fk_cfg == "auto" and fk_w is None:
            fk_w, ratios, norms = fk_calib(777, 200)
            fk_info |= {"weight": fk_w, "rule": "median over 200 batches (sampler/generator seed + 777) of the motion / contact-FK "
                        "gradient-norm ratio on decoder params, at initialisation, no optimizer step; fixed for the whole run",
                        "ratios": ratios, "grad_norms_motion_fk": norms, "median": fk_w}
            print(f"contact_fk weight (auto) = {fk_w:.4g} (median of {len(ratios)} ratios, "
                  f"range {min(ratios):.3g}-{max(ratios):.3g})", flush=True)
        elif fk_cfg != "auto":
            fk_info |= {"weight": fk_w, "rule": "fixed in config"}
        fk_json.write_text(json.dumps(fk_info, indent=1))
    log = open(out / "log.jsonl", "a")
    sgen = torch.Generator().manual_seed(cfg["seed"] + 1000 + step)

    def save(path, extra_=None):
        tmp = Path(str(path) + ".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "best": best,
                    "config": cfg, "model_config": mc.to_dict(), "set_config": scfg.to_dict(), "kind": "joint_contact_get"} | (extra_ or {}), tmp)
        os.replace(tmp, path)

    t_last, n_last = time.time(), 0
    model.train()
    while step < Tt and not stop["flag"]:
        idx = sampler.sample(cfg["batch_size"]).to(dev)
        rb = train.index(idx)
        if wide_mult:  # wide-start arm: a mix of KITE-noise and wider starts (training batches only)
            sg, wide = draw_start_mix(bank, rb, sgen, noise, wide_mult, float(cfg["start_wide_frac"]))
        else:
            sg, wide = draw_start(bank, rb, sgen, **noise), None
        z = psi(rb.c_pos, jitter_normals(rb.c_normal, cfg["normal_noise_rad"], sgen), rb.c_valid)
        r_tr = rb.r
        sg_nat, Qb = sg, None
        if aug:  # same physical example, base frame re-expressed by Q_b (world poses unchanged; contacts and z unchanged)
            b = torch.randint(0, len(Qaug), (len(rb.r),), generator=sgen).to(dev)
            Qb = Qaug[b]
            r_tr = rb.r + b * n_rob
            sg = type(sg)(**{**{k: getattr(sg, k) for k in sg.__dataclass_fields__}, "R1": sg.R1 @ Qb, "Rg": sg.Rg @ Qb})
        elif canon and jit:  # same physical example in its robot's canonical frame tilted by J_b (training only)
            b = torch.randint(0, len(Jaug), (len(rb.r),), generator=sgen).to(dev)
            Qb = Qc[rb.r] @ Jaug[b]
            r_tr = rb.r + b * n_rob
            sg = type(sg)(**{**{k: getattr(sg, k) for k in sg.__dataclass_fields__}, "R1": sg.R1 @ Qb, "Rg": sg.Rg @ Qb})
        elif canon:  # same physical example in the canonical palm frame of its robot (CF-O: and its canonical origin)
            Qb = Qc[rb.r]
            upd = {"R1": sg.R1 @ Qb, "Rg": sg.Rg @ Qb}
            if Oc is not None:
                Ob = Oc[rb.r][..., None]
                upd |= {"t1": sg.t1 + (sg.R1 @ Ob).squeeze(-1), "tg": sg.tg + (sg.Rg @ Ob).squeeze(-1)}
            sg = type(sg)(**{**{k: getattr(sg, k) for k in sg.__dataclass_fields__}, **upd})
        out_m = model(bank_tr, r_tr, sg.qc1, sg.R1, sg.t1, z)
        tr_root, tr_j = target_residuals(bank_tr, r_tr, sg.qc1, sg.R1, sg.t1, sg.qcg, sg.Rg, sg.tg, mc)
        lm, lr_, lj_ = residual_loss(out_m, tr_root, tr_j, bank_tr.coord_valid[r_tr], fam_tr[r_tr], len(families))
        lm_gt = lm_t = None
        if teachers is not None:
            td_root, td_j = distill_targets(teachers, bank, bank_tr, rb.r, r_tr, sg_nat, sg, Qb if canon else None, z, mc, tmatch)
            if wide is not None and cfg["start_wide_teacher"] == "label":
                # the teachers only ever trained on KITE-noise starts, so on wide starts their targets are out of distribution:
                # those examples use the ground-truth target in the teacher term too (label-only); KITE-noise examples unchanged
                td_root = torch.where(wide.view(-1, *([1] * (td_root.dim() - 1))), tr_root, td_root)
                td_j = torch.where(wide.view(-1, *([1] * (td_j.dim() - 1))), tr_j, td_j)
            lm_t, _, _ = residual_loss(out_m, td_root, td_j, bank_tr.coord_valid[r_tr], fam_tr[r_tr], len(families))
            lm_gt = lm
            lm = (1 - alpha) * lm_gt + alpha * lm_t
        CL, _ = model.contacts.set_loss(*out_m["contact_set"], *world_labels(rb))
        loss = lm + cfg["contact_weight"] * CL["total"]
        lfk = fk_mm = fk_viol = None
        if fk_cfg:  # CF-3f-contact: FK contact loss on the predicted goal pose (requested contacts, not teacher targets)
            lfk, fk_mm, fk_viol = contact_fk_loss(bank_tr, r_tr, out_m, rb.c_tpl, rb.c_pos, rb.c_valid, fam_tr[r_tr], len(families),
                                                  float(cfg["contact_fk_unit_m"]))
            loss = loss + fk_w * lfk
        log_now = (step + 1) % cfg["log_every"] == 0
        if log_now:  # motion-only gradient reaching the contact predictor (diagnostic, every log step)
            gm = torch.autograd.grad(lm, list(model.contacts.parameters()), retain_graph=True, allow_unused=True)
            g_motion_contacts = float(torch.cat([g.flatten() for g in gm if g is not None]).norm()) if any(g is not None for g in gm) else 0.0
            gc = torch.autograd.grad(CL["total"], list(model.contacts.parameters()), retain_graph=True, allow_unused=True)
            g_contact_contacts = float(torch.cat([g.flatten() for g in gc if g is not None]).norm())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn_d = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), cfg["grad_clip"])
        gn_c = torch.nn.utils.clip_grad_norm_(model.contacts.parameters(), cfg["contacts_grad_clip"])
        opt.step(); sched.step()
        step += 1
        n_last += len(idx)
        if log_now:
            v = out_m["contact_valid"]
            row = {"step": step, "lr": sched.get_last_lr()[0], "lr_contacts": sched.get_last_lr()[1], "train/loss": loss.item(),
                   **({"train/wide_frac": float(wide.float().mean())} if wide is not None else {}),
                   "train/motion": lm.item(), **({"train/motion_gt": lm_gt.item(), "train/motion_teacher": lm_t.item()} if lm_t is not None else {}), "train/root": lr_.mean().item(), "train/joint": lj_.mean().item(),
                   "train/contact": CL["total"].item(), "train/contact_pos": CL["pos"].item(), "train/contact_normal": CL["normal"].item(),
                   "train/contact_exist": CL["exist"].item(), "train/contact_weighted": cfg["contact_weight"] * CL["total"].item(),
                   **({"train/contact_fk": lfk.item(), "train/contact_fk_mm": fk_mm.mean().item(), "train/contact_fk_weighted": fk_w * lfk.item(),
                       "train/limit_violation": float(fk_viol)} if lfk is not None else {}),
                   "grad/decoder": float(gn_d), "grad/contacts": float(gn_c), "grad/contacts_from_motion": g_motion_contacts,
                   "grad/contacts_from_contact": g_contact_contacts,
                   "count/pred_mean": float(v.sum(1).float().mean()), "count/true_mean": float(rb.c_valid.sum(1).float().mean()),
                   "count/acc": float((v.sum(1) == rb.c_valid.sum(1)).float().mean()),
                   "examples_per_s": n_last / (time.time() - t_last),
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0.0}
            log.write(json.dumps(row) + "\n"); log.flush()
            print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in row.items()}), flush=True)
            t_last, n_last = time.time(), 0
        if step % cfg["eval_every"] == 0 or step == Tt:
            res = evaluate(evm, psi, bank, val, val_sg, cfg, families)
            key = (res["all/model/success"], -res["all/model/contact_mm"])
            for k, v_ in extra.items():
                res |= {f"{k}:{m}": x for m, x in evaluate(evm, psi, bank, v_, extra_sg[k], cfg, families).items()}
            res["contacts/val"] = contact_eval(model, psi, val)
            res |= {f"contacts/{k}": contact_eval(model, psi, v_) for k, v_ in extra.items()}
            log.write(json.dumps({"step": step, "eval": res}) + "\n"); log.flush()
            brief = {k: round(x, 3) for k, x in res.items() if isinstance(x, float) and k.split("/")[-1] in ("contact_mm", "success")
                     and "robot_macro" not in k and "/wrist_only/" not in k and "/noop/" not in k}
            print(json.dumps({"step": step, **brief, "contacts/val": res["contacts/val"]}), flush=True)
            if best is None or key > tuple(best["key"]):
                best = {"key": list(key), "step": step}
                save(out / "best.pt", {"selected_on": "val macro success, then macro contact_mm"})
        if fk_cfg == "auto" and step == 10000 and "ratio_10k" not in fk_info:  # diagnostic only; the weight stays fixed
            w10, r10, _ = fk_calib(778, 200)
            fk_info |= {"ratio_10k": {"median": w10, "ratios": r10, "note": "same rule at step 10k, sampler/generator seed + 778; not used"}}
            fk_json.write_text(json.dumps(fk_info, indent=1))
            print(f"contact_fk ratio at 10k (diagnostic) = {w10:.4g}; training weight stays {fk_w:.4g}", flush=True)
        if step % cfg["ckpt_every"] == 0:
            save(out / "last.pt")
    save(out / "last.pt")
    if step >= Tt:
        (out / "DONE").write_text(json.dumps({"best": best, "wall_end": time.time()}))


if __name__ == "__main__":
    main()
