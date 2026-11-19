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

J1_DEFAULTS = dict(contact_weight=1.0, contacts_lr=1e-3, contacts_grad_clip=1.0, contacts_weight_decay=1e-4)


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
        sg = draw_start(bank, rb, sgen, **noise)
        z = psi(rb.c_pos, jitter_normals(rb.c_normal, cfg["normal_noise_rad"], sgen), rb.c_valid)
        out_m = model(bank, rb.r, sg.qc1, sg.R1, sg.t1, z)
        tr_root, tr_j = target_residuals(bank, rb.r, sg.qc1, sg.R1, sg.t1, sg.qcg, sg.Rg, sg.tg, mc)
        lm, lr_, lj_ = residual_loss(out_m, tr_root, tr_j, bank.coord_valid[rb.r], fam_index[rb.r], len(families))
        CL, _ = model.contacts.set_loss(*out_m["contact_set"], *world_labels(rb))
        loss = lm + cfg["contact_weight"] * CL["total"]
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
                   "train/motion": lm.item(), "train/root": lr_.mean().item(), "train/joint": lj_.mean().item(),
                   "train/contact": CL["total"].item(), "train/contact_pos": CL["pos"].item(), "train/contact_normal": CL["normal"].item(),
                   "train/contact_exist": CL["exist"].item(), "train/contact_weighted": cfg["contact_weight"] * CL["total"].item(),
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
            res = evaluate(model, psi, bank, val, val_sg, cfg, families)
            key = (res["all/model/success"], -res["all/model/contact_mm"])
            for k, v_ in extra.items():
                res |= {f"{k}:{m}": x for m, x in evaluate(model, psi, bank, v_, extra_sg[k], cfg, families).items()}
            res["contacts/val"] = contact_eval(model, psi, val)
            res |= {f"contacts/{k}": contact_eval(model, psi, v_) for k, v_ in extra.items()}
            log.write(json.dumps({"step": step, "eval": res}) + "\n"); log.flush()
            brief = {k: round(x, 3) for k, x in res.items() if isinstance(x, float) and k.split("/")[-1] in ("contact_mm", "success")
                     and "robot_macro" not in k and "/wrist_only/" not in k and "/noop/" not in k}
            print(json.dumps({"step": step, **brief, "contacts/val": res["contacts/val"]}), flush=True)
            if best is None or key > tuple(best["key"]):
                best = {"key": list(key), "step": step}
                save(out / "best.pt", {"selected_on": "val macro success, then macro contact_mm"})
        if step % cfg["ckpt_every"] == 0:
            save(out / "last.pt")
    save(out / "last.pt")
    if step >= Tt:
        (out / "DONE").write_text(json.dumps({"best": best, "wall_end": time.time()}))


if __name__ == "__main__":
    main()
