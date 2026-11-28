# CF-3f extra arms on Aryan's compute: big300k and contact (predeclared 2026-11-28, before any result)

Two more single-change arms against frozen CF-3e, judged exactly like the other CF-3f arms (`docs/cf3f_proposal.md` on
`getting-kite-universal`). Code and configs: branch `extender-getting-kite-universal` (off `getting-kite-universal` 1244f9e;
merges cleanly into `main`). Each run records the launch commit in its `provenance.json`; every run of an arm is launched from
the same commit. Compute: AWS EC2 Spot (L40S, g6e), run dirs synced to S3. Delivery stays CF-3e seed 0 unless an arm passes.

## The arms

| Arm | Change from CF-3e (nothing else differs; asserted by the config generators) | Configs |
|---|---|---|
| big300k | model 256 wide x 6 layers x 8 heads (as CF-3f-big), `total_steps` 300,000 (500-step warm-up, cosine to 0 at 300k). A clean full-length size test (no warm restart, unlike bigext). | `cf3f_big300k_hf{,_s1,_s2}.json` (`make_cf3f_extra.py`) |
| contact | + forward-kinematics contact loss on the predicted goal pose (`train_joint.contact_fk_loss`), `contact_fk_weight: "auto"`; 100k steps | `cf3f_contact_hf{,_s1,_s2}.json` (`make_cf3f_extra.py`) |

### Contact loss, as implemented

- Per example: mean over valid contacts of the distance between each requested contact and the template point meant to touch
  it, on the hand posed at the predicted goal, divided by 5 mm; mean per family, then over families (as the pose loss).
  Template correspondences are supervision only (as in the benchmark's `corr_mm`).
- Joints are clamped to their limits exactly as the benchmark does (`train_scratch.pose_metrics`), so training matches
  evaluation; out-of-limit joints are corrected by the residual pose loss. The pre-clamp violation rate is logged
  (`train/limit_violation`) and reported (`limit_violation_preclamp`).
- Computed in the canonical training bank and frame; world geometry is identical there (unit-tested).
- Weight ("auto"): before step 0 of a fresh run, at initialisation and with no optimizer step, the motion / contact-FK
  gradient-norm ratio on the decoder parameters over 200 batches drawn from their own sampler and generator (seed + 777, inside
  `torch.random.fork_rng`, so the training stream is untouched); weight = median ratio, fixed for the whole run. All ratios, the
  median and the rule are saved to `contact_fk.json` and reused on resume. The same ratio at 10k steps is logged as a diagnostic
  only.
- Goal-pose self-check before training: FK at the goal pose reproduces the requested contacts to < 0.1 mm on up to 2 records of
  every training robot (every family, every generated body, including bodies repaired by `repair_prim_fps`); per-family
  max/mean saved to `contact_fk.json`. The run stops if it fails.
- The FK loss's gradient also reaches the contact predictor, through the decoder input (positions and normals), the same path the
  motion loss already uses. This is one change, not two.
- With `contact_fk_weight` 0 the code path is skipped. Checked by `scripts/decoder/dlevels/check_contact_weight0.py`: a 200-step
  CPU run of the modified and the unmodified trainer (1244f9e) must give identical logged losses and parameters.
- With distillation at alpha 0.5, the FK term partly overlaps the ground-truth pose term, so a small effect is plausible.

## Readout (fixed now)

`scripts/decoder/benchmark/cf3f_gate.py big300k` and `... contact`: each arm vs CF-3e of the same seed, paired bootstrap over
records within seed, seeds never pooled. CF-3e is re-scored in the same benchmark run on the same hardware
(`transfer_v1_cf3f{big300k,contact}{,_leapval105}.json`). Checkpoints selected on source validation only.

| Check | Rule |
|---|---|
| Q1 source | no significant loss in corr_mm or compat_success on Allegro / Barrett / LEAP validation, any seed |
| Q2 105 held-out LEAP | corr_mm lower than CF-3e in 3/3 seeds, significantly lower in >= 2/3, significantly higher in none |
| Q3 Wuji | corr_mm and wrist_mm not significantly higher in any seed |
| Q4 held-out generated bodies | arm - CF-3e <= +0.10 mm in every seed (training-log eval `prim_heldout:all/model/contact_mm` at each run's selected checkpoint; CF-3e's logs from Anushka's Drive) |
| Q5 Shadow (soft) | for corr_mm and wrist_mm, mean of the 3 per-seed differences <= 0 and significantly higher in <= 1 seed |
| **Q6 wrist (contact arm only)** | wrist_mm and wrist_deg not significantly higher than CF-3e on Allegro / Barrett / LEAP validation and on the 105 held-out LEAP pool, in any seed |

An arm passes only if every applicable check passes (Q1-Q5; Q1-Q6 for contact). Reported, not gating: vs CF2 / D1-bv; wrist_mm
and wrist_deg on every pool; limit violations; the 7 external hands (`external_hands_eval.py --arms cf3e cf3fbig300k
cf3fcontact`); the 360-hand pool if run (L40S has 48 GB). Wuji, Shadow and the external hands are new-family validation, not
test data. The test splits are untouched (Simon scores them). No ensembles.

## Before the full runs

1. Data rebuilt from `docs/cf3e_delivery.md` and checked against CF-3e's `data_sha16`; CF-3e s0 reproduces 8.6146 mm on the
   105-hand pool (`predict_j1.py --check-config`).
2. Unit tests (`tests/decoder/test_contact_fk.py`) and the weight-0 check pass.
3. GPU smoke runs of both arms (finite losses, `contact_fk.json` written, resume from `last.pt` works); speed and memory measured
   with one vs three runs sharing a GPU.
4. Expected total cost reported to Aryan before the six jobs start.
