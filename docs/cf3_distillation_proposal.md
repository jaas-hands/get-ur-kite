# CF-3 step 1: distil the mixed D1-bv + CF2 ensemble into one CF model (proposal, 2026-11-26)

Status: **approved by the user 2026-11-26 ~03:10 and launched** (jobs 33020 / 33021 / 33023, `cf3d_long_hf{,_s1,_s2}`).
Auto-evaluation: `slurm/cf3d_chain.sh` (benchmarks + readout `scripts/decoder/benchmark/cf3d_gate.py` → `${GETKITE_RUNS}/cf3d_gate.json`,
log `slurm/logs/cf3d_chain.log`) and `slurm/cf3d_external_chain.sh` (external hands, reported only). Nothing else is launched by them.

**Interruption (2026-11-26):** the first submission (03:00 EST) used joint.sbatch's default 2 h wall time (CF2 used 5 h); the wrapper
forwards SIGTERM without requeueing, so all three seeds stopped at 04:58 EST at steps ~29.6k / 30.9k / 30.0k and exited "COMPLETED"
without DONE. Resumed unchanged from last.pt at 11:19 EST (jobs 41867 / 41868 / 41869, `-t 08:00:00`; resumed at steps 29,650 /
30,851 / 30,039). Same configs, teachers and readout. Future joint.sbatch submissions of 100k-step runs need an explicit `-t`.

## Why

- Averaging the six D1-bv + CF2 checkpoints at inference beat every 3-seed ensemble on every pool (§4 of
  docs/variety_and_ensemble_checks.md): held-out LEAP 8.16, Wuji 7.82, Shadow 8.13 mm (contact error). The 5-seed test (§3 result)
  showed member count explains little of that, so most of the gain comes from combining two models that make different errors.
- But six checkpoints cost about 6× inference (about 121 ms per request at batch 1 on a 2080 Ti vs about 21 ms for one).
- Knowledge distillation (Hinton et al. 2015; policy distillation in robotics) trains one student to match an ensemble teacher and
  usually keeps a good share of the ensemble's gain at the cost of a single model.
- The student is a CF model (canonical axes, native origin), so it keeps CF's Wuji/Shadow structure, and through the teacher it
  can learn D1-bv's behaviour on LEAP-like and generated hands without training on variety data itself.

## What changes (one change vs CF2)

Everything as CF2 (`cf2_long_hf*`: same data, sampler, start noise, 100k steps, optimiser, contact loss, canonical frame rule E,
native origin, seeds 0–2), except the motion loss target:

    L = (1 − α) · L_motion(student, ground-truth goal) + α · L_motion(student, teacher pose) + contact_weight · L_contact (unchanged)

- **Teacher:** the six frozen checkpoints `d1_bv_long_hf{,_s1,_s2}` + `cf2_long_hf{,_s1,_s2}`, loaded as in the benchmark
  (`load_system`: CF2 teachers wrapped in CanonicalFrame). For each training example and its sampled start, every teacher
  predicts in the native frame. The teacher pose is the mean joint coordinates, mean translation and chordal-mean rotation, the same
  averaging that was scored. Computed online under `no_grad` (starts are re-sampled every step, so labels can't be cached).
- **Teacher target in the student's frame:** the teacher's (R_T, t_T) gets the same relabelling as the ground-truth goal
  (R_T Q_r; t unchanged; no origin shift), and the residual target is `target_residuals(start, teacher pose)`, the same normalised
  residual as the ground-truth term.
- **α = 0.5**, fixed in advance and not tuned.
- **Data:** CF2's standard training data, with no variety bodies. Variety enters only through the D1-bv teachers. Adding generated bodies
  as extra (teacher-labelled) inputs is a possible later arm, not part of this one.
- **Selection:** best.pt on source validation, as always. No selection on held-out LEAP, Wuji or Shadow.
- **Not touched:** ψ, KITE checkout, test splits.
- **Known wrinkle:** student seed s and teachers cf2_s / d1bv_s share a seed number but no state; all six teachers are used for
  every student.

Config keys (train_joint): `distill_teachers` (list of 6 checkpoint paths), `distill_alpha` (0.5). Runs `cf3d_long_hf{,_s1,_s2}`.
Slurm: `slurm/joint.sbatch`, default_partition, usual exclude list and constraint. Estimated 4–6 h per seed (CF2 took about 2 h; six
teacher forward passes per step). Auto-evaluation chain as for CF-O (benchmark on the standard pools plus the 105 held-out LEAP hands,
Wuji, Shadow; relabelling test).

## Implementation checks (before launch, short smoke runs)

1. α = 0 reproduces CF2's training loss for the first 200 steps to fp tolerance (same seed). This shows the code path is otherwise
   unchanged.
2. The teacher ensemble, run through the training-side code on the source validation pool, reproduces the scored mixed-ensemble
   numbers to within 0.01 mm.
3. Teacher targets in the canonical frame: mapping back with Q_rᵀ recovers the native teacher pose (max error < 1e-5).

## Implementation checks: all passed (2026-11-26, job 32977; `${GETKITE_RUNS}/cf3d_checks.json`)

1. α = 0 vs plain CF2, 300 steps, same seed: motion loss relative difference ≤ 1.2e-5 at every log step (PASS).
2. Training-side teacher on source val reproduces the scored mixed ensemble exactly (Allegro 5.4705, Barrett 5.3117, LEAP 7.4917) (PASS).
3. Single teacher CF2 s0: distillation target in the canonical training frame equals CF2 s0's own re-encoded output (max 1.2e-5 root,
   0 joints) (PASS).
Code: `train_joint.py` (`load_teacher`, `teacher_pose`, `distill_targets`; keys `distill_teachers`, `distill_alpha`), configs from
`scripts/decoder/dlevels/make_cf3d.py`, checks `scripts/decoder/dlevels/cf3d_checks.py` + `slurm/cf3d_checks.sbatch`.
Throughput: ~1,330 examples/s vs ~3,850 for CF2 (about 2.9× slower; about 6 h per seed with requeues).

## Predeclared readout (CF-3d vs CF2, paired within seed, records bootstrap)

Pass = all of:
- **P1, no source loss:** no significant loss in corr_mm or compat_success on source Allegro, Barrett or LEAP, any seed.
- **P2, held-out LEAP gain:** corr_mm on the 105 held-out LEAP hands lower than CF2 in 3/3 seeds and significantly lower in ≥ 2/3.
- **P3, unseen not worse:** on Wuji, corr_mm and wrist_mm not significantly higher than CF2 in any seed.

Reported, not gating:
- Shadow, which stays evaluation only and is never used to select.
- The 360-hand second held-out LEAP pool (`transfer_v1_cf3d_leap360`; topologies unseen by CF2 / D1-long / CF-3d; D1-bv trained
  on these topologies). Added 2026-11-26 before any CF-3d result existed.
- The external evaluation hands (Ability, Inspire, XHand, Schunk SVH, MetaHand, Sharpa Wave, Tetheria Aero; unseen by ψ and every
  decoder; `docs/external_hands.md`). Added 2026-11-26 before any CF-3d result existed.
- CF-3d vs the D1-bv seeds.
- CF-3d vs the six-checkpoint teacher, which shows how much of the ensemble gain survives.
- The CF-3d 3-seed ensemble vs the CF2 3-seed ensemble (does distillation still help once ensembled?).
- Delivery bar vs matched D1-long on every family (Primary 3 of earlier proposals).

If it passes, CF-3d becomes the base for step 2 (learned frame correction, only if step 0's oracle shows a gain worth having).
If it fails, distillation is reported as such, with no α sweep unless the user asks.

## Step 0 result: oracle per-hand frame (2026-11-26 02:50 EST; job 32879; evaluation only)

`scripts/decoder/benchmark/oracle_frame_eval.py` → `${GETKITE_RUNS}/cf2_oracle_frame/oracle_frame.json`. CF2 on the 105 held-out
LEAP hands, with each hand's rule-E frame replaced by the mean training-LEAP frame (24 of 105 hands are > 6° off).
- Perfect frames for all hands: CF2 improves by 0.11 / 0.13 / 0.13 mm contact error (all significant). Fixing only the 24 tilted
  hands gives 0.075–0.09 mm.
- vs D1-long: rule E +0.06 / +0.14 / +0.15 → oracle −0.08 / +0.01 / +0.03 (s0 significantly better, s1 and s2 not significant).
  **Frame error accounts for essentially all of CF2's held-out-LEAP gap to D1-long.**
- vs D1-bv: oracle still +0.32 / +0.04 / +0.05 behind (s0 and s2 significant). A frame fix alone does not reach D1-bv.
- Tilted hands: 10–15° bin goes from +0.54 to +0.60 → −0.18 to −0.01; > 15° from +1.13 to +1.38 → +0.06 to +0.16.
Reading: a learned frame correction (step 2) could gain at most about 0.1 mm on held-out LEAP. The oracle uses knowledge that the hand is
a LEAP variant, so it is a ceiling. Worth doing after step 1, not before.

## Extra evaluation pools available (checked 2026-11-26)

- **552 train-split LEAP hands used by no training run** (kite_code contact data exists for all). 360 of them are topologies CF2
  and D1-long never trained on, but D1-bv trained on all 552 topologies. So this pool is a fair held-out set only for CF2 vs D1-long
  vs CF-3d, not vs D1-bv. Would widen held-out LEAP from 105 to 465 hands for those comparisons.
- Test splits (test_finger_subset 36, test_combined 216, test_joint_count 378, test_geometry 40) stay untouched until the user
  decides.
- New hand families: the KITE checkout ships URDFs only for Allegro, Barrett and Wuji (plus a Robotiq gripper). KITE's own
  collision-primitive files show its authors also had Ability, Inspire, Schunk, XHand, MetaHand (SPIDER asset set), and EZGripper,
  Robotiq 3-finger, a real LEAP hand and Shadow (DRO-Grasp asset set). ψ was trained on Wuji, Allegro, Barrett, Robotiq gripper
  and MANO only, so all of those would be unseen by both ψ and the decoder. Using them needs downloading the public assets outside
  the KITE checkout and building eval data as was done for Shadow.

## 360-hand pool baseline (2026-11-26; job 32982; `${GETKITE_RUNS}/transfer_benchmark_v1_leap360`)

CF2 − D1-long contact error on the 360 hands: +0.07 / +0.18 / +0.19 mm (all significant); on the 105 val hands it was
+0.06 / +0.14 / +0.15. The held-out-LEAP gap replicates on 3.4× more hands. D1-bv − D1-long: −0.43 / −0.01 / −0.04 (D1-bv trained
on these topologies, so not a held-out comparison for it).

## Rule-E 90° flips (found 2026-11-26; details in docs/external_hands.md)

On the 360-hand pool, 9 hands (2.5 %) get a rule-E frame ~90° off (n and a swapped), costing CF2 +2.5 to +3.0 mm on average versus
D1-long; they carry most of the pool's gap. MetaHand (external) has the same failure. This changes step 2's priority: the 90°
ambiguity matters more than the < 15° tilts the oracle measured on the 105 hands. Candidate (needs approval, evaluation only, existing
checkpoints): per request, run CF2 under rule E's frame and its 90° / 180° / 270° turns about d and keep the prediction whose hand
surface lies closest to the requested contact positions (no correspondences, so only deployment-time inputs), then score corr_mm on
the 360 hands, the 105 hands and source. Must not be tuned on MetaHand or any external hand.

## Related work (pointers)

- Ensemble / knowledge distillation: Hinton, Vinyals, Dean 2015, "Distilling the Knowledge in a Neural Network".
- Learned canonicalization with a prior (for step 2): Kaba et al. ICML 2023 (arXiv 2211.06489); Mondal et al. NeurIPS 2023;
  Panigrahi & Mondal 2024 (arXiv 2405.14089).
- Interaction-centric targets for cross-embodiment grasping (for the later FK-contact-loss arm): D(R,O) Grasp, arXiv 2410.01702.
- Cross-embodiment dexterous policies where zero-shot transfer to unseen hands remains the hard part: CrossDex (ICLR 2025,
  arXiv 2410.02479), DexFormer (arXiv 2602.08278).

## Result (2026-11-26 15:53 EST; gate `${GETKITE_RUNS}/cf3d_gate.json`; benchmark 50898, 105-pool 50900, ensembles 50902, external 50951)

**CF-3d FAILS the predeclared readout (P1 pass, P2 pass, P3 fail).** Distillation is reported as such; no α sweep.
- P1 pass: no significant source loss; CF-3d − CF2 corr_mm Allegro −0.04 / −0.09 / −0.14, Barrett −0.35 / −0.12 / −0.14,
  LEAP −0.26 / −0.26 / −0.13 (several significantly better).
- P2 pass: 105 held-out LEAP −0.38 / −0.29 / −0.19 mm, all significant; compat +0.03 all significant; wrist −0.17 / −0.08 / −0.07.
- P3 fail: Wuji corr +0.14 / **+0.30*** / −0.19, wrist **+0.22*** / −0.10 / −0.15 (seed 1 contact and seed 0 wrist significantly worse).
- Reported: Shadow corr −0.14 / +0.75* / −0.65*. vs D1-bv: 105 held-out +0.06* / −0.12* / −0.04*; Shadow better 2/3. vs D1-long
  better on every pool.
- Seed 1 is the outlier: significantly worse than CF2 on Wuji, Shadow and all 7 external hands; seeds 0 and 2 are mostly ≤ CF2.
- Ensembles (corr_mm, 3-seed): held-out LEAP CF2 8.49 / D1-bv 8.24 / CF-3d 8.39 / mixed 6-ckpt 8.16; Wuji 8.12 / 8.02 / 8.37 / 7.82;
  Shadow 8.55 / 8.97 / 8.63 / 8.13. CF-3d ensemble − CF2 ensemble: held-out −0.11*, Wuji +0.24*, Shadow +0.08 (n.s.). The mixed
  ensemble beats the CF-3d ensemble significantly on every pool (+0.08 to +0.55).
- Teacher gain kept: held-out LEAP seed mean 8.93 (CF2) → 8.64 (CF-3d) vs teacher 8.16: ~38 % of the gap. Wuji: none.
- External hands (ensemble corr_mm CF2 / D1-bv / CF-3d / mixed): Ability 8.78 / 8.25 / 9.22 / 7.86; Inspire 9.63 / 14.38 / 10.31 /
  10.42; XHand 10.00 / 13.32 / 9.89 / 10.57; Schunk 9.92 / 10.42 / 10.37 / 9.68; MetaHand 11.65 / 7.09 / 11.77 / 8.45; Sharpa
  9.86 / 9.57 / 10.02 / 9.23; Aero 9.23 / 9.76 / 9.11 / 8.94. CF-3d ensemble − CF2 ensemble significant only where worse (Ability
  +0.43, Inspire +0.68, Schunk +0.45).
- 90° frame flips: not fixed. MetaHand stays at CF2's level (seeds 11.8–13.0 vs D1-bv 7.5–8.0). The 360-hand pool (9 flip hands)
  first ran out of GPU memory on a 24 GB card (job 50901); the retry on a ≥ 48 GB card completed, see "CF-3d on the 360 held-out
  LEAP pool" below (flip hands 10.39 / 10.49 / 10.34 mm vs D1-long ~8.1).
Reading: distillation transferred D1-bv's LEAP-family behaviour (the gain is largest on LEAP-like hands) but not the ensemble's
gain on unseen families, and it added seed variance there. It does not touch the frame-flip failure.

## Frame-selection test (2026-11-26 16:19 EST; job 51932; `scripts/decoder/benchmark/frame_select_eval.py` → `${GETKITE_RUNS}/frame_select_v1/frame_select.json`; evaluation only)

Rule (fixed before running, no parameters): per request, CF2 under rule E's Q and Q·R_d(90/180/270°); keep the prediction with the
lowest surf_mm (mean distance from each requested contact position to the nearest predicted surface point; deployment-time inputs
only). θ = 0 reproduces the benchmark exactly (e.g. Allegro 5.8849 / 5.9611 / 5.9006). Contact error, select − rule E, seeds 0/1/2:
- **Hurts on every LEAP-family and source pool:** Allegro +0.83 / +0.79 / +0.77, Barrett +0.52 / +0.72 / +0.69, LEAP +0.96 /
  +0.82 / +0.71, 105 held-out +1.12 / +1.13 / +0.91, 360 non-flip +0.98 / +1.03 / +0.85 (all significant). It picks a turned frame
  for 21–37 % of requests on these hands, where rule E is right, so surf_mm cannot tell the right frame from a wrong one there.
- **360 flip hands (9 hands, 1,125 records):** −0.54 / −0.63 / −0.79 (significant), but still +1.9 to +2.4 behind D1-long / D1-bv.
- **MetaHand:** 11.98 / 13.16 / 12.19 → 8.67 / 8.98 / 8.64 (−3.3 to −4.2), still +0.3 to +1.5 behind D1 (7.4–8.3). It picks the
  270° turn for 57 % of requests, in line with the frame probe.
- **Other unseen hands (frames not flipped):** large gains. Wuji 8.23 / 8.78 / 9.00 → 7.81 / 8.61 / 8.40; Shadow 8.97 / 9.20 /
  9.47 → 8.40 / 9.01 / 8.95; Ability −1.1 to −1.4; Inspire −0.7 to −1.2; Schunk −0.6 to −1.2; Sharpa −0.5 to −1.4; Aero −0.2 to
  −0.8; XHand ≈ 0. Here CF2 + selection beats D1-bv per seed on every external hand except MetaHand, and on Shadow 3/3, Wuji 2/3.
  It picks a turned frame for 60–77 % of requests, on hands whose frame rule E gets right, so this is not frame correction. It
  works as test-time search: four different predictions, keeping the one whose surface fits the requested contacts best. That
  helps where CF2 is unsure (unseen families) and hurts where its rule-E prediction is already good (LEAP family).
- Oracle best-of-4 by corr_mm (a ceiling, uses correspondences) is 0.3–1.4 mm below rule E on LEAP and 1.3–5 mm on unseen hands.
- The mixed 6-checkpoint ensemble is still better than CF2 + selection per seed on Wuji (7.82), Shadow (8.13), Ability (7.86),
  MetaHand (8.45) and all LEAP pools.
Reading: as predeclared, the rule fails (worse on source and held-out LEAP). The unseen-family gain is a post-hoc observation. Any
rule that uses it (e.g. a selection gate decided from geometry alone) would need to be declared and checked on source / LEAP pools
only, never tuned on Wuji, Shadow or external hands.

## Coverage check (2026-11-26 16:37 EST; job 52547; `scripts/decoder/benchmark/student_teacher_gap.py` → `${GETKITE_RUNS}/student_teacher_gap/gap.json`; evaluation only)

Distance from each student to the six-checkpoint teacher pose (mean surface-point displacement, mm), and the relative reduction
CF2 → CF-3d, seeds 0/1/2:
- Source LEAP 0.17 / 0.11 / 0.11; 105 held-out LEAP 0.16 / 0.07 / 0.08; Allegro 0.11 / −0.01 / 0.05; Barrett 0.14 / −0.02 / 0.03.
- Wuji −0.01 / −0.15 / +0.13; Shadow +0.08 / −0.23 / +0.05; external hands −0.36 to +0.09 (seed 1 negative on every one).
- Teacher spread (each teacher to the teacher mean): 2.1–2.9 mm on source / held-out LEAP, 3.4 Wuji, 4.6–6.9 on Shadow and the
  external hands. The teachers disagree most where the ensemble gains most.
- Distance to the teacher tracks accuracy: on Wuji, seed 2 moved toward the teacher (+0.13) and is the only seed better than CF2
  (−0.19 mm); seed 1 moved away (−0.15) and is the worst (+0.30).
**Predeclared criterion: not met as declared.** Wuji and Shadow at most half the held-out LEAP reduction holds in seed 1, is borderline
in seed 0 (Shadow 0.079 vs 0.078), and fails in seed 2. Reading: distillation pulled the student toward the teacher mainly on
LEAP-type hands (even Allegro and Barrett gained little), and it did not do so reliably on unseen hands. But one seed did transfer to
Wuji, and seed 1 actively moved away from the teacher on every unseen hand, so there is a seed-specific instability on top of any
coverage effect. The evidence for CF-3e (teacher-labelled variety bodies) is suggestive, not established.

## CF-3e: distillation with generated-body variety (approved by the user 2026-11-26 ~16:50 EST)

One change vs CF-3d: the generated-body family's training data becomes D1-bv's set, P0000–P0799 × 25 records (CF-3d: P0000–P0049 ×
400). Same record budget and 25 % family share under balanced sampling. Everything else as CF-3d: six teachers, α = 0.5, Allegro /
Barrett, CF2's 48 LEAP hands (D1-bv's extra LEAP topologies are deliberately not added, so the 105- and 360-hand held-out LEAP pools
stay held out), model, optimiser, 100k steps, seeds, best.pt on source validation.
- Why: the coverage check showed the student moved toward the teacher mainly on LEAP-type hands and not reliably on unseen hands,
  where the teachers disagree most. 800 diverse bodies put the teacher target on many more shapes.
- Deviation from the wording "teacher-labelled only": the new bodies get the same loss as every record (0.5 ground truth + 0.5
  teacher), as D1-bv used them with ground truth. This keeps it a pure data change vs CF-3d with no new code. Body variety alone is
  already measured by CF-bv / D1-bv, which can be compared.
- Never used: P0800–P0899 (prim_heldout eval), Wuji, Shadow, external hands, test_* splits.
- Configs `configs/decoder/scratch/cf3e_long_hf{,_s1,_s2}.json` (`scripts/decoder/dlevels/make_cf3e.py`; changed keys vs CF-3d:
  `_note`, `out_dir`, `robots` 154 → 904, `train`). Explicit `-t 08:00:00`.
- Predeclared readout: the same P1 / P2 / P3 vs CF2 as CF-3d (`cf3d_gate.py cf3e` → `${GETKITE_RUNS}/cf3e_gate.json`). Also
  reported, not gating: CF-3e vs CF-3d on every pool (the Wuji question), Shadow, the 360 pool, external hands, ensembles.
- Launched 2026-11-26 17:00 EST: jobs 53124 / 53126 / 53129 (`-t 08:00:00`); smoke run 52877 passed (904 robots, 6 teachers, ~1,230 examples/s).
- Chains: `slurm/cf3e_chain.sh` (benchmarks incl. CF-3d next to CF-3e; 360 pool on ≥ 48 GB GPUs) and `slurm/cf3e_external_chain.sh`.

### CF-3d on the 360 held-out LEAP pool (added 2026-11-26 20:30 EST; reported, not gating)
Gate unchanged: P1 ✓, P2 ✓, P3 ✗ → FAIL. CF-3d − ref corr_mm per seed s0/s1/s2 (sign: 1 = CF-3d significantly better, −1 = significantly worse):
- vs CF2: −0.38 / −0.25 / −0.21 [1,1,1]; wrist −0.16 / −0.08 / −0.08 [1,1,1].
- vs D1-bv: +0.12 / −0.05 / +0.02 [−1,1,−1].
- vs D1-long: −0.31 / −0.07 / −0.01 [1,1,0].
9 flip hands (> 45° from training-LEAP frame; 1125 records): CF-3d 10.39 / 10.49 / 10.34 vs CF2 10.96 / 11.13 / 10.75 (CF-3d better, significant in all seeds)
but still ≈ 2.0–2.4 mm behind D1-long (8.44 / 8.09 / 8.12) and D1-bv (7.98 / 8.23 / 8.06). The frame flip is not fixed by distillation.

### CF-3e result (2026-11-27 02:15 EST): PASS (P1 ✓, P2 ✓, P3 ✓)
Training 53124/53126/53129 all DONE (100k steps, fresh). CF-3e − CF2 corr_mm per seed s0/s1/s2 (paired bootstrap; [1 = significantly better]):
- Source: Allegro −0.16/−0.21/−0.18 [1,1,1]; Barrett −0.42/−0.23/−0.25 [1,1,1]; LEAP −0.37/−0.17/−0.21 [1,0,1]. P1 ✓ (better, not just no loss).
- 105 held-out LEAP: −0.42/−0.23/−0.33 [1,1,1] (wrist [1,1,1]). P2 ✓.
- Wuji: corr −0.89/−1.14/−1.12 [1,1,1]; wrist −0.17/−0.40/−0.38 [1,1,1]. P3 ✓ (significantly better, not just not worse).
Reported only:
- vs CF-3d: Wuji −1.03/−1.44/−0.92 [1,1,1]; Shadow −0.91/−1.60/−0.08 [1,1,0]; 105 −0.03/+0.06/−0.14 [1,−1,1]; source n.s. except Barrett s2.
- vs D1-bv: Wuji −1.07/−0.93/−1.02, Shadow −1.32/−1.84/−1.30 (all sig. better); 105 +0.03/−0.06/−0.18 [0,1,1]; 360 +0.09/−0.04/−0.15 [−1,1,1].
- Shadow vs CF2: −1.05/−0.86/−0.73 [1,1,1]. 360 pool vs CF2: −0.41/−0.24/−0.38 [1,1,1].
- 9 flip hands (360): CF-3e 10.14/10.02/10.03 vs CF2 10.96/11.13/10.75 (sig. better) but still ≈ 1.7–2.0 mm behind D1-long 8.44/8.09/8.12. Frame flip not fixed.
- External hands (corr_mm per seed, CF2 → CF-3e; CF-3e − CF2 significant better in all 3 seeds unless noted):
  ability 9.60/9.81/9.73 → 8.29/8.22/8.45; inspire 10.10/10.55/10.58 → 9.17/9.07/9.88; xhand 10.59/10.71/10.82 → 9.71/9.78/10.24;
  schunk 10.41/11.13/10.54 → 9.94/10.04/10.32 (2/3); metahand 11.98/13.16/12.19 → 10.34/11.13/11.64 (D1 ≈ 7.4–8.3, flip unfixed);
  sharpa 10.65/10.35/10.89 → 9.96/9.84/9.79; aero 9.92/9.77/10.02 → 8.70/8.79/9.56.
- Single CF-3e seeds on Wuji (7.34/7.64/7.88) and Shadow (7.92/8.35/8.74) already match or beat the 6-model D1-bv+CF2 ensemble (7.82 / 8.13).
  3-seed ensembles: Wuji CF-3e 7.24, Shadow 7.79, 105 8.33 (mixed 8.16).
Reading: wider body variety did carry the teacher's gain to unseen hands; the student now beats its teachers off-distribution. Mechanism not isolated
(variety alone = D1-bv, which does not show this; the combination does).
