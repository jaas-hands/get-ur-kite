# CF-J proposal: make CF robust to small errors in its own canonical frame (2026-11-25)

Status: Step 0 run 2026-11-25 (job 19825) and MET in 3/3 seeds; Step 1 not launched (needs go-ahead).
Update (2026-11-27): Step 1 (CF-J) was launched on 2026-11-25 and did NOT meet its readout; CF-J was not promoted
(see "Step 1 result" below).

## What the 105-hand evaluation showed (evaluation only; configs/decoder/dlevels/cf2_readout/leapval105_*.txt)

Held-out LEAP widened from 4 to all 105 dataset val-split hands (13,125 records; transfer_benchmark_v1_leapval105). Source
numbers reproduce transfer_v1_bv exactly.
- **The gap is real but small.** CF2 − D1-long contact error: +0.06 / +0.14 / +0.15 mm (hand-bootstrap CI excludes 0 in seeds
  1–2). CF2 − frozen D1: −0.00 / −0.28 / −0.20 (CF2 lower on 92 and 89 of 105 hands in seeds 1–2). CF2 − D1-bv: +0.45 / +0.17 / +0.16.
  The original 4 hands were not representative (seed 1–2 sign flips).
- **It is caused by the canonical frame's per-hand error.** Gap vs the hand's canonical-frame angle to the mean training LEAP
  frame (CF2 − D1-long, seeds 0/1/2):
  | angle | hands | gap (mm) |
  |---|---|---|
  | < 3° | 45 | −0.01 / +0.01 / +0.05 |
  | 3–6° | 36 | −0.04 / +0.12 / +0.08 |
  | 6–10° | 16 | +0.05 / +0.21 / +0.25 |
  | 10–15° | 5 | +0.60 / +0.54 / +0.58 |
  | > 15° | 3 | +1.38 / +1.26 / +1.13 |
  CF2's own error rises with the angle (r +0.33 to +0.42, hands ≤ 15°); D1's does not (r +0.04 to +0.13). D1 sees every LEAP
  hand in one fixed native frame, so it has no per-hand frame error. The same pattern holds for CF-bv vs D1-bv.
- This reverses audit §63 diagnosis 1 ("not the frame rule's per-variant error"): that test used 16 hands, almost all < 7°.
  It also explains why B did not help CF: more training hands add more frame scatter (8 of 414 B hands are > 30° off).
- Not explained by frame angle: D1-bv's lead over CF2 at < 3° in seed 0 (+0.33). Body variety helps D1 in a way CF did not use.

## Step 0 (evaluation only, ~30 min GPU; needs go-ahead): dose-response check

Confirm the angle is causal, not a correlate of unusual geometry. On the 45 held-out hands with angle < 3°, evaluate CF2
(3 seeds) with its canonical frame deliberately rotated by 0 / 2.5 / 5 / 10 / 15° (fixed axes: about n, a, d, each sign).
Code: an eval-only `frame_perturb_deg` option in CanonicalFrame (Q_r -> Q_r J), off by default.
- **Predeclared:** causal if contact error rises monotonically with the rotation and the 10° increase is ≥ 0.3 mm in 3/3
  seeds (the observed 10–15° bin is +0.55). Otherwise stop: the angle is a marker, not the cause, and frame jitter is not proposed.

## Step 0 result (2026-11-25; scripts/decoder/benchmark/frame_perturb_eval.py, readout configs/decoder/dlevels/cf2_readout/frame_perturb_step0.txt)

45 held-out hands with frame angle < 3° (5,625 records); angle 0 reproduces the benchmark exactly (8.908 / 8.771 / 8.741).
Mean CF2 contact-error increase over the 6 axis/sign directions, seeds 0 / 1 / 2:

| rotation | contact error | wrist rotation | compat success |
|---|---|---|---|
| 2.5° | +0.05 / +0.06 / +0.05 mm | +0.05° | −0.005 |
| 5° | +0.21 / +0.23 / +0.21 mm | +0.20° | −0.02 |
| 10° | +0.81 / +0.91 / +0.81 mm | +0.73–0.79° | −0.07 to −0.09 |
| 15° | +1.56 / +1.83 / +1.58 mm | +1.36–1.50° | −0.14 to −0.18 |

Monotonic and ≥ 0.3 mm at 10° in 3/3 seeds: **the rule is met; frame error causes the error increase.** The cost grows
roughly with the square of the angle and comes through the wrist rotation. Tilts about the palmar axis n cost about a third
as much as tilts about a or d. The induced cost is larger than the natural 10–15° bin (+0.55 mm), as expected: a hand with a
naturally tilted frame also differs in geometry, which the decoder partly uses.

## Step 1 (training, needs go-ahead): CF-J = CF2 + small frame jitter in training

- **One change vs CF2 (A data, 100k steps, seeds 0/1/2):** during training only, each example's canonical frame is rotated by a
  random small rotation J (random axis, angle uniform in [0°, 15°]), via a fixed set of k = 32 jittered canonical banks
  (same mechanism as W2-b's base_frame_aug, but around the canonical frame instead of fully random). Validation, checkpoint
  selection and all evaluation use the unjittered frame, so deployment is unchanged and relabelling invariance stays exact.
- **Why 15°:** covers the observed spread (p90 9°, all non-flipped hands ≤ 20°). W2-b's fully random frames cost source accuracy;
  a 15° cone keeps the frame meaning.
- **Not fixed by this:** the rare flipped frames (> 30°; 1 of 105 held-out hands at 84°). That is a frame-rule failure;
  report it, don't patch it in the same run.
- **Cost:** 3 runs × ~3.6 h, in parallel on default_partition, then the benchmark + 105-hand evaluation + relabelling test
  (~30 min). About 4.5 h end to end if GPUs are free.

## Predeclared readout for CF-J (fixed before any run)

- **Held-out LEAP, 105 hands (target):** CF-J − D1-long contact error with its hand-bootstrap CI including 0 or below in 3/3
  seeds, and the 6–15° bins' gap reduced by ≥ 50% vs CF2 in 3/3.
- **Primaries 1–3 vs D1-long** on transfer_v1 (unchanged definitions, docs/cf2_longer_training_proposal.md), incl. Shadow wrist
  significantly better 3/3 (CF's gain must survive).
- **Secondary:** CF-J vs CF2 everywhere; CF-J vs D1-bv; relabelling spread ≈ 0.
- **Caveat:** the 105 held-out hands informed this design, so they are no longer untouched. A clean check of CF-J would use an
  untouched hand set; the dataset's test_geometry split (40 hands) is the natural one, but it is reserved and needs the
  user's decision.

## Not proposed now

- Changing the frame rule itself (would also need retraining; Step 0 tells us whether the frame is the lever at all).
- Combining jitter with B's data (one change at a time; B did not help CF).

## Step 1 as launched (2026-11-25, user go-ahead)

- Code: `canonical_jitter` option in src/getkite/decoder/train_joint.py (`canonical_jitter_bank`; off unless set; requires
  canonical_frame). k = 32 frames: block 0 = identity, 31 tilts about a random axis by U[0°, 15°] (seed 20261125 + run
  seed; drawn angles 0.0–14.9° in seed 0). Each training example draws a block uniformly. Evaluation, checkpoint selection
  and checks use the unjittered canonical frame; canonical frames identical to CF2's (max diff 0).
- Configs cfj_long_hf{,_s1,_s2} = cf2_long_hf* + canonical_jitter (diff: _note, canonical_jitter, out_dir only), made by
  scripts/decoder/dlevels/make_cfj.py. Smoke test (200 steps, job 20001): checks pass, ~3,700 examples/s on an RTX 3090.
- Jobs 20044–20046 (default_partition, 5 h limit). Auto-evaluation slurm/cfj_chain.sh (log slurm/logs/cfj_chain.log):
  transfer_v1_cfj (4-hand held-out + Shadow; Primaries 1–3), transfer_v1_cfj_leapval105 (105-hand target), and the
  relabelling test (cfj_frame_relabel).
- Readout: cf2_summary.py --cand cfj --ref d1long (and --ref d1, --ref cf2); leapval105_summary.py --a cfj --b d1long /
  --a cfj --b cf2 with --pe on the cfj_leapval105 per_example.pt; leapval105_angle.py <per_example.pt> cfj-d1long cf2-d1long.

## Step 1 result (2026-11-25, evaluation complete 19:40 EST; readouts configs/decoder/dlevels/cf2_readout/*cfj*)

Training: jobs 20044–20046, preempted 16:46–16:53 at ~60k / 60k / 71k steps, resumed 17:56 from last.pt, DONE 19:15. Best
steps 100k / 95k / 82.5k. Relabelling spread 0.002 (≈ 0, as expected; jitter is training-only).

**Predeclared readout: NOT MET. CF-J is not promoted.**
- 105-hand held-out LEAP, CF-J − D1-long: −0.11 [−0.16, −0.06] / **+0.09 [+0.05, +0.14] / +0.17 [+0.12, +0.22]** mm
  (hand bootstrap). Must include 0 or below in 3/3 → fails in seeds 1 and 2.
- 6–15° hands' gap vs CF2 (hand-weighted): seed 0 +0.18 → −0.07 (removed), seed 1 +0.29 → +0.18 (−39%), seed 2 +0.33 → +0.25
  (−23%). Must shrink ≥ 50% in 3/3 → fails.
- Primaries vs D1-long on transfer_v1: Primary 1 FAILS (held-out LEAP corr seed 2, significant), Primary 2 HOLDS (Shadow wrist
  significantly better 3/3, trans R²_id higher 3/3), Primary 3 FAILS.

What jitter did and didn't do:
- **It reduced frame-angle sensitivity.** CF-J's own error vs frame angle r +0.15 to +0.25 (CF2 +0.33 to +0.42); on hands
  > 10° off it beats CF2 by 0.3–0.8 mm in every seed.
- **It did not close the gap.** In seeds 1–2 a gap to D1-long remains on hands < 6° off (+0.02 to +0.20 mm), so part of CF's
  held-out cost is not frame error.
- **Costs:** Allegro +0.09 mm vs CF2 (3/3 direction, 1 significant). Wuji / Shadow became more seed-variable: seed 1 worse
  (Shadow 9.21 → 10.65 mm), seed 2 better (9.47 → 8.49); Shadow trans R²_id lower than CF2 in 3/3.
- vs frozen D1: better on source (3/3, significant) and held-out LEAP (3/3 lower); Wuji / Shadow 2/3.

Consequences (as set out beforehand): CF-J not promoted; item 8 (jitter + body variety) dropped; CF-O (docs/canonical_origin_proposal.md)
is based on **CF2**; the delivery choice stays D1-bv vs CF2.
