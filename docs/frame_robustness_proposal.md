# Frame robustness: proposal (2026-11-28; design only, nothing launched)

Agreed with Aryan as the first research direction after the running gates (docs/plan_2026-11-28.md). Every step below needs the
user's go-ahead, a commit to `getting-kite-universal` and launch from that commit. No step uses Wuji, Shadow or the external
hands to choose anything; they are reported only (new-family validation). The rule for any switch margin is fixed here, before
running; the post-hoc 0.25 mm margin from the first per-hand test is validated only on the sealed new-family set.

## Evidence (verified 2026-11-28 from existing files; no new runs)

- **What the flips are.** Rule E picks d as the palm principal axis best aligned with the mean finger direction; when that
  direction lies between the two in-plane palm axes, d and a swap. Measured on the 9 flip hands of the 360 pool (rule-E frame vs a
  well-framed sibling of the same topology): **7 of 9 are ~80° rotations about n** (the palm normal; d and a swapped) and **2 of 9
  are ~90° about d** (I0-M4-R0-T4_g1, I2-M0-R0-T4_g1).
- **Why the per-hand frame test closed only half the gap.** It searched turns about d only (0/90/180/270), so it could fix only the
  2 about-d hands, and it switched exactly 2 of 9 (both 90°): flip hands 10.0-10.1 -> 9.0-9.1 mm vs D1-bv 7.9-8.2. MetaHand
  (about d) was fixed: 10.4-11.7 -> 7.8 mm. False switches: Sharpa 3/3 seeds (2 significantly worse, +0.3-0.4 mm), Inspire 1 seed
  (+0.5 mm).
- **The training data is partly mis-framed.** In CF-3e's training set, 63 of 800 generated bodies (7.9%) have a rule-E frame
  > 30° from the bodies' shared native convention (110 > 15°), and 1 of 48 LEAP hands (I0-M4-R0-T3_g0) is flipped 85° vs its
  topology. CF is causally sensitive to frame error (CF-J step 0), so the model learns partly inconsistent frames.
- **What has already failed:** a continuous small frame jitter in training (CF-J: lowered sensitivity, failed its readout);
  a canonical origin (CF-O); new axis rules G and K (failed their geometry gates; the user dropped the axis-rule line
  2026-11-25). None of the steps below is a new axis rule.

## Step A: per-hand frame test v2 (evaluation only, frozen CF-3e; ~10 GPU-min)

Same rule as the first test (`frame_perhand_eval.py`): per hand, its first K = 32 records are sample requests; keep the candidate
frame with the lowest mean surf_mm over them (deployment-time inputs only, no correspondences); score the hand's remaining records
with it. **One change: the candidate set adds the two 90° turns about n** (d <-> a), so 6 candidates: rule E; 90/180/270° about d;
+90/-90° about n.
- Margin: 0 (predeclared, as before). Margins 0.25 / 0.5 mm reported descriptively, not chosen.
- Gate G1 (per seed, paired bootstrap): corr_mm and wrist_mm not significantly worse than rule E on the 105 pool and source LEAP.
- Reported only after G1: the 9 flip hands (target: the 7 about-n hands), MetaHand, the other 351 hands of the 360 pool, false
  switches per family, Wuji / Shadow / external hands, plus success, median and p90 per hand.
- Audit before running:
  - More candidates = more chances of a false switch (Sharpa / Inspire already switched wrongly with 4). G1 checks well-framed
    LEAP; false switches on new families are counted and reported, and the margin question is settled only on the sealed set.
  - The 360 pool's topologies were never trained on by CF-3e (D1-bv did train on them; it is a reference only, not compared on).
  - 80° vs 90°: a 90° turn about n leaves a ~10° residual tilt on the about-n hands. CF loses accuracy with tilt (CF-J step 0),
    so the expected gain is less than the full gap. Reported, not tuned (no continuous angle search).
  - Adoption is an inference-procedure change (32-request calibration per new hand): the user's call, as before.

## Step B: clean training frames (training arm; after Step A; 3 seeds x ~4-6 h)

CF-3e with one change: **during training only**, a training hand whose rule-E frame is > 30° from its reference frame is
re-expressed in the reference frame (generated bodies: their shared native convention, which rule E targets; LEAP: the median
rule-E frame of its topology's other hands). 63 bodies + 1 LEAP hand change; everything else, including evaluation (rule E, or
rule E + Step A's per-hand choice if adopted), is unchanged.
- Hypothesis: the model learns one consistent frame instead of a mix, so it is sharper on well-framed hands, and test-time
  per-hand choice (Step A) becomes more reliable.
- Audit:
  - Uses training hands' native frames: training-only information, never used at evaluation. No evaluation hand is touched.
  - The 30° threshold is fixed here, not tuned. 63 bodies is about 2 % of training records (the body family is 25 % of every batch).
  - Possible cost: the model never sees a mis-framed hand in training, so on a flipped test hand without per-hand choice it may do
    worse. Reported per family; the flip hands are read with and without Step A's choice.
  - One change vs CF-3e: same data, teachers, alpha, model, steps, seeds, selection. The teachers' targets are frame-independent
    (computed in the native frame), so they are unaffected.
- Scored both without and with the per-hand frame choice (rule M) on every pool (with = how it would be deployed; Aryan's
  review). The gate is decided on the without-choice scores (one change vs CF-3e); with-choice is reported alongside.
- Gate: Q1-Q5 vs CF-3e with the measurement fixes of plan_2026-11-28 (success, median, p90 reported; tolerances anchored to
  CF-3e's seed spread), frozen in a commit before launch.

## Step C (not proposed first): discrete-turn augmentation

Train with a random turn from Step A's candidate set on a fixed fraction of examples, so the model tolerates the ambiguity itself.
Not first: the canonical-frame work found the decoder is hurt by a frame that means different things on different hands, so this
risks losing accuracy on the well-framed majority; Steps A-B attack the cause with less risk.

## Cost and order

A: ~10 GPU-min, can run any time (frozen CF-3e; no dependency on the running work). B: 3 x ~4-6 h, after A, and only with the
user's go-ahead. Both are independent of CF-3g, long, bigext and big300k.

## Step A result and margin rule fixed before the 360-pool readout (2026-11-28 01:31 EST, user-approved)

Step A (job 149085, commit 380c2f9): **G1 FAILS** (s0 pass, s1 fail, s2 pass). 105 pool, perhand(m0) - rule E corr_mm:
s0 +0.003 [-0.006, 0.013]; s1 +0.021 [0.010, 0.032]; s2 -0.015 [-0.022, -0.009]. Cause: the about-n candidates switch 1-3 of
the 105 hands per seed (v1: none); on LEAP palms the two in-plane axes are nearly equal, so surf_mm barely separates rule E from a
90° turn about n and margin 0 lets noise decide. Source LEAP: no switches. External hands as v1 (MetaHand 10.4-11.7 -> 7.8 mm;
false switches on Sharpa / Inspire / Ability / XHand / Schunk, all removed at margin 0.25 — descriptive). The 360-pool part was cut
off by a preemption and is being rerun from the same commit (job 149292), NOT seen at the time this rule was written.

**Margin rule M (fixed now, before any v2 360-pool number exists):** switch away from rule E only if the best candidate's mean
surf_mm over the hand's 32 sample requests is lower by more than **0.25 mm**. Judged on the v2 360-pool readout (CF-3e s0/s1/s2,
paired bootstrap over scored records, seeds never pooled). Rule M is supported if BOTH hold in every seed:
1. the 9 flip hands: perhand(m0.25) - rule E corr_mm, 95 % CI upper bound < 0 (better);
2. the other 351 hands: perhand(m0.25) - rule E, corr_mm and wrist_mm: **95 % CI upper bound < +0.05 mm** (clearly not worse than
   a 0.05 mm tolerance; replaces "not significantly worse", per the agreed tolerance fix).
   *Amended 2026-11-28 ~01:40 EST (user-approved, Aryan's catch), still before any v2 360-pool result existed: the first wording
   ("point estimate <= +0.05 and CI lower bound < +0.05") had a redundant lower-bound clause; the intended test is the upper bound.
   The bootstrap is over scored records (~30k here; the 105 pool's intervals were about +-0.01 mm), so this is not noise-limited;
   it ignores clustering by hand, as all our gates do.*
Caveat (written now): the margin value was chosen after seeing v1 (including v1's 360-pool numbers, about-d candidates only), so this
is partly-fresh evidence; the sealed new-family set is the clean test. Adoption stays the user's decision either way.
