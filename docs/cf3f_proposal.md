# CF-3f: full-scale step up from CF-3e (proposal v3, DRAFT; not approved, nothing launched; 2026-11-27)

CF-3e is **frozen** as the delivery fallback: `${GETKITE_RUNS}/frozen_cf3e_2026-11-27/` (3 seeds' best.pt, configs, logs,
gate and benchmark results, SHA256SUMS, read-only). Interim delivery candidate: CF-3e seed 0, single model. All new work is
named **CF-3f** and writes to new folders; nothing under the frozen folder or the CF-3e run folders is touched. The six teacher
checkpoints (d1_bv_long_hf*, cf2_long_hf*) are inputs to CF-3f training and must not be moved or deleted.

History: v1 bundled several changes per run; v2 took in Aryan's suggestions (360-pool overlap check, one change per arm judged
against CF-3e, D1-bv's 414 LEAP hands as the main candidate); v3 renames to CF-3f, adds the Shadow guard (soft), and spells out
edge cases, contingencies and practice checks. Sent to Aryan for feedback before any launch.

## 1. Why CF-3e, and what CF-3f must keep
CF-3e = CF2 recipe (J1 decoder, frozen KITE encoder ψ, rule-E canonical palm frame) + six-checkpoint teacher (D1-bv s0–s2 + CF2
s0–s2), α 0.5, + 800 generated bodies × 25 records; 48 LEAP hands, Allegro, Barrett; balanced 4-family sampler; 128 × 4 layers;
100k steps, cosine LR to 0.

| Past effort | Strength kept | Shortcoming |
|---|---|---|
| D1 / D1-bv (native frame) | body variety helps held-out LEAP (−0.39) and generated bodies | weak on new families (Shadow, external) |
| CF / CF2 (canonical frame) | strong on new families (Shadow wrist) | small held-out LEAP gap; variety alone didn't help (CF-bv) |
| CF-J (frame jitter), CF-O (canonical origin), frame selection at inference | — | all failed their gates |
| CF-3d (teacher, 50 bodies) | held-out LEAP gain | didn't carry to Wuji |
| **CF-3e** (teacher + 800 bodies) | **better than CF2 on every pool, 3/3 seeds; beats both teacher families on unseen hands** | held-out LEAP only ≈ D1-bv; rule-E 90° flips unfixed |

CF-3f keeps every CF-3e ingredient and changes one thing per arm.

## 2. Verified facts the design rests on (2026-11-27)
- **Training curves:** source-val contact error at 75k / 90k / 100k: s0 6.21 / 6.11 / 6.09; s1 6.21 / 6.14 / 6.12; s2 6.17 / 6.09 /
  6.07. Best checkpoint at the last or second-to-last eval. The LR is a cosine to zero at the final step, so part of the late gain
  is the schedule ending; a longer schedule is still the right test.
- **LEAP data:** 138 train-split topologies × 7 geometry variants. D1-bv's 414 = 3 variants per topology (includes CF-3e's 48). The
  105-hand pool (15 val topologies) is disjoint from all training and all teachers. The 360-hand pool (4 variants of the 90
  topologies CF-3e never trained on) shares topologies with D1-bv, one of CF-3e's teachers, but no hand files.
- **Prior evidence:**
  - Body variety: D1-bv beat D1 on held-out LEAP by 0.39 mm (3/3 significant); CF-bv (CF + the same variety, no teacher) did not
    improve over CF2.
  - Longer training (50k → 100k): helped held-out LEAP for both lines (D1 −0.32, CF −0.37) but hurt Shadow on average
    (D1 +0.78, significantly worse 1/3; CF +0.29, significantly worse 2/3 and significantly better 1/3).
- **Cost:** CF-3e 100k = 3 h 52 min (A40) to ≈ 6 h (2080 Ti), 2.3 GB GPU memory, checkpoint every 1k steps, eval every 2.5k.
  Queue wait last time ≈ 2.5 h. Evaluation chain ≈ 30 min.

## 3. Design

| Stage | Arm (run name) | Single change from CF-3e | Steps | Seeds | Wall time |
|---|---|---|---|---|---|
| 1 | **CF-3f-L** (`cf3f_leap_hf*`) | LEAP data = D1-bv's 414 hands × 48 records (same LEAP budget ≈ 19.9k, same 25 % share) | 100k | 0, 1, 2 | ≈ 4–6 h |
| 1 | **CF-3f-S** (`cf3f_long_hf*`) | none; 300k-step schedule (warm-up and cosine stretched to 300k) | 300k | 0, 1, 2 | ≈ 12–18 h |
| 2 | **CF-3f** (`cf3f_full_hf*`) | the Stage-1 changes that passed, at 300k | 300k | 0, 1, 2 | ≈ 12–18 h |

Stage 2 rule (fixed now):
- L and S pass → CF-3f = L at 300k; launched as soon as S passes.
- Only S passes → CF-3f-S is the full-scale model; no Stage 2.
- Only L passes → CF-3f-L (100k) becomes the candidate; L at 300k needs a new go-ahead (S's failure means 300k is not supported).
- Neither passes → CF-3e stays; next candidate single change: CF-3e added to the teacher group (Aryan's second suggestion).

Delivery rule (fixed now; single models, seed 0, no best seed, no seed averaging):
- Preference: CF-3f (Stage 2) if it passes → else CF-3f-S if it passed → else CF-3f-L if it passed → else CF-3e.
- Stage 2 must also not be significantly worse than the Stage-1 arm it extends on the 105 pool, Wuji or Shadow in ≥ 2/3 seeds;
  otherwise that Stage-1 arm is delivered.

## 3a. Stage 1 as launched (user go-ahead 2026-11-27: "do the runs that test what should be added to the full model")
Five single-change test arms, each = frozen CF-3e + one change, 3 seeds, 100k steps unless stated, all judged by the same gate
(section 4) against CF-3e of the same seed. Their results decide what goes into the full-scale CF-3f run.

| Arm (runs) | Change | Question it answers for the full model |
|---|---|---|
| **leap** (`cf3f_leap_hf*`) | LEAP data = D1-bv's 414 hands x 48 records | include the wider LEAP set? |
| **long** (`cf3f_long_hf*`) | 300k-step schedule | train longer? |
| **teach** (`cf3f_teach_hf*`) | teacher group + the 3 frozen CF-3e checkpoints (9 teachers, equal weight) | add CF-3e as a teacher? (Aryan's second idea) |
| **big** (`cf3f_big_hf*`) | model 256 wide x 6 layers x 8 heads (CF-3e 128 x 4 x 4) | scale the model? |
| **rec** (`cf3f_rec_hf*`) | generated bodies x 100 records each (CF-3e x 25) | use more of the generated data on disk? |

Plus one evaluation-only test (no training): **per-hand frame choice** on frozen CF-3e (Aryan's flip suggestion), section 3b.

Full-scale rule (fixed now): the full-scale CF-3f run combines every Stage-1 change that passes its gate, at 300k steps if the
long arm passes (100k otherwise); it is judged by the same gate against CF-3e and must not be significantly worse than the
best-supported single arm it contains on the 105 pool, Wuji or Shadow in >= 2/3 seeds. Launching it needs the user's go-ahead.
Caveat: changes that pass separately can interact; the combined run is the test of that.

Tooling (written and checked before any result): `scripts/decoder/dlevels/make_cf3f.py` (asserts only the arm's keys differ from
CF-3e and that CF-3e's configs equal the frozen copies), `scripts/decoder/dlevels/audit_cf3f.py` (leakage audit: passes on all 15
configs; the leap arm overlaps the 360 pool's 90 topologies, as expected, and no 360 or 105 hand files), `scripts/decoder/benchmark/
cf3f_gate.py` (self-test passes: self-comparison gives zero; CF-3d vs CF-3e reproduces cf3e_gate.json negated),
`slurm/cf3f_chain.sh` (divergence check, DONE checks, benchmarks, gate, external hands at the end), 300-step smoke runs per arm.
Q4 implementation: training-log held-out generated-body contact error at each run's selected checkpoint, point estimate per seed,
arm - CF-3e <= +0.10 mm in every seed (CF-3e: 5.156 / 5.126 / 5.023).

## 3b. Per-hand frame test (evaluation only; frozen CF-3e)
Rule fixed before running: per hand, its first 32 records are sample requests; keep the turn of rule E's frame (0/90/180/270
about d) with the lowest mean surf_mm over them (deployment-time inputs only); score that hand's remaining records with it.
Margins 0.25 / 0.5 mm reported descriptively only. Gate G1 (per seed): corr_mm and wrist_mm not significantly worse than rule E on
the 105 pool and source-validation LEAP. Only then are the flip hands, MetaHand and the other unseen hands read. If it passes, it
is an inference-procedure change that needs the user's view against the simplicity ruling before adoption.
Script: `scripts/decoder/benchmark/frame_perhand_eval.py` -> `${GETKITE_RUNS}/frame_perhand_cf3e_v1/`.

## 4. Predeclared pass test (each arm vs CF-3e of the same seed; paired bootstrap over records; seeds never pooled)
- **Q1 source:** no significant loss in contact error on Allegro, Barrett or LEAP validation in any seed.
- **Q2 105 held-out LEAP:** lower than CF-3e in 3/3 seeds, significant in ≥ 2/3; not significantly worse in any.
- **Q3 Wuji:** contact and wrist error not significantly worse in any seed.
- **Q4 held-out generated bodies (P0800–P0899):** training-log contact error at the selected checkpoint, arm − CF-3e ≤ +0.10 mm in every seed (no per-record bootstrap exists for this pool).
- **Q5 Shadow guard (soft; Aryan's suggestion):** contact and wrist error not worse on the 3-seed mean of per-seed differences,
  and not significantly worse in 2 of 3 seeds. A guard, not a target.
- Reported, never gating or selecting: 7 external hands, flip hands, success rates, wrist degrees, CF2 / D1 references, the
  360 pool (CF-3f-S only; CF-3f-L trains on its topologies), CF-3f vs CF-3e on the in-training evals.
- Checkpoint selection: best on source validation only, as for CF-3e.

## 5. Edge cases and contingencies (decided now)
| Situation | Action |
|---|---|
| A job hits its time limit or is preempted | `-t 24:00:00`, `--requeue`, resume from last.pt (checkpoint every 1k). Watchers check for DONE markers, not job states (CF-3d's 2 h default-limit loss). |
| A seed cannot finish by the cutoff | No 2-seed verdict: the arm is not adopted. |
| Divergence (NaN loss, or source-val worse than CF-3e's 25k value by > 1 mm at the same schedule fraction: 25k for 100k arms, 75k for the long arm; fix from Aryan's review) | Logged by the chain and reported to the user; no silent restarts or retuning. |
| Evaluation OOM on the 360 pool | Typed A6000-class GPU request (fix from CF-3d). |
| Mixed results (e.g. Q2 passes, Q5 fails) | Fail. No re-reading of the gate after the fact. |
| CF-3f-S passes but Shadow is worse on mean (soft-guard pass) | Pass, but reported prominently; Stage 2 keeps the guard. |
| Stage 2 not finished by Sunday 20:00 EST | Deliver by the rule above from what has finished; Stage 2 continues and is judged later. |
| Teachers' predictions on the 366 new LEAP hands | Expected fine (D1-bv trained on them; CF2 generalises); in-training eval tracked. |
| In-training "new_hands_fresh" eval hands overlap with the 414 | That in-training metric is contaminated for CF-3f-L; it is reported only and not used. |
| Test splits | Untouched until the user decides. Then: the delivered model only, once, regime kite_code, stored start pose, single seed. |

## 6. Practice checklist (before launch)
- **Preregistration:** this doc, frozen before launch; the gate script for CF-3f written and dry-run on CF-3e vs itself (must
  give 0 differences) before any CF-3f result exists.
- **One variable per arm;** a frozen baseline (CF-3e) with checksums.
- **Leakage audit:** script asserting that no training entry references the 105 val topologies, Wuji, Shadow, external hands,
  P0800–P0899 or test splits; the config generator asserts only the listed keys change from CF-3e.
- **Smoke test:** a few hundred steps per arm (loss finite, LR schedule as intended, teachers load, records counted).
- **Reproducibility:** provenance (config, code snapshot, data hashes) saved per run; fixed eval seeds and start draws as for CF-3e.
- **Statistics:** 3 seeds, paired bootstrap within seed, decisions only from the predeclared gates; everything else is
  descriptive (many pools → no claims from reported-only numbers).
- **Sanity floors:** no-movement baseline and CF2 / CF-3e references in every readout.
- **Deployment checks on the delivered model:** inference latency and memory vs CF-3e (same architecture, expected equal); only
  deployment-time inputs reach the decoder.
- **Honest scope:** results hold for the 9 unseen families tested and simulated kinematic contact targets; no real-robot
  evaluation; ψ stays frozen (Wuji is unseen by our decoder but seen by ψ).

## 7. Known limits CF-3f does not address
- Rule-E ~90° frame flips (9 of 360 LEAP hands, MetaHand ≈ 2 mm behind D1): needs an axis-rule fix, proposed separately later.
- New generated bodies beyond 800 and model size: not this weekend.
- Seed averaging: excluded (user ruling); may be revisited after CF-3f as its own question.

## 8. Compute and timeline
- GPU-hours: CF-3f-L ≈ 15–18; CF-3f-S ≈ 36–54; Stage 2 ≈ 36–54. Total ≈ 90–125 GPU-h; 1 small GPU per job; `ampere|ada`.
- If launched Friday midday: L readout Friday evening; S readout Saturday midday; Stage 2 Saturday midday → Sunday morning.
- Prep ≈ 1–1.5 h without GPU: config generator, leakage audit, gate script + self-test, chain scripts, smoke runs.

## 9. Decisions for the user
1. Approve Stage 1 (CF-3f-L and CF-3f-S, 6 jobs), or CF-3f-L only.
2. Confirm CF-3e seed 0 as the interim delivery model.
3. Confirm the soft Shadow guard (or strict).
4. Confirm the Sunday 20:00 EST delivery cutoff.

Status: v3 2026-11-27; CF-3e frozen; Aryan's feedback received (divergence-check fix adopted). Stage 1 test arms approved by the user and launched 2026-11-27 (see 3a); full-scale run awaits their results and the user's go-ahead.

## Result: per-hand frame test (2026-11-27 03:30 EST; job 85375, 5.5 min)
G1 PASSES in 3/3 seeds. Margin 0 (predeclared): never switched on any well-framed LEAP hand (105 pool, source LEAP, 351 non-flip
360 hands), nor Allegro, Barrett, Wuji, Shadow, Aero -> identical to rule E there.
- MetaHand: switched in all seeds; corr 10.38 / 11.13 / 11.66 -> 7.81 / 7.79 / 7.78 (sig.), ≈ D1-bv 8.02 / 7.55 / 7.56.
- 9 flip hands: 2/9 switched (both 90°); corr 10.09 / 9.97 / 9.98 -> 9.14 / 9.13 / 8.98 (sig.), D1-bv 7.88 / 8.19 / 8.03 (≈ half the gap).
- False switches: Sharpa 3/3 seeds (sig. worse +0.32 / +0.38 in 2), Inspire 1 seed (sig. worse +0.49), Ability 3/3 (mixed),
  XHand, Schunk 1 seed each (n.s.).
- Descriptive (post hoc): margin 0.25 mm removes all false switches and keeps all MetaHand / flip gains. Not a valid test (chosen
  after seeing); no fresh unseen family left to confirm on.
Adoption = inference-procedure change (32-request calibration per new hand): the user's call.

## Result: rec arm (prim rows [0,100], 100 records per generated body instead of 25) — FAIL (2026-11-27 10:53 EST)
Q1 F (Barrett corr sig. worse s0), Q2 F (105 pool: +0.013 / −0.022 / +0.076 mm, sig. worse s2), Q3 F (Wuji corr sig. worse s0,
+0.235), Q4 T (held-out generated bodies −0.111 / +0.067 / +0.051), Q5 F (Shadow wrist mean +0.038; corr +0.29 / +0.20 / −0.71).
Reading: more records per generated body gives no gain over CF-3e; noise-level differences, a few sig. worse. Still better than
CF2 and D1-bv everywhere (as CF-3e is). -> not included in the full-scale run. Gate file ${GETKITE_RUNS}/cf3f_rec_gate.json.

## Result: leap arm (D1-bv's 414 LEAP hands instead of 48) — FAIL (2026-11-27 11:55 EST)
All of Q1–Q5 fail. 105 pool mixed: corr −0.064 / −0.151 / +0.042 (sig. better s0, s1; sig. worse s2). New families hurt:
Wuji corr +1.35 / +1.19 / +0.32 (sig. worse 3/3; s0 also sig. worse than CF2), Shadow +1.57 / +1.12 / −0.03 (sig. worse 2/3);
held-out generated bodies +0.26 / +0.21 / +0.26 (limit +0.10); Barrett sig. worse 2/3.
Reading (CORRECTED after Aryan's question): the per-family share did NOT change — sampler is family-uniform over 4 families
(allegro / barrett / leap / prim); replay of the sampler on both runs' real records gives 0.25 each; LEAP budget 19,872 vs 20,016
records (414 × 48 vs 48 × 417). Only LEAP breadth changed. Every family got worse in training-log val (Allegro, Barrett, generated
bodies +0.1–0.26), consistent with a harder LEAP quarter pulling capacity from the rest (opinion); teacher mismatch on the 366 LEAP
hands CF2 teachers never saw is a possible contributor (not tested). -> not included in the full-scale run. Gate file ${GETKITE_RUNS}/cf3f_leap_gate.json.

## Result: big arm (d_model 256, 6 layers, 8 heads) — FAIL on Q3 / Q5 (2026-11-27 12:16 EST)
Q1 T, Q2 T, Q4 T; Q3 F, Q5 F. Large, consistent gains on seen families: source corr −0.19 to −0.46 (sig. better 9/9 pool×seed);
105 pool −0.319 / −0.472 / −0.254 (sig. better 3/3, the best of any arm so far); held-out generated bodies −0.31 / −0.29 / −0.18.
New families split by seed: Wuji corr +0.354 / +0.009 / −0.188 (sig. worse s0, sig. better s2); Shadow corr +0.60 / +0.07 / −0.33
(sig. worse s0, sig. better s2), wrist mean +0.18. vs CF2 / D1-bv still better on Wuji and Shadow corr in all seeds.
Predeclared verdict: FAIL. Discussion point (not a re-scoring): the failure is carried by seed 0, the delivery seed. External-hands
eval for big submitted early (reported only) to add new-family evidence.
External hands for big (job 107175; reported only; per-seed point estimates, no paired CI vs CF-3e in this eval): big − CF-3e corr
on 9 unseen-family hands (7 external + Wuji + Shadow), mean +0.47 / +0.29 / −0.07 mm; worse on 7/9 hands in s0 and s1, better on
6/9 in s2. MetaHand +1.58 / +1.47 / +0.56. Still better than CF2 on 8–9/9 hands per seed. Confirms the Q3/Q5 failure: the larger
model gives up part of CF-3e's new-family gain (most in s0, the delivery seed).

## Result: teach arm (9 teachers: CF-3e's 6 + the 3 frozen CF-3e seeds) — FAIL (2026-11-27 13:07 EST)
All of Q1–Q5 fail. Worse than CF-3e almost everywhere: 105 pool +0.075 / +0.005 / +0.057 (sig. worse s0, s2); Wuji corr +0.33 /
+0.81 / +0.18 (sig. worse 3/3); Shadow +0.40 / +0.86 / −0.22 (sig. worse s0, s1); held-out generated bodies +0.05 / +0.39 / +0.10;
Allegro / Barrett sig. worse in some seeds. -> not included in the full-scale run. Gate file ${GETKITE_RUNS}/cf3f_teach_gate.json.
Big — seed-variation check (2026-11-27 13:15 EST; CORRECTS the "more seed-dependent" reading above): big's absolute new-family
error is MORE consistent across seeds than CF-3e's (per-hand seed range over the 9 unseen-family hands: big 0.30 mm mean vs CF-3e
0.63; Wuji big 7.69 / 7.65 / 7.69 vs CF-3e 7.34 / 7.64 / 7.88). The swing in the paired differences comes from CF-3e's seeds (s0 is
CF-3e's best on Wuji / Shadow, s2 its worst). No overfitting signature in training logs: held-out LEAP, held-out generated bodies
and fresh LEAP hands keep improving to 85–100k in every seed; selected checkpoints at 85–95k. Big is consistently ~0.3 mm worse
than CF-3e's average on unseen families while ~0.3–0.5 mm better in-family.

## Result: teacher-target check on the leap arm (user-approved; job 109275, 6 min; evaluation only) — mismatch SUPPORTED
6-teacher mean target vs ground truth on the leap arm's LEAP training records (fixed noisy start), corr_mm:
- 48 hands all teachers trained on (2,304 records): 6-teacher 6.86; D1-bv-only 6.93; CF2-only 7.11. 6-teacher − D1-bv-only
  −0.073 [−0.103, −0.044] (ensemble helps).
- 366 hands only D1-bv trained on (17,568 records): 6-teacher 7.35; D1-bv-only 7.02; CF2-only 8.09 (individual CF2 teachers
  8.44–8.63). 6-teacher − D1-bv-only +0.327 [+0.313, +0.341] (ensemble hurts).
- 6-teacher target error, 366 − 48 hands (bootstrap over hands): +0.49 [+0.32, +0.67].
Predeclared (a) and (b) both hold -> teacher mismatch is a real factor: on 88 % of its LEAP records the leap arm was pulled toward
targets ~0.3 mm worse than its best available teachers. It does not by itself explain the losses on families whose targets did
not change (Allegro, Barrett, generated bodies); the capacity explanation remains a separate, untested contributor.
Implication (opinion): any future LEAP-breadth arm needs teachers that trained on its hands (e.g. D1-bv-only targets for hands
outside CF2's set). Not for this weekend. Output ${GETKITE_RUNS}/teacher_target_leap_v1/teacher_target.json.

## Long-arm snapshot curve (user-approved 2026-11-27 13:25 EST; evaluation only)
Question: does longer training improve CF-3e-size new-family transfer, and is it still improving? (Decides whether a big-size
×300k arm is worth testing.) Snapshots of each long seed's last.pt at 175k / 225k / 275k (slurm/cf3f_long_snap.sh) + the final
selected model; external-hands eval (9 unseen-family hands incl. Wuji / Shadow) on each, vs frozen CF-3e (100k). 100k / 150k
were not capturable (seed 1 already past; intermediate checkpoints are not kept). Caveat: mid-run snapshots are at a high
learning rate on the 300k cosine schedule, so they understate a finished model at that length; the curve's shape is the signal.

## Big continued 100k -> 200k ("bigext"; user request 2026-11-27 13:30 EST)
Resumes each big seed's last.pt (step 100k; optimizer state and best-checkpoint record carried over) into new dirs
cf3f_bigext_hf[_sN] (big's own runs untouched); total_steps 200000, so the cosine schedule warm-restarts at 0.5× peak LR at 100k
and decays to 0 at 200k. Jobs 112108–112110. When all 3 are DONE, slurm/cf3f_bigext_chain.sh runs the standard + 105-pool
benchmarks and external hands for: CF-3e, big@100k, bigext selected (best.pt over 0–200k, source-val rule) and bigext final
(200k last.pt). Read: does more training close big's new-family gap (vs CF-3e and vs long's curve) or widen it?
Caveat: a warm restart is not identical to a single 200k schedule; it answers "does continuing help", not "is 200k from scratch better".

## Teacher-target check on CF-3e's generated bodies (user-approved 2026-11-27 ~13:50 EST; evaluation only) — predeclared
Records: CF-3e's own 800 generated bodies × 25 training records. CF2 trained on 50 of them (rows 0–400), D1-bv on all 800 (rows
0–25), so "seen" = the 50 (all 6 teachers trained on them), "unseen" = the other 750 (only D1-bv did). Same targets, same score and
same reading as the leap check, fixed before running: matched teachers are a promising lever only if BOTH (a) on the 750, the
6-teacher target has significantly higher corr_mm than the D1-bv-only target (paired over records) while not on the 50, and (b) the
6-teacher target's corr_mm on the 750 exceeds that on the 50 (bootstrap over bodies, CI above 0). Otherwise not supported.
Script scripts/decoder/benchmark/teacher_target_eval.py prim -> ${GETKITE_RUNS}/teacher_target_prim_v1/teacher_target.json.
Result (job 113160, 8.7 min): SUPPORTED — (a) and (b) both hold. corr_mm of the target vs ground truth:
- 50 bodies all teachers trained on (1,250 records): 6-teacher 4.58; D1-bv-only 4.69; CF2-only 4.77. 6 − D1-bv-only −0.107
  [−0.145, −0.069] (ensemble helps).
- 750 bodies only D1-bv trained on (18,750 records): 6-teacher 4.88; D1-bv-only 4.66; CF2-only 5.52 (individual CF2 teachers
  5.99–6.14). 6 − D1-bv-only +0.225 [+0.212, +0.237] (ensemble hurts); wrist +0.121 [+0.112, +0.129].
- 6-teacher target error, 750 − 50 bodies (bootstrap over bodies): +0.30 [+0.05, +0.54].
Reading: CF-3e itself was distilled toward ~0.22 mm worse-than-available targets on 94 % of its generated-body records. Candidate
arm "matched teachers": per record, average only teachers that trained on that hand/body (D1-bv-only for the 750; all 6
elsewhere). Needs a small code change + checks; awaiting user approval. Effect on the student not predictable from this alone.

## Arm "match" (matched teachers) — launched 2026-11-27 ~14:00 EST (user-approved)
CF-3e + distill_matched: per record, the teacher target averages only the teachers whose training set contained that robot
(train_joint.teacher_match: kite entries by robot name, LEAP / generated bodies by npz hand_id; robots no teacher trained on keep
all 6). Effect on CF-3e's training robots: 100 (Allegro, Barrett, 48 LEAP, CF2's 50 bodies) keep all 6; 750 generated bodies use
the 3 D1-bv teachers. Config diff vs frozen CF-3e = {distill_matched, out_dir, _note} only (asserted).
Checks (job 115636, all PASS): patterns as expected (0 mismatches); all-True match = unmatched pose (1.9e-6); matched = D1-bv-only
mean on the 750-type bodies (1.8e-6) and = 6-teacher mean elsewhere (1.9e-6); 300-step smoke finite.
Gate = the same Q1–Q5 vs CF-3e (cf3f_gate.py match); divergence check at 25k; chain slurm/cf3f_match_chain.sh; external hands after.

## Origin checks (user-approved 2026-11-27 ~14:15 EST)
1. Does origin distance matter on real hands? (read-only, existing external-hands results). Rule-O1 origin->palm distance for the
   9 unseen-family hands: Wuji 4.52, Shadow 4.41, Ability 4.75, Inspire 7.01, XHand 5.73, Schunk 4.19, MetaHand 4.03, Sharpa 4.85,
   Aero 4.86 cm (Allegro 4.02, Barrett 3.14; LEAP ~6.1). Spearman vs mean-over-seeds error (n = 9): CF-3e corr −0.20 (p 0.61),
   CF-3e wrist −0.47 (p 0.21), CF-3e corr / no-movement corr −0.42 (p 0.26), CF2 corr −0.07 (p 0.86); D1-bv (native frame) corr
   +0.70 (p 0.04). Reading: no sign that origin distance hurts the canonical-frame models on real hands (the hardest hands are
   MetaHand, a frame-flip case, and Schunk, both with near-typical origins); the native-frame D1-bv does get worse with distance.
   Caveat: n = 9, distances 4–7 cm, one hand at 7 cm. -> origin robustness is a lower priority for the CF line than the synthetic
   2 cm test suggested.
2. Measurement check (scripts/decoder/benchmark/origin_measure_eval.py; predeclared in its docstring): CF-O vs CF2 wrist error
   re-scored at base / O1 / O2 on Wuji, Shadow and the 7 external hands. Note: CF-O's contact-error deficit cannot be a
   measurement effect (corr_mm does not depend on the reference point); this check can only explain the wrist part.
   Result (job 116277, 1.4 min): NOT a measurement effect (predeclared: base deficit reproduced = true, gone at O1 and O2 = false).
   CF-O − CF2 wrist, mean of 3 seeds (sig. worse seeds): Wuji base +0.72 (3) -> O1 +0.41 (3) / O2 +0.49 (3); Shadow +1.23 (3) ->
   +0.89 (3) / +0.92 (3). Across the 7 external hands the deficit shrinks at the palm point (e.g. Aero +1.09 -> +0.15, XHand
   +1.09 -> +0.29, Inspire +2.03 -> +0.83) but stays positive on all except Aero at O1. So roughly a third to half of CF-O's
   base-origin wrist deficit is lever arm; the rest (and its contact deficit) is a real model loss. CF-3e is better than CF2 at
   every reference point on all 9 hands, more so at the palm point (Wuji −0.32 base -> −0.56 O1; Shadow −0.29 -> −0.41).
3. Origin-shift training arm: NOT warranted now — (1) origin distance doesn't track CF-model error on real hands, (2) CF-O's
   failure is a real model loss, not measurement. Revisit only if a new family with an unusual origin shows up badly.

## In-training signal analysis (user-approved 2026-11-27 ~14:30 EST; read-only, CPU; scripts/decoder/benchmark/proxy_signal_analysis.py)
Question: which training-log metric at the selected checkpoint predicts real new-family error? Target = mean of Wuji and Shadow
corr_mm from the benchmarks (69 models with logs; 0 inconsistent repeats across benchmark files); also the 7 external hands (18).
- Across all 69 models every signal correlates (Spearman +0.53..+0.74; source val +0.74, held-out bodies +0.67), but that is
  dominated by large quality gaps (early D-level models vs CF-3e).
- Decision-relevant test: same-seed pairs of the 30 canonical-frame models whose Wuji+Shadow error differs by >= 0.15 mm
  (123 pairs): the signal picks the right winner — held-out generated bodies 87.8 %, Barrett val 86.2 %, source val 85.4 %,
  LEAP val 83.7 %, fresh LEAP hands 82.9 %, held-out LEAP 75.6 %, train loss 70.7 %. (External hands: 16 pairs, 75–88 %.)
- Today's decisions: every signal agreed with Wuji+Shadow for leap, teach (2/3), CF-3d and CF2 vs CF-3e, but EVERY signal was
  wrong for big s0 and s1 (big better on all in-training metrics, worse on real new families), and mixed for rec.
Reading (CORRECTS the earlier "held-out bodies don't predict real new families"): held-out bodies are the best available signal
and usually right; they fail when a change improves in-family fit (model size), which no logged metric catches. -> keep held-out
bodies for selection; for capacity-type changes require real new-family validation (the named validation hands) before adoption.
Body-count evidence from older native-frame 50k runs (no teacher): 50 bodies 9.94 mm vs 800 bodies 10.01 (seed spread ~1–2 mm)
-> uninformative for CF-3e's setting; the 400-vs-800 test remains open.

## Queued for discussion (user 2026-11-27 ~14:40 EST): more generated bodies via a new teacher — decide after the "match" result
Facts (verified): kite_data/build_prim.py is deterministic per (seed, index), so --n > 900 adds P0900+ without changing P0000–P0899
(to be confirmed by hash before use); 900 bodies took 197 s, their records 51 s; some templates needed the FPS repair pass
(fps_repair.json), which new bodies must also get; the exclusion rule keeps Wuji's 5×4 layout and the nearest-to-Shadow layout out.
Constraint: no existing teacher trained on new bodies, so (per both teacher-target checks) their distillation targets would be
noisy; GT-only on new bodies ≈ CF-bv (no gain). Plan sketch (not approved): (1) generate e.g. 800 more bodies (P0900–P1699) +
records + checks; (2) stage 1: retrain a D1-bv-type teacher on 1,600 bodies (3 seeds × 100k, ~8 h); (3) stage 2: CF-3e-type student
with matched teachers (new D1-bv teachers on all 1,600 bodies; CF2 where it trained), gate Q1–Q5 vs CF-3e (~8 h). Total ≈ 18–20 h
incl. benchmarks. Go / no-go depends on: match passing (teacher quality matters) and the timeline; the 400-vs-800 body-count test
(not approved) would say whether more bodies are likely to help at all.

## Queued for discussion (user 2026-11-27 ~14:50 EST): fair LEAP-breadth retest — decide after the "match" result
leap arm failed from (verified) teacher mismatch on 366/414 hands + (likely) capacity. Fair retest sketch (not approved): 414 LEAP
hands with matched teachers (D1-bv-only targets where CF2 never trained), probably the big model size, judged with Q1–Q5 AND
confirmation on the named new-family validation hands (capacity-type change; in-training signals missed big). Only if match passes.

## Aryan's conditions on the two-stage more-bodies plan — checks (2026-11-27 ~15:00 EST; read-only)
- Sampler (verified, multifamily.BalancedSampler): family-uniform, then body-uniform. 1,600 bodies × 25 records keeps the
  generated-body share at 25 % and LEAP / Allegro / Barrett unchanged; each body is drawn ~4,000 times in 100k steps (batch 256)
  instead of ~8,000. Body records double (40,000); that is not a second change to the family mix.
- CF-3d -> CF-3e (verified from configs): both 20,000 body records, 48 LEAP hands × 417, 100k steps, same 6 teachers; bodies 50 × 400
  -> 800 × 25. Teacher coverage on bodies differs: CF-3d's 50 bodies are all in CF2's training set (all 6 teachers matched); 750 of
  CF-3e's 800 are not (+0.22 mm noisier targets, verified). CF2 (50 bodies, no teacher) -> CF-bv (800, no teacher): no gain.
- Finger-layout overlap (verified; per-finger count of non-fixed joints from the palm group to the tip, mimic joints included):
  Wuji (4,4,4,4,4), Shadow / Sharpa (4,4,4,5,5), Schunk (3,4,4,5): 0 generated bodies match. Exact matches among CF-3e's 800
  training bodies: Ability (2,2,2,2,2) 2 [P0158, P0529]; Inspire (2,2,2,2,4) 1 [P0777]; XHand (2,2,2,3,3) 5 [P0021, P0127, P0316,
  P0390, +1]; MetaHand (4,4,4,4) 2 [P0593, P0770]; Aero (3,3,3,3,4) 4 [P0135, P0167, P0195, P0398]. 14 / 800 (1.75 %); none in the
  50 test / 50 held-out bodies. Primitive geometry differs from the real hands, so this is layout-level, not shape-level, overlap;
  but 5 of the 7 external families are not strictly layout-unseen for any model trained on the 800 bodies (D1-bv, CF-3e, CF-3f).
