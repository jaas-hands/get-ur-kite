# Item 3 design: canonical base origin (CF-O) (2026-11-25; draft for review, nothing implemented or launched)

> **Status (2026-11-27): implemented, trained and failed.** CF-O (runs cfo_long_hf*) failed its gate (Wuji / Shadow worse); see
> "Result" below. A 2026-11-27 check showed the wrist deficit is a real loss, not a measurement effect
> (docs/cf3f_proposal.md, "Origin checks", on the getting-kite-universal branch). Origin line closed.


## Why (evidence, docs/multipass_origin_checks_proposal.md, item 2)

- Every model (D1, D1-long, CF2, D1-bv) is strongly sensitive to where a hand's base origin sits. Moving the origin 2 cm with the
  physical hand unchanged costs +1.9 to +4.0 mm source contact error (3/3 seeds); 1 cm costs 0.5–2.4 mm.
- Families place the origin differently relative to the palm: the palm-point centroid sits 3.8–4.1 cm from the native origin
  on Allegro, Wuji and Shadow, and ~6.0 cm (median) on LEAP variants. So an unseen hand arrives with its origin centimetres
  from where the training families taught the decoder to expect it.
- CF canonicalised the axes (rotation) and fixed most of the base-frame convention dependence found by W2-a; it kept the
  native origin on purpose (one change at a time). This is the matching change for translation.
- Hypothesis: part of the remaining wrist error on unseen hands (Wuji, Shadow) is origin-convention mismatch. CF-O tests it.

## The canonical origin (deterministic; deployment-time inputs only)

Computed once per robot from its description and surface template, the same inputs CF uses; never records, contacts, example
wrist poses or anything learned.
- **Palm points P** = template points on links rigidly attached to the base (fixed joints only), exactly CF's P.
- **Rule O1 (proposed): o = centroid of the palm points' bounding box in the canonical frame** (mid-range of P along n, a, d).
  The box centre depends only on the palm's extent, not on how densely the template samples each part of the palm, which a
  plain mean would.
- **Fallback rule O2 (only if O1 fails the Step 1 gate): o = mean of P.**
- Rejected: the mean of the finger-root joint origins. It moves when a finger is removed (test_finger_subset-style hands), so
  the "same palm" would get different origins.
- The robot is re-expressed with the origin moved to o and axes rotated by Q (CF's rule E): base-link template points
  p → Qᵀ(p − o); joints attached to the base T → (Qᵀ, −Qᵀo)·T; start / goal wrist pose (R, t) → (R Q, t + R o). The output
  is mapped back: R = R' Qᵀ, t = t' − R o. World geometry is unchanged (checked numerically, as for CF and the item-2 test).
  Scoring stays in the native bank.

Properties to verify (implementation checks, not results):
- **Exactly invariant to base-frame relabelling (rotation)** and **to base-origin shifts**: o and Q move with the description,
  so the canonical frame is the same physical frame. Re-running the item-2 origin-shift test and the 24-rotation relabelling
  test on CF-O must give a spread of ≈ 0.
- No effect on psi's input (contacts are world-frame).

## Step 1: geometry check (CPU only, before any training; predeclared)

`scripts/decoder/dlevels/canonical_origin_check.py`, on the 900 generated bodies, all 1,811 LEAP variants (train / val split
only; test splits not read), Allegro, Barrett, Wuji and Shadow.
- **Generated bodies:** their palm is a box whose centre is known by construction: (0, 0, length/2) in the palm frame, with
  the native origin at the wrist end of the palm (`kite_data/prim_hand.py`; `spec.palm_size` in each hand.json; used only for
  checking). Gate: O1 within 5 mm of the box centre for ≥ 95% of bodies.
- **LEAP variants:** the origin's position relative to the palm should change smoothly with geometry: across geometry draws of
  the same topology, report the spread of o in the canonical frame; flag any variant > 1 cm from its topology's median
  (a sign of a degenerate palm, like CF's flipped frames).
- **Named hands:** report o in the native frame and its distance from the native origin (expected 3.8–4.1 cm on Allegro /
  Wuji / Shadow, ~6 cm on LEAP), and check visually against the palm mesh.
- If the gate fails: try O2 on geometry only and report. No tuning on decoder results, nothing on Wuji or Shadow.

## Step 1 result (2026-11-25, job 24850, CPU; configs/decoder/dlevels/cfo_origin_check.json)

**Gate passed; rule O1 adopted.**
- Generated bodies (900): O1 within 5 mm of the true palm-box centre for 100% (median 0.03 mm, max 0.39 mm). O2 (plain mean)
  also passes but is 40× less exact (median 1.2 mm), as expected from uneven template sampling.
- LEAP variants (1,071 train + val hands, 153 topologies; test splits not read): in the native frame, O1 moves smoothly with
  geometry: deviation from its topology's median 0.9 mm median, 1.8 mm p90, 3.7 mm max; no hand > 1 cm. Measured in canonical
  coordinates (as predeclared), 79 hands deviate > 1 cm; all of these are CF axis flips (the canonical axes of that hand flip
  relative to its topology, so the same physical point gets different canonical coordinates), not origin errors. Side
  note (corrected 2026-11-25 by docs/steady_axis_rule_proposal.md): the 79 flags are hands tilted roughly ≥ 10° relative to
  their topology; only 1.9% are > 30° (true flips).
- O1 distance from the native origin: Allegro 4.0 cm, Barrett 3.1, Wuji 4.5, Shadow 4.4; LEAP 5.8–6.3 cm (median 6.1).
- Invariance: O1 of a rotated + shifted description is the same physical point (max error 1e-16 m). No errors on any hand.

## Step 2: training arm (needs go-ahead)

- **CF-O = the current best CF model + one change: canonical origin.** Base decided by CF-J's result (2026-11-25: CF-J did NOT meet
  its readout, so **base = CF2**, configs `cfo_long_hf*` from `cf2_long_hf*`). The rule as written beforehand:
  - if CF-J meets its predeclared readout → CF-O = CF-J + origin (configs `cfo_long_hf*` from `cfj_long_hf*`);
  - otherwise → CF-O = CF2 + origin (from `cf2_long_hf*`).
- 100k steps, seeds 0/1/2, A data, same partition / constraint / exclude list. Config flag `canonical_origin: "o1"`
  (CanonicalFrame wrapper and train_joint; off by default; diff check: only canonical_origin / out_dir / _note change).
- **Optional extra arm (listed for approval, not assumed): D1-O** = D1-long + canonical origin only (no rotation), to separate
  the origin effect from CF. Without it, the comparison is CF-O vs its base.
- Cost: 3 runs × ~3.6 h (6 if D1-O is approved), in parallel, + ~30 min evaluation. Preemption adds delay (CF-J lost ~1 h today).

## Predeclared readout (fixed before any run)

Benchmarks: transfer_v1 (4 held-out LEAP + Shadow) and the 105-hand held-out LEAP pool; systems frozen D1, D1-long, D1-bv,
the base CF model, CF-O, paired by seed.
- **Implementation checks:** origin-shift spread ≈ 0 and relabelling spread ≈ 0 for CF-O.
- **Primary 1 (non-inferiority vs its base):** no significant loss on source Allegro / Barrett / LEAP or held-out LEAP (105)
  corr_mm / compat_success in any seed.
- **Primary 2 (the hypothesis):** on Wuji (unseen by our decoder, seen by KITE's encoder ψ) and Shadow (unseen by decoder and ψ),
  wrist_mm lower than the base in 3/3 seeds, significant in ≥ 2/3, and corr_mm not significantly higher in any seed.
- **Primary 3 (delivery bar, as before):** contact error lower than the matched D1 on every family in 3/3 seeds.
- Reported, not decided on: transfer beyond no movement per pool; trans_R2_id; the 105-hand gap by frame angle.
- **Delivery:** CF-O becomes the default only if Primaries 1–3 hold; otherwise report blocking families and the choice goes to
  the user.
- **Caveats:** Wuji and Shadow have been the readout for many rounds (never trained, selected or tuned on). The training starts
  are generated by rotating about the native origin, so in the canonical origin the start-noise structure differs per hand;
  this is the same physical data, but it means the origin effect cannot be read off the item-2 test alone — only from the
  unseen-hand readout above. A clean claim needs an untouched set (e.g. the reserved test_geometry split; user's decision).

## Not proposed (one change at a time)

Learned origins; origin jitter (consider only if CF-O shows the same per-hand sensitivity CF had, as with CF-J); combining with
body variety; changing KITE's start-noise recipe.

## Conditional combined run (user approval 2026-11-25)

Approved in advance, conditional on CF-O passing. Launch condition (scripts/decoder/benchmark/cfo_gate.py): implementation checks
(relabelling and origin-shift spread ≤ 0.01) AND Primary 1 AND Primary 2 above; Primary 3 is reported, not gating.
`slurm/cfo_gate_launch.sh` waits for CF-O's evaluation, runs the gate and, only on a pass, submits `cfo_bv_long_hf*` (CF-O + B-scale
body variety: cf_bv_long_hf* + canonical_origin; configs from scripts/decoder/dlevels/make_cfo_bv.py) and its evaluation chain
`slurm/cfobv_chain.sh` (transfer_v1_cfobv, the 105-hand pool, relabelling and origin-shift tests). Log: slurm/logs/cfo_gate.log.
The gate was dry-run on CF-J's results (as a stand-in) and reproduced their known numbers.

## Result (2026-11-26 00:06 EST; jobs 25006–25008 trained, evaluation 27656–27659; gate cf2_readout/cfo_gate.json)

**CF-O fails; the combined run was correctly NOT launched (gate launch=false).**
- Implementation checks pass: relabelling spread 1e-4, origin-shift spread 1e-4 (a 2 cm shift costs CF-O 0.00 mm vs CF2's
  +1.6 to +2.2 mm on source). CF-O is exactly origin- and rotation-invariant, as designed.
- **Primary 2 fails, in the wrong direction: Wuji and Shadow get much worse.** CF-O − CF2, wrist mm: Wuji +1.07 / +0.51 / +0.56,
  Shadow +1.45 / +1.19 / +1.05; contact error Wuji +0.77 / +0.35 / +0.65, Shadow +1.03 / +1.76 / +0.55 — significantly worse in
  3/3 seeds. Shadow wrist 8.62–8.95 mm (no movement 7.96; CF2 7.50–7.74); Shadow translation R²_id −0.21 to −0.28 (CF2 +0.03 to
  +0.09); Wuji translation R²_id 0.01–0.07 (CF2 0.16–0.25).
- Primary 1 fails: held-out LEAP (105) +0.07 / +0.01 / +0.15 mm (seeds 0, 2 significant); LEAP source seed 2 +0.21; Allegro
  compat seed 0 −0.04. Source otherwise ≈ CF2. Primary 3 fails (held-out LEAP higher than D1-long 3/3).

**Reading (interpretation, consistent with the caveat above).** The native origin is not just a naming convention: KITE's start
perturbation rotates the hand about its native base origin, so in the native frame rotation and translation corrections are
decoupled. Moving the origin to the palm centre couples them through a lever arm (the palm-centre offset, 3–6 cm and different per
family). The decoder must infer that offset from geometry, which it does on the training families but not on Wuji / Shadow, whose
offsets differ. So the origin sensitivity found in item 2 partly reflects real task structure (where the perturbation's rotation
centre is), and canonicalising the origin removes useful information. Not pursued further without the user.

## Follow-up: was the rotation centre the cause? (2026-11-26 02:20 EST; job 32469; evaluation only)

Test (`scripts/decoder/benchmark/start_centre_eval.py`, `${GETKITE_RUNS}/start_centre_v1/start_centre.json`): CF2 vs CF-O, seeds
0–2, same records and noise draws, three start-noise conditions — native (KITE's recipe), norot (rotation noise 0) and palm (same
rotation draws about the palm centre o: t1 = t2 + (R2 − R1) o + noise). Predicted if the rotation-centre reading held: CF-O's
Wuji/Shadow deficit mostly vanishes under norot and reverses under palm.

**Result: the rotation-centre reading is refuted as the main cause.** CF-O − CF2, wrist mm (all significant):
- Wuji: native +1.07 / +0.51 / +0.56; norot +1.13 / +0.61 / +0.70; palm +0.78 / +0.32 / +0.39.
- Shadow: native +1.45 / +1.19 / +1.05; norot +1.52 / +1.23 / +1.08; palm +0.95 / +0.76 / +0.51.
- Contact error: Wuji worse in all conditions 3/3; Shadow norot mixed (+0.03, +1.35, −0.20), palm +0.37 / +1.43 / +0.13.
- Source and held-out LEAP: same small CF-O cost (+0.0 to +0.3) in every condition.
The deficit persists with no rotation at all and only shrinks (does not reverse) with palm-centred rotation, so at most a minor part
of CF-O's loss comes from the perturbation's rotation centre. The earlier "Reading" paragraph above is superseded.
Remaining candidates (untested): (a) part of the wrist_mm penalty is measurement — wrist_mm scores the native origin, and CF-O's
rotation error moves that point through the 4–5 cm lever arm (δR × o), but contact error (frame-free) is also worse on Wuji, so
not all of it; (b) the bounding-box palm centre is a less consistent anatomical landmark across families, so on unseen hands the
canonical origin lands somewhere the decoder has not learned to reason from. No clear retraining path; origin line stays closed.
