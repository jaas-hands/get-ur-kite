# CF proposal: a shared palm frame computed from each hand's own geometry (2026-11-22)

Status (2026-11-22): proposal written. Step 1 gate failed for the first rule; revised to rule E on geometry only (see "Step 1 outcome"). Implementation and tests done. CF × 3 trained and evaluated 2026-11-25 (audit §63): Primary 2 holds, Primary 1 fails by one seed (held-out LEAP corr); not promoted.

## Why (from §61)

- W2-a: D1's wrist relies on each training family's fixed base-frame convention. Relabelling a hand's base frame destroys its
  wrist translation, even on the training hands.
- W2-b (random base frame per example) removes that dependence and fixes Shadow's wrist in 3/3 seeds. It beats no movement in
  every seed, translation R²_id goes from about −.08 to +.17, corr improves by 1.6 mm and compat by .14. But source and
  held-out LEAP get worse on every metric in 3/3 seeds (corr +1.4–1.9 mm), so it fails non-inferiority.
- Reading: the decoder benefits from a **consistent** frame. It is hurt by a frame that means different things on different
  hands (Shadow) and by a frame that is random (W2-b, which has to learn the mapping for every orientation).
- CF gives every hand the **same** frame meaning, computed from its description and not learned, so the decoder sees one
  convention everywhere.

## The canonical frame (deterministic; deployment-time inputs only)

(Steps 4–6 below are the first rule, "rule A". It failed the Step 1 gate and was replaced by rule E; see "Step 1 outcome".)

Computed once per robot from its description (URDF / descriptor + surface template), which is a deployment-time input for any
robot. It never uses records, contacts, the wrist poses of examples, or anything learned. All quantities are in the robot's
native base frame.

1. **Palm points P:** template points on links rigidly attached to the base link (connected only through fixed joints).
   Barrett's base is a virtual link with no points, so this picks up bh_base_link.
2. **Distal points D:** template points on links that have points and no descendant link with points (the fingertip links).
3. **Reference pose q₀:** all coordinates 0, clamped to the joint limits. **Workspace:** S = 512 joint configurations drawn
   uniformly within the limits from a fixed seed (mimic joints follow their sources).
4. **Distal axis d** = normalise(mean D at q₀ − mean P).
5. **Palmar axis n** = normalise(the component orthogonal to d of mean_S(mean D) − mean D at q₀): the direction in which the
   fingertips move on average over the joint range ("curl side").
6. **Across axis a** = d × n, so (n, a, d) is a proper right-handed frame.
7. **Q** = [n a d] (columns in the native base frame). The robot is re-expressed with `relabel_base(desc, Q)`, the same
   operation as W2-a / W2-b, which is verified to leave world geometry unchanged (6e-8 m). The new base axes are x = palmar,
   y = across, z = distal, the generated bodies' native convention. Only the rotation changes; the base origin is kept, as one
   change at a time.

Properties:
- **Exactly invariant to relabelling.** If a hand's description is relabelled by any R, its points move by Rᵀ, so n, a and d
  move by Rᵀ and the canonical frame is the same physical frame. Re-running the W2-a relabelling test on CF must give drop = 0
  up to float error. This is an implementation check, not a result.
- Used identically for training and evaluation: the start and goal wrist rotations become R·Q_r and the output is mapped back
  as R'·Q_rᵀ. Scoring stays in the original bank, so every metric is computed on the same physical quantities as every
  other arm.

## Step 1: frame check (CPU, before any training; predeclared)

`scripts/decoder/dlevels/canonical_frame_check.py`, run on the 900 draw-A generated bodies, the training LEAP variants, Allegro,
Barrett, Wuji and Shadow.
- **Generated bodies:** their convention is known by construction (x = palmar, y = across, z = distal), so the angle between
  the canonical frame and the native frame is a direct correctness check. This uses generator metadata only for checking;
  it never reaches the decoder.
- **Named hands:** report Q, |pre-normalisation n| (how well determined the curl side is), and which native axis each
  canonical axis lies closest to. Shadow is expected to map to x ← y, y ← ±x, z ← z (its convention, §60a).
- **Gate:**
  - Proceed if ≥ 90% of generated bodies are within 30° of their native frame, and every named hand's d points from the palm
    towards the fingers and n to the palm side, checked against the hand's known geometry and reported per hand.
  - Otherwise report and revise the rule before any training. No tuning on Wuji or Shadow results: the check is geometric.

## Step 1 outcome and rule revision (2026-11-22)

The first rule (d = palm centroid → distal centroid; n = curl orthogonal to d; "rule A" above) **failed the gate**. Only 76.4%
of generated bodies came within 30° (median 15.9°, p90 45.4°), and the 52 LEAP variants, which share one native frame, were
up to 49° from their mean frame. The distal axis is pulled sideways by where the fingers sit on the palm.

As the gate prescribes, the rule was revised **on geometry only** (generated-body convention and LEAP-variant consistency).
No decoder results were used, and nothing on Wuji or Shadow. Candidates, over all 900 generated bodies:

| rule | within 30° | median | p90 | LEAP spread to mean frame (median / max) |
|---|---|---|---|---|
| A palm centroid → distal centroid | .764 | 15.9° | 45.4° | 16.9° / 49.0° |
| B mean per-finger direction (root joint → fingertip) | .792 | 17.1° | 38.5° | 11.9° / 39.8° |
| C as B, dropping the most deviant finger (≥ 3 fingers) | .772 | 15.0° | 38.9° | 9.9° / 59.3° |
| D as B, with per-finger curl | .798 | 17.0° | 41.0° | 10.1° / 34.5° |
| **E palm principal axes, picked and signed by B's finger direction and the curl** | **.917** | **2.8°** | **22.8°** | **3.1° / 84.0°** |

**Rule E (adopted):** V = principal axes of the palm points (native base frame). d = the axis of V best aligned with the
mean per-finger direction (sign from it). n = the remaining axis best aligned with the curl (sign from it). a = d × n.
- **Passes the gate:** 91.7% of generated bodies are within 30°.
- **Named hands:** d palm → fingers and n to the palm side on every named hand, all within 16° of a native axis except
  Allegro (31°). Final per-hand numbers: configs/decoder/dlevels/cf_frame_check.json.
- **Known weakness:** one LEAP variant (I0-M4-R0-T3_g0, two fingers) picks a different palm axis for d (87° from the
  others). Its mean finger direction lies between two palm axes. The other 51 are within 17.4° of each other (median 3°).
  This affects one of the 52 training LEAP variants. It is reported, not patched, to avoid tuning the rule further.
- **Invariance to relabelling** still holds exactly: the principal axes rotate with the description, and their signs come
  from rotating vectors. The code and tests use rule E.

## Step 2: CF training arm (needs go-ahead)

- **CF** = D1 at seeds 0/1/2 plus one change: config `canonical_frame: true`. Every robot is expressed in its canonical frame
  for training, validation, checkpoint selection and the benchmark. `base_frame_aug` stays off.
- Configs `cf_canon_hf{,_s1,_s2}` are generated from D1. A diff check confirms that only canonical_frame / out_dir / _note
  change.
- Cost: three D1-size runs, with the same partition, constraint and exclude list.
- **Benchmark:** transfer_v1 + Shadow with G1 / D1 / W2-b / CF, paired by seed. Plus the relabelling test on CF (expected
  drop ≈ 0).
- **Predeclared readout:**
  - **Primary 1 (non-inferiority):** CF vs D1 on source Allegro / Barrett / LEAP and held-out LEAP corr mm and compat. Not
    worse in 3/3 seeds on any of them.
  - **Primary 2 (Shadow):** CF vs D1 on Shadow wrist mm and translation R²_id, better in 3/3 seeds.
  - **Secondary:** Shadow and Wuji corr / compat vs D1 and vs W2-b; source vs W2-b.
  - **Promotion:** only if both primaries hold. Otherwise report it as is, and no follow-up variant without the user.
  - 3/3 counts. No seed is chosen, no examples are pooled, nothing is selected on Wuji or Shadow. Wuji is reported separately
    as "unseen by our decoder, seen by KITE's encoder (ψ)".
- **Caveat:** Shadow has been the readout for D1c, W2-b and now CF. It has never been used to train, select or tune, but a
  clean claim would need a further unseen hand, which needs assets.

## Not proposed

Canonicalising the base origin (e.g. palm centroid), mixing CF with W2-b augmentation, longer training, or wrist head / loss
changes. One change at a time; only after CF results.
