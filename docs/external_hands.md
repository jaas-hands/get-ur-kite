# External evaluation hands (2026-11-26)

Purpose: more hand families to test generalisation on, beyond Wuji (seen by ψ) and Shadow. **Evaluation only: never used for
training, selection or tuning** (same status as Shadow). ψ (KITE's encoder) was trained on Wuji, Allegro, Barrett, Robotiq gripper
and MANO (KITE `conf/kite.yaml`), so all hands below are unseen by ψ and by every decoder.

## What the KITE checkout has

URDF/MJCF robot files only for Allegro, Barrett, Wuji (+ Robotiq gripper, MANO without the model files). KITE's
`assets/collision_primitives/` shows its authors also used Ability, Inspire, Schunk, XHand, MetaHand (SPIDER asset set) and
EZGripper, Robotiq 3-finger, LEAP, Shadow (DRO-Grasp asset set); those robot files are not shipped. The KITE checkout is not
modified; all downloads live in `$GETKITE_EXTERNAL` (default `../external_assets/` next to this repo); fetch them at the pinned commits with
`bash scripts/decoder/external_hands/fetch_assets.sh`, then build the adapters with `slurm/external_hands.sbatch` and
`slurm/shadow_records.sbatch` (after `scripts/decoder/shadow/make_shadow_urdf.py`).

## Sources (downloaded 2026-11-26)

| Hand | File | Licence | Independent coords / physical joints | Self-collision geometry |
|---|---|---|---|---|
| Ability | SPIDER `ability/right.xml` | CC-BY-NC | 10 / 10 | KITE's fitted primitives |
| Inspire | SPIDER `inspire/right.xml` | CC-BY-NC | 12 / 12 | KITE's fitted primitives |
| XHand | SPIDER `xhand/right.xml` | CC-BY-NC | 12 / 12 | KITE's fitted primitives |
| Schunk SVH | SPIDER `schunk/right.xml` | CC-BY-NC | 20 / 20 | KITE's fitted primitives |
| MetaHand (4 fingers) | SPIDER `metahand/right.xml` | CC-BY-NC | 16 / 16 | KITE's fitted primitives |
| Sharpa Wave | Menagerie `sharpa_wave/right_hand.xml` | Apache-2.0 | 22 / 22 | Menagerie collision geoms |
| Tetheria Aero | Menagerie `tetheria_aero_hand_open/right_hand.xml` | Apache-2.0 | 11 / 16 (5 couplings as mimic) | Menagerie collision geoms |

SPIDER repo commit 1d607af (2026-11-17); Menagerie sparse checkout. Shadow DEX-EE was dropped (its fingers are not under a palm body).
SPIDER files model coupled joints as independent; kept independent, as KITE does for Barrett.

## Pipeline

`scripts/decoder/external_hands/make_hand_adapter.py` (generalises `scripts/decoder/shadow/`): MJCF → KITE-style `*_extended` URDF
(root body = floating base, six virtual joints, hinges from the compiled MJCF, joint equality constraints → mimic), a 4096-point
visibility-filtered surface template, and 500 records in the kite_code regime (as Shadow): wrist U[-0.4, 0.4]^3 and uniform
rotation, KITE's joint mixture, self-colliding poses rejected, 2–6 contacts on uniformly chosen regions (palm + one per finger).
Checks per hand (`<hand>_records_check.json`): record round trip exact, decoder FK vs MuJoCo ≤ 3e-8 m, no goal out of limits,
every finger covered by the decoder's surface sample, KITE primitives enclose 92–100 % of template points within 3 mm.
Smoke run (30 records each) passed for all seven; full build job 33115 (CPU). Evaluation: `scripts/decoder/benchmark/external_hands_eval.py`
(D1-long, CF2, D1-bv per seed, 3-seed ensembles, mixed ensemble; job 33117) and, after CF-3d finishes, the same with CF-3d
(`slurm/cf3d_external_chain.sh`).

## Caveats

- Not KITE's own sampler (KITE's registry does not include these hands); same caveat as Shadow.
- Self-collision geometry differs by source (KITE primitives vs Menagerie geoms); acceptance rates differ by hand (1–80 %).
- Hand sizes differ from the training families; no rescaling.

## Baseline results (2026-11-26 03:11 EST; job 33117; `${GETKITE_RUNS}/external_hands_v1/external_hands.json`)

All 7 hands built and passed every check (job 33115). Contact error (mm), seeds 0/1/2, and 3-seed ensembles; no movement for reference.

| Hand | No movement | D1-long | CF2 | D1-bv | Ens D1-long / CF2 / D1-bv / mixed D1-bv+CF2 |
|---|---|---|---|---|---|
| Ability | 13.75 | 13.38 / 9.33 / 11.48 | 9.60 / 9.81 / 9.73 | 8.99 / 9.38 / 9.34 | 9.89 / 8.78 / 8.25 / **7.86** |
| Inspire | 18.46 | 22.89 / 15.05 / 17.94 | 10.10 / 10.55 / 10.58 | 15.52 / 16.83 / 13.82 | 16.61 / **9.63** / 14.38 / 10.42 |
| XHand | 15.47 | 18.24 / 14.11 / 14.86 | 10.59 / 10.71 / 10.82 | 14.75 / 15.61 / 13.42 | 14.38 / **10.00** / 13.32 / 10.57 |
| Schunk SVH | 14.45 | 15.42 / 11.56 / 12.00 | 10.41 / 11.13 / 10.54 | 11.19 / 11.30 / 11.14 | 11.69 / 9.92 / 10.42 / **9.68** |
| MetaHand | 12.61 | 8.33 / 7.82 / 7.38 | 11.98 / 13.16 / 12.19 | 7.99 / 7.53 / 7.59 | 7.14 / 11.65 / **7.09** / 8.45 |
| Sharpa Wave | 14.11 | 13.96 / 10.20 / 10.80 | 10.65 / 10.35 / 10.89 | 10.05 / 10.65 / 10.38 | 10.33 / 9.86 / 9.57 / **9.23** |
| Tetheria Aero | 14.22 | 13.86 / 10.50 / 13.57 | 9.92 / 9.77 / 10.02 | 9.88 / 11.27 / 10.70 | 11.45 / 9.23 / 9.76 / **8.94** |
| Wuji (ref.) | 14.19 | 9.55 / 9.03 / 9.76 | 8.23 / 8.78 / 9.00 | 8.41 / 8.57 / 8.90 | 8.73 / 8.12 / 8.02 / **7.82** |
| Shadow (ref.) | 13.97 | 12.36 / 10.15 / 11.17 | 8.97 / 9.20 / 9.47 | 9.24 / 10.18 / 10.03 | 9.96 / 8.55 / 8.97 / **8.13** |

Reading:
- CF2 beats no movement on every new hand and is the most stable model across seeds. It is significantly better than D1-long on
  Inspire, XHand, Schunk and Aero in 3/3 seeds (by 0.4 to 12.8 mm) and better than D1-bv on Inspire, XHand and Schunk.
- D1 models are very seed-variable on new hands (D1-long on Inspire 15.1–22.9 mm) and sometimes worse than no movement.
- **MetaHand is CF2's clear failure**: CF2 11.98–13.16 mm, about no movement (12.61), while both D1 models reach 7.4–8.3. MetaHand
  has 4 fingers. Frame probe (job 33187) checks whether rule E picks the wrong axes for it.
- Mixed D1-bv+CF2 ensemble is best on Ability, Schunk, Sharpa, Aero (and Wuji, Shadow); the CF2 ensemble is best on Inspire and XHand,
  where D1-bv is poor; D1-bv is best on MetaHand.
- Wrist error: CF2 beats no movement (7.96) on every hand except Schunk (8.2–8.5) and MetaHand (8.7–9.0).

## Frame probe (2026-11-26 03:19 EST; job 33187; `${GETKITE_RUNS}/external_frame_probe/frame_probe.json`; diagnostic only)

CF2 scored with rule E's frame Q replaced by each of the 24 axis permutations / sign flips Q·P. Rule E's frame is the best of the 24
on Inspire, XHand, Schunk, Sharpa, Aero, Wuji and Shadow, and second on Ability (within 0.3 mm). **MetaHand: rule E's frame ranks
8th; a 90° turn about the finger axis d (n and a swapped) brings CF2 from 11.98 / 13.16 / 12.19 to 8.18 / 8.52 / 8.21 mm, near D1
(7.4–8.3).** So CF2's MetaHand failure is a rule-E frame error, not a decoder failure. MetaHand has the lowest finger/axis agreement
(f·d = 0.69; others 0.90–1.00). The frame that fixes it was found by looking at MetaHand, so it must not be used to pick a rule.

## Same failure on held-out LEAP (360-hand pool, 2026-11-26; CPU analysis of job 32982 outputs)

Frame angle of each of the 360 hands to the mean training-LEAP frame (`transfer_benchmark_v1_leap360/leap_frame_angles.json`):
137 hands < 3°, 106 at 3–6°, 80 at 6–10°, 25 at 10–15°, 3 at 15–45°, **9 hands (2.5 %) at ~82–90°** (a MetaHand-type n/a swap).
CF2 − D1-long per angle bin (s0 / s1 / s2): < 3° −0.09 / +0.02 / +0.04; 10–15° +0.35 / +0.46 / +0.50; **> 45°: +2.53 / +3.04 / +2.64 mm**
(worst hands +4.5 to +6.9). Excluding the 9 flipped hands the pool gap falls from +0.07 / +0.18 / +0.19 to +0.01 / +0.11 / +0.13.
All 5 worst hands have f·d 0.71–0.78, but low f·d alone does not mark a flip (median angle of the 35 hands with f·d < 0.85 is 3.5°).
The 105-hand val pool has no flipped hand (max ~20°), which is why the flips were not seen before.
