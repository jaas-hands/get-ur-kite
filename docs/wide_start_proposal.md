# Wide-start arm (predeclared 2026-11-28, before any training)

Run on Aryan's compute, from branch `extender-getting-kite-universal`. Made by `scripts/decoder/dlevels/make_cf3f_widestart.py`.

## Why

CF-3e learns one correction from a start near the goal: KITE's recipe draws the start as the goal plus Gaussian noise of
5 mm per wrist axis, 5 deg per rotation axis and 5 % of each joint's range (`train_scratch.start_noise`). In Janice's first
Oracle-Intent rollout (Barrett, mug, episode 55) CF-3e missed the grasp where KITE's decoder succeeded. The replayed intents
keep advancing whatever the hand does, so once the hand lags further behind than the training starts cover, CF-3e works
outside its training data. The true-contact bound (`true_contact_bound_summary.py`) puts nearly all remaining error in the GET
decoder, not the contact stage, so a decoder-side change is the right place to act.

## The one change vs CF-3e (`cf3e_long_hf*`)

- Training batches only: each example's start is drawn with KITE's noise with probability 0.5, or with every noise scale
  x3 (15 mm / 15 deg / 15 %) with probability 0.5 (`train_joint.draw_start_mix`).
- On the wide examples the teacher term uses the ground-truth target (`start_wide_teacher: "label"`): the six teachers only
  trained on KITE-noise starts, so their targets from wide starts are out of distribution. KITE-noise examples keep CF-3e's
  six-teacher distillation unchanged.
- Everything else as CF-3e: data, teachers, alpha 0.5, model, 100k steps, seeds 0-2, canonical frame. Validation and
  checkpoint selection use KITE's noise (unchanged `noise`), so selection is comparable with CF-3e.
- Configs: `configs/decoder/scratch/cf3f_widestart_hf{,_s1,_s2}.json`; the generator asserts only the three `start_wide_*`
  keys differ from CF-3e.

## Readout (one job: `scripts/decoder/benchmark/run_widestart_eval.sh`)

The arm and frozen CF-3e of the same seed are scored in the same benchmark run, on the same records:

- near-goal starts (KITE noise): standard pools + Shadow, and the 105 held-out LEAP pool;
- wide starts (`start_noise_mult` 3.0): the same pools and records with starts at 3x KITE noise;
- the 7 external hands, near-goal (reported only).

Paired bootstrap over records within seed; seeds never pooled.

**Pass = all of** (`cf3f_gate.py widestart`):

- Q1, Q3, Q4, Q5 as for every CF-3f arm (no loss on source pools; Wuji not significantly worse; held-out generated bodies
  within +0.10 mm; Shadow soft guard);
- Q2 as non-inferiority: on the 105 held-out LEAP pool at near-goal starts, corr_mm not significantly higher than CF-3e in
  any seed (the arm is not expected to gain near the goal, only not to lose);
- W1 (the gain): at 3x-noise starts, on the 105 pool, corr_mm lower than CF-3e in 3/3 seeds, significantly lower in >= 2/3,
  significantly higher in none;
- W2: at 3x-noise starts, Wuji and Shadow corr_mm not significantly higher in any seed.

Reported, not gating: success, median and 90th percentile; wrist errors; the external hands; Janice's closed-loop episodes
(Placing / Keyboard, same recorded intents for both decoders) once they run. Delivery stays CF-3e seed 0 unless the arm passes.
