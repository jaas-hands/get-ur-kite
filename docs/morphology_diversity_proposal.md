# Morphology-diversity test (D-levels): plan (draft, 2026-11-19)

Status: proposal only. Nothing has been built, generated or launched. Awaiting the user's final go-ahead.

This is distinct from G2 in `track_g_morphology_manifest.md`, which means 4× goals on the same hands.

## Question

Does training on many structurally varied hands (not only LEAP variants) improve transfer to unseen families?
- Wuji: unseen by our decoder, seen by KITE's encoder ψ.
- Shadow: unseen by both.

## Background

- J1 / G1 train on 3 families: 1 Allegro, 1 Barrett, and 12 / 48 LEAP variants.
- G1 beats J1 on Wuji in 3 of 3 seeds, but by a small amount (compat success +.075 ± .036).
- Within-family held-out LEAP transfers well.

## Levels

Each level has the same total of about 60k records; only the number and kind of bodies changes.

| level | training bodies | role |
|---|---|---|
| D0 | G1 (Allegro, Barrett, 48 LEAP variants) | exists |
| D1 | about 50 generated bodies | same count as G1, broader variety. Separates the kind of variety from its amount. |
| D2 | about 200 generated bodies | deferred; only if D1 and D3 differ |
| D3 | about 800 generated bodies | the maximum-diversity test |

- Agreed first step: D1 and D3, seed 0, in parallel.
- Seeds 1 and 2, and D2, only after the seed-0 results.

## Unchanged from G1

Copied from `configs/decoder/scratch/g1_leapdiv50k_hf.json`:
- Model: d_model 128, 4 layers, 4 heads.
- Contact predictor and loss: contact_weight 1.0.
- Frozen ψ (KITE release).
- Optimiser: lr 3e-4, warmup 500, batch 256.
- 50k steps; start noise resampled with KITE values.
- hand-frame surface sampling.
- Checkpoint selected on source validation only.
- G1 took 36 GPU-minutes.

## New: the generator

An extension of `get-ur-kite-dataset-kite-families`; the existing LEAP outputs stay unchanged.
- Primitive geometry: box palm, capsule links. LEAP's thumb housing is fused into its palm mesh.
- Fingers: 2–5.
- Joints per finger: 2–4, with an optional base abduction joint and different axis orders.
- Link lengths: 0.6–1.5×.
- Palm: 0.7–1.4×.
- Finger spacing and mount angle.
- Thumb base position and orientation.
- Validation reuses the repo's tests:
  - FK against MuJoCo;
  - collision checks;
  - surface template;
  - the record format the decoder reads.
- Exclusion: drop bodies too close to Wuji or Shadow. The rule is still to be defined.
- Also needed: 500 evaluation-only Shadow records (the adapter is validated; there are no records yet).

## Evaluation

The fixed benchmark:
- source validation;
- held-out generated bodies;
- Wuji;
- Shadow.

Per-seed reporting, no best seed, no pooling.

## Execution

1. Build and validate the generator.
2. Generate the D1 and D3 bodies and records, the held-out bodies, and the Shadow records.
3. Write the configs and run a smoke test.
4. Report: bodies, checks and the measured budget.
5. On the user's go-ahead, launch D1 and D3 together.

Runtime:
- `--requeue` and the USR1 handler resume from `last.pt` (saved every 1,000 steps).
- The existing exclude list and GPU constraint apply.
- A watcher checks all runs. It resubmits any run that dies without `DONE`, and adds bad nodes to the exclude list.

## Open decisions

- Exact variation ranges, especially the joint-axis layouts.
- The rule for "too close to Wuji or Shadow".
- Final body counts.
