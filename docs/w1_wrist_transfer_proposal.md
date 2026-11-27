# W1: source-only wrist-transfer diagnostic (proposal, 2026-11-19). NOT LAUNCHED.

> **Status (2026-11-27): launched and completed on 2026-11-19**; results in docs/decoder_audit.md sections 49, 50 and 54.


**Question.** Fingers already at the goal, true request contacts as input, only the wrist perturbed: does a wrist correction
learned on source hands (Allegro, Barrett, 12 LEAP) transfer to Wuji?

## Configuration

Config: `configs/decoder/scratch/w1_wristonly_true50k_hf.json`. Trainer: `getkite.decoder.train_scratch` via
`slurm/scratch.sbatch`. Out dir: `runs/w1_wristonly_true50k_hf`.

**Base.** `c3_raw50k_cf8_10x_wrist_hf`, i.e. J1's motion component (raw-contact GET, exact start-wrist transform,
hand_frame surface sampling) fed the true contacts. Inherited unchanged:
- source records (J1's: Allegro/Barrett 20k each, 12 LEAP);
- seed 0, BalancedSampler;
- architecture (d128, 4 layers, Fourier 8+8, contact Fourier 8);
- residual loss (root + joint, family-balanced);
- AdamW 3e-4, warmup 500, cosine;
- batch 256, 50k steps, normal jitter 0.1 rad;
- selection on source val (macro success, then contact mm).

**Exactly what changes:**

| item | reference | W1 |
|---|---|---|
| `start_joint_frac` (train, val and eval starts) | 0.05 × range | **0**: fingers start exactly at the goal joints |
| wrist start perturbation | N(0, 5 mm) xyz, N(0, 5°) per Euler angle | unchanged; the same draws as the frozen benchmark (the joint noise is still drawn, then multiplied by 0) |
| joint residual target | goal − start | 0 (follows from the above); the joint loss term is kept, not removed |
| input | true contacts (diagnostic reference, not a deployment input) | unchanged |
| out_dir | scratch_c3_raw50k_cf8_10x_wrist_hf | w1_wristonly_true50k_hf |

Wuji is never loaded in training or selection.

**Measured budget.**
- Timing probe: 1,000 steps on an RTX 2080 Ti, the same GPU type as the reference run; probe output in
  `runs/_timing_w1`, throwaway.
- Throughput 2,810 examples/s, peak 1.03 GB.
- 50k × 256 ≈ 76 min of training plus about 20 validation passes ≈ **1.3–1.4 GPU-h**, one job.
- Evaluation afterwards (`scripts/decoder/benchmark/wrist_transfer_eval.py`): under 1 GPU-min.

## Evaluation (source and Wuji reported separately)

**Protocol.** Wrist-only starts (`fixed_start(val_start_seed, joint_frac=0)`) on:
- source val, training goals and held-out LEAP (source learning);
- Wuji near-goal (transfer; unseen by our decoders, seen by KITE's ψ).

**Systems:**
- no movement;
- `raw_ref` (same model trained with the standard starts; control);
- W1;
- KITE released (in-family reference; trained on Wuji).

Paired bootstrap per record: W1 − no movement and W1 − raw_ref.

## Pre-launch control (existing checkpoints only; `runs/w1_wrist_transfer_eval_pre`)

Wrist-only starts, true contacts:

| pool | no movement wrist mm | raw_ref wrist mm (Δ vs no movement, 95% CI) | raw_ref trans R²_id | KITE released wrist mm |
|---|---|---|---|---|
| val Allegro | 7.95 | 4.57 (−3.37 [−3.65, −3.11]) | .64 | 3.36 |
| val Barrett | 8.02 | 4.44 (−3.58 [−3.86, −3.30]) | .65 | 3.53 |
| val LEAP | 7.75 | 5.56 (−2.18 [−2.45, −1.93]) | .45 | – (no head) |
| held-out LEAP | 7.96 | 5.97 (−1.99 [−2.23, −1.73]) | .42 | – |
| **Wuji** | 7.96 | **10.33 (+2.38 [+1.96, +2.81])** | **−.75** | 2.83 mm contact / 3.85 wrist |

Even with the fingers at the goal and the true contacts given, the existing source-trained decoder moves Wuji's wrist the
wrong way, while it corrects every source and held-out LEAP hand. W1 asks whether training on exactly this problem
changes that.

## Predeclared reading (single seed; diagnostic, not a claim)

- **Source learned:**
  - W1 val wrist error is lower than raw_ref's on all three source families (paired CI excludes 0).
  - Otherwise, "removing finger noise does not improve the source wrist mapping".
- **Wuji transfer:**
  - **Transfers:** Wuji wrist_mm *and* corr_mm are below no movement (CI excludes 0) and translation R²_id > 0. Then
    the J1/G1 Wuji failure lies in the finger/contact parts, not wrist inference.
  - **Fails:** Wuji wrist_mm is above no movement (CI excludes 0), as raw_ref is now. With source learned, this means
    wrist inference itself is morphology-specific: how the new hand's geometry is read, independent of ψ, contact
    prediction and finger control.
  - **Otherwise:** partial. No follow-up without approval.
