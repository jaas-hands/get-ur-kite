# T1: translation-only diagnostic for the direct decoder (proposal, 2026-11-19). NOT LAUNCHED.

> **Status (2026-11-27): launched and completed on 2026-11-19**; results in docs/decoder_audit.md sections 52 and 53.


Separate from W1 and from the multi-family runs. Code: `src/getkite/decoder/translation_diag.py`. Config:
`configs/decoder/scratch/t1_translation_allegro.json`. Out dir: `runs/t1_translation_allegro/{get,mlp}`.

## Motivation: where the movement path stops responding to intent

Layer-by-layer z-sensitivity probe (`scripts/decoder/diagnostics/direct_zperm_probe.py --layers`, output
`runs/direct_zperm_probe/probe_layers_val_start_seed.json`):
- 384 source-val goals at the benchmark starts; z permuted across goals.
- Each stage's relative sensitivity = mean |x(z) − x(z perm)| / across-goal SD. 0 means it ignores z; about 1.1–1.4 means
  it is fully z-dependent.

| stage | J1 (positive control) | Track D | v1 best | A1 |
|---|---|---|---|---|
| trunk h / J1 contact stage | 1.15 | 1.10 | 1.10 | 1.10 |
| intent tokens at GET input | 1.25 | 1.08 | 1.03 | 0.83 |
| root token after block 0 / 1 / 2 / 3 | 0.32 / 0.94 / 1.22 / 1.41 | 0.20 / 0.32 / 0.35 / 0.41 | 0.31 / 0.42 / 0.44 / 0.47 | 0.36 / 0.37 / 0.36 / 0.40 |
| final root token: z-delta / across-goal SD | 0.96 / 0.72 | 0.06 / 0.14 | 0.08 / 0.16 | 0.06 / 0.14 |
| wrist output res_root: z-delta / across-goal SD | 1.12 / 0.60 | 0.0006 / 0.003 | 0.0013 / 0.005 | 0.0005 / 0.002 |

The intent tokens stay fully z-dependent through all four blocks in every model, so z does reach them. The movement path
stops responding at two places:
1. **Blocks 1–3 (intent → root propagation).** In J1 the root token's z-dependence builds from 0.32 to 1.41. In the
   direct models it plateaus at 0.35–0.47, and the final root token barely varies across goals (SD 0.14–0.16 vs 0.72).
2. **The root head.** It maps that small variation to an almost constant wrist output (SD 0.002–0.005 vs J1 0.60).

## Design

**One Allegro morphology** (KITE nominal URDF, the J1 robot entry), hand_frame surface sampling.

**One base goal** (`allegro_right_train.npz`, row 0):
- fixed finger configuration, fixed wrist orientation, and its 6 contacts.
- Contacts reproduce through our FK to 9e-6 mm, and still to 1.7e-5 mm after translation.

**Targets:**
- the whole goal rigidly translated by Δ ~ U[−0.10, 0.10]³ m: goal wrist t2 + Δ, contacts c + Δ; normals,
  orientation and fingers unchanged.
- Train: 20,000 targets (seed 11). Validation: 2,000 *fresh* targets (seed 22); the nearest training target is 0.55 mm
  away (continuous draws). A fixed 2,000-target subset of the training targets is scored as "training goals".

**Starts:**
- translation only: goal + N(0, 5 mm) per axis (KITE's translation noise; measured SD 4.97 mm).
- Orientation and fingers exactly equal to the goal's (checked: 0.0 / 1.2e-7).
- Resampled every batch in training; fixed (val_start_seed) for evaluation.

**Inputs at inference:**
- frozen z = ψ(request contacts) (normal jitter 0.1 rad in training only, as in all runs);
- the start state;
- the robot description.
- No reconstructed contact coordinates enter the movement path. The GET's aux contact head is built but has zero effect
  on outputs and receives no gradient (both checked).

**Target:** KITE's normalised root residual, the same for both models: translation in the start-wrist frame / res_xyz_m,
plus a 6D rotation part that is 0 here (checked at 1.4e-6).

| | direct GET | z + start-pose network |
|---|---|---|
| architecture | direct_worldaux_v1: z-only trunk (512 × 3) → 4 intent tokens + start-wrist world-pose feature → GET (d128, 4 layers, wrist frame), fresh init | [standardised z, the same start world-pose feature (t1 · pos_scale, 6D R1, Fourier 8/8)] → 3 × [Linear 512, LayerNorm, GELU] → 9 root residuals |
| loss | GET residual loss: root (mean of 9) + joint (target 0; zero-init joint head stays 0) | root loss (mean of 9) |
| optimiser | v1's groups: GET + token projections lr 3e-4; trunk lr 1e-3; AdamW wd 1e-4; clip 1.0 each | AdamW lr 1e-3, wd 1e-4, clip 1.0 |
| trained params | 1,956,938 | 742,921 |

**Shared settings:**
- z standardisation from the training targets;
- batch 256, **10,000 steps** (2.56 M examples), warmup 500 + cosine over 10k, seed 0;
- evaluation every 1k;
- best checkpoint kept by validation translation error.

**Reported** (training subset and validation separately):
- translation error mm, mean and median;
- no-movement translation error (7.9 / 8.0 mm);
- how far the wrist moved;
- rotation error;
- correction correlation in the start-wrist frame: slope, R²_lin, R²_id, and per-axis Pearson r between predicted and
  required translation.

**Reference (evaluation only, already computed in the checks):**
- a closed-form ridge readout of the target wrist position from standardised z recovers it to **0.0017 mm** on
  validation (0.0017 on training targets).
- In this task the target is linearly available in z almost exactly. A failure would be about learning or using it, not
  about information.

## Measured budget (timing probe, 500 steps each, RTX A5500, throwaway output `runs/_timing_t1`)

- **Direct GET:** 3,300 ex/s, 1.25 GB peak → 10k steps ≈ 13 min, plus 10 evaluations.
- **Network:** 19,450 ex/s, 0.43 GB → ≈ 2.5 min.
- **Total:** about 20 GPU-min for both models, run sequentially in one job (time limit 1 h).
- At 500 steps (during warmup) both were still at no movement. This is timing only, not a result.

## Predeclared reading

- **Network learns, GET does not:** with intent linearly available and only translation to infer, the GET's
  intent → root path is the bottleneck, consistent with the probe.
  - Network learns: validation translation ≤ 50% of no movement, i.e. ≤ 4.0 mm, and slope ≥ .5.
  - GET does not learn: within 10% of no movement, i.e. ≥ 7.1 mm, or slope < .1.
- **Both learn:** the multi-family direct failure is not a basic inability of the GET to read translation from intent
  tokens. It points to the multi-family / rotation / finger setting.
- **Neither learns:** suspect the z standardisation, target scale or budget before drawing conclusions. No conclusion
  about GET.
- Anything else: partial. Single seed; no follow-up without approval.
