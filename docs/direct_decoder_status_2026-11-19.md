# Direct-decoder investigation: status (2026-11-19)

Scope: the direct path z = ψ(request) → intent tokens → GET → motion, with no explicit contact stage in the motion
path. Separate from W1 (wrist-transfer diagnostic) and from J1/G1. Updated the same day: the z-shuffling probe was made durable, with token statistics added, and the learned-anchor
diagnostic A1 was implemented and run once (10k steps), as approved. No further run is in progress.

## A. Completed experiments

| # | experiment | config | selected checkpoint | report | key finding |
|---|---|---|---|---|---|
| 1 | **direct_worldaux_v1**: z-only trunk → intent tokens (+ start-wrist world pose) → GET; training-only world-frame contact head on the trunk (never feeds motion). 50k × 256, joint motion + aux contact loss, J1 decoder optimiser. | `configs/decoder/scratch/direct_worldaux_v1.json` (code `src/getkite/decoder/direct_worldaux.py`, `train_direct_worldaux.py`) | `runs/direct_worldaux_v1/best.pt`, step 35,000 of 50k, selected on source val (success, then −contact mm): .1304 / 13.18 mm | `runs/transfer_benchmark_v1_direct/report.md`; audit §43 | The motion path collapsed to no movement on every pool, including its own **training goals** (val success .130 vs no movement .123; training goals .117 vs ≈.11; J1 .586 / .619). The wrist moves 0.02 mm. The aux contacts were learned (3.4–3.8 mm, count acc .95–.98). The trunk gradient from motion is 20–2000× smaller than from contact; the trunk clip was active on 100% of steps. Under the frozen benchmark it equals no movement everywhere (corr −0.04 to −0.17 mm; success within ±.016). Cross-body is 0.26%, the same as no movement. |
| 2 | **Tiny overfit check** (part of v1's pre-training checks): 64 fixed examples, 3,000 steps, same model and losses. | same config, overfit mode of `train_direct_worldaux.py` | n/a (check only) | `runs/direct_worldaux_v1_overfit64/overfit.json`, `runs/dwa_overfit.txt` | Contact error 13.31 mm → **0.015 mm**, success .0625 → **1.0**; motion loss 1.196 → 3e-6; aux 365 mm → 0.0002 mm. The motion path *can* carry a goal from z when it can memorise. Learnability at scale is not tested by this. |
| 3 | **Track D, frozen-trunk control**: v1's trunk, aux head and z stats (step 35k) loaded and frozen (eval mode, bit-identical, sha 4a37f7c39cf8a5b7 unchanged at the end). Fresh intent projection, pose conditioning and GET from v1's initial weights. **Motion loss only**; lr schedule identical to v1 (50k horizon); stopped at 10k. | `configs/decoder/scratch/direct_frozentrunk_d1.json` (code `src/getkite/decoder/train_direct_frozentrunk.py`) | `runs/direct_frozentrunk_d1/best.pt`, step 2,500 (val .1297 / 13.19 mm; later steps tie or are marginally lower); `last.pt` = step 10k | `runs/direct_frozentrunk_d1/report.md`; audit §44 | Identical to v1 at every matched step: val success .130/.128/.128 at 2.5k/5k/10k vs v1 .128/.128/.128, J1 .154/.221/.283. Training-goal success .113–.115 (no movement .110). Train motion loss equals v1 to 3 decimals. Removing the contact gradient and trunk drift changed nothing measurable. |
| 4 | **z-shuffling sensitivity probe** (no training): 384 source-val goals; z permuted across the batch; outputs compared. Checkpoints: D last, v1 best, v1 last. | `scripts/decoder/diagnostics/direct_zperm_probe.py` (made durable 2026-11-19; was an ad hoc scratchpad script) | n/a | `runs/direct_zperm_probe/probe_val_start_seed.json` (benchmark starts) and `probe.json` (original seed-0 starts) | Intent tokens change by 0.65 (magnitude 0.71), so z reaches the tokens. The wrist residual changes by **0.0006** against a near-constant 0.02 output; the joint residual by 0.023. Same in all three checkpoints. The motion network largely **ignores the intent tokens**. |
| 5 | **A1 learned-anchor diagnostic**: Track D with the intent tokens replaced by 6 learned world anchors, transformed exactly into the start wrist frame; motion loss only; 10k steps | `configs/decoder/scratch/direct_anchor_a1.json` (code `src/getkite/decoder/train_direct_anchor.py`) | `runs/direct_anchor_a1/best.pt`, step 10,000 (val .131 / 13.17 mm) | `runs/direct_anchor_a1/report.md` | **Fails** by the gate: equal to no movement and to Track D; the wrist moves ≤0.08 mm. The anchors spread out (to 297 mm within an example) but stay about 300 mm from the contacts. Does not rule out alignment (see C). |

Earlier ψ-token direct decoders that this builds on (audit §§12, 21, 35, 36; superseded configs, listed for context only):
- `scratch_c3_psi50k_10x_wrist` (§21): fails at no movement. Root output ≈0, only the joints move.
- `scratch_c3_aux50k_10x_wrist` (§35): direct with aux; no improvement over two-stage.
- Probes (§21 `runs/c3_psi_zprobe`, §36 `runs/aux_probe`):
  - z encodes absolute contact location linearly to about 1 mm.
  - Precise world position is lost already at the intent projection (1.8 → 36–45 mm on fresh goals).
  - The final tokens memorise training goals' wrist-frame centroid (0.2–1.8 mm) but not fresh ones (≈45 mm).

## B1. Verified implementation checks (`checks.json`, all `pass: true`)

| check | v1 | Track D |
|---|---|---|
| ψ frozen, eval mode | yes | yes |
| surface sampling hand_frame; surf_idx sha 6e9d3d5a912a0086 = J1's | yes | yes |
| new heads zero-initialised | yes | yes |
| labels are not inputs (perturbing labels changes motion by) | 0.0 | 0.0 |
| aux head removed / replaced → motion changes by | 0.0 / 0.0 | 0.0 / 0.0 |
| motion gradient into aux head | 0.0 | – (frozen) |
| contact-loss gradient into intent tokens / GET | 0.0 / 0.0 | – |
| mixed-robot batch vs single-robot | 1.2e-7 | 1.2e-7 |
| decode round-trip | 1.8e-7 | 1.8e-7 |
| set loss exact at labels; 1 cm shift → pos loss | ≈0; 1.0 | – |
| frozen weights and z stats equal v1; frozen grads none; frozen modules stay in eval under `.train()`; trainable set excludes frozen | – | all yes |
| aux contacts at load (val) | – | 3.68 mm, count acc .971 |
| 64-example overfit | success 1.0, 0.015 mm | – |

z-probe:
- Now version-controlled and rerun at the benchmark starts (`val_start_seed` 12345) and at the original seed-0 starts.
- Outputs are saved, and they reproduce the reported numbers: Track D wrist change 0.00063 vs output 0.0226; intent-token
  change 0.649.

## B2. Observed training failures

1. **Failure to fit, not failure to generalise** (v1): training-goal success .117 ≈ no movement .11, val .130 ≈ .123.
   The curve is flat from 2.5k to 50k.
2. **Wrist output is a near-constant offset:**
   - Movement is 0.02 mm / 0.05°.
   - Wrist slope ≈0 and R²_lin negative on every pool.
   - The joints move slightly (median progress .03–.06).
3. **Gradient imbalance in v1:**
   - Trunk gradient from contact 111 → 1.3, from motion 0.00 → 0.06. Trunk clip active on 100% of steps; decoder clip
     0%.
   - At init (checks): motion→trunk 0.0029 vs contact→trunk 1764.
4. **Tiny intent-pathway gradient in both v1 and D:**
   - Motion gradient into the intent-token projection is 0.002–0.007, against 0.15–0.46 into the GET blocks.
   - This persists in D, where no contact gradient exists.
5. **D reproduces v1's motion loss to 3 decimals** at every window, on both training and validation.

## B3. Explanations supported or weakened by the interventions

| explanation | status | evidence |
|---|---|---|
| The motion path cannot represent the mapping at all | **weakened** | 64-example overfit reaches 0.015 mm / success 1.0 |
| Shared-trunk contact-gradient dominance or trunk feature drift causes the failure | **weakened** (not sufficient) | Track D removes both (frozen trunk, motion loss only) → identical to v1 |
| The trunk features lack the spatial information | **weakened** | Frozen trunk features decode contacts to 3.68 mm (aux head); z is linearly decodable to about 1 mm (§21 probe) |
| Overfitting / generalisation gap | **weakened** | Training goals fail as badly as validation |
| The motion network does not *use* the intent tokens | **supported** | z-permutation changes the tokens (0.65) but not the wrist output (0.0006); tiny gradient into the intent projection |
| World-frame abstract intent is not turned into a wrist-frame correction (frame/state alignment) | **supported, not established** | consistent with §21 (raw contacts succeed once R0ᵀ(x − t0) is applied explicitly; ψ tokens fail), §36 (precise position lost at intent projection; wrist-frame centroid only memorised), and with J1 working because its intent tokens are positions in the same wrist frame as the robot tokens. No intervention has isolated it yet. |

## B4. Remaining hypotheses and tests (status updated 2026-11-19)

| hypothesis | proposed test | status |
|---|---|---|
| H1 Frame alignment: intent needs a geometric slot expressed in the start-wrist frame | **Learned-anchor diagnostic A1** (section C) | **run once (10k): fails by the gate; not ruled out** (frozen trunk saw contact supervision; anchors never approached the hand) |
| H2 Intent-pathway optimisation: the tokens get too little gradient or are swamped (token scale vs robot tokens) | Probe of token norms and per-token motion-loss gradients at the first GET block (`direct_zperm_probe.py`) | **run; weakened.** Intent token norms 9–11 vs robot tokens 12–19. Per-token gradient into intent tokens is about equal to the root token's and 10–18× the surface tokens' (D: 7.8e-6 intent, 6.5e-6 root, 4.4e-7 surface, 2.0e-5 joint). The tokens are neither scaled away nor gradient-starved; attention mass is still unmeasured. |
| H3 Data/compute scale: KITE's decoder reads z after about 2.7B samples trained jointly with ψ; ours had 12.8M | Previously only indirect (§11: the pretrained KITE trunk fine-tuned on LEAP decodes z, 8.2 mm at 30% of its run); a longer run is a budget decision | not proposed |
| H4 Mean-regression collapse: the near-constant wrist output is the residual mean | Probe: compare the output to the training-set mean residual; per-example variance | not run |

## C. Learned-anchor (intent–geometry alignment) diagnostic A1

**Status: implemented, checked and run once (10k steps, approved). Result: fails by the predeclared gate.**
- Code: `src/getkite/decoder/train_direct_anchor.py`.
- Config: `configs/decoder/scratch/direct_anchor_a1.json`.
- Report: `runs/direct_anchor_a1/report.md`.
- Selected checkpoint: `runs/direct_anchor_a1/best.pt`, step 10k (val .131 / 13.17 mm), preserved with `last.pt`.

**Design.** One change from Track D: the intent tokens.
- The frozen v1 trunk h feeds a fresh head that outputs 6 world-frame anchors plus a free 3-vector each.
- Each anchor is transformed exactly into the start wrist frame.
- The tokens use J1's contact-token layout, with Fourier features on the learned positions only.
- Training uses the motion loss only, with no contact labels, matching or existence.
- **Caveat:** the frozen trunk was trained with a contact loss in v1. A1 therefore tests the combined spatial-slot
  interface, not frame alignment alone.

**Result:**
- Source val success .129/.125/.128/.131 at 2.5/5/7.5/10k. Track D: .130/.128/.128 at 2.5/5/10k. No movement .123. J1 at
  10k: .283.
- Training goals .115 (no movement .110).
- The wrist moves 0.04–0.08 mm. Wrist error equals no movement (val 7.91, training goals 7.86).
- Training motion loss is identical to Track D (1.194 at 9.5–10k).

**Anchors:**
- They move and spread out: within an example 38 → 297 mm, across examples 29 → 203 mm, so they depend on z.
- They stay far from the hand: the nearest anchor is about 300 mm from every true contact, and the anchors are about
  0.5 m from the start wrist.
- Anchor-head gradient 1e-4 → 5e-2, roughly 10× Track D's intent projection. No clipping on any step.

**Does it preserve a direct path rather than recreate J1?** Yes. It is one motion-trained path, z → h → anchors → GET,
with no contact supervision on the anchors. It borrows J1's token interface, as noted in the design.

**Reading.** This does **not** rule out the alignment hypothesis. Untested reasons for the failure:
- the anchors start about 0.37 m from the hand and receive no pull toward it under motion-only loss;
- 10k steps may be too few;
- cold start: a model that does not move gets little informative gradient.

No follow-up has been launched.

## C2. Where the movement path stops responding (layer probe, 2026-11-19)

`direct_zperm_probe.py --layers`; output `runs/direct_zperm_probe/probe_layers_val_start_seed.json`; J1 as positive
control.
- The intent tokens stay fully z-dependent through all GET blocks in every direct model (relative sensitivity
  0.8–1.1).
- **Root token:** the z-dependence reaching it plateaus at 0.35–0.47 after blocks 1–3, vs J1 rising 0.32 → 1.41. The
  final root token barely varies across goals (SD 0.14–0.16 vs J1 0.72).
- **Root head:** it maps that small variation to an almost constant output (res_root SD 0.002–0.005 vs J1 0.60).
- T1 translation-only diagnostic (run 2026-11-19; runs/t1_translation_allegro/report.md):
  - z + start-pose network: 0.10 mm (slope 1.00).
  - Direct GET: 6.17 mm (no movement 7.91), slope .36, R²_id .35, per-axis r .89 / .46 / .22; plateaus; train = val.
  - Partial: the GET's intent → wrist bottleneck persists even for pure translation of one hand.
- T1 layer probe (audit §53; runs/t1_translation_allegro/layer_probe.json):
  - The required translation is already only partly encoded in the intent tokens at the GET input (R² x / y / z .60 /
    .11 / .02).
  - Blocks and head pass it on without loss (root token .78 / .18 / .01; output .78 / .20 / .05). Root attention on
    intent is .35–.49.
  - So the loss is at intent-token formation (additive z-projection + start world-pose feature).
  - Multi-family direct checkpoints also show weak intent → root routing (attention .09–.24 vs J1 .45–.66).

## D. Files

- Reports: `runs/transfer_benchmark_v1_direct/report.md`, `runs/direct_frozentrunk_d1/report.md`; audit §§43–44 (and
  context §§12, 21, 35, 36).
- Runs: `runs/direct_worldaux_v1` (best 35k, last 50k), `runs/direct_worldaux_v1_overfit64`,
  `runs/direct_frozentrunk_d1` (best 2.5k, last 10k), plus `_smoke` and `_checks` directories.
- z-probe: `scripts/decoder/diagnostics/direct_zperm_probe.py`, outputs `runs/direct_zperm_probe/`.
- A1: `runs/direct_anchor_a1` (report, best/last, log), `runs/direct_anchor_a1_checks`.
