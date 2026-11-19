<!-- Working log of the multi-family scratch decoder (sections 1-42, 2026-11-14 to 2026-11-18), moved here from a local note. Run directories named runs/... are under ${GETKITE_RUNS}. -->

# get_universal_scratch: Stage A audit and file map (2026-11-14)

This audit covers the diverse-embodiment brief (Stage A). Every fact below was checked in local code or files. Where a fact comes from KITE, the file:line is given relative to `~/code/get-ur-kite/KITE`.

## 1. KITE revision, checkpoint, encoder

| Item | Value |
| --- | --- |
| KITE repo | `helloqxwang/KITE` main @ `136159544aea68f5e8b6cc64f17d221e825e3eee` (2026-11-08), unmodified |
| Released decoder bank | `checkpoints/decoder/release.pt`, sha256 `a8c7deb7…54d`, matching `asset_downloads.json`. It is epoch 550, global step 4,297,150, of a 1,200-epoch cosine schedule |
| Dataset metadata | `checkpoints/decoder/dataset_meta.pt`, sha256 `6f85c30e…b52` |
| Runtime assets | `download_assets.py --only runtime`, 1,343 files, every sha verified (downloaded 2026-11-14) |
| ψ | `_ContactOnlyEncoder` (`networks/contact_probe_policy_paired.py:334-401`). Details below |

ψ in detail:
- Input: (B, 6, 6) world-frame [xyz, outward normal] plus a validity mask. There is no scaling or centering inside ψ.
- Architecture: Linear(6→256), then 2 self-attention blocks whose Q/K also see raw xyz, then a CLS cross-attention.
- Output: z ∈ ℝ²⁵⁶.
- Training: ψ was trained jointly with the decoder bank from scratch ("cotrain"), and is shared by aliasing across heads. Our loader verifies that all heads hold identical weights.

## 2. Encoder exposure: what this ψ was trained on

From `dataset_meta.pt` (`manipulators`) and `conf/kite.yaml decoder_bank.embodiments`, both checked locally: **wujihand_right, allegro_right, barrett, robotiq_gripper_stick, mano_right**.
- LEAP, Shadow, Inspire, Ability, XHand, Schunk, MetaHand and EZGripper are **not** in this checkpoint's training.
- The runtime bundle has collision primitives for several of them. That is not exposure, because ψ never saw their contacts.

## 3. KITE's released pipeline: facts the new decoder follows

Config key names are translated by `paper_config.py`. Note in particular that `supervision_mode: contact_relevant_joints` → `full_mix`.

**Batching** (`train_kinematic_decoder.py:397-430`, `data_process/contact_policy_cotrain_dataset.py:31-82`):
- One buffer of 1M samples per embodiment.
- Each step takes a stratified sub-batch of 128 from every embodiment, so 640 per step.
- The loss is the mean of the 5 per-embodiment losses.
- AdamW lr 1e-4, cosine schedule, 2k warmup, grad clip 1, bf16, EMA (weights used for evaluation).

**q_contact (q2)** (`robot_model/robot_model.py:2896-2935`, `contact_policy_probe_dataset.py:1753-1818`):
- Joints come from a mixture distribution within limits.
- The base is U[-0.4, 0.4]³ m, with a uniform SO(3) rotation stored as intrinsic-XYZ Euler.

**Contacts** (`contact_policy_probe_dataset.py:1994-2048, 2347-2395`):
- Sparse mode takes 2–6 points straight from the 4,096-point template at q2.
- `region_uniform`: pick a non-empty region uniformly, then a point within it uniformly.
- Normals are the template normals posed by FK.
- Training only: each normal is rotated by a rotation vector drawn from N(0, 0.1²) per component.

**q_init (q1), gaussian mode** (`contact_policy_probe_dataset.py:3043-3088, 1077-1087`), with q1 = q2 + noise:
- Translation: + N(0, (5 mm)²).
- Rotation: + N(0, (5°)²) on **each Euler angle**, then wrapped.
- Joints: + N(0, (0.05·range)²), then clamped to limits.
- q1 is redrawn every epoch. The contact-error-mixture options are inactive in the release.

**Goal and supervision** (`contact_policy_probe_dataset.py:3090-3128`, `contact_policy_probe.py:1360-1364`):
- goal base = q2.
- A joint's goal is q2 if it is *contact-relevant* (any contacting link is its descendant), otherwise q1.
- **Every coordinate is supervised.** Inactive joints are trained to stay put. "Contact-relevant" describes how the target is built; it is not a loss mask.

**Residual** (`networks/residual_action_utils.py:231-363`):
- Layout: [Δxyz, 6D(R_goal R_q1ᵀ), Δq].
- Normalised as Δxyz/5 mm; (6D − I₆)/5°; Δq/(0.05·range).
- Decoding: R = R6d_sym(·) R_q1.

**Loss** (`contact_probe_policy_base.py:150-156`, `contact_probe_policy_paired.py:1904-1916`):
- Plain masked MSE on the normalised residual, per sample, then over the batch.
- **`contact_point_normal_weight: 0.025` is not used by the released run.** It only feeds the FK-margin supervision modes, and `full_mix` returns before reaching them. The released training therefore has **no FK contact loss**.

**Robot model** (`robot_model/robot_model.py:1681-1697`):
- pytorch_kinematics, with the base given as six `virtual_joint_*` joints.
- **`<mimic>` is never parsed.** Barrett's 8 joints, 4 of them URDF mimics, are 8 independent DOFs in KITE's action space and in its synthetic data.

**Adaptation**: KITE has no code or recipe for adding an embodiment to a trained bank. The only documented path is retraining the whole bank.

**Evaluation**: KITE's code has no success thresholds. Its training metrics are residual MSE plus contact point/normal MSE. The threshold of every contact within 10 mm and 15° is ours, kept from the LEAP work.

## 4. Assets available locally

| Family | Model + meshes | KITE surface template | In ψ training | Pilot role |
| --- | --- | --- | --- | --- |
| Allegro (right) | `assets/ohra_urdf_robot/allegro` URDF + MJCF | `surface_template_graph_n4096_ddcc74fe8fec.pt` | yes | train |
| Barrett | `assets/dro_urdf_robot/barrett` URDF + MJCF | `…_n4096_3e0ba32e061c.pt` | yes | train |
| Wuji | `assets/robots/wuji_hand`, `assets/spider_robots/wujihand` | to check | yes | later source family |
| LEAP variants | our kite_data generator: 1,811 hands, 253 topologies | our 4,096-point templates (kite_data) | **no** | train (12 hands) / held-out structure test (4 hands) |
| Robotiq + stick | `assets/composite/*` | to check | yes | deferred (composite, two roots) |
| MANO | needs the MANO licence; `.pkl` not distributed | 778 vertices | yes | deferred |
| Shadow, Inspire | **collision primitives only**; no URDF or meshes in the bundle | none | no | external candidates; need outside models, licence check and coupling support |

## 5. The earlier fresh-decoder failure (preserved; not assumed to be fixed)

The runs are preserved: `runs/raw_world_flat_lr3e-4`, `diag_raw_overfit_lr*`, `diag_rawworld_resample`, `diag_get_psi_wrist*`, plus configs `raw_world.json` and `get_psi_wrist*.json`.

- **World frame, raw contacts (no ψ):**
  - 64 fixed examples with a fixed q1 memorise to 0.01 mm.
  - With q1 redrawn on the same 64 targets, the wrist stays at the no-op 8.0 mm and contact error at 13.6 mm.
  - On full data, the error was flat for 15k steps.
- **wrist_init frame with KITE ψ, with or without the world start pose and with or without surface tokens:** same picture. The wrist stays at 8.0 mm and contact error at about 13.8 mm.
- **wrist_init frame with raw contacts:** learns, reaching 3.2 mm and 94% on the resample test, and 5.6 mm and 72% on full validation.
- **What these tests show:** memorisation works, but learning the q1-dependent correction did not happen within 4k–15k steps when a fresh network had to compute millimetre differences between world-frame quantities.
- **What they do not show:** they don't rule out a world-frame scratch decoder at another scale or with another architecture. KITE itself learned in the world frame, but with 4.3M steps of 640 samples.
- **What differs in the new model:**
  - New architecture: a 128-wide, 4-layer GET.
  - A shared point encoder for surface tokens.
  - Intent goes in only through projected tokens.
  - KITE's exact loss and supervision: all coordinates and the contact-relevant goal mix. The old GET used a contact-relevant mask in places.
- **The decisive gate is C2**, where q1 is redrawn. C1 is expected to pass from memorisation alone.
- `pos_fourier` is an off-by-default diagnostic: high-frequency features of world positions. It is held in reserve for a *specific* C2 failure, not as a planned ablation.

## 6. Padding, IDs and canonical leaks

- LEAP joint counts genuinely vary, from 4 to 16. The old `HandBank` pads to 16 and carries `slot` (canonical LEAP id) and `finger` fields. Only `slot` and `finger_id` read them, and neither is reachable from the new model.
- The new path (`robots.py`, `get_scratch.py`):
  - pads to `J_MAX = 24` physical joints;
  - has no canonical, joint, family or embodiment ids;
  - uses Embeddings only for graph-distance buckets and the 4 token types.
- Tests check: padding has no effect, a mixed batch gives the same result as one robot alone, reordering joints reorders the outputs to match, and no per-slot tables exist.

## 7. Interface and conventions (new)

- **Robot description** (`RobotDesc`): links/joints from URDF or LEAP hand.json, with mimic as (source, multiplier, offset), plus the template (link-local points and normals).
  - Physical joints are graph and FK nodes.
  - *Coordinates* are the independent joints, in native action order. That is the unpadding order, and for KITE robots it is KITE's joint order.
- **Base**: a floating base (R, t) in KITE's world frame, intrinsic-XYZ Euler. The pilot has one root; composite roots are deferred.
- **Mimic**: supported in FK. For Barrett the pilot follows **KITE's released convention** (`ignore_mimic=True`, 8 independent coordinates). Reasons:
  - KITE's data samples those joints independently, so a coupled model could not reproduce KITE's contacts.
  - Comparisons with the released Barrett decoder stay on the same action space.
- **Records** (`multifamily.Records`): q2 base and coordinates, the contact-relevant mask, world contacts at q2, and contact template indices (provenance for scoring). q1 and the goal are built by KITE's gaussian recipe:
  - redrawn every batch in training;
  - seeded and fixed for evaluation pools.
- **FK validation**:
  - Our generic FK reproduces KITE's stored template points at KITE's canonical pose to <0.1 mm (Allegro, Barrett).
  - It matches our old LEAP FK to <0.01 mm.
  - It reproduces KITE-generated contacts at q2 to 6e-5 mm.
- **The one deliberate deviation**: LEAP records come from kite_data's `kite_code` regime, which uses the same recipe. It keeps a self-collision filter and its own contact regions; KITE infers regions from link names, which fails for LEAP's numeric names.

## 8. Pilot manifest (fixed; `pilot_data/manifest.json`)

- **Allegro, Barrett:** KITE's own sampler (`jobs/pilot/gen_kite_family.py`, released config).
  - Train 2,000 records (seed 101); validation 500 fresh goals (seed 202).
  - File hashes are in each run's provenance.
- **LEAP:**
  - 12 training hands from 12 distinct topologies in the LEAP train split. Training rows [0, 167) give about 2,000 goals; validation rows [900, 942) give fresh goals on the same hands.
  - **4 held-out hands** from 4 distinct topologies in the LEAP val split, 125 records each, reported separately (`leap_heldout`) and never used for selection.
- **Sampling:** family uniform → morphology uniform → example.
- **Selection:** validation macro success over families, with macro contact error as tie-break.
- **Untouched:** Shadow and Inspire are not used. No test-family data is used.

## 9. File-change map (all new files; no existing experiment changed)

| File | Purpose |
| --- | --- |
| `kite_model/robots.py` | Robot adapters (`urdf_desc`, `leap_desc`), padding, mimic expansion, batched FK, graph matrices |
| `kite_model/get_scratch.py` | `GETScratch` + `ScratchConfig` (128-wide, 4 blocks, 4 heads, MLP×4; 1.12M params), KITE residual encode/decode |
| `kite_model/multifamily.py` | Record schema, KITE q1/goal recipe, balanced sampler |
| `kite_model/train_scratch.py` | Training/evaluation, frozen ψ (asserted every call), per-family E1 metrics with no-op and wrist-only references, provenance, resume, source-only selection |
| `tests/test_scratch.py` | FK vs KITE, mimic, LEAP adapter, residual round trip, padding, permutation equivariance, no id tables |
| `configs/scratch/c{1,2,3}_*.json` | Pilot gates C1/C2/C3 |
| `jobs/pilot/gen_kite_family.py` (+ `shims/pytorch3d`) | KITE-native record export. The shim is a pure-torch `sample_farthest_points`, the only pytorch3d function KITE imports, with the same algorithm |
| `jobs/pilot/make_manifest.py`, `scratch.sbatch`, `gen_kite.sbatch` | Manifest/configs, GPU pilot job (not watchdog-managed), CPU generation job |
| `~/envs/kite_gen` | Overlay venv: the get_ur_kite base plus KITE's pinned data dependencies. The training env is untouched |

## 10. Not yet done (in order)

1. C1 → C2 → C3 gates.
2. The released KITE reference on the same fixed Allegro/Barrett validation records. This needs a wrapper that feeds our records through KITE's own `predict_action`.
3. E2 (cross-embodiment): needs a target-surface assignment procedure and a feasibility reference.
4. Wuji as an extra source family.
5. External families: need outside models and licences.

## 11. Pilot results (2026-11-14)

All on an RTX 2080 Ti with 1.12–1.16M params. Gate C1 is fixed q1; C2 redraws q1 each batch and scores on fresh fixed q1 draws.

| Run | Allegro mm / succ | Barrett | LEAP | Note |
| --- | --- | --- | --- | --- |
| no-op reference (C2 records) | 11.3 / 0.19 | 14.2 / 0.00 | 13.8 / 0.06 | wrist 7.6–8.6 mm, 6.4–7.8° |
| C1, 48 fixed examples, 2k steps | 0.06 / 1.00 | 0.05 / 1.00 | 0.05 / 1.00 | residual loss down >99.98%. Passes (memorisation) |
| C2 baseline, 6k steps | 11.1 / 0.31 | 10.6 / 0.12 | 11.8 / 0.19 | **wrist stays at no-op**; joints learn |
| C2 + position Fourier (8), 6k | 8.0 / 0.56 | 11.0 / 0.19 | 11.0 / 0.25 | wrist translation about halves; rotation stays at no-op |
| C2 + position + rotation Fourier (8+8), 6k | 8.0 / 0.75 | 6.7 / 0.62 | 7.5 / 0.62 | rotation 1.5–2°; translation 6–7.5 mm |
| same, 20k steps | **1.01 / 1.00** | **0.86 / 1.00** | **0.98 / 1.00** | wrist 0.4–0.8 mm / 0.4–0.6°. **C2 passes** |
| **C3** (2k goals/family, fresh val goals, Fourier 8+8, 10k steps × 256) | 11.51 / 0.17 | 14.81 / 0.09 | 14.49 / 0.07 | **fails: at or below no-op** (11.55 / 14.45 / 13.93). Held-out LEAP 15.33 vs no-op 14.98. Train root loss flat at about 0.76 |
| KITE released decoder, same C3 val records | 4.09 / 0.92 | 3.70 / 0.92 | n/a | reference; wrist 3.5–3.6 mm, 3.2–3.4° |

- **C3 throughput:** 2,709 examples/s at batch 256, 0.98 GB peak, 10k steps in 16 min.
- **Diagnosis:** C2 succeeded by memorising 48 absolute goals keyed by z; the high-frequency features gave enough resolution on the start pose. C3 needs the decoder to *read the absolute goal pose out of z* for unseen contacts, and in 2.6M samples it does not start to learn this at all.
- **Scale gap:** KITE trained the decoder jointly with ψ for about 2.7B samples.
- **Other evidence:**
  - The pretrained KITE trunk fine-tuned on LEAP does decode z (8.2 mm at 30% of its run).
  - Raw contacts in the wrist frame learn (5.6 mm).

## 12. C3 diagnostics, 50k steps (2026-11-15)

Two runs, identical except the intent input and its encoder: `scratch_c3_long50k` (frozen ψ → 4 intent tokens) and
`scratch_c3_raw50k` (diagnostic reference: one token per raw world contact, ψ bypassed). Same records, start-pose
stream, geometry inputs (Fourier 8+8, world frame), backbone and initial backbone weights (intent encoder built last;
test), AdamW 3e-4 cosine, 50k × 256. Step-100 training losses identical. Both preempted once and resumed.
Full tables: `runs/c3_diag_report.md` (`jobs/pilot/report_c3.py`).

Macro contact mm, training goals / validation goals (no-op ≈ 13.3 / 13.3):
ψ 13.29/13.22 (2.5k) → 12.79/13.67 (10k) → 11.62/14.78 (20k) → 8.87/17.61 (50k);
raw 13.35/13.22 → 13.07/13.41 → 11.72/14.56 → 8.33/18.48.
Training-goal wrist translation: ψ 7.91 mm = no-op (never moves); raw 6.2 mm. Wrist rotation: ψ 2.8°, raw 4.0° (no-op 7.9°).
Validation: both worse than no-op in every family (wrist rotation ~11.5° vs 7.7°).

Interpretation: mainly a generalisation problem (memorises ~2k goals/family; validation diverges from step ~5k);
not ψ-specific, since raw contacts overfit the same way. Secondary: wrist translation is not decoded from ψ's z even
on training goals, while raw contacts give partial translation. LEAP counts: pilot slices (2,004 / 504 / 500) of the
1,000-per-hand dataset. Skipped tests: 7, all in test_kite_trunk.py (needs KITE_DATA_DIR); all pass when set.

## 13. Generalisation diagnosis, steps 1–2 (2026-11-15)

Scripts: `jobs/pilot/diag_c3.py` (→ runs/c3_diag_step1.{json,txt}), `jobs/pilot/diag_c3_vis.py` (→ runs/c3_diag_vis/).

**Step 1 passes.** Loss = mean_9(root²) + mean_valid(joint²) reproduced by hand. q_init: trans ≈ 1.0, rot ≈ 0.62,
joint 0.28–0.54 (all families, all sets). q_goal: loss 0, contact error 0.0000–0.0001 mm, decode error ≤ 2e-6 mm /
1e-7 rad (rotation 0.03° = float32 acos floor). Half-way: loss 0.25×, contact error ≈ half. Train/val parity: same
contacts per example (≈4.0), contact spread (55–57 mm), world spread, active fraction and residual-target mean/RMS;
0 duplicated goals; only difference is training-time normal noise (0.1 rad, KITE recipe), and training-goal metrics
are the same with a training-style draw + noise and with the fixed evaluation start. Per term at 50k:
ψ model trans term = q_init exactly on every set (never predicts translation); rot 0.09–0.11 train vs 1.30–1.48 val
(noop 0.62); joints 0.03–0.10 train vs 0.67–1.59 val (noop 0.28–0.54). Raw: trans 0.63–0.68 train vs 1.51–1.55 val.

**Step 2: masks pass, raw spatial representation fails.** Invalid contact slots are zero and changing them changes the
output by exactly 0. But shifting the whole scene (contacts + start pose; targets unchanged) by 1 mm changes the raw
model's root residual by 0.94–0.99 (output norm 1.8–2.4), 3 mm by 1.9, i.e. the prediction is effectively
re-drawn per millimetre on training and validation goals alike. ψ model: 0.02 per mm (smooth). Cause: contact tokens
get absolute world xyz plus Fourier octaves down to 6.25 mm wavelength and no hand-relative feature, which lets the
decoder key each training goal by its exact contact positions. Figures: validation predictions often move contact
points away from their targets.

## 14. Contact-encoding runs and decision rule (2026-11-15)

Runs: `scratch_c3_raw50k_cf0` (no contact Fourier) and `scratch_c3_raw50k_cf5` (contact octaves with wavelength
≥ 5 cm only); everything else as `scratch_c3_raw50k` (hand/wrist tokens keep 8 octaves). Setting:
`model.contact_fourier` (-1 = follow pos_fourier).

Interpretation notes (user): scaled coordinates keep millimetre precision numerically; Fourier features change what
is easy to learn. Passing the shift test is necessary, not sufficient (no-movement passes trivially): the chosen
variant must also beat no-movement on validation contact error and success. Fine hand/wrist features may still
contribute to sensitivity; if removing contact frequencies does not help, a broader coordinate-representation
problem is not ruled out.

Decision rule:
1. smoother + better validation → more independent goals with that encoding;
2. smoother, validation still poor → inspect the train/val gap; more data or wrist-relative geometry;
3. still shift-sensitive → wrist-relative raw geometry before scaling data;
4. both variants improve similarly → prefer the simpler encoding (cf0).

Open, separate: ψ model never learns wrist translation (translation term = no-movement on every set, including
training goals). First probe (400 Allegro val examples, CPU): ψ's z does carry absolute position (linear probe of the
contact centroid: 3.0 mm held-out RMSE over a 417 mm spread), but it moves only 0.023 std units per mm of global shift
against ~22 between examples, so the 5 mm wrist correction is a very small signal in z. Not yet explained.

Results (50k steps; shift = mean change of predicted root/joint residual for a rigid scene shift, ψ model 0.02/0.02 per mm):

| contact encoding | shift 1 mm | 10 mm | train-goal macro mm | val macro mm (no-op 13.3) |
|---|---|---|---|---|
| 8 octaves (scratch_c3_raw50k) | 0.94–0.99 | 2.4–2.5 | 8.33 | 18.48 |
| ≥5 cm octaves (cf5) | 0.15–0.16 | 1.3–1.4 | 8.62 | 17.91 |
| none (cf0) | 0.09 | 0.85–0.86 | 8.36 | 18.45 |

Shift sensitivity drops 6–10× without the fine contact octaves (cf0 still ~4× ψ; remaining sensitivity may come from the
hand/wrist Fourier features), but validation is unchanged: worse than no-op in every family, success 2–6%, and the
train/val divergence starts at the same step (~7.5–10k) with the same slope in all three runs. Decision rule → case 2
(smoother, validation still poor); with case 4, cf0 is the preferred encoding. Report: runs/c3_diag_report_cf.md,
runs/c3_diag_vis_cf{0,5}/.

## 15. 10× independent goals, raw contacts, no contact Fourier (2026-11-15)

Run `scratch_c3_raw50k_cf0_10x`, launched from the reorganized repo (`get-ur-kite/get-ur-kite`, config
configs/decoder/scratch/c3_raw50k_cf0_10x.json). Only the training pool differs from `scratch_c3_raw50k_cf0`:
original goals + new draws → Allegro 20,000 (2,000 seed 101 + 18,000 KITE sampler seed 303), Barrett 20,000 (same),
LEAP 20,004 on the same 12 hands (original seed-0 rows 0–899 and 942–999, never 900–941 = validation, + 709 rows/hand
of new seed-1 kite_code draws). Data: getkite_data/multifamily_10x/ (job gen_10x.sbatch, ~2 min CPU). Checked: no
train/val or train/held-out overlap, no duplicates, original 6,004 goals contained. Evaluated each 2,500 steps on
val (selection), original training subset (`train_goals`), 2k/family of the new goals (`train_goals_new`), held-out LEAP.
Next per user: if validation improves meaningfully over cf0 and beats no movement → same pool with frozen ψ;
otherwise stop and investigate wrist-relative geometry (no further data increase). ψ translation issue tracked separately.

Result (step 50k; full tables runs/c3_10x_report.md): flat at no movement on every pool for the whole run, no
memorisation. Validation contact mm / success: Allegro 11.47 / 18% (no-op 11.55 / 16%), Barrett 14.21 / 11%
(14.45 / 12%), LEAP 13.74 / 9% (13.93 / 9%); held-out LEAP 14.75 (14.98). Original training subset ≈ new training goals
≈ no-op (11.8 / 14.0–14.1 / 14.0 mm). Per-family training loss 0.95–1.28 at 50k vs 0.98–1.26 at step 100. Wrist
7.7–8.0 mm / 7.7–7.9° = no-op. → "remains poor": no ψ run on this pool, no further data increase.

## 16. Wrist-frame raw geometry (2026-11-15, reorganized repo)

`model.frame = "wrist"` (raw_contacts only): contacts, normals, hand FK poses, joint axes and surface points are
expressed as R0ᵀ(x − t0) in the start-wrist frame; the root token becomes the identity pose; the wrist update is
predicted in that frame (t = t0 + R0·Δ, R = R0·R6d(·)) and decoded back to world for evaluation; joint residuals and the
loss form unchanged (translation-residual norm identical to world; rotation residual = R0ᵀR_goal instead of R_goal R0ᵀ,
same angle). Tests (tests/decoder/test_scratch.py): decode(target) round trip, untrained = q_init, and a random rigid
transform of the whole scene leaves targets and predicted residuals unchanged while the decoded world goal moves with
the scene (world-frame model fails the same test, as a sanity check). 39 decoder tests pass.

Gates (model as cf0 + frame=wrist: pos/rot Fourier 8, no contact Fourier):
- C1 (48 fixed, fixed start, 3k steps): **pass**, 0.02–0.03 mm, 100%, loss 1.0–1.8 → 0.0000.
- C2 (same 48, start redrawn, 10k steps, batch 48): **not passed**. Residual loss vs no-op −48% / −60% / −56%
  (Allegro / Barrett / LEAP; gate ≥80%); contact 10.44 / 11.34 / 11.85 mm (no-op 11.26 / 14.20 / 13.78), success
  31 / 12 / 25%. Joints learned (joint loss 0.008); wrist rotation partly (7.2–7.8° → 4.7–6.2°); wrist translation
  not at all (7.6–8.6 mm = no-op). Still improving at 10k. (Preempted once; resumed from step 4.5k.)
Larger wrist-frame run (configs/decoder/scratch/c3_raw50k_cf0_10x_wrist.json) prepared, not launched.
- C2 at 20k steps (fresh run, `scratch_wrist_c2_20k`): **not passed, fails only on wrist translation.** Step 20k:
  residual loss vs no-op −62% / −75% / −74%; contact 8.65 / 7.50 / 7.70 mm (no-op 11.26 / 14.20 / 13.78), success
  69 / 75 / 75%; wrist rotation 0.37 / 0.33 / 0.48° (no-op ~7.5°); joint loss 0.0006; wrist translation 8.59 / 7.61 /
  7.68 mm = no-op (7.6–8.6) at every evaluation. The residual loss plateau is the untouched translation term.
  Wrist translation has now stayed exactly at no movement in every variant except the world-frame raw run with fine
  contact Fourier (hash-like, 6.2 mm on training goals only) and the world-frame ψ C2 with start-pose Fourier.
  Hypotheses (untested): the translation signal is a few-mm uniform shift of contact positions with no normal change,
  i.e. ~1% of a coarse coordinate feature (rotation also turns the normals, a ~9% signal), possibly lost to
  learnability and/or bf16 autocast in the blocks.

## 17. Translation path checks (2026-11-15; scripts/decoder/diagnostics/{diag_translation,translation_checks}.py)

Baseline = `scratch_wrist_c2_20k` (wrist frame, cf0, bf16 blocks). Results: runs/diag_translation_baseline.json,
runs/translation_checks_baseline.{json,txt}.
- Translation output ≈ 0: RMS 0.026–0.032 vs target 0.94–1.08, cosine 0.02–0.03 (rotation cosine 0.996–0.999);
  translation loss = no-op (0.89–1.18 vs 0.88–1.17); rotation 0.002, joints 0.001.
- Quoted gradients are gradients of the training loss w.r.t. the root head's last-layer *parameters* (translation rows
  0:3 incl. bias vs rotation rows 3:9), not output gradients. Single batch: translation 0.25 vs rotation 0.01 (trained),
  0.15 vs 0.24 (init). Across 50 batches (batch 48) coherence ||mean g|| / mean ||g||: translation 0.22 at init and
  trained, rotation 0.45 at init (random directions 0.14) → the translation gradient mostly cancels across batches from
  the start.
- fp32: everything except the bf16 autocast around the transformer blocks is float32 (records, bank, FK, frame
  transform, targets, heads, decode, loss); model.amp=False makes the whole path fp32 (dtype hooks confirmed on CPU;
  GPU trace pending).
- Check 1 (input shift, 3 mm start-wrist translation, exact model inputs via hooks): contact xyz input changes by
  −R1ᵀd·pos_scale (max error 6e-8); normals, padding, root/joint/surface/neighbour inputs unchanged (0); translation
  target changes by −R1ᵀd/5 mm (error 3e-6); rotation/joint targets unchanged. Trained model: root feature (norm 8.8)
  moves 0.07 for a 3 mm shift vs 1.54 for a 3° rotation.
- Check 2 (preprocessing audit): no centering, mean subtraction, normalisation or pooling of contact positions; the
  only pooling is the surface tokens' local-neighbourhood shape encoder, whose centre position is passed separately.
- Check 3 (correspondence oracle, true goal rotation + joints + contact-template correspondence): goal translation
  recovered to 1.5e-5 mm mean / 8.6e-5 mm max on C2 and on the full 10x train (60,004) and val (1,504) pools.
→ Labels, frames and inputs are correct; translation information reaches the model inputs, but the trained network
  barely represents it. Next per rule: translation-only gate (`wrist_c2_transonly`: start rotation and joints at
  their goal values, only wrist translation perturbed) — launched.

## 18. fp32, contact Fourier and translation-only gates (2026-11-15)

All 20k steps, 48 C2 goals (same goals throughout; validation = fresh start perturbations of those goals), wrist frame.
Diagnostics on GPU: runs/diag_translation_all.{json,txt}, runs/translation_checks_<run>.json.

| run | change vs baseline | loss reduction A/B/L | contact mm, success | wrist mm, ° | trans loss (no-op) | trans corr / slope | shift-response slope |
|---|---|---|---|---|---|---|---|
| wrist_c2_20k | — | 62/75/74% | 7.5–8.7, 69–75% | 7.6–8.6, 0.3–0.5 | 0.89–1.18 (0.88–1.17) | ~0 / 0.00 | 0.00 |
| wrist_c2_20k_fp32 | amp off (fp32 blocks) | 63/74/74% | 7.7–8.6, 69–75% | 7.7–8.5, 0.4–0.5 | 0.90–1.16 | ~0 / 0.00 | 0.00 |
| wrist_c2_20k_cf8 | 8 contact Fourier octaves (wrist-relative) | 99/99/99% | 0.94–1.21, 100% | 0.8–0.9, 0.6–0.9 | 0.011–0.017 | 0.99 / 0.95–1.01 | 0.98 |
| wrist_c2_transonly | start rot + joints = goal; translation only | 49/23/46% | 5.4–6.2, 81–94% | 5.4–6.2, 0.03 | 0.52–0.68 | 0.2–0.9 / 0.23–0.49 | 0.32 |

Gradients (root head last layer, parameter gradients, weights / bias separately): init trans 0.150/0.032, rot 0.238/0.051;
50-batch coherence at init: translation weight 0.22, bias 0.22; rotation 0.45/0.45 (random 0.14). With symmetric
translation noise, cancellation of the translation gradient is expected and not evidence of a bug by itself.
Dtype hooks: block outputs are fp32 under amp too (the residual stream stays fp32; qkv/attention/MLP matmuls run in bf16).

Reading: precision is not the bottleneck (fp32 identical). With coarse contact coordinates the translation path can
partly learn when translation is the only task (slope ~0.3–0.5, wrist 8 → 5–6 mm) but not at all next to rotation and
joints; with wrist-relative contact Fourier features the full C2 passes (≈1 mm, 100%, translation slope ≈1, responds
correctly to a known start shift). Caveat: C2 reuses 48 goals; fine features could favour memorisation, so this only
justifies a C3 generalisation test. Prepared: configs/decoder/scratch/c3_raw50k_cf8_10x_wrist.json (10x pool, same
loss, val and 50k budget), not launched.

## 19. C3: wrist-frame raw contacts + contact Fourier (8) on the 10x pool (2026-11-15, job 667633)
Config configs/decoder/scratch/c3_raw50k_cf8_10x_wrist.json; differs from c3_raw50k_cf0_10x only in model.frame=wrist,
contact_fourier=8. Same val records, eval start (seed 12345), loss, optimizer, 50k steps. best.pt (selected on val) = step 50000;
val curve still improving slowly at the end (macro contact 6.53 → 6.48 mm over the last 10k).

Contact mm / success at step 50k (noop on the same records in brackets):
| set | allegro | barrett | leap |
|---|---|---|---|
| val | 5.80 / 0.73 (11.55 / 0.16) | 5.66 / 0.72 (14.45 / 0.12) | 7.99 / 0.43 (13.93 / 0.09) |
| train orig | 5.78 / 0.74 | 5.63 / 0.74 | 7.53 / 0.48 |
| train new (2k) | 5.88 / 0.72 | 5.49 / 0.74 | 7.51 / 0.48 |
| held-out LEAP hands | | | 8.51 / 0.37 (14.98 / 0.07) |
Wrist val: 4.7 / 4.6 / 5.7 mm (noop 8.0 / 8.0 / 7.8), 3.9 / 4.3 / 4.5 deg (noop 7.7 / 7.8 / 7.7).
Earlier runs, val contact mm (allegro/barrett/leap): psi 2k 14.3/19.4/19.1; raw world 2k 15.7/19.8/20.0; raw world cf0 2k
15.9/20.1/19.3; raw world cf0 10x 11.47/14.21/13.74 (= noop).
Translation (val, diag_translation.json): loss 0.38/0.39/0.54 vs noop 0.99/1.02/0.96; cosine 0.73/0.72/0.58; per-axis
corr 0.72–0.85 (allegro, barrett), 0.63–0.69 (leap); slope 0.61/0.60/0.43 (under-predicts). World cf0 10x: corr ≈0, slope 0.01.
Reading: train ≈ val ≈ new-goal subset (gap ≤0.5 mm; LEAP val–train 0.5 mm, held-out hands +0.5 mm more), so no memorisation;
the remaining error is fit, not generalisation. Translation still the weakest component (slope 0.4–0.6), LEAP weakest family.
psi not tested with this frame/encoding; psi translation-head issue still open.

## 20. Frozen psi in the wrist frame (2026-11-16)
Preserved: runs/archive_c3_raw50k_cf8_10x_wrist_step50k/ (best.pt = last.pt = step 50k, resolved config.json, source
config, provenance, log, diag_translation.json; SHA256SUMS; read-only).
Settings, raw C3 (§19) vs world-frame 10x (§15), from resolved configs + checkpoint model_config + provenance: only
model.frame (world -> wrist) and model.contact_fourier (0 -> 8) differ. Data hashes identical; same repo HEAD (dirty tree:
wrist/amp/start-noise edits). Precision identical: both bf16 autocast inside the transformer blocks, fp32 residual stream,
heads, targets and loss (the 10x run predates the model.amp field; the old code enabled autocast whenever CUDA, = amp=True).
Start-noise defaults identical (5 mm, 5 deg/axis, 5 % range). Params 985,418 vs 979,274 (contact Fourier inputs).

psi path (model.world_pose_in, get_scratch.py): psi receives the original world-frame contacts (KITE-compatible, unchanged
FrozenPsi). The decoder gets standardised z -> intent tokens (no transform, no Fourier on z dims), robot geometry / FK /
surfaces in the start-wrist frame, and the start wrist's world pose [t0*2.5, 6D(R0), Fourier8 of each] through a separate
projection added to the root token (built last: backbone init identical to the raw wrist model). Wrist updates predicted in
the start-wrist frame, decoded to world. Guards: psi+wrist requires world_pose_in; contact_fourier refused for psi.
Tests (17/17 pass): untrained identity + round trip; backbone init equal to raw wrist; with z fixed, rigid change of the start
pose leaves joint/surface tokens unchanged and changes only the world-pose input; with that projection zeroed, residuals
are invariant and the decoded goal moves rigidly; decoder inputs are only (robot, start pose, z) and psi sees only contact
positions/normals/validity (goal poses and contact-template correspondences stay supervision/evaluation only).
Note: KITE's start q1 is drawn around the goal (task definition), as in every earlier run.
Gates: psi_wrist_c1 (= wrist_c1 data/schedule), psi_wrist_c2_20k (= wrist_c2_20k_cf8), jobs 703718 / 703720.
C3 prepared: c3_psi50k_10x_wrist (= c3_raw50k_cf8_10x_wrist except model), launched only if both gates pass.
Gate results (2026-11-16):
- psi_wrist_c1 PASS: 3k steps, contact 0.012/0.015/0.014 mm, success 1.0 all families (= raw wrist C1).
- psi_wrist_c2_20k PASS: step 20k contact 0.74/0.75/0.82 mm, success 1.0, wrist 0.68/0.54/0.51 mm (noop 11.26/14.20/13.78 mm).
  Raw cf8 C2 same records/schedule: 1.03/0.94/1.21 mm, wrist 0.87/0.79/0.91. psi learns faster from ~4k steps on.
  Translation (diag_translation, C2 records): psi loss 0.015/0.005/0.005 vs noop 1.17/0.88/0.97; per-axis corr >= 0.99;
  slope 0.96/1.01/0.99. Raw cf8: corr >= 0.986, slope 0.996/1.006/0.944. Caveat as before: C2 reuses 48 goals.
- C3 c3_psi50k_10x_wrist launched: job 704567.

## 21. C3 frozen psi, wrist frame, 10x pool (job 704567; runs/scratch_c3_psi50k_10x_wrist) -- FAILS at no-op
Val contact mm / success (noop): allegro 11.47/0.17 (11.55/0.16), barrett 14.39/0.11 (14.45/0.12), leap 14.12/0.08 (13.93/0.09);
held-out LEAP 15.06/0.08 (14.98/0.07). Train orig/new goals equally at no-op (11.8/13.9-14.0/13.8). Wrist mm identical to noop
(7.95/8.02/7.75) and wrist deg identical: the root output is ~0 (translation RMS 0.002 vs target 0.98; corr ~0; slope 0).
Only joints move (small gain on train, LEAP val residual loss 1.40 vs noop 1.29). Best-on-val = step 7500 (macro 13.15 mm);
curve flat from 2.5k steps. Root head last-layer grads: trans weight 0.0066 vs bias 0.022 (raw C3: 0.129 vs 0.022).
Raw-contact C3 (§19) on the same records: 5.80/5.66/7.99 mm, success 0.73/0.72/0.43.
psi_wrist C2 passed (§20) -> with 48 goals the decoder can map z + world pose to the correction (lookup); it does not at 20k goals.
Linear ridge probe of z (runs/c3_psi_zprobe/probe.txt; fit on 60k train, test on val): contact centroid recovered to 0.85-0.94 mm
(spread ~385 mm) on train and val -> z carries absolute contact location at sub-mm precision, linearly; slot-0 contact
(56 mm) and goal wrist position (41-69 mm) are not linear in z (expected: set encoder; wrist depends on hand + contacts).
Reading: not a generalisation problem (train = val = no-op) and not missing information in z. The decoder never learns to
relate z's world-frame contact location to the start wrist's world pose at mm precision: the same failure as the world-frame
raw 10x run (§15), where contacts and root were both absolute. The raw wrist run succeeds because the subtraction R0^T(x - t0)
is done explicitly before encoding; with psi the decoder must learn it across two separately encoded inputs.

## 22. psi + KITE's frozen scene encoder (intent_source=psi_scene, 2026-11-16)
Why: KITE does not decode z directly: cls_unpacker(z) -> 4 tokens, which interact with KITE's 32 hand tokens (hand_proj
Linear(6->d) on world xyz+normal, link-local FPS, posed at q1) in bottleneck_scene_encoder (xyz-aware attention). These were
co-trained with psi, so z and the hand points share a code; our §21 decoder had to learn that relation from scratch.
Implementation (get_scratch.py): psi unchanged on world contacts; KITE's released hand_proj, cls_unpacker,
bottleneck_scene_encoder loaded (EMA weights), frozen, eval, fp32, no grad; hand points posed in the world (wrist-frame FK
mapped back by R0, t0). Their 32 hand + 4 z output tokens -> new scene_in MLP -> intent tokens; hand tokens also carry
wrist-frame position/normal (+Fourier 8), z tokens a flag. world_pose_in kept; geometry and corrections wrist-relative.
Tests 19/19: parity with KITE's own _encode_scene on the released allegro/barrett heads (max abs diff 5e-4 on features ~30,
float round trip); KITE weights unchanged after an optimiser step and stay in eval; untrained = q_init; backbone init = raw
wrist model. Configs psiscene_wrist_c1 / psiscene_wrist_c2_20k (jobs 707952/707953) / c3_psiscene50k_10x_wrist (prepared).
Fresh-goal check of C2 (runs/c2_fresh_eval; C3 val records of the C2 hands, never selected on): both earlier C2 passes are
48-goal lookups -- raw cf8 C2 fresh 22.2/24.2/25.7 mm, psi C2 24.0/29.0/32.1 mm vs noop 11.6/14.1/14.9. Raw cf8 nevertheless
generalised at C3, so fresh-goal error at C2 does not predict C3; it is reported, not used as a gate.
Gates: psiscene_wrist_c1 PASS (3k: 0.01 mm, success 1.0). psiscene_wrist_c2_20k PASS (20k: 0.78/0.78/0.98 mm, success 1.0,
wrist 0.57/0.49/0.64 mm). Fresh goals (not a gate) 16.3/16.6/25.4 mm vs noop 11.6/14.1/14.9 (raw C2 22-26, psi C2 24-32).
C3 c3_psiscene50k_10x_wrist launched.
Checks requested 2026-11-16 (runs/init_check/init_check.py reproduces train_scratch's construction order):
- psi_scene uses only KITE's shared modules (hand_proj, cls_unpacker, bottleneck_scene_encoder); load_release verifies they are
  identical in every released head. No per-robot KITE parameter is loaded or selected (no joint_identity_embed,
  action_queries, qref_* or head_* in the model). The released allegro/barrett heads appear only in the parity test, whose
  _encode_scene path itself uses only the shared modules.
- KITE hand tokens for every robot, LEAP included: greedy FPS from index 0 over the robot's link-local template points
  (handbank.kite_local_fps), posed by FK; deterministic, no trained or LEAP-specific parameters.
- Initial weights: random initialisation from seed 0; no run loads another run's weights (only its own last.pt on resume).
  CORRECTION: the unit test shows identical backbone init when models are built back to back, but in train_scratch the
  FrozenPsi encoder is built before the model and its KITE module construction consumes RNG draws. So plain-psi and
  psi_scene C3 share identical initial backbones (0/91 tensors differ), but the raw-contact runs started from a DIFFERENT
  random draw (53/91 non-constant tensors differ). Raw-vs-psi comparisons (here and the 2k runs, §12) differ by seed as well.
Token probe (runs/token_probe/; ridge on root token and mean intent token after the input projections (L0) and each block,
fit on 20k train records, tested on val; c_wrist = R0^T(contact centroid - t0), spread 60 mm; target = normalised
translation residual, zero-prediction error 1.58):
- raw wrist C3: c_wrist readable from L0 on (intent mean 18.5 mm, R2 0.87; ~15-20 mm through the blocks); target R2 rises
  layer by layer to 0.51 in the root token at L4.
- plain psi C3: world contact centroid and world t0 are present but coarse (R2 0.6-0.9, errors 100-230 mm; z itself gives
  0.9 mm linearly, so the fresh intent projection keeps only a coarse code); the relation c_wrist only R2 ~0.5 (40 mm) and
  does not improve with depth; target R2 = 0.00 at every layer. The decoder never forms the contact-relative-to-wrist
  quantity at the precision the correction needs (mm against a ~400 mm world range), so it stays at the zero output.
Caveat: linear probe of the mean intent token only; information spread across tokens nonlinearly would be missed.
2026-11-16 20:10: C3 psi_scene job 709248 FAILED at step 14.1k (GPU lost on kuleshov-compute-01, hardware). Resubmitted to resume from last.pt (step 14k, opt/sched state), kuleshov-compute-01 added to excludes.

## §23 Scene-encoder C3 (c3_psiscene50k_10x_wrist) vs plain-psi vs raw, 2026-11-16
Run completed 50k (resumed at 14k after GPU loss). Validation contact mm / success (last step):
| run | allegro | barrett | leap val | leap train goals | held-out LEAP |
|---|---|---|---|---|---|
| raw wrist cf8 (diagnostic reference) | 5.80/0.73 | 5.66/0.72 | 7.99/0.43 | - | 8.51/0.37 |
| plain psi (main objective) | 11.47 | 14.39 | 14.12 | 13.77 | 15.06 |
| psi_scene (pretrained comparison) | 5.35/0.79 | 4.60/0.81 | 10.92/0.20 | 8.92/0.33 | 13.02/0.09 |
| no movement | 11.55 | 14.45 | 13.93 | 14.26 | 14.98 |
KITE released reference (earlier record): 4.09 / 3.70 on Allegro/Barrett.
psi_scene: Allegro/Barrett train = val (no overfit); LEAP val best 10.55 @15k, held-out best 12.59 @22.5k, then both drift
up while LEAP train goals keep falling (8.9) -> LEAP-specific memorisation, weak transfer to unseen LEAP hands.
Translation (diag_translation): corr 0.78-0.86 / slope 0.67-0.68 on Allegro/Barrett; LEAP corr 0.40-0.54, slope 0.27.
Token probe (c_wrist spread 60 mm): psi_scene intent tokens carry c_wrist at L0 (27 mm, R2 0.78) -> root reaches
target R2 0.45 at L4 (raw 0.51). Plain psi: c_wrist R2 0.02 at L0, ~0.5 after L1, flat; target R2 0 at every layer.
Reading: the comparison between hand geometry and the contacts must happen in one frame. Raw: contacts and hand both
wrist-frame. psi_scene: KITE's frozen module compares world-frame hand points with z. Plain psi: hand tokens are
wrist-frame, z is a world-frame code, and the only link is one additive world-pose embedding on the root token; the
decoder never recovers contact-relative-to-wrist at the precision needed. Hypothesis, consistent with probes; not yet
tested by intervention. No new runs launched.

## §24 Spatial bridge test (2026-11-17)
Module src/getkite/decoder/spatial_bridge.py (random-init MLP, 4 layers x512, LayerNorm/GELU; inputs ONLY frozen z
(standardised, no Fourier) + start wrist world pose (scaled + Fourier 8)); label = target contact centroid in the start-
wrist frame R0^T(c - t0), source synthetic contacts, training only. Script scripts/decoder/diagnostics/spatial_bridge.py,
tests tests/decoder/test_spatial_bridge.py (4 pass: forward(z, R0, t0) only; goal pose/config/correspondences/invalid
slots/contacts-with-z-fixed do not change the prediction; label definition; random init).
Data = C3 config (c3_psi50k_10x_wrist): train pool, balanced batches, start redrawn per batch, normal jitter 0.1.
Separation: val 1504 records / held-out LEAP 500: 0 record overlap with train; 0 held-out LEAP robots in train.
Run runs/bridge_c3_20k (20k steps, bs 512, lr 1e-3 cosine, ~4.5 min). Mean L2 error mm (val-selected = last step):
| set | zero | train mean | family mean | linear (ridge on same inputs) | bridge | structured ref. |
|---|---|---|---|---|---|---|
| fresh val allegro / barrett / leap | 41.5/72.9/66.0 | 43.5/69.7/67.6 | 37.1/39.0/44.1 | 43.1/68.5/67.7 | 1.13/1.52/1.41 | 0.86/0.91/0.94 |
| held-out LEAP hands | 68.9 | 70.5 | 43.3 | 71.8 | 1.48 | 0.93 |
| training goals (memorisation ref.) | 60.5 | 60.5 | 40.5 | 60.0 | 0.75 | 0.90 |
Label spread 60 mm (held-out 43 mm). Structured ref. = ridge z->world centroid then exact R0^T(c-t0) (not a candidate).
Bridge still improving at 20k; train/fresh gap 0.75 vs 1.35 mm.
Reading: the relation is learnable from z + start pose by a random-init network, and generalises to fresh goals and
unseen LEAP hands (~1.4 mm vs 37-44 mm family mean). A linear map cannot (it is bilinear in R0 and c). Caveat: the
centroid label is hand-independent (contacts + wrist only), so held-out-LEAP success here shows robustness to a new
contact distribution, not embodiment generalisation; the embodiment-dependent part remains the decoder's job. The
coordinate explanation for the C3 failure remains a hypothesis; this shows the information is extractable, not that its
absence was the cause.

## §25 Joint bridge + GET: C1/C2 (2026-11-17)
Model: GETScratch(bridge=True): plain psi-wrist decoder + random-init SpatialBridge(z, R0w, t0w) built last (all shared
params identical init to plain psi-wrist); its PREDICTED centroid (wrist frame, decoder position encoding) + hidden
features -> one extra intent token, always (train and eval). Ground-truth centroid only in the aux loss.
Loss = decoder residual loss + 1.0 * mean_xyz(((c_hat - c) / 5 mm)^2). Tests: 15 pass (test_scratch bridge tests +
test_spatial_bridge). Configs bridge_wrist_c1 / bridge_wrist_c2_20k (= psi_wrist_c1 / psi_wrist_c2_20k + bridge;
extra_eval fresh_goals and leap_heldout reported only).
C1 (3k, fixed start): contact 0.007/0.01/0.007 mm, success 1.0, wrist 0.003/0.004/0.003 mm, bridge ~0. PASS.
C2 (20k, start redrawn, 48 goals): contact 0.91/0.83/0.81 mm, success 1.0, wrist 0.81/0.55/0.50 mm (psi C2 0.74-0.82
contact). PASS. Losses (mean over log rows): steps 50-500 decoder 1.235 / centroid 3.34; 550-2000 0.813 / 0.945;
18050-20000 0.0075 / 0.0117 (weighted = raw, weight 1.0).
Centroid error mm, A/B/LEAP: joint C2 on its goals 0.77/0.45/0.53; fresh goals 60.0/72.8/81.4; held-out LEAP 81.6.
Matched control (standalone bridge, same 48 goals, bs 48, lr 3e-4, 20k; runs/bridge_alone_c2data): own goals
0.77/0.50/0.62; fresh 59.1/71.9/80.7; held-out 81.6; family mean on fresh 44.7/41.2/45.5.
Decoder fresh-goal contact 23.7/27.9/28.9 (noop 11.6/14.1/14.9), held-out 29.7 (noop 15.0): C2 memorisation, as before.
Reading: at matched data, joint training leaves the bridge's accuracy unchanged (in-distribution and fresh). 48 goals
cannot make the centroid generalise (standalone needs the full pool: 1.4 mm fresh). Whether joint training PRESERVES
the full-pool 1.4 mm accuracy is untested; that requires training on the C3 pool. C3 not launched.

## §26 Joint bridge + GET C3 (c3_bridge50k_10x_wrist, 2026-11-17)
Completed 50k (1h24, no preemption). Validation / held-out contact mm (noop):
allegro 11.70 (11.55), barrett 13.84 (14.45), leap 13.80 (13.93); held-out LEAP 15.25 (14.98). = plain psi (11.47/14.39/14.12,
15.06). Training goals improve slightly (10.7/11.5/11.6 vs noop 11.9/14.3/14.3) and val drifts up from ~13k: goal-specific fit.
Bridge inside the joint model, centroid mm: val 1.15/1.52/1.49, held-out LEAP 1.45, training goals 0.70/0.74/0.73
(standalone full-pool bridge 1.13/1.52/1.41, 1.48) -> joint training preserves the bridge's spatial accuracy.
Losses: steps 100-1000 decoder 1.195 / centroid 57.6; 24-26k 1.097 / 0.080; 46-50k 1.035 / 0.0093 (weight 1.0).
Translation: corr 0.18-0.34, slope 0.13-0.17 (raw 0.6-0.85 / 0.43-0.61; scene 0.78-0.86 / 0.67).
Token probe: root token carries c_wrist from L1 (R2 0.97, 9 mm) but target R2 <= 0.05 at every layer.
Centroid sufficiency (runs/token_probe/centroid_sufficiency.py; ridge, val): translation target from the TRUE c_wrist
R2 0.02 (+family 0.03); from c_wrist and the centroid of the corresponding hand points at the start (correspondence =
supervision-only) R2 0.49; their difference alone 0.49.
Reading: the wrist-relative contact centroid was the wrong quantity. Its 60 mm spread is grasp geometry (where the
contacts sit relative to the wrist at the goal); the correction is the ~5-8 mm offset between the contacts and the hand
points meant to touch them. That needs the contact SET compared against the hand SURFACE (implicit correspondence), which
raw (per-contact tokens attending to surface tokens, same frame) and psi_scene (KITE's hand points x z tokens) both have.
The "coordinate" hypothesis in its centroid form is refuted as a sufficient fix; the earlier c_wrist probe was not
diagnostic of success (raw had it for free).

## 27. Standalone contact-set bridge (2026-11-17, job 754018, runs/set_bridge_c3_20k)

Random-init ContactSetBridge (same trunk as the centroid bridge: 512x4), inputs frozen z + start wrist world pose only.
Output 6 unordered slots (wrist-frame position, unit normal, existence); count = #exist>0.5 (no GT count at inference).
Exact 720-assignment matching, shared cost/loss (pos cm^2 + 10(1-cos) + BCE). C3 pool, 20k steps, batch 512, lr 1e-3;
selection on val (fresh goals) macro Chamfer; best step 18k (= last within noise). Separation: 0 overlap, 0 held-out robots.

| pool | count acc | matched pos mm | normal deg | missing@5 | missing@10 | dup@5 | exact set@5 | Chamfer mm |
|---|---|---|---|---|---|---|---|---|
| fresh A/B/L | .54/.59/.60 | 25.8/26.5/26.8 | 44 | .97 | .86 | .00 | .003 | 22.9/23.4/23.2 |
| held-out LEAP | .60 | 28.2 | 46 | .98 | .88 | .00 | .002 | 24.5 |
| train goals | .60 | 14.7 | 30 | .91 | .64 | .00 | .027 | 14.6 |
| chance (random same-family set, own wrist frame), fresh | .19 | 67.5 | 87 | 1.0 | .99 | - | 0 | 64.7 |
| z-NN retrieval + exact transform, fresh | .73 | 212 | 61 | 1.0 | 1.0 | - | 0 | 183 |

Decomposition (best.pt, analysis/shape_vs_centroid.txt): predicted-set centroid error 12-13 mm; matched error after
aligning centroids 25.6-27.6 mm (vs 58-64 mm for "all contacts at the true centroid"). Error is shape, not placement.
Val plateaus from ~10k steps; training goals 14.5 mm, i.e. the bridge does not even memorise -> this run cannot
separate "psi does not retain individual contacts" from "this set head/loss/capacity cannot extract them".
Verdict: not sufficiently accurate for the frozen raw-decoder substitution test (need << 5-8 mm correction scale).
Proposed (not run) control: identical set head fed raw contacts instead of z (upper bound on the head itself).

## 28. Set-bridge diagnostics: raw-contact control, fixed-example overfit, centroid gap (2026-11-17)

Inputs: z bridge = (z, R0, t0); raw control = (request c_pos, c_normal, c_valid [world frame, the tensors psi itself
consumes], R0, t0) through a random-init permutation-invariant encoder (per-contact MLP on scaled pos + Fourier(8) +
normal, 2 masked self-attention layers, masked mean+max pool -> 256 -> LayerNorm) replacing z; same set head/loss/
schedule. Neither receives label validity/count: the head predicts existence; the request mask reaches the raw encoder
only as the set's own membership (as for psi). Tests: permutation invariance, padded contents ignored, a 2 mm move or a
normal flip of ANY valid contact changes the encoding; bridge_args never passes labels (12 tests pass).

| run (fresh goals A/B/L unless noted) | count acc | pos mm | normal deg | Chamfer mm | train goals Chamfer |
|---|---|---|---|---|---|
| z bridge, full C3 (§27) | .54/.59/.60 | 25.8/26.5/26.8 | 44 | 23 | 14.5 |
| raw control, full C3, 20k (754666) | .30/.31/.29 | 63.8/74.3/73.9 | ~80 | 61/69/68 | 44.0 |
| z overfit 64 fixed goals+start, 5k (754667): own examples | 1.00 | 0.00 | 0.0 | 0.00 | - |
| z overfit 1024 fixed, 10k (754668): own examples | 1.00 | 0.00 | 0.0 | 0.00 | - |
| (overfit runs on fresh goals) | ~.4-.5 | 61-81 | - | 55-69 | 64 |

- Raw control FAILED: training loss still falling at 20k (pos 14.6 cm^2 vs 2.0 for z), training-goal Chamfer 60->44 mm
  while fresh goals got worse (52->61 mm, ~chance 65). It memorises and does not generalise. Likely (untested) cause:
  Fourier(8) on world contact positions reaches a 6 mm period -> lookup-table features. So it does NOT validate the
  set machinery under the full regime; it also says nothing about psi.
- Overfit: the z set bridge reproduces 64 and 1024 fixed examples exactly (count, positions, normals), so the head /
  matching / loss can represent exact sets and z separates those goals. The full-run gap is fitting at scale (60k goals,
  start redrawn every batch: training goals 14.5 mm) plus generalisation (fresh 26 mm).
- Centroid gap (analysis/centroid_gap.txt, val; held-out similar): set centroid 12.4 mm; with correct count 10.2, wrong
  count (43% of goals) 15.3; true-count top-k (diagnostic) 11.3 -> count errors explain ~1-2 mm only. Existence-weighted
  9.1, plain mean of all 6 slots 8.1, linear probe from the set bridge's own trunk features 7.7 mm. Per-contact rms
  31.7 mm / sqrt(4) = 15.9 if independent -> errors partly cancel. Conclusion: the set objective is dominated by 30 mm
  per-contact errors and never asks for the shared offset, so its features carry the centroid only to ~8 mm; the same z
  gives 1.5 mm with a dedicated centroid loss. Not evidence of missing information in z.

## 29. Decisive raw control: request mapped analytically to the start-wrist frame (2026-11-17, job 755558)

Change vs §28 raw control: the request's positions/normals are transformed by R0^T(c - t0), R0^T n BEFORE the learned
encoder (raw_frame=wrist) and the contact Fourier features are removed (raw_fourier=0); pose still also given to the
head. Everything else identical (random init, 20k steps, batch 512, lr 1e-3, start redrawn, normal jitter on input).
(The world-frame cf0 rerun 755363 was cancelled before running.)
Input distinction: the raw request's validity mask is an input and legitimately carries its contact count (it defines
the request set). The z bridge sees the request only through psi's z; it receives no mask or count.

Transform-and-copy reference (request -> wrist frame, existence = request mask) on fresh goals, held-out LEAP and
training goals: count 1.000, position 0.00 mm, normal <=0.01 deg, exact set 1.000, Chamfer 0.00 -> label frame and
evaluation verified.

| learned raw control (step 20k = val-selected) | count acc | pos mm | normal deg | missing@5 | missing@10 | dup@5 | exact@5 | Chamfer mm |
|---|---|---|---|---|---|---|---|---|
| fresh A/B/L | .97/.96/.96 | 2.38/2.41/2.56 | 4.5/4.1/4.3 | .09/.09/.10 | .04 | <=.002 | .77/.75/.72 | 2.09/2.08/2.25 |
| held-out LEAP | .96 | 2.91 | 4.9 | .12 | .04 | 0 | .69 | 2.43 |
| training goals | .96 | 2.37-2.56 | 4.3 | .09 | .04 | <=.001 | .75-.78 | 2.05-2.17 |
Train losses at 20k: pos 0.170 cm^2, normal 0.162, exist 0.021 (still falling: not converged). Train ~= fresh.
PASS: the set head / matching / loss learn accurate, generalising reconstructions from a local-frame input, so the set
output machinery is validated. This says nothing about information in psi.

Loss balance of the z set bridge (best.pt, evaluation, training assignment; analysis/centroid_gap.txt): position term
7.16 cm^2 = shared offset 1.11 (16%) + shape 6.05 (84%); normal (x10) 2.48; existence BCE 0.34 (held-out: 8.35 =
1.18 + 7.17; 2.59; 0.32). Training losses at 20k: pos 2.01, normal 1.31, exist 0.29. The positional set loss DOES
penalise centroid displacement; it is a ~16% share dominated by shape error, and normals add ~1/3 more. The earlier
centroid gap (set centroid 12 mm vs 1.5 mm from a centroid-only loss) is therefore a learning / loss-balance issue,
not an absence of centroid supervision.

## 30. World-frame z -> contact set, analytic transform afterwards (2026-11-17, job 756341, runs/set_bridge_zworld_c3_20k_b)

ContactSetBridge with --target world: the network receives ONLY frozen z (its pose inputs held at identity/origin, test
enforced) and predicts world-frame positions, normals, existence. Evaluation maps every predicted slot exactly into the
start-wrist frame with the evaluation start pose: R0^T(p - t0), R0^T n. GT count/validity = supervision only. Same data,
splits, schedule as §27 (random init, 20k steps, batch 512, lr 1e-3, normal jitter on psi input). Selection on fresh
goals (val) Chamfer -> step 20k (=last). Separation: 0 overlap, 0 held-out robots. (Duplicate 756127 on slow node acadia
cancelled at step 4k, consistent with 756341.)
Both frames scored on the same predictions: max |wrist - world| over every metric, pool and family = 0 (rigid invariance).

| pool (both frames identical) | count acc | count bias | pos mm | normal deg | missing@5 | missing@10 | dup@5 | exact@5 | Chamfer mm |
|---|---|---|---|---|---|---|---|---|---|
| fresh A/B/L | .958/.964/.962 | ~0 | 2.70/2.92/2.91 | 7.0/7.1/7.0 | .13/.14/.13 | .05/.06/.06 | <=.003 | .69/.68/.67 | 2.29/2.48/2.44 |
| held-out LEAP | .980 | 0 | 2.99 | 7.5 | .15 | .07 | .002 | .65 | 2.42 |
| training goals A/B/L | .978/.972/.976 | ~0 | 2.33/2.27/2.37 | 5.8/5.7/6.2 | .10-.11 | .03 | .001 | .72-.73 | 1.92-1.99 |
Train loss at 20k: pos 0.131 cm^2, normal 0.266, exist 0.018 (fresh still improving: 4.1 mm at 12k -> 2.8 at 20k).

Compared with §27 (same z, wrist-frame target, pose input): 26 mm / 44 deg / count .57. Compared with the local raw
control §29: 2.4-2.9 mm / 4-5 deg / .96. => z supports per-contact reconstruction (to ~3 mm, normals ~7 deg) that
generalises to fresh goals and unseen LEAP hands; the §27 failure was learning the world->wrist transform inside the
network, not missing information. Small train/fresh gap (2.3 vs 2.8 mm). Not converged.
Meets the threshold for the next diagnostic (predicted contacts -> frozen raw-contact decoder vs true contacts).

## 31. Frozen raw-contact decoder fed bridge-reconstructed contacts (2026-11-17, runs/frozen_decoder_substitution)

Frozen decoder scratch_c3_raw50k_cf8_10x_wrist/best.pt (step 50k, selected in training) + frozen world-frame z bridge
set_bridge_zworld_c3_20k_b/best.pt (step 20k, selected on fresh-goal Chamfer). Nothing tuned on these results.
Identical records and identical fixed starts (val_start_seed + KITE start noise) for every path. Predicted path: z =
psi(request) -> bridge -> world-frame slots; validity = existence > 0.5 (>= 1), predicted count = its sum; packed exactly
as RawContacts packs the true request; the decoder's own wrist preprocessing (+ contact Fourier) applied once, inside the
decoder. All paths scored with pose_metrics against the ORIGINAL target contacts (positions, normals, template
correspondences) and goal wrist; success = all contacts <= 10 mm / 15 deg. Sanity: true path with slot order shuffled
differs by <= 4e-5 mm (decoder is slot-order invariant). Tests: tests/decoder/test_frozen_decoder_substitution.py.

| pool | path | contact mm | success | wrist mm | wrist deg | normal deg |
|---|---|---|---|---|---|---|
| Allegro (n 500, pred count acc .958) | true | 5.80 | .726 | 4.72 | 3.88 | 3.48 |
| | predicted | 6.44 | .660 | 5.25 | 4.41 | 3.87 |
| | no movement | 11.55 | .164 | 7.95 | 7.71 | 6.92 |
| | drop | +0.64 | -.066 | +0.53 | +0.53 | +0.39 |
| Barrett (500, .964) | true | 5.66 | .722 | 4.60 | 4.30 | 3.87 |
| | predicted | 6.48 | .638 | 5.24 | 4.79 | 4.34 |
| | no movement | 14.45 | .116 | 8.02 | 7.77 | 8.08 |
| | drop | +0.82 | -.084 | +0.64 | +0.49 | +0.47 |
| LEAP (504, .962) | true | 7.99 | .433 | 5.68 | 4.48 | 5.16 |
| | predicted | 8.39 | .387 | 5.88 | 4.89 | 5.44 |
| | no movement | 13.93 | .089 | 7.75 | 7.67 | 8.28 |
| | drop | +0.40 | -.046 | +0.19 | +0.41 | +0.28 |
| held-out LEAP (500, .980) | true | 8.51 | .370 | 6.08 | 4.86 | 5.48 |
| | predicted | 9.11 | .352 | 6.35 | 5.41 | 5.80 |
| | no movement | 14.98 | .072 | 7.96 | 7.92 | 8.60 |
| | drop | +0.60 | -.018 | +0.27 | +0.55 | +0.32 |

True path reproduces §19 exactly. Share of the true path's contact-error gain over no movement retained with predicted
contacts: Allegro 89%, Barrett 91%, LEAP 93%, held-out LEAP 91%. The deployment pipeline (z + start pose only ->
bridge -> decoder, no retraining) reaches 6.44/6.48/8.39 mm, held-out 9.11 (plain psi decoders: ~11.5/14.4/14.1, 15.1;
psi_scene: 5.35/4.60/10.92, 13.02; KITE on known: 4.09/3.70).

## 32. Frozen two-stage system: held-out Wuji family + cross-body benchmark (2026-11-17, runs/crossbody_heldout_eval)

Protocol fixed beforehand: get-ur-kite-model/CROSSBODY_HELDOUT_PROTOCOL.md (+ Wuji URDF correction: KITE's
right_floating.urdf). Frozen: raw decoder cf8_10x_wrist best.pt (50k) + world-frame z set bridge best.pt (20k).
Bank sanity: adding Wuji leaves decoder outputs on existing records bit-identical (max |diff| 0).

A. Wuji same-body (500 KITE-sampler records, seed 202; never seen by decoder/bridge; seen by psi), correspondence scoring:
| path | contact mm | normal deg | success | wrist mm | wrist deg | limit viol. | motion mm |
|---|---|---|---|---|---|---|---|
| true request | 14.31 | 8.03 | .104 | 9.86 | 8.60 | .144 | 7.9 |
| two-stage (predicted) | 14.38 | 8.03 | .106 | 9.82 | 8.63 | .146 | 7.8 |
| no movement | 14.19 | 7.78 | .118 | 7.96 | 8.08 | 0 | 0 |
The frozen decoder does not transfer to an unseen family: it moves the hand ~8 mm but not toward the goal (contact error
and wrist error slightly WORSE than not moving). Predicted ~= true contacts (bridge is not the limitation here).

B. Cross-body (768 cases: 4 source families x 3 other targets x 64; KITE closest-example starts). Position-nearest
surface distance (valid) -- success / angle / oracle feasibility NOT valid (degenerate rule, see protocol addendum):
| target | true mm (rms) | two-stage mm (rms) | no movement mm (rms) | oracle mm | true motion mm | limit viol. true |
|---|---|---|---|---|---|---|
| Allegro | 6.76 (8.31) | 6.77 (8.33) | 9.43 (11.51) | 1.58 | 9.4 | .18 |
| Barrett | 7.18 (8.77) | 7.26 (8.85) | 10.30 (12.27) | 1.81 | 9.7 | .05 |
| held-out LEAP | 8.14 (9.89) | 8.12 (9.91) | 11.04 (13.54) | 2.01 | 6.8 | .03 |
| Wuji | 12.20 (14.82) | 12.26 (14.87) | 13.18 (16.18) | 2.02 | 9.2 | .17 |
| all | 8.57 (10.45) | 8.60 (10.49) | 10.99 (13.38) | 1.85 | 8.8 | .11 |
Two-stage tracks the true-request path within 0.1 mm everywhere. Both reduce position-nearest distance by ~2.4 mm over
no movement on known/LEAP targets, little on Wuji. Position-only distance is lenient (any face counts); a normal-aware
rescoring is proposed and pending approval before interpreting success.

## 33. Rescored with the approved normal-compatible rule (2026-11-17, runs/crossbody_heldout_eval_v2)

Rule (approved after §32): per requested contact, distance to the nearest template point whose normal is within 15 deg;
success = all <= 10 mm; oracle assigns by the same rule. Check at exact goals: 0.02 mm, success 1.000 (val and Wuji);
at starts: 10.7 / 11.2 mm, success .30 / .21. Same records, starts, weights as §32 (bank sanity again 0).
Caveat: a contact with no normal-compatible point counts 1 m in the compat mean (1.0-1.4% of decoder-path cases;
inflates those means by ~10-14 mm) -> success rates are the primary number.

A. Wuji same-body: success (compat) true .234, two-stage .240, no movement .200; corr success .104/.106/.118;
compat mm 10.55/10.62/11.16. Small gain under the lenient-to-faces rule, none under correspondence scoring.

B. Cross-body (768):
| target | oracle-feasible | true success | two-stage success | no-move success | oracle motion mm/deg | decoder motion mm/deg |
|---|---|---|---|---|---|---|
| Allegro | .469 | .005 | .005 | .005 | 54.5 / 35.9 | 9.4 / 10.3 |
| Barrett | .385 | .026 | .026 | 0 | 58.0 / 37.2 | 9.7 / 8.6 |
| held-out LEAP | .380 | 0 | 0 | 0 | 53.3 / 31.9 | 6.8 / 8.2 |
| Wuji | .552 | .010 | .005 | .005 | 44.4 / 28.5 | 9.2 / 7.6 |
| all | .447 | .010 | .009 | .003 | 52.5 / 33.4 | 8.8 / 8.7 |
On the oracle-feasible subset: true .023, two-stage .020, no movement .006 (oracle 1.000 at 2.0 mm).
Reading: ~45% of these cross-body requests are reachable from KITE-style starts, but reaching them needs ~5 cm / 33 deg
of motion; the frozen decoder (trained on near-goal starts: ~1 cm corrections) moves ~9 mm / 9 deg and almost never
succeeds. The two-stage system matches the true-request path (the bridge is not the bottleneck); the limitation is the
decoder's start distribution / same-body training, and transfer to an unseen family (Wuji A).

## 34. Direct GET decoder + training-only auxiliary contact-set head: C1/C2 (2026-11-17)

Model: plain psi-wrist GET (frozen z, start state, robot description) + aux_head (MLP on mean final intent tokens +
root token -> 6 wrist-frame slots: position, normal, existence), exact set-matching loss x aux_set_weight 0.05; labels
from training contacts only; motion path never reads it (test: perturbing aux_head leaves res_root/res_coord/q/t/R
identical; aux loss reaches the trunk, not the motion heads); built last (backbone init identical to plain psi).
1,337,844 trainable params (plain psi-wrist 1,195,466 + aux 142k). Configs aux_wrist_c1 / aux_wrist_c2_20k (= psi_wrist
C1/C2 + aux); c3_aux50k_10x_wrist prepared, not launched. slurm/env.local.sh (gitignored) updated to the uni-kite paths.

| gate | own-goal contact mm A/B/L | success | wrist mm | aux matched pos rms | aux count acc |
|---|---|---|---|---|---|
| C1 (3k, fixed start) | 0.013/0.013/0.014 | 1.0 | 0.007/0.006/0.008 | 0.04 mm | 1.0 |
| C2 (20k, start redrawn, 48 goals) | 0.76/0.54/0.53 | 1.0 | 0.62/0.36/0.41 | 1.0 mm | 1.0 |
| ref plain psi-wrist C2 | 0.74/0.75/0.82 | 1.0 | 0.68/0.54/0.51 | - | - |
| ref joint bridge C2 | 0.91/0.83/0.81 | 1.0 | 0.81/0.55/0.50 | - | - |
Losses (C2): steps 50: decoder 1.400, aux 55.72 (pos 47.73, normal 7.33, exist 0.67), weighted aux 2.79;
step 10k: decoder 0.0126, aux 0.059; step 20k: decoder 0.0039, aux 0.0169 (weighted 0.0008).
Both PASS. Fresh goals / held-out LEAP from 48 training goals: contact 22.4/27.4/27.5, 28.2 mm (worse than no movement,
as every C2 model), aux 65-78 mm, count .30-.36 -- memorisation gates, not generalisation evidence.
Wuji same-body start perturbations verified identical in distribution to the source families
(runs/crossbody_heldout_eval_v2/perturbation_check.txt: wrist 7.96 mm / 8.08 deg / joint 3.7% of range vs 7.7-8.0 /
7.7-7.9 / 3.5-3.9%; perturbed-joint share .36 within .31-.54; only contact-relevant joints are perturbed, KITE recipe).

## 35. Direct decoder C3 vs frozen two-stage, matched records (2026-11-17, runs/scratch_c3_aux50k_10x_wrist, runs/crossbody_eval_direct_c3)

Budgets: direct 1,337,844 learned params, 50k x 256 (same data/schedule as plain psi C3); two-stage 1.75M (raw decoder
985,418, 50k x 256 + set bridge 759,850, 20k x 512). Direct selected on source validation only: best.pt = step 7.5k
(same selection as plain psi C3); last (50k) indistinguishable on val. Bank sanity 0 for both systems. Identical
records and fixed starts per pool; aux head ignored at evaluation.

1) Near-goal, source families (correspondence scoring):
| pool | two-stage contact mm / success | direct | true-request ref | no movement |
|---|---|---|---|---|
| fresh Allegro | 6.44 / .660 | 11.49 / .172 | 5.80 / .726 | 11.55 / .164 |
| fresh Barrett | 6.48 / .638 | 14.31 / .112 | 5.66 / .722 | 14.45 / .116 |
| fresh LEAP | 8.39 / .387 | 13.79 / .105 | 7.99 / .433 | 13.93 / .089 |
| held-out LEAP hands | 9.11 / .352 | 14.82 / .084 | 8.51 / .370 | 14.98 / .072 |
Direct wrist error = no movement (7.95/8.02/7.75/7.96 mm); mean motion 0.10 mm: it does not move.

2) Near-goal Wuji (unseen by our decoders; seen by KITE's encoder psi; same start perturbation distribution):
two-stage 14.38 / .106, direct 14.19 / .114, true-request 14.31 / .104, no movement 14.19 / .118 (compat success
.240 / .202 / .234 / .200). No system transfers; direct simply does not move.

3) Cross-body (768; optimizer found solutions for 44.7%): success two-stage .009, direct .003, true-request .010,
no movement .003; on cases with a found solution .020 / .006 / .023 / .006. Direct motion 0.10 mm / 0.07 deg.

Aux head (fresh goals): matched position rms 45 mm at the selected step (50 mm at 50k), count acc .23-.28 (~chance);
training goals 39 -> 19 mm (memorises, does not generalise). Train losses: decoder 1.197 -> 1.099; aux 55.4 -> 7.27
(pos 4.55, normal 2.23, exist 0.49 ~ chance); weighted aux 2.77 -> 0.36.
Conclusion (matched data/budget): the training-only auxiliary contact-set head does not make the direct psi decoder
use z; it stays at no movement like plain psi. The two-stage system (analytic world->wrist step between a world-frame
z reconstruction and the raw-contact decoder) is the only psi-based system here that moves the hand correctly.
Open (not run): why the aux head cannot even recover the count from the intent tokens (the standalone world-frame
bridge reaches .96 from the same z).

## 36. Where contact information is lost in the direct psi decoder (2026-11-17, runs/aux_probe; probes only)

Fresh 2-layer MLP probes on FROZEN features (train pool 30k at a fixed start; scored on train 5k / fresh val 1504).
| features (direct C3 best@7.5k) | count acc train/val | world centroid mm train/val | wrist centroid mm train/val |
|---|---|---|---|
| z (standardised) | 1.00 / .989 | 1.3 / 1.8 | 14.8 / 76.8 (no pose in z) |
| intent tokens entering transformer | 1.00 / .972 | 20.0 / 41.8 | 12.5 / 75.3 |
| aux-head input (final intent mean + root) | 1.00 / .883 | 55.8 / 121.9 | 7.3 / 44.4 |
| all final intent tokens + root | 1.00 / .933 | 13.8 / 80.9 | 1.8 / 45.0 |
(50k checkpoint: same pattern, sharper memorisation: final wrist centroid 0.17 mm train vs 44.5 val. Plain psi C3
best@7.5k: intent_in world 24.9/36.4, final wrist 10.2/46.2 -- the same.) Count majority baseline .204.
- intent_proj already discards precise world position (1.8 -> 36-45 mm on fresh goals) in both psi decoders.
- final tokens encode the wrist-frame centroid of TRAINING goals (to 0.2-1.8 mm) but not fresh ones (~45 mm):
  memorisation, not a generalising world->wrist computation -> no movement on fresh goals.
- Aux head count: its own .25 vs a CE probe on the SAME input .88. An offline refit of an identical head with the same
  set loss on the frozen features also stays at .25-.27 (exist BCE ~.52) even on training goals -> the count deficit
  is a property of the head + set-matching objective when position fit is poor (existence learning is coupled to
  position matching), NOT evidence that the tokens lack count information. (runs/aux_probe/offline_head.json)

## 37. Independent review follow-up (2026-11-17): surface tokens, Wuji geometry, aux metric, qualified conclusions

A. Surface-token selection bug (CONFIRMED, robots.py:298). For URDF robots (Allegro, Barrett, Wuji) desc.surf_idx is
None and the fallback runs FPS on `surface_template_points_local` -- link-LOCAL coordinates, not one hand frame (the
comment "FPS in the zero pose" is wrong). LEAP uses the dataset's precomputed FPS (unaffected). Coverage at the zero
pose (runs/surface_coverage/coverage.json + PNGs):
| hand | current: palm/base tokens, per finger | links hit | covering radius | common-frame FPS: palm, per finger | radius |
|---|---|---|---|---|---|
| Allegro | 24; 2/0/3/3 | 5/21 | 126.5 mm | 13; 5/4/5/5 | 35.6 mm |
| Barrett | 21; 3/5/3 | 7/10 | 68.6 mm | 12; 7/6/7 | 32.6 mm |
| Wuji | 28; 0/1/1/2/0 (two fingers none) | 5/26 | 96.9 mm | 12; 4/5/4/3/4 | 24.5 mm |
| LEAP (dataset FPS) | 17; spread | 9/9 | 33.7 mm | 18 | 33.4 mm |
Every trained decoder saw the palm-heavy token sets for Allegro/Barrett; Wuji is described almost only by its palm.
NOT established as the cause of the Wuji failure (several structural differences change at once). Fixing it changes
model inputs -> requires retraining; no run launched.

B. Wuji geometry verified independently (runs/surface_coverage/wuji_geometry_check.json): on all 2,500 generated
records our FK + template reproduce KITE's sampled contacts at q2 to <= 0.0001 mm / 0.04 deg; coordinate order equals
KITE's export; 0 goal joints outside our parsed limits; 20 coords, 4096 template points, 26 template links.

C. Aux metric bug (CONFIRMED, fixed): train_scratch.evaluate assigned a batch-mean matched pos rms to every example
(mixing families). Replaced by per-example aux_set_metrics (matched mean position mm, normal deg, count); test added.
Recomputed (runs/aux_probe/aux_metrics_recomputed.json):
| checkpoint / pool | aux pos mm A/B/L | normal deg | count acc |
|---|---|---|---|
| C2 own goals | 0.84/0.70/0.66 | 0.7-0.8 | 1.0 |
| C2 fresh / held-out | 58.7/65.2/67.3 ; 71.2 | 76-79 | .30-.36 |
| C3 best@7.5k fresh / held-out | 40.7/41.0/42.6 ; 46.3 | 67-74 | .23-.28 |
| C3 best@7.5k training goals | 35.3/34.8/35.0 | 63-64 | .24-.28 |
| C3 last@50k fresh / training goals | 45.0/44.4/46.5 ; 17.4/19.3/18.7 | 67-72 ; 33 | .25-.31 |
Motion/contact results (pose_metrics) were never affected. §34/§35 aux numbers are superseded by this table.

D. Qualified conclusions (replace stronger wording in §31/§35):
- Supported: the tested staged system works better than this particular direct implementation. NOT supported: that a
  single decoder cannot learn this.
- The comparison changes more than the number of networks: the working bridge predicts WORLD-frame contacts directly
  from z and then applies an exact transform; the direct model's aux head predicts WRIST-frame contacts from tokens
  after the transformer (the harder task that already failed standalone in §27). Sampled training examples: two-stage
  ~23.0M (bridge 20k x 512 = 10.24M at lr 1e-3 + decoder 50k x 256 = 12.8M at lr 3e-4) vs direct 12.8M (lr 3e-4);
  objectives differ.
- Morphology coverage of all trained models: 60,004 training records = 1 Allegro, 1 Barrett, 12 LEAP variants (data
  hashes match across runs). The earlier 966-hand campaign was a separate experiment. Wuji adds a fifth finger and 20
  controllable joints (training max 16) and other differences at once; its failure is meaningful but not attributable
  to one factor.
- Cross-body evaluation assumptions: one correction step from starts far from training's near-goal starts; start
  poses chosen from banks of TARGET-hand synthetic examples (KITE closest-example) -- target-hand example data is used
  at test time (not for training) and must be disclosed as required target information.

## 38. Surface-sampling correction: old vs corrected raw-contact decoder (2026-11-18, runs/scratch_c3_raw50k_cf8_10x_wrist_hf, runs/crossbody_eval_corrected_hf)

Retrained `scratch_c3_raw50k_cf8_10x_wrist` with only `surface_sampling: hand_frame` changed (best@47.5k on source validation; same data, seed, loss, 50k budget). Same records, starts and frozen world-frame translator (`set_bridge_zworld_c3_20k_b`). Full table: `runs/crossbody_eval_corrected_hf/old_vs_corrected.md`.

- Source val / held-out LEAP: small, consistent gains (held-out true 8.51→8.28 mm, predicted 9.11→8.89 mm). Allegro limit violations up 0.152→0.170.
- Wuji near-goal (unseen by our decoder, seen by KITE's encoder ψ): worse. True contacts 14.3→15.9 mm, success 0.234→0.158; no-movement reference 14.2 mm / 0.200. The corrected decoder now does worse than not moving.
- Cross-body: unchanged within noise (all-target success 1.0%→0.8%); the optimizer found solutions for 44.7% in both (independent of the decoder).
- Reading: coverage is a confirmed input correction but does not explain the Wuji failure; better finger coverage did not give morphology transfer. Single seed; per-example paired tests not computed.

## 39. Per-example Wuji diagnosis: why neither raw-contact decoder beats no movement (2026-11-18, runs/wuji_per_example)

Scripts: `scripts/decoder/diagnostics/wuji_per_example.py`, `wuji_failure_plots.py`. Same records, starts and frozen translator as §38; numbers reproduce §38 exactly. Wuji is unseen by our decoder, seen by KITE's encoder ψ.

Success metrics (both reported; the §38 "success" column was the second):
- corr_success: every contact's fixed-correspondence point (the record's template point) within 10 mm AND 15°; error = corr_mm.
- compat_success (protocol score): every contact within 10 mm of the nearest template point whose normal is within 15°; error = compat_mm.

Wuji, true contacts (old → corrected); no movement corr 14.19 mm / 0.118, compat 11.16 mm / 0.200:
- full: corr 14.31→15.85 mm, 0.104→0.084; compat 10.55→11.34 mm, 0.234→0.158.
- wrist only (predicted wrist, start fingers): corr 14.37→15.90 — the whole full-pose result, old and corrected.
- fingers only (start wrist, predicted fingers): corr 14.06→14.04 ≈ no movement.
- references: goal wrist only 3.28 mm / 0.78; goal fingers only 13.59 mm / 0.16. The error budget is the wrist, as on source hands (Allegro: goal wrist 2.53, goal fingers 10.91).
- Paired (500, bootstrap 95%): corrected − old corr_mm +1.54 [1.03, 2.06]; old − no-move +0.12 [−0.45, 0.70]; compat_success corrected − old −0.076 [−0.116, −0.036]. The regression in §38 is real, and entirely in the wrist path.

Wrist tracking — fraction of the predicted correction explained by the needed correction (R², start-wrist frame):
| | rotation R² | rotation gain | translation R² | |pred rot| when need < 3° |
|---|---|---|---|---|
| Allegro | 0.75 | 0.6–0.7 | 0.63 | 3.0° |
| Barrett | 0.68 | 0.6–0.7 | 0.62 | 3.2° |
| LEAP | 0.66 | 0.5–0.7 | 0.44 | 3.7° |
| Wuji old / corrected | 0.01 / 0.06 | 0.2–0.3 | −0.15 / −0.18 | 5.6° / 6.0° |

Diagnosis: on Wuji the decoder emits wrist corrections of normal size (6–7°) that are essentially uncorrelated with the needed correction, so it does not infer the wrist error from the Wuji contact–surface relation. Fingers contribute nothing either way. Joint-limit violations are small (0.2–3% per joint, mean overshoot < 0.1% of range) and fingers-only ≈ no movement, so limits do not explain it. Per finger, the error is spread across all five fingers and the palm, as expected from a wrist error. Candidate cause (untested): Wuji's surface sits ~9 cm from its base origin (Allegro ~1 cm, Barrett/LEAP ~5 cm), so the contact-error → wrist-correction relation is outside the training range; better surface coverage does not supply it. A cheap no-training test: re-anchor Wuji's base frame (same world starts and goals) and re-evaluate.

## 40. Coordinate-sensitivity test: Wuji base frame re-anchored, no training (2026-11-18, runs/wuji_reanchor)

Script `scripts/decoder/diagnostics/wuji_reanchor.py`. Frozen old and corrected raw-contact decoders, frozen translator and ψ. Only Wuji's base-frame origin moves (rotation unchanged), via a new root link with a fixed joint (translate(−o)); starts/goals transformed t' = t + R o, so physical starts and goals are identical; predictions mapped back (t = t' − R' o) and scored with the original bank. Offsets fixed from robot geometry before predictions: hand_centroid = centroid of all template points at q = clip(0) (−0.62, 0.72, 8.75) cm; palm_centroid = centroid of base-link template points (−0.29, 0.06, 3.85) cm.

Correction to §39: the palm-centroid distance from the base origin is 3.9 cm for Wuji vs 4.1 (Allegro), 3.1 (Barrett), 6.0 cm (LEAP); the "~9 cm" in §39 was the whole-surface centroid with extended fingers, so the lever-arm candidate was overstated.

Geometry checks (pass, both banks, start and goal poses, all 500 records): template points ≤ 1.2e-4 mm, normals 0, contact points ≤ 6e-5 mm, identical surface-token selection, other robots unchanged.

Wuji, true contacts (no movement: corr 14.19 mm / 0.118, compat 11.16 mm / 0.200):
| anchor | decoder | corr_mm | corr_success | compat_mm | compat_success | rot R² | trans R² |
|---|---|---|---|---|---|---|---|
| original | old | 14.31 | 0.104 | 10.55 | 0.234 | 0.01 | −0.15 |
| original | corrected | 15.85 | 0.084 | 11.34 | 0.158 | 0.06 | −0.18 |
| hand centroid | old | 14.93 | 0.068 | 11.47 | 0.148 | −0.01 | −0.10 |
| hand centroid | corrected | 13.71 | 0.082 | 10.46 | 0.214 | 0.07 | −0.09 |
| palm centroid | old | 13.38 | 0.116 | 10.19 | 0.254 | 0.03 | −0.11 |
| palm centroid | corrected | 13.54 | 0.116 | 10.09 | 0.248 | 0.10 | 0.02 |
Paired vs original anchor (corr_mm, 95%): corrected −2.14 [−2.75, −1.54] (hand), −2.31 [−2.88, −1.75] (palm); old +0.62 [0.05, 1.17] (hand), −0.93 [−1.34, −0.51] (palm). Predicted contacts give the same pattern.

Reading (coordinate sensitivity, not an explanation): predictions depend on the base-frame convention by up to ~2.3 mm at identical physical geometry, and the old-vs-corrected ordering of §38 flips with the anchor, so that regression is not a robust ranking. No anchor restores wrist inference (rot R² ≤ 0.10, trans R² ≤ 0.02 vs source 0.44–0.75), and the best case is ~0.8 mm better than no movement. The base origin offset does not explain the Wuji failure.

## 41. J1: one jointly trained decoder with an explicit internal contact stage (2026-11-18, runs/j1_joint50k_hf)

Code: `src/getkite/decoder/joint_contact_get.py` (JointContactGET, pack_contacts), `src/getkite/decoder/train_joint.py`, config `configs/decoder/scratch/j1_joint50k_hf.json`, `slurm/joint.sbatch`. Structure: frozen ψ z → world-frame contact-set predictor (config of set_bridge_zworld_c3_20k_b; random init) → packing (slots by existence logit, hard mask = inference mask) → raw-contact GET (config of c3_raw50k_cf8_10x_wrist_hf, hand_frame surface sampling; random init; its own exact start-wrist transform). One forward interface, one checkpoint. The internal contact representation is kept explicit, not eliminated.
- Loss = motion residual loss + contact_weight × contact set loss (world labels from training contacts; supervision only). contact_weight = 1.0: the components have disjoint parameters, so the weight only sets, for the contact predictor, the balance between its own set loss and the motion gradient; at 1.0 the reference objectives are recovered when the motion gradient is zero.
- Optimiser: decoder group lr 3e-4, wd 1e-4, clip 1.0 (reference decoder); contact group lr 1e-3, wd 1e-4, clip 1.0 (reference translator); one warmup-500 + cosine schedule over 50k; batch 256; seed 0; same data and start sampling; normal jitter on ψ's input only.
- Pre-training checks (`checks.json`, run on a deep copy with small random output heads because the decoder's root/joint heads are zero-initialised, which makes the motion gradient into the contact stage exactly zero at step 0): ψ frozen; packing = two-stage packing (max abs 0); motion gradient reaches predicted positions (|g| 3.4e-3) and normals and the contact-predictor parameters (1.0e-2); zero to existence logits and invalid slots; contact loss gives zero gradient to the decoder; central directional derivative vs autograd rel. error 0.1%. Initial losses: motion 1.13, contact 1583 (pos 1573, normal 9.0, exist 0.76).
- Smoke (600 steps): contact loss 432 → 2.7; motion-gradient norm into the contact predictor 1e-4 (step 50) → ~0.05 (step 500–600) vs contact-loss gradient 28–990 (clip 1.0); predicted/true active contacts 4.12 / 3.97, count acc 0.25; val contact error 15.9 → 9.0 mm. Cross-body evaluation path verified on the smoke checkpoint.
- Reference for comparison: corrected two-stage (frozen translator + c3_raw50k_cf8_10x_wrist_hf): val Allegro/Barrett/LEAP 6.34/6.53/8.31 mm, held-out LEAP 8.89 (true-contact bound 5.71/5.60/7.90, 8.28).

## 42. J1 results: source goals first, then Wuji and cross-body separately (2026-11-18, runs/j1_joint50k_hf, runs/crossbody_eval_j1)

Job 786951 (one preemption at ~2.4k, resumed from last.pt), 50k steps, settings unchanged mid-run. Best checkpoint = step 50000 (selected on source val macro success, then contact_mm). J1 = one jointly trained decoder with two internal learned components; this tests whether separate training is necessary, not whether the explicit contact representation can be dropped. Scripts: `crossbody_heldout_eval.py --direct` (J1 path), `j1_paired.py` (paired per-example J1 vs two-stage vs no movement + wrist R²), `j1_contact_compare.py` (internal contact stage vs standalone translator, same metric code).

**1. Fresh source goals (deciding result).** Same records and starts; two-stage = frozen translator + c3_raw50k_cf8_10x_wrist_hf (step 47.5k), predicted contacts.
| pool | system | corr_mm | corr_success | compat_mm | compat_success | rot R² | trans R² |
|---|---|---|---|---|---|---|---|
| Allegro val | two-stage | 6.34 | 0.674 | 3.98 | 0.854 | 0.66 | 0.56 |
| | J1 | 6.31 | 0.666 | 3.91 | 0.884 | 0.70 | 0.58 |
| Barrett val | two-stage | 6.53 | 0.628 | 4.04 | 0.838 | 0.58 | 0.53 |
| | J1 | 6.31 | 0.684 | 3.87 | 0.880 | 0.62 | 0.55 |
| LEAP val | two-stage | 8.31 | 0.405 | 5.92 | 0.647 | 0.58 | 0.42 |
| | J1 | 8.30 | 0.413 | 6.02 | 0.647 | 0.59 | 0.41 |
| held-out LEAP | two-stage | 8.89 | 0.344 | 6.75 | 0.576 | 0.54 | 0.36 |
| | J1 | 8.85 | 0.312 | 6.76 | 0.570 | 0.56 | 0.37 |
No movement: corr 11.55/14.45/13.93, 14.98 mm. True-contact bound (two-stage decoder, true contacts): 5.71/5.60/7.90, 8.28 mm.
Paired J1 − two-stage, corr_mm (95%): Allegro −0.04 [−0.19, 0.12], Barrett −0.22 [−0.44, −0.01], LEAP −0.01 [−0.19, 0.19], held-out −0.04 [−0.26, 0.17]. compat_success: +0.03 [0.00, 0.06], +0.04 [0.01, 0.08], 0.00 [−0.04, 0.04], −0.01 [−0.05, 0.03]. Held-out corr_success −0.03 [−0.07, 0.00].
Reading: similar to the reference (Barrett slightly better; others within noise). One jointly trained structured decoder reproduces the two-stage source result; separate training is not necessary for it.

**Internal contact stage (fresh goals, same metric code, `contact_compare.json`).** J1 vs standalone translator: val pos 2.19 vs 2.35 mm, normal 5.15 vs 5.30°, count acc 0.968 vs 0.961, predicted/true active contacts 3.99/3.99; held-out LEAP 2.25 vs 2.46 mm, 5.28 vs 5.58°, 0.986 vs 0.980, 4.03/4.03. (These numbers use train_joint.contact_eval's matching; the translator's own results.json uses a different matching and reports 2.85/2.99 mm.) Joint optimisation reproduced, slightly improved, the standalone translator.

**Training dynamics (logged every 50 steps; gradient norms measured separately on the same batch, before clipping).** contact_weight 1.0.
| steps | motion loss | contact loss | |g| into contact predictor from motion | from contact loss | motion/contact | count acc |
|---|---|---|---|---|---|---|
| 0–2.5k | 1.18 | 3.4 | 0.6 | 130 | 0.03 | 0.75 |
| 5–10k | 0.94 | 1.12 | 3.2 | 36 | 0.14 | 0.94 |
| 20–30k | 0.71 | 0.52 | 2.5 | 9.2 | ~0.3 | 0.98 |
| 42–50k | 0.64 | 0.37 (pos 0.11, normal 0.25, exist 0.012) | 2.0 | 2.3 | ~0.85 | 0.98 |
Clipping frequency (pre-clip norm > 1.0 at logged steps): contact predictor 100% throughout (updates always rescaled); decoder 0% before 20k, 1–2.5% after. Initial loss imbalance (contact 1583 vs motion 1.13) did not prevent motion learning; the motion share of the contact-predictor gradient rose steadily to ~parity.

**2. Wuji near-goal (unseen by our decoder, seen by KITE's encoder ψ), n=500.** No movement: corr 14.19 / 0.118, compat 11.16 / 0.200.
| system | corr_mm | corr_success | compat_mm | compat_success | rot R² | trans R² |
|---|---|---|---|---|---|---|
| two-stage (predicted) | 16.03 | 0.080 | 11.48 | 0.154 | 0.06 | −0.19 |
| J1 | 13.44 | 0.104 | 10.04 | 0.226 | 0.12 | −0.13 |
Paired J1 − no movement: corr_mm −0.75 [−1.32, −0.14], compat_mm −1.12 [−1.60, −0.63], corr_success −0.014 [−0.05, 0.02], compat_success +0.026 [−0.02, 0.07]. J1 − two-stage: corr_mm −2.59 [−3.10, −2.11], compat_success +0.07 [0.03, 0.11].
Reading: J1 is less harmful than the two-stage system on Wuji and slightly reduces mean error vs no movement, but success rates are not distinguishable from no movement and wrist inference is still absent (rot R² 0.12, trans R² negative vs 0.36–0.70 on source). The J1-vs-two-stage gap (2.6 mm) is of the same size as the base-frame coordinate sensitivity in §40 (up to 2.3 mm), so it is not evidence of morphology transfer. Wuji morphology transfer still fails.

**3. Cross-body (start-distribution test; separate diagnosis), normal-compatible scoring.** Aggregate success: J1 0.9%, two-stage 0.8%, true contacts 0.8%, no movement 0.3%; compat_mm 42.6 / 42.9 / 41.9 / 47.2. Per target, J1 = two-stage within noise (64 pairs each). Optimizer found solutions for 44.7% (§33), unchanged. Cross-body failure unchanged.

Conclusion: J1 answers the source question positively (similar or better than the corrected two-stage reference, internal contacts as accurate as the standalone translator). Wuji and cross-body remain failures, as separate diagnoses. No further runs launched.
