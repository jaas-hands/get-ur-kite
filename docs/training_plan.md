# Training plan: the LEAP head and the GETyourKITE decoder

Status on Nov 18, 2026. Spotlight talk Nov 28 – Dec 3, report Dec 4–11.

## What exists

| Piece | Where | State |
|---|---|---|
| LEAP hand family: 1,741 hands, hand-level splits, 2.4M KITE-recipe examples | `third_party/get-ur-kite-dataset` (submodule) | done, regenerates in ~3 min |
| KITE's frozen intent encoder ψ | `getkite.leap.intent.IntentEncoder` | done; reproduces KITE's stored latents to 4e-5 relative error |
| Batches: z, contacts, start/goal poses, joint features, joint frames, 32 hand tokens, graph | `getkite.leap.data.LeapSplit` | done; ~16k examples/s on the laptop GPU |
| Batched forward kinematics over the family | `getkite.leap.hands.HandBank` | done; reproduces the dataset's contacts to 5e-8 m |
| Floating, actuated LEAP MJCF in KITE's format, for any hand | `assets/leap/`, `scripts/make_leap_mjcf.py` | done; matches the FK to 1e-6 |
| KITE's decoder trunk (scene encoder, action decoder), 4 other hands' heads | `third_party/KITE/checkpoints/decoder/release.pt` | released |
| LEAP in KITE's simulator, tasks, start poses | – | missing (step S below) |
| Trained interaction policy π | – | not released |

## Shared pieces (needed by both)

**M. Open-loop metrics** (`getkite/leap/metrics.py`), computed on any split:
- *contact error:* FK of the prediction at the contacts' template points vs. the target contacts
  (position in mm, normal angle in degrees). This is what the decoder is for.
- *success rate:* all contacts within a tolerance (start with 1 cm; the dataset's open question).
- *joint error* on the masked joints, and *drift* of the unmasked ones (should stay at q_init).
- reported per test split: `test_finger_subset`, `test_joint_count`, `test_geometry`, `test_combined`.

**S. LEAP in KITE's simulator** (for Oracle-Intent rollouts; not needed for open-loop results):
1. An `EmbodimentSpec` for `leap_right` (as `allegro_right` in `experiment/tabletop_sim/robot.py`),
   using `assets/leap/<hand>_floating.xml`, and a scene hook like `_build_floating_allegro_spec`.
   KITE stays unmodified: register it from `getkite` at import time (as `ScoringSession` patches the env).
2. Start poses per task, in the format of `allegro_right_keyboard_press_start_pose.pt`: put LEAP's
   palm where Allegro's is.
3. A rollout runtime that calls our decoder instead of KITE's head (the rollout script's
   low-level predict call), with q in KITE's layout: wrist as `virtual_joint_*` (x, y, z and
   intrinsic x-y-z Euler angles, `getkite.leap.mjcf.wrist_to_virtual`), then fingers in URDF order.
4. Contact settings (friction, collision model): start from Allegro's, then check grasps with
   the demo viewer and the scores.

Estimate: 2–3 days. Without it, everything below is evaluated open-loop only.

## 1. The LEAP head (KITE's own decoder for LEAP)

**Why.** KITE's release has no LEAP decoder, so there is nothing to compare against on LEAP. A
KITE-style head for the nominal LEAP is the reference "KITE on LEAP", and it gives the baselines:

| Baseline | From the LEAP head |
|---|---|
| B0: KITE, nominal LEAP | the head itself, on the nominal hand |
| B1: KITE, sliced | for a lesioned hand, drop the locked joints' rows of `joint_identity_embed` and the hand tokens move with the URDF; zero-shot |
| B2: KITE, shifted | for a hand with a finger removed, give the remaining fingers the rows of their structural neighbours (index missing: middle ← index, ring ← middle), so the hand keeps the full hand's finger roles; zero-shot |
| B3: KITE, head-only fine-tune | fine-tune a copy of the head on the test hand, curve over minutes / examples |
| B4: KITE, per-hand head | one head trained on each test hand (upper bound; on a subset of test hands) |

**What is trained.** Only the per-hand parts of one head of the decoder bank (~143k parameters,
for 16 joints): `joint_identity_embed`, `qref_joint_proj`, `head_joints`, the pose
projections and `action_queries`. Everything shared (ψ, `cls_unpacker`,
`bottleneck_scene_encoder`, `action_decoder`, `hand_proj`) is loaded from `release.pt` and
frozen, so Z and the other hands stay exactly as released.

**Data.** The nominal hand `I4-M4-R4-T4_g0` with KITE's budget: generate a separate split with
only this hand, ~200k–1M examples (`kite_data.generate --splits <name> --per-hand <name>=N`, its
own seed), plus 2,000 validation examples. Collision rejection makes generation ~1 h on 16 cores
for 1M; start with 200k.

**Code.**
1. `getkite/kite_head.py`: build one `ContactProbePolicyPairedTransformer` with KITE's config
   (`conf/kite.yaml` → `per_joint` tokens, `paired_transformer_mlp`), `action_dim = 6 + 16`,
   alias and freeze the shared modules from `release.pt`.
2. An adapter from a `LeapSplit` batch to KITE's head batch:
   `contact_cloud` (B, 6, 6) + `contact_valid_mask`, `hand_points_world` / `hand_normals_world`
   (the 32 hand tokens), `q1_padded` (start pose in KITE's layout), `action` (goal),
   `contact_joint_mask`; action normalizers as KITE builds them (`contact_policy_common._build_action_normalizer_meta`:
   joint limits → [−1, 1], wrist translation ±0.4 m, rotation ±π).
3. Training loop: KITE's settings (Adam 1e-4, weight decay 1e-6, cosine schedule, 2,000 warm-up
   steps, EMA, bf16), loss = the head's own `compute_loss` (`full_mix` = L2 on the masked target,
   as KITE's released config).
4. Save in KITE's checkpoint format (`model` / `ema_model` / `manipulators`), so KITE's scripts
   can load it once step S exists.

**Checks before trusting it.**
- On KITE's own synthetic test metric, the head should be about as good on LEAP as the released
  Allegro head is on Allegro (same metric, same contact recipe).
- B1 needs the head's joint rows to line up with a lesioned hand's joints: map by joint name
  (LEAP motor IDs), not by position.
- B2 needs a fixed rule chosen before looking at test results: shift the remaining non-thumb
  fingers towards the index, keeping their order; the thumb keeps its rows. It only differs from
  B1 on hands with a whole index/middle/ring finger removed (`test_finger_subset`, part of
  `test_combined`); report it there. It is KITE's best hand-engineered zero-shot answer, so the GET
  decoder should beat B1 *and* B2. Expect it to matter most with intents made for the full hand
  (closed loop, step S); open-loop contacts come from the lesioned hand itself, where the shifted
  rows may clash with the hand tokens' geometry.

**Compute.** ~143k trainable parameters on a frozen ~16M trunk: hours on the laptop GPU
(RTX 5050, 8 GB) for 200k examples × tens of epochs. Fine-tuning curves (B3) are minutes each.

**Optional fidelity check.** Train the same head with KITE's own data generator
(`train_kinematic_decoder.py`) after adding LEAP to KITE's robot assets. That confirms our data
and adapter match KITE's pipeline, at the cost of KITE's 17 GB-per-hand cache.

## 2. The GETyourKITE decoder

**Model.** KITE's decoder with the per-hand parts replaced, so every weight is shared:
- *Kept and initialised from `release.pt`:* ψ (frozen), `cls_unpacker`, `bottleneck_scene_encoder`,
  `action_decoder`, `hand_proj`, the wrist-pose projections.
- *Joint tokens:* an MLP over the batch's `joint_features` (URDF origin, orientation, axis,
  limits, depth, leaf), `joint_pos_init` / `joint_axis_init` (expressed in the wrist frame, so the
  token does not depend on where the hand is) and the joint's start angle. This replaces
  `joint_identity_embed` + `qref_joint_proj`.
- *Graph bias:* GET-Zero's learned scalars per head and layer, indexed by `spd`, `anc` and
  `desc`, added to the attention logits of the action decoder's self-attention
  (`_SpatialAttnBlock` takes an additive mask; the wrist pose tokens are graph node 0).
- *Joint positions in cross-attention:* let joint tokens query the hand tokens with their xyz
  (today `query_has_xyz=False`).
- *Output:* one shared per-joint head (residual added to q_init, normalized per joint by its range
  from `joint_features`), and KITE's wrist head (translation + 6D rotation residual).
- *Optional:* GET-Zero's self-model loss (predict each joint's position).

**Loss.** As KITE: L2 on the normalized residual towards the masked target; optionally the
contact error of the prediction (FK is differentiable: `HandBank.fk`).

**Stages.**
1. *Sanity (½ day):* overfit the nominal hand; open-loop error should reach the LEAP head's.
2. *Train split (1–2 days):* the 1,015 training hands, ~1M examples; validation split for model
   selection. Run the scaling test early (1k / 4k / 16k examples per hand on ~50 hands) to decide
   how much data to generate.
3. *Test splits:* open-loop metrics per split, against B0, B1, B3 (and B4 on a subset).
4. *Ablations (one flag each):* no graph bias; no joint-position features; joint index
   embedding instead of URDF features (= KITE's table, cannot transfer); frozen vs. fine-tuned
   trunk; with / without self-model loss.
5. *Closed loop (after step S):* Oracle-Intent Keyboard Pressing on nominal and held-out LEAP
   variants, scored with `getkite` scores and watched with `scripts/demo_task.py`.

**Compute.** ~16M parameters, batch 1,024: development and small runs on the laptop; the full
train split and ablations on the cluster. Data loading is not the bottleneck (16k examples/s,
the whole split on the GPU).

## Order of work and owners

| Days | Work | Depends on |
|---|---|---|
| Nov 18–19 | M (metrics), LEAP-head adapter, 200k nominal-LEAP examples | – |
| Nov 19–21 | Train the LEAP head; B1 slicing, B2 shifting; GET model code | M |
| Nov 21–24 | GET sanity run, scaling test, full training; B3 curves | LEAP head for comparison |
| Nov 24–27 | Test splits, ablations, figures for the talk; start S | trained GET |
| Nov 28 – Dec 3 | Talk; S and Oracle-Intent rollouts on Keyboard Pressing | S |
| Dec 4–11 | Closed-loop results, B4 subset, report | – |

Four people can work in parallel: (1) metrics + LEAP head and baselines, (2) GET model and training,
(3) step S (simulator integration), (4) data: scaling test, extra splits, GET-Zero's hands as an extra
test set (the dataset's "What's left" #1).

## Open decisions

- **Tolerance for success** in the open-loop metric (1 cm?).
- **How much data per hand** (the scaling test decides).
- **Joint tokens in the wrist frame or the world frame.** The wrist frame makes them pose-invariant;
  KITE's hand tokens are in the world frame, which the trunk expects.
- **Fine-tune the trunk or not.** Frozen keeps Z and all of KITE's hands intact; fine-tuning may be
  needed for capacity. Run both.
