# Training plan: the LEAP head and the GETyourKITE decoder

Status on Nov 19, 2026. Spotlight talk Nov 28 – Dec 3, report Dec 4–11.

## What exists

| Piece | Where | State |
|---|---|---|
| LEAP hand family: 1,741 hands, hand-level splits, 2.4M KITE-recipe examples | `third_party/get-ur-kite-dataset` (submodule) | done, regenerates in ~3 min |
| KITE's frozen intent encoder ψ | `getkite.leap.intent.IntentEncoder` | done; reproduces KITE's stored latents to 4e-5 relative error |
| Batches: z, contacts, start/goal poses, joint features, joint frames, 32 hand tokens, graph | `getkite.leap.data.LeapSplit` | done; ~16k examples/s on the laptop GPU |
| Batched forward kinematics over the family | `getkite.leap.hands.HandBank` | done; reproduces the dataset's contacts to 5e-8 m |
| Floating, actuated LEAP MJCF in KITE's format, for any hand | `assets/leap/`, `scripts/make_leap_mjcf.py` | done; matches the FK to 1e-6 |
| KITE's decoder trunk (scene encoder, action decoder), 4 other hands' heads | `third_party/KITE/checkpoints/decoder/release.pt` | released |
| LEAP in KITE's robot model, data generator and trainer | `scripts/install_kite_leap.py`, `scripts/train_leap_head.py` | done (pilot); full B0 run next |
| LEAP in KITE's simulator, tasks, start poses | – | missing (section 1, step 8) |
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

## 1. The LEAP head and the KITE baselines (owner: Simon)

KITE's release has no LEAP decoder. We train one with KITE's own training script, on KITE's
released trunk, and derive every baseline of the evaluation matrix from it:

| Baseline | What | Training |
|---|---|---|
| KITE / B0 | the LEAP head on the nominal hand; on a jammed joint, run unchanged with that joint clamped | once, cluster |
| B1 sliced | drop the rows of missing or locked joints (matched by joint name) | none |
| B2 rows reassigned | a missing finger's neighbours take over its rows (middle missing: ring ← middle; thumb keeps its own) | none |
| B3 head fine-tuned | B0's per-hand parameters fine-tuned on the test hand; error against minutes / examples | minutes per hand, laptop |
| B4 head retrained | B0's recipe from scratch on the test hand, a few hands only (upper bound) | one cluster job per hand |

### Go / no-go gates

Each gate either passes or tells us to switch to the Allegro hand. KITE ships Allegro's head, so the
Allegro fallback skips gates G1–G4 and starts at the baselines.

| Gate | Question | Status | If it fails |
|---|---|---|---|
| G1 | Does KITE load LEAP (URDF, surface template, 22 DoF)? | **passed** Nov 19 | – |
| G2 | Do KITE's data generator and training script run on LEAP, with the trunk frozen? | **passed** Nov 19 (pilot) | – |
| G3 | Is the LEAP head as good on LEAP as KITE's Allegro head is on Allegro (KITE's own test metrics)? | pilots: frozen trunk no (1.4× vs 8×), trunk trained promising (3.7×, still improving); cluster run Nov 20–21 | if the full trunk-trained run stays far off by Nov 22: Allegro |
| G4 | Does the head work on the team dataset's nominal hand as on KITE's own test data? | **failed** Nov 24 (5.2× vs 1.4×); B0-ds trained on dataset data instead | train B0 on the dataset's nominal-hand examples instead; no switch needed |
| G5 | Does LEAP grasp/press in KITE's simulator with Oracle intents? | Nov 25 | closed loop on Allegro; the open-loop matrix stays on LEAP |

**Decision date for the team: Nov 22** (G3, G4). G5 only affects the closed-loop part.

### What the pilot showed (Nov 19)

- **KITE already knows LEAP.** Its robot registry has `leaphand_right` (`dro/leaphand/leap_hand_right`)
  and ships LEAP's fitted collision primitives; only the URDF and meshes are missing from the release.
  They are DRO-Grasp's (MIT), the same source as KITE's Barrett, and `scripts/install_kite_leap.py`
  installs them (40 MB read from DRO-Grasp's 1 GB archive).
- **Same robot as the team dataset.** DRO's LEAP and the dataset's nominal hand have the same joints
  (names `0`–`15`, LEAP motor IDs), the same order, parents, axes and limits; origins agree to 0.1 mm
  (DRO rounds). Two differences: the thumb's base link is `thumb_temp_base` (dataset: `pip_4`), and
  DRO mounts the palm under the virtual wrist joints with a fixed offset
  (`xyz 0 0.038 0.098`, `rpy 0 −π/2 0`), so wrist poses convert with that transform.
- **Surface template:** built once (5 min). KITE's builder needs more than 14 GB of RAM for LEAP's
  meshes; our scripts run its closest-point query in chunks (identical result, 9.7 GB peak).
- **Training:** `scripts/train_leap_head.py` runs KITE's `train_kinematic_decoder.py` unchanged, except
  that the modules KITE shares across hands (ψ, `hand_proj`, `cls_unpacker`, `bottleneck_scene_encoder`,
  `action_decoder`, ...) are loaded from `release.pt` and frozen. Trainable: 141,834 parameters (joint
  table, joint and pose projections and heads, action queries, action-token norm), the same as each
  released head. After training, all 331 shared tensors are bit-identical to the release; the checkpoint
  has KITE's format, so KITE's visualisers and rollout scripts load it.
- **Pilot numbers** (20k examples, 3 epochs, laptop, 2 min 47 s in total): train loss 21.2 → 0.74,
  KITE's test action MSE 0.41 → 0.081. Not converged; it shows the pipeline works.
- **Laptop pilots, 100k examples, 12 epochs** (Nov 19). Metric: KITE's `contact_point_mse` on its own
  test set (1,000 examples; mean squared contact position error per coordinate), with the start pose as
  the "no movement" floor. KITE's `action_mse_valid` is not usable as a yardstick: it does not wrap the
  wrist's Euler angles, and even the released Allegro head scores worse than no movement on it.

  | Run | contact_point_mse | vs. no movement | Dataset nominal hand: contact / success |
  |---|---|---|---|
  | Allegro, released head (reference) | 8.0e-6 (RMS 4.9 mm) | 8× better than 6.3e-5 | – |
  | LEAP, trunk frozen, random init | 9.4e-5 | 1.1× | – |
  | LEAP, trunk frozen, Allegro init | 7.1e-5 | 1.4× | 14.7 mm / 6 % (no movement 14.5 mm / 8 %) |
  | LEAP, **trunk trained, ψ frozen**, Allegro init | **2.7e-5** (RMS 9.0 mm) | **3.7×** (floor 1.01e-4) | 12.2 mm / 17 % |

  With the trunk frozen the LEAP head barely learns (train loss flat from epoch 4). Training the trunk
  too, with ψ frozen so z stays KITE's, works and was still improving at the end. **So B0 trains the
  trunk** (`GETKITE_B0_UNFREEZE=trunk`), like Anushka's decoder (trunk fine-tuned at 0.1× lr); this
  changes the trunk for KITE's other hands, which does not matter for a LEAP-only baseline. The
  frozen-trunk head stays as an ablation. For comparison, the GET decoder reaches ~7.6 mm / 48 %.
- **KITE's contact regions were wrong for LEAP** (fixed in our wrapper). KITE labels regions by matching
  Shadow/Wuji-style names; for LEAP only the thumb matched, so 51 % of contacts fell on the thumb and
  index, middle and ring got ~26 % together (Allegro: everything is "palm"). `scripts/train_leap_head.py`
  now gives LEAP five regions, sampled uniformly (20 % each), the team dataset's recipe. The pilots above
  ran before this fix.
- **Hand tokens:** on the dataset's nominal hand the head needs KITE's own template points (12.2 mm);
  the dataset's points make it worse than no movement (15.4 mm). For B1/B2 the tokens of a variant hand
  should be KITE's template moved by the variant's FK (its links are kept), not the dataset's template.
- **Laptop runs, Nov 24** (cluster down; 300k examples, 60 epochs, corrected regions):
  - `b0-trunk` (trunk trained, ψ frozen, Allegro init): KITE's LEAP test set `contact_point_mse`
    2.25e-5 against a no-movement floor of 1.17e-4, **5.2× better** (Allegro's released head: 8× on its
    own test set). Flat over the last 15 epochs; 2.7 h, 9.1 GB RAM.
  - **G4 fails:** the same head on the team dataset's nominal hand is only **1.4×** better than no movement
    (11.4 mm / 25 % success against 14.5 mm / 8 %). The adapter is not the cause: run on KITE's own test
    examples, our inference path gives 2.35e-5 (trainer: 2.25e-5). So KITE's and the dataset's LEAP data
    differ in ways that matter (contacts on DRO's visual meshes vs. Menagerie's collision geometry, ~3 mm;
    KITE's contact/start-pose sampler vs. the dataset's).
  - Per the plan, **B0-ds**: the head continued on 300k nominal-hand examples generated with the dataset's
    own code (`scripts/train_kite_head_dataset.py`, KITE's loss and normal noise). B0-ds is the B0 for the
    matrix; B1, B2 and B3 start from it, so all baselines and GET use the same data recipe.
  - New tools: `getkite.kite_baselines` (clamped, B1, B2; KITE-style hand tokens for any variant from
    KITE's template + the dataset's link deformations, identical to KITE's for the nominal hand),
    `scripts/eval_kite_baselines.py` (matrix over the test splits), `scripts/finetune_kite_head.py` (B3),
    `scripts/kite_test_floor.py`.
- **B0-ds result (Nov 24).** Supervising only KITE's masked joints let the other joints drift (0.016 rad):
  loss fell while the contact error rose (`b0-ds-masked`). Supervising all joints (the label keeps the
  others at their start value) fixed it. 10 epochs on 300k dataset examples of the nominal hand, 27 min:

  | Head, on 1,000 of the dataset's nominal-hand examples | Contact | Success | Normal |
  |---|---|---|---|
  | no movement | 14.47 mm | 7.6 % | 9.1° |
  | b0-frozen (KITE's data, trunk frozen) | 14.39 mm | 5.8 % | 8.8° |
  | b0-trunk (KITE's data, trunk trained) | 11.44 mm | 25.3 % | 5.9° |
  | b0-ds-masked (dataset data, KITE's mask) | 10.07 mm | 25.2 % | 5.9° |
  | **B0-ds** (dataset data, all joints) | **7.36 mm** | **49.3 %** | **4.9°** |

  B0-ds is the B0 of the matrix: on the nominal hand as good as the GET decoder on the test splits.
- **Zero-shot baselines from B0-ds** (`b0-ds.baselines.json`, all hands, 200 examples each; contact / success):

  | Split | No movement | Clamped | B1 sliced | B2 reassigned | GET (Anushka, docs/decoder.md) |
  |---|---|---|---|---|---|
  | test_finger_subset (36) | 14.74 / 7.3 % | n/a | 13.50 / 11.8 % | 13.63 / 11.1 % | 7.54 / 49.7 % |
  | test_joint_count (378) | 14.29 / 7.6 % | 12.84 / 16.1 % (189 hands) | 13.52 / 13.5 % | 13.51 / 13.5 % | 7.48 / 50.6 % |
  | test_geometry (40) | 14.22 / 7.6 % | 13.30 / 12.6 % (12 hands) | 13.82 / 11.6 % | 13.99 / 10.9 % | 7.53 / 50.0 % |
  | test_combined (216) | 14.74 / 6.9 % | n/a | 13.56 / 13.1 % | 13.85 / 11.5 % | 7.67 / 48.4 % |

  KITE's head does not transfer: 7.4 mm / 49 % on the nominal hand, 13–14 mm / 11–16 % on every variant.
  Geometry alone breaks it: the full hand with link scales 0.8–1.2 (g1–g6) gets 11.9–16.0 mm against
  7.5 mm at g0. B2 is no better than B1. The GET numbers come from Anushka's runs (same splits, her
  evaluation, not yet run through `eval_kite_baselines.py`): the comparison is preliminary until both are
  scored by the same script on the same examples.
- **B3** (fine-tuning per test hand, `b3/`, lr 1e-4): the best point comes after ~15 s (e.g. 13.9 → 11.4 mm
  on the middle-finger-removed hand), then it overfits (18–25 mm after 5 min). Best values 9.8–12.7 mm with
  1,000 examples, all far from GET's zero-shot ~7.5 mm. With lr 1e-5 (`b3_lr1e-5/`) the curves are stable,
  best after ~2 min, then slightly worse at 5 min; the best values are the same (9.75–12.6 mm, e.g.
  13.9 → 11.4 mm on the middle-finger-removed hand). So head-only fine-tuning with 1,000 examples of a
  test hand stays ~4 mm behind the GET decoder's zero-shot result.
  Picking the best point on the held-out examples flatters B3; a fair version needs a separate
  validation set.
- **Cluster results (Nov 26, `slurm/submit_kite_baselines.sh`; copied to `outputs/leap_head/cluster/`).**
  - **Template caveat.** KITE's LEAP surface template is seeded, but its random draws differ between
    machines (laptop md5 `f3bf89f7…`, cluster `6bc79cf1…`), and a head only works with the template it was
    trained with: the cluster's B0-ds scores 6.9 mm with the cluster's template, 11.0 mm with the
    laptop's; the laptop's b0-trunk 11.2 mm with its own, 18.4 mm with the cluster's. All numbers below
    use the cluster's template (kept in `outputs/leap_head/cluster/template/`; select it with
    `GETKITE_KITE_TEMPLATE`). The GET decoder does not use KITE's template.
  - **B0-ds, 1M examples, 20 epochs** (3.2 h on an A6000; started from the laptop's b0-trunk, i.e. from
    18.4 mm because of the template): **6.94 mm / 55.7 %** on the nominal hand (laptop, 300k: 7.36 / 49.3 %).
  - **Zero-shot baselines** (all 2,000 examples per hand; contact / success):

    | Split | No movement | Clamped | B1 sliced | B2 reassigned | GET (Anushka) |
    |---|---|---|---|---|---|
    | test_finger_subset (36) | 14.74 / 7.0 % | n/a | 17.11 / 4.4 % | 17.17 / 4.1 % | 7.54 / 49.7 % |
    | test_joint_count (378) | 14.27 / 7.8 % | 9.97 / 27.1 % (189 hands) | 11.24 / 22.3 % | 11.25 / 22.2 % | 7.48 / 50.6 % |
    | test_geometry (40) | 14.17 / 8.1 % | 11.00 / 21.0 % (12 hands) | 11.34 / 21.8 % | 11.84 / 19.8 % | 7.53 / 50.0 % |
    | test_combined (216) | 14.72 / 7.1 % | n/a | 17.11 / 5.3 % | 16.97 / 5.2 % | 7.67 / 48.4 % |

    The stronger B0-ds is better on variants that keep all fingers (11.3 mm against 13.5 on the laptop)
    and worse than no movement when a finger is removed. B2 is again no better than B1.
  - **B3** (fair protocol: training examples generated for the hand, checkpoint chosen on 500 separate
    validation examples, scored on the hand's 2,000 official test examples; lr 1e-5, 40 min; 3 hands per
    split). Test error at the selected checkpoint, range over the split's hands:

    | Training examples | finger_subset | joint_count | geometry | combined |
    |---|---|---|---|---|
    | 0 | 16.7–17.2 mm | 9.5–15.7 | 10.4–13.3 | 16.5–19.1 |
    | 1k | 13.6–14.0 | 9.0–13.1 | 9.4–11.3 | 13.6–14.4 |
    | 4k | 13.0–13.3 | 8.9–12.6 | 8.9–10.9 | 13.1–13.5 |
    | 16k | 12.6–12.8 | 8.5–12.0 | 8.5–10.4 | 12.5–13.0 |

    The validation-selected point is almost always the test optimum. With 1k examples the head overfits
    (22–27 mm after 40 min); 16k examples are stable. B3 stays 1–5 mm behind GET's zero-shot result.
  - **B4** (B0-ds's B1 head retrained on 200k examples of the hand, trunk too, 20 epochs, 30–40 min;
    official test examples): finger_subset I4-M0-R4-T4_g0 7.41 mm / 49.6 %, joint_count I3-M0-R0-T2_g0
    6.56 / 65.5 %, geometry I0-M2-R0-T2_g7 6.44 / 66.4 %, combined I0-M0-R2-T2_g1 6.73 / 64.0 %. With
    200k examples of its own hand, KITE's head matches or beats GET's zero-shot result.
  - Still open: score the GET decoder with `scripts/eval_kite_baselines.py`'s metrics on the same examples.
- **Scale:** KITE's budget is 1M examples per hand; its buffer is ~18 GB in RAM (369 MB per 20k), so the
  full run is a cluster job. Generation runs at ~380 examples/s (1M ≈ 45 min).

### Steps

1. **Reference numbers for G3** (*done Nov 19*, table above; laptop, ~1 h). Score KITE's released Allegro head on KITE's own
   Allegro test set (1,000 examples, same generator) with the metrics the training script logs
   (`test/<hand>/*`: action MSE on the masked joints, contact errors). That is the bar the LEAP head
   should reach on LEAP.
2. **Full B0 run** (cluster). KITE's settings unchanged: 1M train / 1,000 test examples, batch 128,
   AdamW 1e-4, cosine schedule with 2,000 warm-up steps, EMA, bf16, compiled heads, `full_mix` loss.
   The trunk was trained for 550 epochs; a head on a frozen trunk needs far fewer. Plan for up to
   100 epochs, keep the best test checkpoint, stop when the test metric is flat for 10 epochs.
   Three jobs, in parallel (commands in `slurm/leap_head.sbatch`), all starting the per-hand weights
   from Allegro's (same shapes; faster in the pilots) and with the corrected LEAP regions:
   - `b0-trunk`: trunk trained, ψ frozen; **the B0 we use**;
   - `b0-frozen`: trunk frozen (ablation; how much of KITE's per-hand claim holds for a new hand);
   - `b0-trunk-200k`: 200k examples, to see whether KITE's 1M budget is needed (sets B3/B4's budget).
3. **Adapter to the team dataset** (*done Nov 19*: `getkite.kite_head.KiteHead`, `scripts/eval_kite_head.py`,
   `tests/test_kite_head.py`): loads a KITE checkpoint with KITE's own loader and runs one head on
   `LeapSplit` batches. Joints map 1:1 by name. Wrist poses go through DRO's palm offset and KITE's Euler
   convention (intrinsic x-y-z); checked against KITE's robot model over all 17 links (0.11 mm, 0.06°).
   ψ is the same frozen encoder on both sides. **Open point: hand points and contact surfaces.** The
   two LEAP templates differ: KITE's comes from DRO's visual meshes, the dataset's from Menagerie's
   collision geometry; per link they are 2.5–3 mm apart on average and 7–10 mm at worst (palm 22 mm).
   This matters twice: KITE's head sees *its* 32 hand points, and dataset contacts lie on the
   collision surface, ~3 mm from the surface the head was trained on, against a 10 mm success
   tolerance. `--dataset-tokens` feeds the dataset's hand points instead; G4 measures both.
4. **G4** (laptop): open-loop contact error and success (10 mm, 15°, as `docs/decoder.md`) of B0 on the
   dataset's nominal-hand validation examples, against B0 on KITE's own test set. A large gap means the
   two contact recipes differ; then also train B0 on the dataset's nominal-hand examples (same script,
   data from the adapter) and use whichever matches the GET decoder's training data.
5. **B1 and B2** (laptop, no training): in KITE's per-joint mode the only joint-specific weights are the
   16 rows of `joint_identity_embed` (16 × 256) and the per-dimension normalizers; `qref_joint_proj`
   (1 → 256) and `head_joints` (256 → 1) are shared by all joints of a hand. So B1 builds a head with
   `action_dim = 6 + k` for a hand with `k` movable joints and copies those rows by joint name; B2
   permutes the rows of whole fingers first. "KITE clamped" keeps all 16 tokens and holds the locked
   joints at their value. Run on every matrix row through the adapter.
6. **B3** (laptop): fine-tune B0's per-hand parameters on each test hand with the dataset's examples of
   that hand (1k / 4k / 16k examples); log error against wall-clock minutes. A few hands per scenario.
7. **B4** (cluster): step 2's recipe from scratch on 4 test hands (one per scenario 1–4), 200k examples
   each (or what `B0-200k` says is enough).
8. **LEAP in KITE's simulator (G5, for closed loop)**:
   - Simulation model in KITE's frames (*done Nov 19*): `python scripts/make_leap_mjcf.py I4-M4-R4-T4_g0
     --kite-frame` writes the floating, actuated MJCF with the palm below the virtual chain by DRO's
     offset, so its qpos is KITE's LEAP configuration; `tests/test_kite_head.py` checks all LEAP bodies
     against KITE's robot model. Collision geometry is Menagerie's (MuJoCo-tuned), as in the dataset.
   - KITE's scene code has Allegro-only contact settings (`env.allegro_*`: fingertip pads, friction,
     solref; `experiment/tabletop_sim/factory.py`); LEAP needs its own equivalents, starting from Allegro's.
   - Register an `EmbodimentSpec` for `leaphand_right` from `getkite` at import time (as Allegro's in
     `experiment/tabletop_sim/robot.py`); KITE stays unmodified.
   - Start poses for Keyboard Pressing: LEAP's palm where Allegro's is.
   - Run Oracle-Intent Keyboard Pressing with B0, scored and watched with `scripts/demo_task.py`.
     Pass: LEAP presses chords at a rate comparable to Allegro's 15/20.

### Cluster chain (Nov 24)

`bash slurm/submit_kite_baselines.sh` submits the remaining baseline work as dependent jobs
(`slurm/kite_baselines.sbatch`, partition `fang`); results go to `$GETKITE_RUNS/leap_head/cluster/`:
- `gen` (CPU): with the dataset's code, 1M nominal-hand examples, 16k + 500 val for each of 12 B3 hands
  (3 per test split), 200k + 500 val for each of 4 B4 hands (`scripts/gen_kite_head_data.py`).
- `b0ds`: B0-ds at 1M examples, 20 epochs, from the laptop's b0-trunk (`outputs/leap_head/local/b0-trunk-init`,
  copied to `$GETKITE_RUNS/leap_head/init/b0-trunk/`), then G4.
- `matrix`: clamped / B1 / B2 on every test hand with all 2,000 examples per hand.
- `b3`: the fair B3: 1k / 4k / 16k training examples of the hand, checkpoint chosen on separate
  validation examples, scored on the hand's official test examples; lr 1e-5, up to 40 min.
- `b4`: B0-ds's B1 head retrained on 200k examples of the hand (trunk too, 20 epochs), the upper bound.
- optional `--with-kite-1m`: b0-trunk on KITE's own data at KITE's 1M budget (G3 at full scale).

### Cluster request

| Job | GPU | CPU / RAM | Time | Disk |
|---|---|---|---|---|
| B0, B0-allegro-init (each) | 1 × A100 / L40S / A6000 class (≥ 24 GB) | 8 cores, 64 GB | ≤ 24 h | 30 GB |
| B0-200k, B4 (each) | 1 GPU, same class | 8 cores, 32 GB | ≤ 8 h | 10 GB |

Step time on the cluster is not measured yet; the laptop runs ~5 steps/s uncompiled (7,800 steps per
epoch at 1M examples). The first hour of the B0 job gives the real number; adjust the epoch cap then.

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

Owners since the Nov 19 meeting: **Anushka** the GET decoder (architecture, training), **Aryan** the
training set (length variations and more), **Simon** the baselines (LEAP in KITE, B0–B4).

| Days | Baselines (Simon) | Depends on |
|---|---|---|
| Nov 19 | LEAP in KITE's robot model and trainer, pilot (G1, G2 passed) | – |
| Nov 20–21 | Allegro reference numbers; full B0 runs on the cluster; adapter to the dataset | cluster access |
| Nov 22 | **G3, G4: LEAP or Allegro, team decision** | B0 runs |
| Nov 22–24 | B1, B2 on all matrix rows; B3 curves; start the simulator integration | B0, adapter |
| Nov 25 | **G5: closed loop on LEAP or Allegro** | simulator integration |
| Nov 25–27 | B4 jobs; matrix and figures for the talk, with Anushka's GET numbers | – |
| Nov 28 – Dec 3 | Talk; Oracle-Intent Keyboard Pressing on LEAP variants | G5 |
| Dec 4–11 | Closed-loop results, report | – |

## Open decisions

- **Tolerance for success** in the open-loop metric: 10 mm and 15° (as `docs/decoder.md`) unless the team changes it.
- **How much data per hand** (the scaling test decides).
- **Joint tokens in the wrist frame or the world frame.** The wrist frame makes them pose-invariant;
  KITE's hand tokens are in the world frame, which the trunk expects.
- **Fine-tune the trunk or not.** Frozen keeps Z and all of KITE's hands intact; fine-tuning may be
  needed for capacity. Run both.
