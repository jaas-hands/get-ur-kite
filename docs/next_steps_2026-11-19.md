# Shadow inventory, Allegro/Barrett prototype estimate, alignment proposal (2026-11-19)

## 1. Shadow Hand E3M5 (reserved external final target; not integrated)

**Source.**
- MuJoCo Menagerie `shadow_hand/`, commit 370c439 (2026-11-10). Sparse clone at
  `~/code/uni-kite/external_assets/mujoco_menagerie/shadow_hand`.
- Files: `right_hand.xml`, `left_hand.xml`, `keyframes.xml`, scenes, 14 OBJ meshes.
- **License: Apache-2.0**, asset provided by Shadow Robot Company. Redistribution and modification are allowed with the
  license and notices kept.
- ψ never saw it. KITE's ψ was trained on wujihand_right, allegro_right, barrett, robotiq_gripper_stick and mano_right.
- Not used for training, selection or tuning. Nothing has been loaded into the decoder.

**Kinematics** (right hand; compiled with mujoco 4.0.1):
- 26 bodies, 24 hinge joints, 20 position actuators, 4 fixed tendons, no equality constraints.
- Wrist: WRJ2 and WRJ1, between forearm and palm.
- First finger, middle finger, ring finger (FF/MF/RF): J4 abduction ±0.35, J3 −0.26..1.57, J2 0..1.57, J1 0..1.57.
- Little finger (LF): an extra J5 metacarpal joint (0..0.785), then J4..J1.
- Thumb (TH): J5 ±1.05, J4 0..1.22, J3 ±0.21, J2 ±0.70, J1 −0.26..1.57.
- **Coupling:**
  - Tendons `FFJ0/MFJ0/RFJ0/LFJ0` = J1 + J2 (coefficients 1, 1).
  - Actuators `A_*J0` drive that tendon length over 0..π, so J1 and J2 are underactuated.
  - The split between them is decided by dynamics, not a fixed ratio.
  - All other joints are driven directly.
  - 22 hand joints, 18 independently actuated hand coordinates (+2 wrist).

**Adapter decisions to make** (no integration yet):

| item | proposal | why |
|---|---|---|
| floating base | the palm (forearm + WRJ2/WRJ1 removed, wrist fixed at 0) | matches KITE's own Shadow handling (`robot_model.py`: `probe_palm_root`, forearm removed, WRJ* fixed at zero) and our other hands (wrist = palm frame) |
| coupled J1/J2 | J1 = J2 mimic (multiplier 1, offset 0); coordinate = the J0 actuator / 2 | the decoder supports mimic (`mimic_src/mult/off`); gives 18 coordinates, matching the 18 hand actuators. The alternative, J1 and J2 independent (22 coordinates), over-states the actuation. |
| coordinate count | 18 coordinates, 22 physical joints (≤ J_MAX 24) | fits the padding |
| coverage | 5 fingers, 18 coordinates, little-finger metacarpal | outside every source body (max 4 fingers, 16 DOF): an intentional extrapolation, like Wuji |
| description format | write a KITE-style `*_extended` URDF (6 `virtual_joint_*` + palm subtree, meshes) from the MJCF | lets KITE's own tooling build the 4096-point surface template (`robot_model_process.py`), the collision primitives (`generate_collision_primitives.py`) and KITE-recipe contacts (`gen_kite_family.py` via `ContactPolicyProbeDataset`), the same recipe as Allegro/Barrett |
| goals | 500 eval-only goals, KITE recipe, 2–6 contacts; plus a target example bank for cross-body starts | as the other pools |

**Validation needed before use:**
- URDF FK vs the native MJCF (MuJoCo) at random configurations, all link frames < 1e-6.
- Mimic expansion vs the tendon constraint.
- Limits.
- Surface coverage (every finger, covering radius < 45 mm).
- Contact reproduction from the records.

**Estimate.** About 1–1.5 days: URDF writer and FK check ½ d; KITE template, primitives and goal generation ½ d; decoder
adapter, mimic and tests ½ d. CPU only.

## 2. Allegro / Barrett: minimal prototype (one validated variant each); estimate only

KITE builds a new robot's surface template and collision primitives from a URDF with meshes (cached by robot-path hash).
So a variant needs only a valid edited URDF; no template code of our own.

| step | Allegro variant | Barrett variant | time |
|---|---|---|---|
| URDF writer | one finger scaled uniformly (joint origins along the chain and `<mesh scale>`, e.g. index × 1.1); index mount shifted 3 mm along the palm; distal joint of ring locked (→ fixed) | one finger's link lengths × 1.1 (origins + mesh scale); spread joint kept; mimic kept | ½ d (shared code) |
| KITE assets | template + primitives via KITE tools; register the variant path for `gen_kite_family.py` | same | ½ d |
| records | 2,000 train + 500 val each (seconds) | same | – |
| validation | URDF vs MuJoCo FK, limits, coverage, contact reproduction, no self-collision at zero pose; reuse `scripts/decoder/g1/validate_hands.py` checks | + mimic expansion | ½ d |
| decoder smoke | the bank loads both; J1 checks on a 64-example batch; no training run | same | ¼ d |

- **Total:** about 1.5–2 days for one validated variant of each.
- **Scaling to the full ~25-variant campaign:** about ½–1 day more (sampling rules, batch template builds, manifest).
- **Risks:**
  - Scaled meshes colliding at the zero pose (fix: smaller ranges).
  - KITE's registry may assume fixed robot keys; a variant needs its own key or a path override.
  - Barrett mimic multipliers must be kept.
- Not started, as requested.

## 3. Proposed bounded intervention: intent–geometry alignment (not launched)

**Evidence.**
- Track D (audit §44): with a frozen trunk that encodes world contacts to 3.7 mm and no competing loss, the motion network
  still ignores the intent tokens.
- z-permutation changes the wrist residual by 0.0006 against a near-constant 0.02 output.
- This matches §21/§36: abstract tokens derived from z in the world frame are not turned into wrist-frame corrections.
- J1 succeeds because its intent tokens are positions, moved into the start-wrist frame by an exact rigid transform, in
  the same frame and positional encoding as the robot's joint/surface tokens.

**Intervention: unsupervised geometric intent anchors.** Change one thing relative to Track D:
- The frozen trunk features h feed a fresh head that outputs M = 6 latent anchors, each a world-frame point a_m plus a
  feature vector f_m.
- The anchors are moved into the start-wrist frame by the exact rigid transform:
  a_m' = R0wᵀ (a_m − t0w), using only the deployment-time start pose.
- They are embedded like J1's contact tokens: position scale, Fourier features of the *anchor position*, and f_m.
- These tokens replace the abstract intent tokens.

**What stays fixed:**
- No contact loss and no supervision of the anchors. No existence or masks.
- Motion loss only.
- Everything else as Track D: frozen trunk, data, schedule, 10k steps, about 15 GPU-min.
- Fourier features apply only to the transformed anchor positions, never to z. z is not treated as coordinates.

**What it tests.** Whether giving intent a frame-aligned geometric slot (structure, not contact supervision) is enough for
the motion network to learn. It differs from J1: no contact labels or set matching, and the anchors are free latents.

**Predeclared gates** (at 10k, source val; J1 at 10k = .283 success / 9.61 mm; Track D / v1 = .128 / 13.17 mm; no movement
.123 / 13.31):
- **Learns:** val success ≥ .20 *and* training-goal success ≥ .20, with wrist error clearly below no movement (≤ 7.4 mm).
- **Fails:** within ±.02 of no movement. That would point at the GET's use of intent tokens more broadly, not their frame.
- **In between:** reported as partial. No follow-up without approval.

**Not included:** no loss-weight change, no new data, no wide starts, no end-to-end variant.
