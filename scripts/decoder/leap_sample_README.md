# LEAP-family data sample (`leap_sample_v1`)

This is a small slice of the data our decoder trains and evaluates on (`getkite.decoder`). It contains only
the files and fields our code reads.
- Every file is unmodified from get-ur-kite-dataset's `data/` on our `kite-families` branch, the data our
  runs used. Example files are only cut to their first 32 rows.
- Keys and dtypes match the full files, so our loaders run on this folder unchanged.
- Provenance and checksums of the full files are in `splits.sample.json` → `sample`.

## Contents

```
hands/<hand_id>/hand.json      hand descriptor (joints in canonical order, limits, axes, links, regions)
hands/<hand_id>/template.npz   4,096 surface points: link, pos, normal (link frame), region, fps, names
examples/kite_code/<split>/<hand_id>.npz   32 examples per hand (full files: 1,000 train/val, 2,000 test)
examples/kite_code/summary.json            generator settings of the full set
splits.sample.json   same schema as splits.json, restricted to the hands here; plus full counts
splits.full.json     the full split manifest our runs used (1,741 hands)
multifamily_pilot/   32-row slices of the Allegro / Barrett validation records our pilot uses, and its manifest
```

| Split | Hand | DoF |
| --- | --- | --- |
| train | I0-M2-R0-T2_g0 | 4 |
| train | I4-M4-R4-T4_g0 (full LEAP layout) | 16 |
| val | I0-M2-R0-T4_g0 | 6 |
| test_finger_subset | I0-M0-R2-T2_g0 | 4 |
| test_combined | I0-M0-R2-T2_g1 | 4 |
| test_joint_count | I3-M0-R0-T2_g0 | 5 |
| test_geometry | I0-M2-R2-T0_g7 | 4 |

Hand id `I{a}-M{b}-R{c}-T{d}_g{n}`: active joints in the index, middle, ring and thumb (0 = finger absent),
geometry draw n. Our splits have 15 validation layouts (966 train / 105 val hands). The dataset repo's main
branch README still lists the earlier 8-layout split.

## Example fields

All positions are in metres in the world frame.

| Key | Shape | Contents |
| --- | --- | --- |
| `k` | (n,) | number of contacts, 2–6 |
| `template_idx` | (n, 6) | template points making the contacts; −1 in unused slots (= invalid contact) |
| `contact_pos`, `contact_normal` | (n, 6, 3) | contact targets |
| `joints_contact`, `joints_init`, `joints_goal` | (n, n_dof) | finger joints (rad), hand's canonical order |
| `wrist_pos_*`, `wrist_quat_*` | (n, 3), (n, 4) | wrist pose for contact / init / goal; quaternion w, x, y, z |
| `mask` | (n, n_dof) | joints that move (upstream of a contact) |
| `wide`, `hand_id`, `split`, `regime`, `seed`, `kite_data` | | flags / provenance |

The task: given the contacts, a start pose (`*_init`) and the hand, predict the goal pose (`*_goal`).
- Joints: `joints_goal = mask ? joints_contact : joints_init`.
- Wrist: the goal wrist is the contact wrist.

## How our code uses it

- **LEAP experiment, evaluation:** contacts, the stored start pose (`*_init`) and the stored goal. The
  final LEAP numbers are on the `val` / `test_*` splits, plus `train@holdout` (the last 10% of each training
  hand's rows; this sample's rows 0–31 are in the training part).
- **LEAP experiment, training:** the stored start pose is not used. Every batch draws a new one from the
  contact pose with KITE's noise (joints + N(0, (5% range)²); wrist + 5 mm and 5° per Euler angle), and the
  goal is recomputed with `mask`.
- **Multi-family pilot:** reads `joints_contact`, `wrist_*_contact`, `mask` and the contacts. It draws the
  start pose itself, with a fixed seed for evaluation. The Allegro/Barrett records hold the same
  information in KITE's format.

## Metrics

To score a predicted pose:
1. Clamp the joints to their limits.
2. Run FK on the hand.
3. Move each contact's template point (`template_idx`) with its link.
4. Compare the result with the targets.

| Metric | Definition |
| --- | --- |
| contact error (mm) | mean over valid contacts of ‖p − contact_pos‖ |
| success | every valid contact within 10 mm and its normal within 15° |
| wrist error | translation (mm), geodesic rotation (deg) vs the goal wrist |
| no-movement reference | the same metrics with the start pose as the prediction |

On this sample, through our evaluation code: the goal pose scores 0.00 mm / 100% success; no movement scores
13.3–16.5 mm and 0–9% success per split, with wrist error ~8 mm / ~8° (full validation split: 14.76 mm /
6.8%). These are 32-row slices, so treat them as a smoke test, not reference numbers.

## Loading

```python
from kite_data.dataset import load_split            # pip install -e get-ur-kite-dataset
d = load_split("val", root="leap_sample_v1/examples/kite_code")   # joints padded to 16; d["dof_valid"]
hand_id = d["hand_ids"][d["hand"][i]]                 # -> hands/<hand_id>/hand.json, template.npz

# or through our decoder (get-ur-kite, main):
#   GETKITE_DATASET=leap_sample_v1 and getkite.decoder.data.SplitData(cfg, "val")
```

## Allegro / Barrett records (`multifamily_pilot/`)

Their robot models are KITE's released assets (`assets/ohra_urdf_robot/allegro/allegro_hand_right_extended.urdf`,
`assets/dro_urdf_robot/barrett/model_extended.urdf`, plus KITE's templates). They are not copied here.

| Field | Contents |
| --- | --- |
| `q2_t`, `q2_euler` | goal base pose (intrinsic XYZ Euler) |
| `q2_joints` | contact joints, in the order given in `meta` |
| `joint_active` | KITE's mask |
| `c_pos`, `c_normal`, `c_valid`, `c_tpl` | contacts |
