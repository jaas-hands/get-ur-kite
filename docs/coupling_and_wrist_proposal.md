# Proposal: coupled joints in the body generator, and the Shadow wrist (2026-11-21; not started)

Status (2026-11-21): Phases 0–2 done (docs/decoder_audit.md §60, §62). D1c: no effect on Shadow, small costs on Allegro corr / Wuji compat in 2/3 seeds; not promoted.

## What §56–§59 imply for the next step

| finding (3 seeds, paired) | consequence for the design |
|---|---|
| 50 generated bodies ≫ 1 body, and replicate on a second draw (§58, §59 D1b) | the lever is **variety of bodies**; a new generator feature should add a new axis of variety |
| one 5-finger body ≈ one 2-finger body; 50 non-5-finger ≈ 50 mixed (§59 C5, N5) | do **not** build Shadow look-alikes; keep the feature general and randomised |
| 800 bodies ≈ 50 (§56, §57); 4× records per body ≈ 1× (§59 R4) | keep D1's budget: **50 bodies × 400 records**; no larger runs needed for this test |
| Wuji wrist: every 50-body arm beats no movement in 3/3 seeds (−0.8 to −1.0 mm) | the wrist is not broken in general; body variety already fixed Wuji's wrist |
| Shadow wrist: no arm reliably beats no movement | the remaining gap is specific to Shadow; see below |

Wrist breakdown for D1 seed 0 (benchmark metrics; the same pattern holds in every arm and seed). Columns: translation slope / R² against identity, then rotation slope / R² against identity.

| pool | wrist mm (no movement) | trans slope / R²_id | rot slope / R²_id |
|---|---|---|---|
| source Allegro / Barrett | 5.20 / 5.12 (7.95 / 8.02) | .55 / .55 | .61 / .64 |
| source LEAP, held-out LEAP | 6.01 / 6.36 (7.75 / 7.96) | .37 / .36 | .53 / .55 |
| Wuji | 7.14 (7.96) | .30 / .17 | .47 / .30 |
| Shadow | 8.03 (7.96) | .21 / **−.03** | .47 / .38 |

On Shadow the **rotation transfers** (6.2° vs 7.8° for no movement; rot R²_id is about .38 in every 50-body arm). The **translation does not** (R²_id is about 0). The decoder moves the wrist by about as much as on the source hands (move_mm 5.1). The translation direction is wrong, not the step size. So "wrist-path work" should target translation on Shadow, and it should first find out why.

Shadow has two features no generated body has: (a) coupled distal joints (J1 = J2 on the four long fingers), and (b) a palm-embedded little-finger joint (LFJ5) plus a 5-DOF thumb. (a) is the one already named as a generator gap. (b) is out of scope here, to keep to one change at a time.

## Phase 0: wrist diagnostics (evaluation only, existing checkpoints, no training)

Run on D1 s0/s1/s2 (and G1 for reference), on the frozen pools (source val, held-out LEAP, Wuji, Shadow). The tooling extends `wrist_transfer_eval.py` / `w1_checks_eval.py`, which already have the fingers-at-goal and wrist-only machinery. Output goes to runs/wrist_diag_v1.

- **W-a. Per-axis translation error in the palm frame:** x is the palmar normal, y runs across the palm, z is distal. Also slope and R² per axis. A single dominant axis, for example z (distal), points to hand size or proportion, not coupling.
- **W-b. Stratified by where the contacts are on Shadow:** records with ≥1 contact on a coupled distal link (ff/mf/rf/lf distal) vs none; thumb-only; palm. Translation R²_id and wrist mm per stratum, with record counts.
- **W-c. Oracle split:** (1) fingers set to the goal, the decoder's wrist only; (2) wrist set to the goal, the decoder's fingers only; (3) full. This tells apart a wrong wrist that compensates for fingers that can't be realised from a wrong wrist mapping.

**Predeclared decision rule** (written before looking):
- Shadow translation error concentrated in records with coupled-distal contacts (W-b), or W-c shows the wrist compensating for fingers → coupling is the likely cause. Phase 1–2 tests it, and the Shadow wrist is a Phase 2 readout.
- Error dominated by one palm axis (W-a) and not by contact location → hand size or proportion coverage. Write a separate size-range proposal; Phase 1 still proceeds on its own merits.
- Neither → a decoder-side wrist problem. Write a W2 proposal on the decoder side; don't change the generator for it.

Cost: CPU/GPU evaluation only, about 1 GPU-h in total. Shadow stays evaluation-only, and nothing is selected on it.

## Phase 1: coupled joints in the generator (code + data, CPU only)

**Generator** (`get-ur-kite-dataset/kite_data/prim_hand.py`):
- New coupling draw from a **separate RNG stream** `default_rng([seed, index, attempt, crc32(b"couple")])`. Every other spec feature of a body is then exactly draw A's, and the only difference between the new draw ("A-c") and draw A is coupling. That gives a clean paired comparison against D1, with the same bodies and the same records for uncoupled bodies.
- Per body: coupled with p = 0.5. In a coupled body, each finger (long fingers and thumb, so it isn't Shadow-shaped) with ≥2 flexion joints has its **last flexion joint follow the previous flexion joint** with p = 0.75: `q_last = m · q_prev`, m ~ U(0.5, 1.2), offset 0. The coupled joint's limits become m × the source's limits, so they can never be violated.
- `FingerSpec` gains `mimic: list` (per joint: None or (source k, m)). The body ID and topology string add a `-cN` suffix (N = coupled joints) so topologies stay distinct.
- `ensure_finger_tokens`, the template and the MJCF geometry don't change (coupling doesn't change the geometry).

**Kinematics** (`kite_data/hand.py`, `PrimHand`):
- `HandJoint` gains `mimic=(src_dof, mult, off)`. Mimic joints are left out of `dof_joints` (like locked joints), so `n_dof` = independent coordinates.
- `fk` uses `mult·q[src]+off` for them.
- `link_dof_ancestors` maps a mimic joint to its source DOF, so KITE's `joint_active` mask stays correct.
- `descriptor()` writes `"mimic": {"joint", "multiplier", "offset"}` per joint.
- `to_mjcf()` adds `<equality><joint joint1 joint2 polycoef="off m 0 0 0"/>` for validation.

**Records:** `kite_data.sample.HandSampler` needs no change, because it samples `n_dof` independent coordinates with KITE's mixture, and collision checking and contacts go through `fk`. This needs confirming in tests, not assuming.

**Decoder** (`getkite/decoder/robots.py::leap_desc`): read `mimic` → `Joint(..., mimic=(src_name, m, off))`, and leave the joint out of `coords`. `RobotBank` already supports one-level mimics (Shadow uses them; `test_barrett_mimic_expansion`), so nothing else changes. Check `J_MAX` = 24 (unchanged: physical joints ≤ 20).

**Tests and validation** (before any data is used):
1. Generated FK vs MuJoCo with the equality constraint, on coupled bodies.
2. Decoder `RobotBank.expand` + FK vs kite_data FK on coupled bodies: max error < 1e-6 m.
3. Stored records reproduce their contacts through the decoder's FK.
4. **Regression:** every uncoupled body in A-c is byte-identical to draw A (hand.json, template.npz, records).
5. `validate_prim.py` gains a coupling check (mimic limits consistent, no mimic chains).
6. Existing kite_data and decoder tests stay green.

**Data:** A-c = 900 bodies (seed 20261119 + the coupling stream) → `$GETKITE_DATASET/prim_c`. Records use the same `record_plan.json`, and only coupled bodies need new records. Validation output: `configs/decoder/dlevels/prim_c_validation.json`. A CPU job like `prim_followup_data.sbatch`.

## Phase 2: one training arm (needs go-ahead)

- **D1c** = D1 at seeds 0/1/2, with prim P0000–P0049 from A-c (×400) instead of draw A. Configs come from `make_followups.py`-style path substitution; a diff check confirms only train / robots / out_dir / _note change. About half of the 50 bodies will be coupled; the exact count is reported before launch. Cost: 3 runs of D1 size, on the same partition, constraint and exclude list as before.
- Benchmark: transfer_v1 + Shadow, with G1 / D1 / D1c × 3 seeds, paired by seed.
- **Predeclared readout:**
  - Primary: D1c vs D1 on Shadow corr mm and compat success.
  - Secondary: Shadow wrist mm and translation R²_id.
  - Non-inferiority: Wuji, held-out LEAP and source must not be worse in 3/3 seeds.
  - Reported 3/3 counts. No seed is chosen, no examples are pooled, nothing is selected on Wuji or Shadow.
- Honest caveat for reporting: Shadow has now been the readout for several comparisons. It has never been used for training, selection or tuning, but repeated looks weaken it as a final test. If D1c is promoted, a further unseen hand would be needed for a clean claim. The other candidates (Inspire, Ability, XHand, …) only have KITE collision JSONs on disk, so that would need assets.

## Not proposed now (one change at a time)

Palm-embedded finger joints (LFJ5) and 5-DOF thumbs; wider size or proportion ranges (only if W-a points there); wrist head or loss changes (only if Phase 0 says the wrist mapping itself is at fault); larger body counts or more records (§56, §59 say no).

## Order and gates

1. Phase 0 diagnostics → report, and apply the decision rule.
2. Phase 1 code + tests + A-c data (can run alongside 1; CPU only) → validation report.
3. Phase 2 needs the user's go-ahead → D1c × 3 + benchmark → audit §60.
