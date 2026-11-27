# W2 proposal: is the wrist-translation failure a wrist-frame convention effect? (2026-11-21)

Status (2026-11-21): W2-a done (frame-dependent, §61a). W2-b done (§61c): frame dependence removed, Shadow wrist / corr / compat better 3/3, but source and held-out LEAP worse 3/3 (non-inferiority fails). Not promoted.

## Why (from §60a)

- The decoder's fingers transfer to Wuji and Shadow (goal wrist → 2.4–2.9 mm). Nearly all remaining error is the wrist,
  on every pool.
- On Shadow, rotation transfers, and translation along the palm normal transfers (R²_id ≈ +.18). Translation across the palm
  and along the fingers is worse than not moving (negative in 12/12 and 10/12 of the 50-body runs). This is not tied to
  coupled-joint contacts, and Shadow's dimensions are inside the generator's ranges.
- The decoder (frame = "wrist") predicts the wrist update **along the start wrist's own axes**, from geometry tokens expressed
  in that frame and from the start wrist's world pose (ψ's z is world-frame).
- Each training family has one fixed palm-frame convention:
  - every generated body: x = palmar normal, y = across, z = distal;
  - LEAP: palm_lower;
  - Allegro and Barrett: their own.
- Shadow's is different (x = across, y = palmar normal, z = distal). Nothing in the model makes its output independent of
  how a hand's base frame is labelled. If the decoder learned convention-specific habits, a hand with a new convention
  would fail on particular axes, which is the pattern seen.

## W2-a: frame-relabelling test (evaluation only; existing checkpoints; no training)

The same physical task with the hand's base frame re-expressed by a fixed rotation Q:
- base-link template points and normals → Qᵀ·p, Qᵀ·n;
- every joint attached to the base link → its origin is pre-multiplied by Qᵀ;
- start and goal wrist rotations → R·Q (translations unchanged; contacts, and hence ψ's z, unchanged);
- the decoder's output is mapped back (R ← R'·Qᵀ) and scored in the original bank with the benchmark's `score`.

World link poses are identical by construction. This is checked numerically (template points in the world at the start
pose, max abs error < 1e-6 m), and so is the identity rotation (it reproduces the benchmark exactly).

- **Rotations:** all 24 axis-aligned proper rotations, applied to every robot in the bank at once.
- **Systems:** D1 s0/s1/s2, plus G1 s0/s1/s2 for reference.
- **Pools:** source val (Allegro, Barrett, LEAP), held-out LEAP, Wuji, Shadow.
- **Reported:** per pool and seed, the identity value and the distribution over the 23 relabelled frames (median, min, max) of
  translation R²_id, wrist mm, corr mm and compat success.
- **Also reported:** the Shadow value under the relabelling that maps Shadow's axes onto the generated bodies' convention
  (x ← y, y ← x, z ← z, with a sign fix to keep it a proper rotation), and whether any relabelling makes Shadow's
  in-plane translation R²_id positive.

Script: scripts/decoder/benchmark/frame_relabel_eval.py. Output: runs/w2a_frame_relabel. Cost: one GPU job, under 1 GPU-h.

**Predeclared rule** (fixed before the run):
- **Frame-dependent:** source Allegro or LEAP translation R²_id, median over the 23 relabelled frames, is ≥ 0.20 below its
  identity value in 3/3 D1 seeds.
  → The decoder relies on frame conventions. Launch W2-b.
- **Not frame-dependent:** that drop is < 0.10 in 3/3 seeds.
  → Frame conventions are ruled out. No W2 training; report it, and the next candidate (translation supervision through the
  contact centroid) goes back to the user as a proposal.
- **In between:** report it; no training without the user.

## W2-b: base-frame augmentation (training; only if W2-a says frame-dependent)

One change from D1, at the same 3 seeds: during training, each example's robot base frame is re-expressed by a random proper
rotation Q (uniform over SO(3)), exactly as in W2-a (geometry tokens, start/goal wrist rotations and targets consistently).
The physical task and losses are unchanged, and evaluation is unchanged (identity frame).

Implementation: an optional per-example base rotation in `RobotBank.fk` and its callers in the trainer, plus a config flag
`base_frame_aug` (default off, so all existing configs reproduce). It is tested by checking that world FK is invariant under Q.

- Benchmark: transfer_v1 + Shadow with G1 / D1 / W2-b, paired by seed.
- **Readout:**
  - Primary: Shadow wrist mm and translation R²_id vs D1, 3/3 counts.
  - Secondary: Shadow and Wuji corr / compat.
  - Non-inferiority: source, held-out LEAP and Wuji not worse in 3/3.
- Nothing is selected on Wuji or Shadow.

## Not proposed

Wrist head / loss redesigns, longer training, other architecture changes (one change at a time; only after W2-a / W2-b).
