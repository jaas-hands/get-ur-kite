<!-- Moved here from a local note; text unchanged. Run directories named runs/... are under ${GETKITE_RUNS}. -->

# Cross-body intent benchmark + held-out family: evaluation protocol (fixed 2026-11-17, before any result)

Written before generating the held-out data or running any evaluation. Learned weights are not changed or tuned.

## Systems (all frozen)
- **two-stage** (deployable): z = KITE psi(request) -> world-frame contact-set bridge
  (`set_bridge_zworld_c3_20k_b/best.pt`, step 20k) -> predicted contacts (existence > 0.5, >= 1) -> raw-contact GET
  decoder (`scratch_c3_raw50k_cf8_10x_wrist/best.pt`, step 50k) for the TARGET robot. Wrist preprocessing applied once,
  inside the decoder.
- **true-request** (diagnostic reference, not deployable as a psi system): the same decoder fed the request contacts.
- **no movement**: the start pose.
- **optimizer oracle** (feasibility reference only, not a learned system): see Feasibility.

## A. Held-out family: Wuji right hand
- `wujihand_right` (KITE `spider/wujihand/right.urdf`, 20 DOF, five fingers; KITE template
  `surface_template_graph_n4096_998694447d62.pt`). Never seen by any GET decoder or bridge (training, selection).
  **Caveat:** it is one of KITE psi's five training embodiments, so it is held out from our models, not from psi.
- Records: KITE's own sampler (get-ur-kite-dataset `gen_kite_family.py`, kite-families branch, read via `git show`,
  no checkout): eval pool n=500 seed 202; example bank n=2000 seed 101 (used only to initialise cross-body starts).
- Same-body protocol identical to the fresh-goal evaluation: start = `fixed_start(val_start_seed)` with KITE start noise
  around the goal. Scored by (i) correspondence scoring (`pose_metrics`, template indices, as before) and (ii)
  target-surface scoring (below).
- Sanity before use: adding Wuji to the robot bank must leave the decoder's outputs on existing records unchanged.

## B. Cross-body intent benchmark
- Source requests: fresh records only (never training): Allegro val, Barrett val, LEAP fresh goals (rows 900-942 of
  the 12 training hands), Wuji eval. 64 requests per source family, fixed seed (7).
- Targets: `allegro_right`, `barrett`, one held-out LEAP hand per request (the four held-out hands, assigned round-robin),
  `wujihand_right`. All pairs with source family != target family: 4 x 3 x 64 = 768 cases per path.
- Start pose (KITE's "closest centered example" protocol, `visualization/decoder_runtime.closest_initial`): among the
  target's example bank, pick the example whose centred valid contacts best match the centred request contacts
  (symmetric mean nearest-neighbour distance); start = that example's goal configuration with KITE start noise (fixed
  seed), base translated so the example's contact centroid equals the request centroid; no rotation. Identical start
  for every path. Example banks: Allegro/Barrett = pilot training records; held-out LEAP hand = its rows 125-1000 (the
  eval rows 0-125 excluded); Wuji = the seed-101 bank. Uses only the request (deployment-visible) and target examples.

## Target-surface scoring (cross-body; also reported for same-body Wuji)
For each requested contact (position c_k, normal n_k): position-nearest point among the target hand's 4096 surface
template points at the predicted pose (joints clamped to limits, as in `pose_metrics`); d_k = distance, a_k = angle
between that template point's normal and n_k (same convention as `pose_metrics`). Reported: mean d (mm), RMS d (KITE's
number), mean angle (deg), success = all d_k <= 10 mm and a_k <= 15 deg, wrist translation/rotation from the start.
No correspondences are used (none exist across bodies).

## Feasibility handling
1. Joints are clamped to limits before scoring; the pre-clamp limit-violation rate is reported.
2. No collision or penetration checks: the records carry no object geometry and self-collision is not modelled.
3. Request feasibility is unknown across bodies, so each case gets a non-learned optimizer oracle: from the same start
   (+3 perturbed copies), Adam on the target configuration (base 6D + joints, clamped to limits), 300 steps,
   minimising mean_k [ (min_j |p_j - c_k| / 5 mm)^2 + 10 (1 - n_j* . n_k) ] (nearest point re-assigned every step).
   A case is **oracle-feasible** if the best restart meets the success criterion.
4. Primary results on ALL cases; secondary on the oracle-feasible subset; oracle metrics shown as a reference.

## Reporting
Per (source family -> target) and per target: the metrics above for each path, the two-stage minus true-request drop,
and the oracle-feasible fraction. Same-body Wuji: correspondence and target-surface metrics, per path.

## Correction (2026-11-17, before any result; job failed at robot-bank construction)
Wuji is loaded from `spider_robots/wujihand/right_floating.urdf`, the file KITE itself uses
(`robot_model.ensure_virtual_floating_base_asset`: right.urdf wrapped with the 6-DoF virtual base whose joint names
match the generated records). Same hand, same template; scoring and feasibility rules unchanged.

## Finding after the run (2026-11-17): the pre-registered angle/success criterion is degenerate
Check (runs/crossbody_heldout_eval/checks/surf_check.py): at exact goal poses target-surface scoring gives 0.02 mm /
0.0 deg (implementation correct). At the start poses (~14 mm correspondence error) the POSITION-nearest template point
is 6.2 mm away but on a differently oriented face: 60-69 deg normal error, versus 7.7 deg at the corresponding point.
So "nearest by position, then compare normals" fails the 15 deg condition for almost any non-exact pose; success and
oracle feasibility under this rule are uninformative (oracle: 1.85 mm but 69 deg). Position distances remain valid.
Proposed revision (NOT applied; awaiting approval): normal-compatible distance d_k = min distance over template points
whose normal is within 15 deg of n_k; success = all d_k <= 10 mm; oracle assigns points by the same rule. Applied
identically to all paths, reported next to the original position-nearest distance.

## Disclosure (2026-11-17, independent review)
Cross-body starts are selected from banks of target-hand synthetic examples (not used for training), so the benchmark
assumes access to target-hand example configurations at test time. Each system makes ONE correction step. All trained
decoders also received palm-heavy surface tokens for Allegro/Barrett/Wuji (audit §37A).
