# Sealed new-family test set: proposal (2026-11-28; design only, nothing downloaded or run)

Why: Wuji, Shadow and the 7 external hands have steered recipe choices since CF-3d, so they are new-family VALIDATION. A clean
measure of generalisation to new hand families needs hands that have never been scored by any of our models, scored once at the
end. It is also the only valid place to settle the per-hand frame margin (0 vs 0.25 mm), which was chosen after seeing results.
Simon's test splits stay separate (LEAP-family test; his scoring).

## Candidates (research 2026-11-28 from the repos' pages; licences and joint counts to be re-verified from the files)

| Hand | Fingers, layout (non-fixed joints per finger, mimic included) | Model | Licence | Layout match among the 800 training bodies (verified) |
|---|---|---|---|---|
| Linker Hand L25 | 5; (4,4,4,4,5), DIP/IP mimic-coupled | URDF + STL, linker-bot/linkerhand-urdf | Apache-2.0 | **0** (layout-unseen) |
| Unitree Dex3-1 | 3; (2,2,3), opposable thumb | URDF + STL, unitreerobotics/unitree_ros | BSD-3-Clause | 18 (+6 in P0800+) |
| Robotiq 3-Finger | 3; (3,4,4), no thumb, underactuated (no mimic tags) | URDF/xacro + STL, ros-industrial/robotiq | BSD-2-Clause | 18 (+3 in P0800+) |
| D'Claw (ROBEL) | 3; (3,3,3), radial, no thumb | URDF, dexsuite/dex-urdf | Apache-2.0 (from dex-urdf's table) | 6 |
| ORCA v2 | 5; (3,3,3,3,4) + a wrist joint to drop | URDF + MJCF, orcahand/orcahand_description | MIT (README) | 4 (same layout as Aero, already used) |

Not proposed: Unitree Dex5-1 (5x4 = Wuji's layout, which the generated bodies exclude on purpose); DexHand (TheRobotStudio) and
Inspire-derived hands (CC BY-NC-SA, no URDF); Kinova 3-finger (licence unclear). Not checked yet: Fourier, BrainCo, OYMotion,
DLR/HIT, RAPID, Faive, Mimic, OmniHand.

Layout overlap with primitive generated bodies is unavoidable for small hands (3 fingers of 2-4 joints); the shapes are real,
the bodies are primitives, so it is layout-level only and is reported per hand, as for the external hands. Linker L25 is the only
candidate that is strictly layout-unseen. To check during asset work: layout matches with LEAP variants (138 topologies).

## Recommended set (4): Linker L25, Dex3-1, Robotiq 3F, D'Claw
Diverse (5 and 3 fingers; with and without thumb; mimic coupling; underactuated), permissive licences. ORCA v2 as a spare only
(same layout as Aero).

## Protocol (fixed before any asset is fetched)

1. **Seal.** Hand list, licences, asset commit hashes and SHA256 recorded in a committed manifest before any record is
   generated. No model is run on these hands until the end-of-project evaluation; no recipe, margin, threshold or checkpoint is
   chosen on them.
2. **Build** with the existing external-hands pipeline (`scripts/decoder/external_hands/`): fetch pinned assets, adapter (joint
   order, mimic handling, base link), surface template, frame probe, records with the same sampler, counts and seeds as the 7
   external hands. Checks: FK of adapter vs the original model (0 error), goal reproduction, cover radius, joint-limit sanity.
   The person building them looks only at geometry checks, never at model outputs.
3. **What is allowed before the end:** automatic validity checks only (adapter FK, goal reproduction, cover radius, joint limits),
   reported pass/fail. Not allowed: any decoder prediction, and no rule-E frame angles (they would reveal which hands flip while
   frame methods are being designed; Aryan's review 2026-11-28).
4. **End-of-project evaluation, once:** the delivery model and its predecessors (D1, D1-bv, CF2, CF-3e, and the final model if it
   differs), each seed separately; corr_mm, corr_success, compat_success, median and p90 per hand, wrist error at base origin and
   at a palm point; per-hand frame choice reported at margin 0 and 0.25 mm (the frame-margin question is decided here).
5. **Reporting:** per hand, per seed, with 95 % intervals; no pooled headline. Linker L25 (layout-unseen) reported separately from
   Dex3-1, Robotiq 3F and D'Claw (layout matches among the training bodies). Labelled "sealed new-family test". ψ status: each
   hand checked against KITE's ψ training data and recorded in the manifest before sealing.

## Cost
Asset fetch and adapters: about half a day of CPU/human time (Robotiq xacro expansion and its malformed palm inertial, Linker
mimic joints, ORCA wrist removal are the known quirks). Records: minutes per hand. Evaluation: minutes of GPU, once.
Needs the user's go-ahead for downloading third-party assets and for the manifest commit.
