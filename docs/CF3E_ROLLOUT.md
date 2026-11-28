# CF-3e: first Oracle-Intent rollout

The connection runs on the Mac. CF-3e made all 120 predictions in a Barrett mug-placement episode, but did not lift the mug. KITE succeeded on the same episode.

| Decoder | Episode | Completed steps | Task success | Staged score |
| --- | --- | --- | --- | --- |
| Released KITE | Barrett, mug, 55 | 120/120 | Yes | 100% |
| CF-3e, seed 0, step 95,000 | Barrett, mug, 55 | 120/120 | No | 2.84% |

These are two individual attempts, not success-rate estimates. The score measures progress through Simon's lift/carry stages; 2.84% is not a success probability. KITE's success rule does not require releasing the object. Barrett is a training family, so this is not a held-out-family experiment.

## What is connected

Recorded oracle intent + current simulated hand state → CF-3e → target wrist/joints → KITE's existing controller → MuJoCo → Simon's score.

Simon already supplied the demo, controller integration, video overlays and task scoring. This change adds a CF-3e adapter at the saved-intent prediction call, plus action traces and verification. It preserves KITE's episode setup, coordinate shift, state feedback and controller. The official KITE source is unchanged.

Only Barrett with saved latent intents and one rollout worker is supported by this first adapter. It does not need a trained high-level policy. Connecting that policy, LEAP or other hands is separate work.

## Files to look at

- `src/getkite/cf3e_rollout.py`: loads CF-3e, constructs the Barrett description, converts wrist/joint inputs and outputs, and temporarily replaces the prediction call.
- `scripts/demo_task.py`: the existing demo, now with `--decoder cf3e --decoder-checkpoint PATH`; also carries over Mac recording and score-file support.
- `scripts/verify_cf3e_rollout.py`: compares the run with saved oracle intents and the team's existing J1 inference code.
- `outputs/cf3e_oracle_smoke/verification.json`: the checks and numerical differences.
- `outputs/cf3e_oracle_smoke/{cf3e,kite}/scores_placing_barrett.json`: task results and run arguments.
- `outputs/cf3e_oracle_smoke/cf3e/cf3e_actions.npz`: all 120 current states, predicted actions and input intents. These are decoder targets, not measured executed poses.
- `outputs/cf3e_oracle_smoke/cf3e/cf3e_adapter.json`: checkpoint identity, joint order and geometry checks.

Videos: [CF-3e](../outputs/cf3e_oracle_smoke/cf3e/placing_barrett_025_mug_ep55.mp4) · [KITE](../outputs/cf3e_oracle_smoke/kite/placing_barrett_025_mug_ep55.mp4). Outputs are local and ignored by Git; uploading code alone will not publish these videos.

The NPZ task result retains KITE's bootstrap metadata. For CF-3e runs, read `action_decoder`, `action_decoder_checkpoint` and `action_decoder_sha256` to identify the model actually producing actions.

## What was checked

- All 120 steps completed with no episode errors, for both decoders.
- Matching episode, contact source, start pose, timestep, coordinate shift and feedback settings.
- Exact match between the saved oracle intents and the adapter's inputs. Re-encoding the contact points with the released encoder also agrees within numerical tolerance (largest absolute difference 0.000044).
- All 4,096 surface points and normals checked at eight random hand poses against KITE's own forward kinematics. Largest position-component difference: about 0.000051 mm.
- Adapter predictions agree with the existing J1 benchmark inference path over all 120 recorded states: translation within 0.000060 mm, joint coordinates within 0.0000015 rad, rotation-matrix entries within 0.0000017.
- Fifteen relevant adapter, scoring and visualization tests passed. Both videos decode as 210 frames.

These checks support that the adapter is connected correctly. They do not establish why CF-3e missed the grasp. In the video it moves toward the bin while the mug stays behind. Contact tracking through the approach/grasp phase is the next diagnostic.

## Run it again

From this checkout, using the existing Mac environment:

```bash
export GETKITE_KITE_ROOT=/Users/janiceshen/Documents/jaas-hands/get-ur-kite/third_party/KITE
export KITE_ROOT="$GETKITE_KITE_ROOT"
export PYTHONPATH="$PWD/src:/Users/janiceshen/Documents/jaas-hands/get-ur-kite-dataset"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 WANDB_MODE=disabled
PY=/Users/janiceshen/Documents/jaas-hands/get-ur-kite/.venv-mac/bin/python
CKPT=/Users/janiceshen/Documents/jaas-hands/getkite_runs/cf3e_long_hf/best.pt

"$PY" scripts/demo_task.py --hand barrett --indices 55 --record --video-size 960x640 \
  --decoder cf3e --decoder-checkpoint "$CKPT" --out outputs/cf3e_oracle_smoke/cf3e -- \
  online_rollout.device=cpu online_rollout.backend=raw env.backend=raw
"$PY" scripts/demo_task.py --hand barrett --indices 55 --record --video-size 960x640 \
  --out outputs/cf3e_oracle_smoke/kite -- \
  online_rollout.device=cpu online_rollout.backend=raw env.backend=raw
"$PY" scripts/verify_cf3e_rollout.py --run outputs/cf3e_oracle_smoke
"$PY" -m pytest tests/test_cf3e_rollout.py tests/test_kite_scores.py tests/test_scoring.py tests/test_viz.py -q
```

## Next steps

1. Confirm with Simon which baseline branch, task settings and small episode list we should share. Nothing from him was needed to run this first check.
2. Review the missed grasp with Anushka/Aryan. Confirm whether this checkpoint should use one prediction per control step or any documented refinement; do not change the protocol just to improve this episode.
3. Agree on a small fixed set of episodes, then compare both decoders and inspect contact tracking during failures. Keep the recorded-intent result separate from learned-policy task success.
4. Verify each additional hand's geometry and action mapping before testing it. LEAP and held-out-family rollouts remain untested here.
