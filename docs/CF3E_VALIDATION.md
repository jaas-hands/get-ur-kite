# CF-3e validation check

CF-3e seed 0 runs on the Mac CPU and its predictions load into our existing
evaluator. This uses the 32 validation records in Anushka's `leap_sample_v1`
bundle: one hand, `I0-M2-R0-T4_g0`, with six movable joints. No test splits were run.

| Method | Contact error | Normal error | Contact success |
| --- | ---: | ---: | ---: |
| No movement | 14.79 mm | 9.56° | 3/32 (9.4%) |
| CF-3e seed 0 | 8.39 mm | 6.53° | 11/32 (34.4%) |
| Stored goal, sanity check | <0.001 mm | <0.01° | 32/32 |

The model reduces mean contact error by about 43%. Success requires every contact
within 10 mm and 15°. Both model and no-movement predictions have zero invalid
poses and zero joint-limit violations on this sample. These are contact results,
not object-picking results or a model ranking.

## What was checked

- Downloaded the delivery checkpoint and matched its SHA256 to the team's manifest.
- Used the released KITE encoder, the model's canonical palm frame and stored starts.
- Confirmed joint order against the model's hand description. This hand is absent
  from the checkpoint's training list and appears in its 105-hand validation config.
- Exported real predictions with row IDs and input hashes into the evaluator format.
- Independently recomputed contact error, normal error, success and raw limit flags.
  All 256 comparisons passed, both in memory and after the saved prediction import
  (512 checks total). Largest differences: 0.000020 mm and 0.000374°; flags match exactly.
- Removed goal fields from the inference input and separately replaced goals with
  NaNs and corrupted scoring-only masks/correspondences. Wrist/joint predictions
  stayed exactly unchanged. Requested contacts and their valid flags stay available
  to the frozen encoder, as intended.
- The stored goal reconstructs the requested contacts to numerical precision.

Contact scoring clamps joints; limit flags use raw predictions with a 1e-6 rad
tolerance. Collision/physical-validity checks are unavailable with this sample's
descriptor-only assets. Compatible-contact success and wrist errors are saved by
the team's predictor but have not been independently cross-checked here.

The 8.61 mm headline result uses 105 hands and newly sampled starts. It is not an
expected value for these 32 stored-start records. A matching small reference from
the training environment would establish cross-machine agreement separately.

## Run it

Use this model branch plus the dataset repo's `eval-analysis` branch. From this
checkout, with the existing sibling Mac environment and downloaded checkpoint:

```sh
export KITE_ROOT="$(cd ../get-ur-kite/third_party/KITE && pwd)"
export PYTHONPATH="$PWD/src:$(cd ../get-ur-kite-dataset && pwd)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

../get-ur-kite/.venv-mac/bin/python scripts/decoder/validate_j1.py \
  --ckpt ../getkite_runs/cf3e_long_hf/best.pt \
  --kite-root "$KITE_ROOT" \
  --kite-ckpt "$KITE_ROOT/checkpoints/decoder/release.pt" \
  --data ../get-ur-kite-dataset/data/evaluation/leap_predictions_2026-11-19/sample_input/leap_sample_v1 \
  --out outputs/my_cf3e_validation
```

Choose a new output folder for each run. The script selects only `val` files.
The predictor's new `--kite-root` and `--kite-ckpt` options replace the Linux paths
saved in the checkpoint; model weights and coordinate transforms are unchanged.

## Files to open

The verified local run is under `outputs/cf3e_validation_verified/`:

- [Comparison](../outputs/cf3e_validation_verified/comparison/report.md): readable results.
- [Plot](../outputs/cf3e_validation_verified/comparison/comparison.png): errors and success.
- [Pose examples](../outputs/cf3e_validation_verified/model/poses/val/I0-M2-R0-T4_g0.png): first row and worst remaining error.
- [Checks](../outputs/cf3e_validation_verified/verification.json): pass/fail and numeric differences.
- `model/examples.csv`: each record's errors, success and limit flags.
- `raw/` and `adapted/`: original predictions and evaluator-format predictions.
- `model.json`, `model/manifest.json`, `crosscheck.csv`, `run.log`, `commands.sh`:
  checkpoint/input/code hashes, package versions, comparisons and commands.

These links refer to local generated files; `outputs/` and checkpoints are not
committed. The reusable code is `scripts/decoder/validate_j1.py`, using the existing
`predict_j1.py` and the dataset evaluator. This offline check did not run training
or simulation. The subsequent [Oracle-Intent rollout](CF3E_ROLLOUT.md) is now working
on Barrett; its task results are reported separately.

## Next steps

- Get the matching full `kite_code` validation bundle and per-hand stored-start
  reference results; check all 105 hands before scoring the final splits.
- Get Simon's baseline predictions/results, exact evaluation records and matching
  KITE surface template. Confirm dataset version before combining tables.
- Agree on the frozen model and whether the proposed frame calibration is included.
  This check uses CF-3e seed 0 without calibration.
- The first Barrett Oracle-Intent connection is done. Coordinate the shared episode
  list with Simon and investigate the missed grasp before expanding the comparison.
  Keep task success separate from these contact metrics.
