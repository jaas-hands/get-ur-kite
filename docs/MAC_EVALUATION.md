# Task evaluation on Mac

The Objects Placing demo runs on Apple Silicon with CPU inference. The original
`setup_env.sh` is for Linux/CUDA, so use the separate `.venv-mac` environment.
This checks KITE's released decoder using recorded intents. Our team's decoder and
a learned task policy have not been evaluated here.

## Demo results

On macOS 15.3 / arm64, Python 3.11, PyTorch 2.9.1 and MuJoCo 3.8.0:

| Hand / episode | KITE success | Peak score | Ever lowered into bin | Video |
| --- | --- | --- | --- | --- |
| Barrett 55, mug | yes | 100% | yes | [Watch](../outputs/mac_check/barrett/placing_barrett_025_mug_ep55.mp4) |
| Barrett 51, mug | no | 7.3% | no | [Watch](../outputs/mac_check/barrett/placing_barrett_025_mug_ep51.mp4) |
| Wuji 37, can | yes | 100% | no | [Watch](../outputs/mac_check/wuji/placing_wuji_010_potted_meat_can_ep37.mp4) |
| Allegro 20, bowl | no | 36.1% | no | [Watch](../outputs/mac_check/allegro/placing_allegro_024_bowl_ep20.mp4) |

These links open the local recordings; they don't copy the videos into this document.
The `outputs/` folder is not pushed to GitHub. To share the videos with the team,
upload them to the team's Drive and use those links instead.

These four episodes were picked to check different behaviours, so don't use them
to estimate overall success rates. The success/failure labels match Simon's demo
descriptions; the exact partial scores differ. This is not numerical reproduction
of his Linux results or the paper.

The Barrett pair was repeated without video; every saved score field matched exactly.
All four videos decode completely (210 frames each). Scorer success labels were
checked against KITE's separate NPZ results for every run.

KITE counts an object held over the bin as success. The separate `placed` flag means
it was also lowered at some point; neither proves release. Peak score keeps credit
for earlier progress after a drop. Hand penetration is reported separately.

## Watch or rerun

From this repo, the environment is already installed. Record the two mug episodes:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 WANDB_MODE=disabled \
  .venv-mac/bin/python scripts/demo_task.py \
  --hand barrett --indices 55,51 --record --video-size 960x640 \
  --out outputs/my_mug_check -- \
  online_rollout.device=cpu online_rollout.backend=raw env.backend=raw
```

The MP4s have scores drawn on screen. Episode 55 should succeed; 51 should fail.
Use a fresh output folder for each run. Run one hand at a time.

To open the interactive viewer, use `.venv-mac/bin/mjpython` in place of `python`
and omit `--record`. Space pauses; drag to rotate the camera; Tab shows controls.
macOS needs `mjpython` for MuJoCo's passive viewer
([MuJoCo docs](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer)).

Change `--hand` / `--indices` to `wuji` / `37` or `allegro` / `20` for the other checks.

## Read the results

```bash
.venv-mac/bin/python scripts/summarize_task_run.py \
  outputs/my_mug_check/scores_placing_barrett.json \
  outputs/my_mug_check/kite_results_placing_barrett.npz \
  --out outputs/my_mug_check/report.md
```

- `report.md`: the readable table. The script rejects incomplete runs and checks
  episode IDs and success labels against KITE's own NPZ output.
- `scores_placing_*.json`: Simon's scores, stage times, penetration and flags.
- `kite_results_placing_*.npz`: KITE's separate results, used as a cross-check.
- `*.mp4`: what happened; `kite_log.txt`: the full run log if something fails.
- `outputs/mac_check/`: the original checks, setup logs and package versions.

## What changed

No KITE source, weights, physics parameters or scoring equations were edited.
CPU inference and KITE's `raw` backend are explicit command-line overrides.
`MjlabBackend` inherits this same physics stepping code; its extra dependency
check is unnecessary for these CPU runs.

The demo uses Mac rendering and normal Python shutdown on macOS. It now saves
score JSON with source episode IDs. The scoring runner also preserves failed exit
statuses. The report script checks its inputs before writing a table.
The viewer's close routine skips the native threads exposed by `mjpython`, which
Python cannot join.

Run the relevant tests:

```bash
.venv-mac/bin/python -m pytest tests/test_scoring.py tests/test_kite_scores.py \
  tests/test_viz.py tests/test_rollout_scored.py tests/test_task_report.py -q
```

## Rebuild the environment

With Python 3.11 and Apple's command-line compiler installed:

```bash
git submodule update --init third_party/KITE
python3.11 -m venv .venv-mac
.venv-mac/bin/python -m pip install -r requirements-mac.txt
PYTORCH3D_FORCE_NO_CUDA=1 MAX_JOBS=2 .venv-mac/bin/python -m pip install \
  --no-build-isolation --no-deps ./third_party/KITE/third_party/pytorch3d
.venv-mac/bin/python third_party/KITE/download_assets.py
.venv-mac/bin/python third_party/KITE/download_assets.py --verify-only
```

PyTorch3D is compiled from KITE's source, with its actual CPU operations; there is
no substitute sampler. This environment is for evaluation, not CUDA training.

## Next steps

- Get the team's decoder checkpoint and connect it to the rollout runner with Simon.
- Agree on a fixed episode list, starting states and success rule before comparing
  KITE with our decoder. The four demos above are setup checks, not a benchmark.
- Report contact accuracy and task success separately, and save a few representative
  videos alongside the scores.
