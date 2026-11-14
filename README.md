# GETyourKITE

CS 6758 (Deep Learning for Robotics, Cornell, Fall 2026) project by Janice Shen, Anushka Vijay, Aryan Patel and Simon Tebeck.

[KITE](https://kite-manip.github.io) (Wang et al., 2026) splits manipulation into a shared
policy that predicts a contact-based *latent intent* and a kinematic decoder per robot that
turns the intent into joint targets. A new hand, or a hand with a broken finger, needs its
own decoder. We replace KITE's per-hand decoder parameters with a graph-conditioned decoder
in the style of [GET-Zero](https://get-zero-paper.github.io) (Patel & Song, 2025): joints
become tokens described by the URDF, with attention biased by the kinematic tree. The aim
is that one set of weights drives unseen and lesioned variants of a hand zero-shot, with
KITE's intent encoder and policy kept frozen.

## Layout

```
GETyourKITE/
├── src/getkite/
│   ├── scoring/          staged, embodiment-agnostic scores (ported from CEBenchmark)
│   ├── tasks/            one score spec per KITE task: objects_placing, keyboard_press, pump_bottle
│   └── kite_bridge.py    finds the KITE checkout; ScoringSession scores KITE rollouts unchanged
├── scripts/
│   └── rollout_scored.py run any KITE rollout/eval script with scores attached
├── tests/                unit tests (no KITE checkout needed)
├── docs/scoring.md       what the scores measure, calibration notes, first numbers
└── third_party/KITE      KITE code (git submodule, helloqxwang/KITE)
```

Planned next to `scoring/` and `tasks/`: `embodiments/` (lesioned and extended hand URDFs),
`decoder/` (the GET decoder) and `baselines/` (KITE decoder with lesioned joints dropped,
head-only fine-tuning).

## Setup

```bash
git clone --recursive <this repo>      # or: git submodule update --init
cd GETyourKITE/third_party/KITE
# Follow KITE's README: conda env `kite`, pytorch3d, then the asset downloads
conda activate kite
python download_assets.py
cd ../..
python -m pip install --no-deps -e .    # getkite into the kite env (it only needs mujoco and numpy)
export MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

Tests: `python -m pip install pytest && python -m pytest`.

## Scoring KITE rollouts

```bash
python scripts/rollout_scored.py --out outputs/scores/barrett.json -- \
    --config-name=rollout_gt_contacts/object_suite_barrett online_rollout.count=10
```

Prints one line per episode (KITE success, staged success, score, quality, max hand
penetration, stages reached or failure) and saves them as JSON. Everything after `--` goes
to the KITE script unchanged. See [docs/scoring.md](docs/scoring.md) for what the scores
mean, how CEBenchmark's scoring was adapted to KITE, and the first numbers.

## Provenance

- `third_party/KITE`: [helloqxwang/KITE](https://github.com/helloqxwang/KITE), used unmodified.
- `src/getkite/scoring/`: copied from CEBenchmark (`cebench/tasks/_base`, commit `ffad82e`),
  only the import paths changed. Keep the two in sync.
