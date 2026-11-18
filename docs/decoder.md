# The GET decoder (`getkite.decoder`)

The decoder takes KITE's latent intent `z = ψ(contacts)`, a start pose `q_init` and a robot description, and
predicts the goal joint configuration. KITE's ψ (contact encoder) is always used frozen. Two lines of work
live here:

| Experiment | Question | Trainer | Configs | Status |
| --- | --- | --- | --- | --- |
| **LEAP family** | Do GET joint tokens (URDF graph) beat a shared-slot baseline on unseen LEAP variants, both on top of KITE's pretrained decoder trunk? | `getkite.decoder.train` | `configs/decoder/*.json` | finished, preliminary (below) |
| **Multi-family scratch decoder** (`get_universal_scratch`) | Can one randomly initialised GET decoder, with ψ frozen, decode across Allegro, Barrett and LEAP variants? | `getkite.decoder.train_scratch` | `configs/decoder/scratch/*.json` | pilot gates C1–C3 (below) |

## Where things are

```
src/getkite/decoder/
  paths.py            where KITE, datasets and runs live (environment variables, see below)
  kite_release.py     loads KITE's released checkpoint: ψ, decoder trunk, normalisers
  rotations.py        6D / Euler / axis-angle helpers shared by everything
  # LEAP-family experiment
  config.py, data.py  run config; loads get-ur-kite-dataset splits (kite_data)
  handbank.py         LEAP hand descriptions (hand.json) as padded tensors, FK
  model.py, kite_trunk.py, intent.py, losses.py, train.py, probe.py
  # multi-family scratch decoder
  robots.py           any URDF (KITE's Allegro/Barrett) or LEAP hand -> one padded RobotBank, generic FK
  get_scratch.py      the scratch GET decoder (joint, surface and intent tokens; tree attention biases)
  multifamily.py      records, KITE's start-pose noise, family-balanced sampling
  train_scratch.py    training/evaluation with per-family metrics and provenance
  kite_reference.py   KITE's released decoder scored on the same records (reference numbers)
  spatial_bridge.py   psi z (+ start wrist) -> contact centroid / contact set ("bridge" / translator)
  joint_contact_get.py, train_joint.py   J1: frozen psi -> contact-set predictor -> raw-contact GET, trained jointly
configs/decoder/      LEAP-family configs;  configs/decoder/scratch/: pilot configs
scripts/decoder/      make_manifest.py (writes the pilot configs), report_c3.py (pilot tables),
                      check_kite_release.py, diagnostics/ (one-off analyses kept for provenance)
slurm/                train.sbatch, scratch.sbatch, joint.sbatch, final_eval.sbatch, watchdog.sh, env.sh
tests/decoder/        decoder tests (skip cleanly when KITE or the dataset is missing)
docs/decoder_audit.md               working log of the multi-family decoder, sections 1-42, with every result
docs/decoder_crossbody_protocol.md  the Wuji / cross-body evaluation protocol, fixed before the results
```

Data generation is not here: LEAP hands and examples come from
[jaas-hands/get-ur-kite-dataset](https://github.com/jaas-hands/get-ur-kite-dataset) (`kite_data`), and the
Allegro/Barrett records from that repo's `scripts/kite_family/` (KITE's own sampler).

## Data and runs live outside the repo

Nothing large is committed. Configs refer to data with placeholders that `getkite.decoder.paths` expands
when a config is loaded:

| Placeholder | What | Default |
| --- | --- | --- |
| `${KITE_ROOT}` | KITE checkout with `assets/` and `checkpoints/` (`python download_assets.py --only runtime decoder`) | `third_party/KITE`, or `../KITE` if the submodule is not initialised |
| `${GETKITE_DATASET}` | get-ur-kite-dataset's `data/` (`hands/`, `examples/`, `splits.json`), ~2 GB | `kite_data`'s data dir (`../get-ur-kite-dataset/data`) |
| `${GETKITE_DATA}` | data made for the decoder, e.g. `multifamily_pilot/` (Allegro/Barrett records) | `../getkite_data` |
| `${GETKITE_RUNS}` | checkpoints, logs, results (several GB) | `../getkite_runs` |

So the expected workspace is

```
workspace/
  get-ur-kite/            this repo
  get-ur-kite-dataset/    the data repo (its data/ is gitignored)
  getkite_data/           generated decoder data
  getkite_runs/           runs
```

To put data or runs elsewhere (a bigger disk, a synced Google Drive folder), move the folder and set the
variable, e.g. `export GETKITE_RUNS=/path/to/drive/getkite_runs`. `python -m getkite.decoder.paths` prints
what is resolved. For Slurm jobs put the exports in `slurm/env.local.sh` (gitignored; start from
`slurm/env.local.example.sh`).

## Setup

```bash
git clone --recursive git@github.com:jaas-hands/get-ur-kite.git
git clone git@github.com:jaas-hands/get-ur-kite-dataset.git
cd get-ur-kite/third_party/KITE && python download_assets.py --only runtime decoder && cd ../..
pip install -e ../get-ur-kite-dataset          # kite_data
pip install -e ".[decoder,dev]"                # getkite + torch
python -m getkite.decoder.paths                # check every location resolves
python -m pytest tests/decoder                 # ~3 min on CPU
```

The LEAP data: `cd ../get-ur-kite-dataset` and follow its README (hands, splits, then
`python -m kite_data.generate --regime kite_code`; `slurm/generate.sbatch` there runs it on a CPU node).
The Allegro/Barrett pilot records: `sbatch scripts/kite_family/gen_kite.sbatch` in the dataset repo
(about 10 min on 8 CPUs).

## Running

All Slurm scripts are submitted from the repo root and log to `slurm/logs/`. Add your cluster's partition and
GPU constraint on the command line (Unicorn: `-p default_partition --constraint="ampere|ada|2080ti"`;
torch cu13x has no Pascal/Volta kernels, and a job on a node whose driver is too old resubmits itself
elsewhere).

```bash
# LEAP family: GET vs shared-slot decoder on KITE's fine-tuned trunk
sbatch -J kr-kite_trunk_get_ft slurm/train.sbatch configs/decoder/kite_trunk_get_ft.json kite_trunk_get_ft
sbatch -J kr-eval-kite_trunk_get_ft slurm/final_eval.sbatch configs/decoder/kite_trunk_get_ft.json kite_trunk_get_ft

# multi-family scratch decoder (pilot)
python scripts/decoder/make_manifest.py        # picks the LEAP hands, writes configs/decoder/scratch/c{1,2,3}*.json
sbatch -J kr-c3 slurm/scratch.sbatch configs/decoder/scratch/c3_long50k.json
sbatch -J kr-j1 slurm/joint.sbatch configs/decoder/scratch/j1_joint50k_hf.json    # J1
python scripts/decoder/report_c3.py $GETKITE_RUNS/scratch_c3_long50k $GETKITE_RUNS/scratch_c3_raw50k

# without Slurm
python -m getkite.decoder.train_scratch --config configs/decoder/scratch/c1_overfit.json --set total_steps=200
```

Every run writes `config.json` (paths expanded), `log.jsonl` (training and evaluation rows), `last.pt`
(resumable), `best.pt` (selected on validation only) and `DONE`; scratch runs also write `provenance.json`
(ψ checksum, git state, data checksums, robot fingerprints, GPU). `slurm/watchdog.sh` run from cron
resubmits unfinished runs listed in `slurm/watch_runs.txt` (see `watch_runs.example.txt`).

## Results so far

Contact error is the mean distance between the goal contacts and the same template points on the predicted
hand; success means every contact within 10 mm and 15°. "No movement" scores the start pose itself.

**LEAP family, preliminary** (ψ frozen, KITE trunk fine-tuned at 0.1× lr, 100k steps, one seed each;
best.pt by validation):

| Split | GET joint tokens | Shared-slot baseline | No movement |
| --- | --- | --- | --- |
| val (105k) | 7.64 mm / 48.0 % | 7.80 mm / 46.3 % | 14.76 mm / 6.8 % |
| test_combined (432k) | 7.67 / 48.4 % | 7.80 / 46.9 % | 14.78 / 7.0 % |
| test_geometry (80k) | 7.53 / 50.0 % | 7.68 / 48.2 % | 14.52 / 7.8 % |
| test_joint_count (756k) | 7.48 / 50.6 % | 7.65 / 48.4 % | 14.41 / 7.6 % |
| test_finger_subset (72k) | 7.54 / 49.7 % | 7.59 / 49.0 % | 14.78 / 7.1 % |

The shared-slot model is a KITE-inspired baseline, not KITE's own decoder. These are LEAP-only references,
not matched baselines for the multi-family experiment.

**Multi-family scratch decoder, pilot** (Allegro, Barrett, 12 LEAP hands; 4 more LEAP hands held out):

| Gate | Result |
| --- | --- |
| C1: 48 fixed examples, fixed start | passes: ~0.05 mm, 100 % in every family |
| C2: same goals, start pose redrawn | fails as specified (wrist never moves); passes with Fourier features of the world pose: ~1 mm, 100 % |
| C3: ~2,000 goals per family, 10k steps | fails: at or slightly worse than no movement on new goals |
| C3, 50k steps, ψ vs raw contacts (diagnostic reference; same data, starts, backbone, budget) | both fit training goals only partly (ψ 8.7–9.1 mm / ~50 %, raw 8.1–8.6 mm / ~50 %) while error on new goals rises well above no movement (ψ 14.3–19.4 mm, raw 15.7–20.0 mm; held-out LEAP 19.6 / 20.2 mm). Raw contacts do no better than ψ, so ψ is not the bottleneck; the decoder memorises instead of generalising at this data size |
| KITE's released decoder, C3 validation records | Allegro 4.09 mm / 92 %, Barrett 3.70 mm / 92 % |

After C3 (Nov 15-18; details and every number in `docs/decoder_audit.md`, sections 14-42):

1. **Wrist frame and more data (§15-21).** On a pool with 10x more independent goals, raw contacts in the starting
   wrist's frame with contact Fourier features no longer memorise (train ≈ validation). Frozen ψ in the same setup stays
   at no movement: the decoder never learns to use z for the wrist correction.
2. **ψ + KITE's frozen scene encoder (§22-23)** does not fix it: hand geometry and contacts have to be compared in
   one frame.
3. **Bridges (§24-31).** A small network recovers the contact set from ψ's z (world frame, then the exact transform
   into the start-wrist frame). Feeding its contacts to the frozen raw-contact decoder gives a two-stage ψ system.
4. **Held-out Wuji hand and cross-body requests (§32-33, protocol in `docs/decoder_crossbody_protocol.md`).** Both fail:
   on Wuji (unseen by our decoders, seen by ψ) the decoder does not beat no movement; across bodies about 1% succeed.
5. **Direct decoder with an auxiliary contact head (§34-36)** and an **independent review (§37)**.
6. **Surface-token fix (§38)**: surface tokens are now chosen in one hand frame (`surface_sampling: hand_frame`) instead
   of per link, which had concentrated them on the palm. Small gains on source hands; it does not explain Wuji.
7. **Wuji diagnosis (§39-40)**: the error is in the wrist path (wrist corrections uncorrelated with the needed ones);
   moving Wuji's base origin changes predictions by up to ~2.3 mm but does not restore it.
8. **J1 (§41-42)**: one jointly trained decoder, frozen ψ -> contact-set predictor -> raw-contact GET, one checkpoint.

| J1 vs two-stage, fresh goals (contact mm / success) | Allegro | Barrett | LEAP | held-out LEAP |
| --- | --- | --- | --- | --- |
| two-stage (frozen translator + raw-contact GET) | 6.34 / 67.4 % | 6.53 / 62.8 % | 8.31 / 40.5 % | 8.89 / 34.4 % |
| J1 | 6.31 / 66.6 % | 6.31 / 68.4 % | 8.30 / 41.3 % | 8.85 / 31.2 % |
| no movement | 11.55 | 14.45 | 13.93 | 14.98 |
| raw-contact decoder given the true contacts | 5.71 | 5.60 | 7.90 | 8.28 |

J1 matches the two-stage system on source hands (Barrett slightly better, the rest within noise), and its internal
contact stage is as accurate as the standalone translator. Wuji and cross-body remain failures. Single seed throughout.

What the decoder is: only the decoder is trained from scratch; ψ is KITE's, frozen. The C3 failure is for this
configuration and data size, not a claim about scratch decoders in general.
