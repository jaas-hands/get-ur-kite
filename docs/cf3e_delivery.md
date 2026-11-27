# CF-3e: the delivered decoder (frozen 2026-11-27)

CF-3e maps a request (contact points with normals on a robot hand, plus the hand's start pose) to a hand configuration: finger
joints and wrist pose that place the hand's surface on those contacts. It is a **J1 decoder**: KITE's released contact encoder ψ
(frozen, unmodified, KITE commit `1361595`, `release.pt`) → a learned contact-set predictor → a GET decoder over the hand's
kinematic graph. One model serves every hand; the hand is given by its description files only.

**Delivery model: seed 0** (`${GETKITE_RUNS}/cf3e_long_hf/best.pt`, fetched with `bash scripts/fetch_checkpoints.sh cf3e_long_hf`). Seeds 1 and 2 are included for evaluation and are judged
separately; seeds are never averaged or pooled.

## What makes up CF-3e

| Ingredient | What it does |
|---|---|
| Canonical palm frame (rule E) | Every hand is re-expressed in a palm frame computed from its own description, so the wrist convention is the same across families. Predictions are mapped back to the hand's native frame. |
| Distillation from 6 teachers (α = 0.5) | Training loss = 0.5 × ground-truth pose + 0.5 × the mean pose of D1-bv seeds 0–2 and CF2 seeds 0–2 (mean joints, mean translation, chordal-mean rotation). Teachers are used in training only. |
| 800 generated bodies × 25 records | Procedurally generated primitive hands (`get-ur-kite-dataset`, `kite_data/build_prim.py`) as a fourth training family, carrying teacher targets. |
| Data mix | Allegro, Barrett, LEAP (48 hands × 417 records), generated bodies: 25 % of every batch each (family-balanced sampler). |
| Model | d_model 128, 4 layers; batch 256; 100k steps; cosine LR to 0 with 500 warm-up steps; best checkpoint chosen on source validation only (macro success, then macro contact error). |

## Results (paired bootstrap over records, per seed; seeds 0 / 1 / 2)

Contact error with correspondences (`corr_mm`, lower is better) and its success rate (10 mm / 15°):

| Pool | corr_mm | success | vs CF2 (corr_mm, all significant) |
|---|---|---|---|
| Allegro (source val) | 5.72 / 5.75 / 5.72 | 74 / 74 / 71 % | −0.16 / −0.21 / −0.18 |
| Barrett (source val) | 5.50 / 5.63 / 5.55 | 76 / 73 / 75 % | −0.42 / −0.23 / −0.25 |
| LEAP (source val) | 7.84 / 7.96 / 7.85 | 46 / 45 / 45 % | −0.37 / −0.17 (n.s.) / −0.21 |
| Held-out LEAP, 105 hands | 8.61 / 8.70 / 8.50 | 36 / 35 / 38 % | −0.42 / −0.23 / −0.33 |
| Wuji (family not trained on) | 7.34 / 7.64 / 7.88 | 59 / 56 / 53 % | −0.89 / −1.14 / −1.12 |
| Shadow (family not trained on) | 7.92 / 8.35 / 8.74 | 51 / 45 / 44 % | −1.05 / −0.86 / −0.73 |

No movement (the start pose): 14.8 mm on held-out LEAP, 14.2 mm on Wuji, 14.0 mm on Shadow. CF-3e is also better than CF2 on all
7 external hands (Ability, Inspire, XHand, Schunk, MetaHand, Sharpa Wave, Tetheria Aero; Schunk in 2 of 3 seeds). Full numbers:
`checkpoints/cf3e/benchmark/` and `docs/cf3_distillation_proposal.md` ("CF-3e result").

## Scope and known limits

- Simulation only; ψ frozen. Wuji is unseen by our decoder but seen by KITE's encoder ψ.
- The 9 unseen-family hands (7 external, Wuji, Shadow) were used to choose between recipes, so treat them as **new-family
  validation**, not untouched test data.
- Layout overlap: 14 of the 800 generated training bodies share the exact finger-joint layout of Ability, Inspire, XHand, MetaHand
  or Aero (none for Wuji, Shadow, Sharpa, Schunk). The bodies' shapes are primitive, so this is layout-level, not shape-level.
- Rule-E frame flips: about 2.5 % of LEAP variants and MetaHand get a palm frame turned by about 90°; those hands are about 2 mm
  behind the native-frame D1 model. An optional per-hand frame choice at inference (32 sample requests per new hand) fixes most of
  MetaHand and half of the flip gap; it is not part of the delivered model.
- Base-origin wrist error depends on each hand's origin convention; also report wrist error at a palm point when comparing
  families (`scripts/decoder/benchmark/origin_measure_eval.py`).

## Using it

```bash
bash scripts/fetch_checkpoints.sh cf3e_long_hf   # downloads ${GETKITE_RUNS}/cf3e_long_hf/best.pt from Google Drive, checks SHA256
source slurm/env.sh   # sets GETKITE_* paths; machine overrides in slurm/env.local.sh (see slurm/env.local.example.sh)
PYTHONPATH=src python scripts/decoder/predict_j1.py --ckpt $GETKITE_RUNS/cf3e_long_hf/best.pt \
    --data $GETKITE_DATASET --regime kite_code --splits val --out predictions/cf3e_s0
```

Predictions start from the start pose stored in each data file and are written per hand (joints in dataset order, wrist position
and w-x-y-z quaternion in the world frame) with our per-example metrics; see the script's header. To reproduce a benchmark number
exactly, use `--check-config configs/decoder/scratch/c3_psi50k_10x_wrist_leapval105.json --check-pool leap_heldout` (seed 0 gives
8.6146 mm on the 105-hand pool, identical to the benchmark). Checkpoints are not in git (`*.pt` is gitignored). `checkpoints/manifest.tsv` lists each one (run folder, Google Drive file
id, SHA256); `bash scripts/fetch_checkpoints.sh [pattern]` downloads them to `${GETKITE_RUNS}/<run>/best.pt`, where every
config and benchmark expects them, and verifies each hash. `cf3e_long_hf*` = CF-3e seeds 0/1/2; `d1_bv_long_hf*` and
`cf2_long_hf*` = the six teachers (needed only to retrain CF-3e). `checkpoints/cf3e/s{0,1,2}/` holds each seed's training
config, provenance (psi / data hashes), canonical frames and distillation settings; `checkpoints/cf3e/benchmark/` the
benchmark results.

## Reproducing it from scratch

1. **Data** (`get-ur-kite-dataset`): LEAP variants and kite_code records (`kite_data.generate`), KITE-family records
   (`scripts/kite_family/gen_kite_family.py`, `scripts/kite_family/gen_10x.sbatch`), generated bodies
   (`python -m kite_data.build_prim hands --n 900`, then `records`), then the template repair and checks in this repo
   (`scripts/decoder/dlevels/repair_prim_fps.py`, `validate_prim.py`, `validate_leap_bv.py`).
2. **Teachers** (or fetch them: `bash scripts/fetch_checkpoints.sh 'd1_bv_*' 'cf2_*'`): `configs/decoder/scratch/d1_bv_long_hf{,_s1,_s2}.json` and `cf2_long_hf{,_s1,_s2}.json` with
   `sbatch -J <name> -t 1-00:00:00 slurm/joint.sbatch <config>` (about 8 h each on one mid-range GPU).
3. **CF-3e**: `configs/decoder/scratch/cf3e_long_hf{,_s1,_s2}.json` the same way (the config lists the teacher checkpoints).
4. **Evaluation**: `slurm/benchmark.sbatch configs/decoder/benchmark/transfer_v1_cf3e.json` (+ `_leapval105`, `_leap360`),
   `slurm/external_hands_eval.sbatch`, and the gate `scripts/decoder/benchmark/cf3d_gate.py cf3e`.

Pass `-t` explicitly to `joint.sbatch`: its default wall time is 2 h, and a job that hits it stops without `DONE` (it resumes
from `last.pt` when resubmitted).

Experimental work after CF-3e (CF-3f arms, matched teachers, longer and larger runs, diagnostics) lives on the
`getting-kite-universal` branch.
