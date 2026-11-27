# Two evaluation-only checks: why body variety helps D1 but not CF, and seed ensembles (2026-11-25, user go-ahead)

No training. Readouts in configs/decoder/dlevels/cf2_readout/ (variety_gain.txt / .json, seed_ensemble.txt).

## 1. Body variety: where D1 gains and CF doesn't (scripts/decoder/benchmark/variety_gain_analysis.py)

Matched comparisons only: D1-bv vs D1-long and CF-bv vs CF2 (both 100k steps; only the training bodies differ).

**Correction to earlier summaries.** "Body variety improved D1 on every family" compared D1-bv with the frozen 50k D1, which
mixes in training length. Against its matched 100k control the picture is:

| family | D1-bv − D1-long (corr mm, s0/s1/s2) | CF-bv − CF2 |
|---|---|---|
| source Allegro / Barrett / LEAP | −0.14 / −0.07 / −0.11 mean (1/3 significant each) | +0.05 / +0.11 / +0.11 |
| held-out LEAP (4 hands) | −0.07 (1/3) | −0.02 |
| held-out LEAP (105 hands) | **−0.39 / −0.03 / −0.01** (seed 0 only; D1-long seed 0 is its weakest seed) | −0.11 / −0.17 / −0.02 |
| **Wuji** | **−0.82 (better 3/3)** | +0.28 (worse 2/3) |
| **Shadow** | **−1.41 (better 2/3)** | +0.45 (worse 2/3) |

- **On held-out LEAP, variety helps D1 and CF about equally and only a little** (mean over seeds −0.15 vs −0.10 mm on the 105
  hands; both seed-variable). Per-hand gains do not track finger count, joint count, topology distance to the training set, palm
  size, origin distance or frame tilt (|Spearman| ≤ 0.24 in every seed), and D1's and CF's per-hand gains are uncorrelated
  (r −0.04 to +0.13). The only visible subgroup: the 7 two-finger hands, where D1 gains −0.19 and CF −0.01. Held-out LEAP's
  CF-vs-D1 gap therefore mostly predates body variety (CF2 − D1-long +0.11 → CF-bv − D1-bv +0.16).
- **The real divergence is on unseen families.** Variety makes D1 much better on Wuji and Shadow but makes CF slightly worse there.
- **Interpretation (not yet tested):** variety and CF are partly substitutes. Many bodies with their own native frames weaken
  D1's reliance on frame conventions (the W2-a relabelling test showed body variety halving D1's frame dependence, G1 → D1),
  which is exactly what CF provides by construction. CF already has that, so extra bodies add little for it on unseen
  families, and with the same record budget spread over 414 instead of 48 LEAP hands it fits each family slightly less well
  (source +0.05 to +0.11 mm). A cheap test: run the relabelling test on D1-bv vs D1-long; if D1-bv is less frame-dependent,
  the substitution reading is supported.

### 1b. Test of the substitution reading: relabelling test on D1-bv vs D1-long (user go-ahead; rule written before running)

`frame_relabel_eval.py` (W2-a): every robot's base frame re-expressed by each of the 24 axis-aligned rotations, physical task
unchanged; systems d1long_s0–2 and d1bv_s0–2 (transfer_v1_bv.json); output `${GETKITE_RUNS}/d1bv_frame_relabel`.
- **Measure:** frame dependence = drop in wrist translation R²_id from the native frame (rotation 0) to the median over the 23
  relabellings, per seed, on source Allegro / Barrett / LEAP (mean of the three) and on held-out LEAP.
- **Supported** if D1-bv's drop is smaller than D1-long's in 3/3 seeds on source. **Not supported** if equal or larger in ≥ 2/3
  seeds. Wuji / Shadow reported, not used for the decision. Contact error under relabelling reported alongside.

## 2. Seed ensemble (job 25549; scripts/decoder/benchmark/seed_ensemble_eval.py; readout cf2_readout/seed_ensemble.txt)

Average of the three seeds' predictions (mean joints and wrist translation, chordal mean of wrist rotations), per arm. All
three seeds are always used; nothing is selected.

**The ensemble beats the individual seeds almost everywhere** (significantly better than the seed in 102 of 108 arm × pool ×
seed comparisons; 5 not significant; worse in 1: CF-J vs its own best Shadow seed, seed 2). Contact error, ensemble vs the mean of the seeds:
source −0.30 to −0.47 mm, held-out LEAP (105 hands) −0.32 to −0.47, Wuji −0.39 to −0.71, Shadow −0.66 to −1.27. Normal-
compatible success rises by 2–7 points (source, held-out) and up to 9 points (Wuji, Shadow).

| ensemble corr mm (wrist mm) | D1 | D1-long | CF2 | D1-bv | CF-bv | CF-J | no movement |
|---|---|---|---|---|---|---|---|
| Allegro | 6.03 | 5.72 | **5.56** | 5.60 | 5.67 | 5.68 | 11.55 |
| held-out LEAP (105) | 8.77 | 8.34 | 8.49 | **8.24** | 8.43 | 8.46 | 14.84 |
| Wuji | 9.03 | 8.73 | 8.12 | 8.02 | 8.40 | **8.02** | 14.19 |
| Shadow | 9.75 (7.90) | 9.96 (7.88) | **8.55 (7.33)** | 8.97 (7.66) | 8.87 (7.43) | 8.80 (7.36) | 13.97 (7.96) |

- **Shadow's wrist now beats no movement for every ensemble** (7.33–7.90 vs 7.96 mm); as single seeds, only the CF models
  did, and D1-long was worse than no movement.
- Ensemble vs ensemble (paired, 95% CI): held-out LEAP D1-bv better than CF2 by 0.26 mm [0.23, 0.29]; Shadow CF2 better
  than D1-bv by 0.42 mm [0.13, 0.70] (wrist 0.33); Wuji and source: ties. The single-seed picture holds: D1-bv leads on
  LEAP-like hands, CF2 on Shadow.
- Deliverable: an ensemble of the three trained seeds costs 3× inference and no training, and is the largest single
  improvement found today (0.3–1.3 mm). Recommended for whichever model is delivered.

### 1b result (job 25623; readout cf2_readout/d1bv_relabel.txt)

World-geometry check passes (6e-8 m). Drop in source wrist-translation R²_id from the native frame to the median relabelling:

| seed | D1-long | D1-bv |
|---|---|---|
| 0 | 1.19 | 0.68 |
| 1 | 0.64 | 0.78 |
| 2 | 0.91 | 0.69 |

**Rule: NOT SUPPORTED** (D1-bv's drop is smaller in 2/3 seeds, not 3/3). On average variety reduces frame dependence
(0.92 → 0.72) and makes it more uniform across seeds, but both models stay strongly frame-dependent: relabelled, source contact
error rises from ~6.7 mm to 11.8–15.6 mm (no movement is ~13.3). So D1-bv's gains on Wuji and Shadow are **not** explained by
it becoming frame-independent; the substitution reading is not supported. Why variety helps D1 on unseen families but not CF
remains open.

## 3. Five-seed ensemble test (user go-ahead 2026-11-25; rule written before training)

Seeds 3 and 4 for D1-bv and CF2 (`d1_bv_long_hf_s3/_s4`, `cf2_long_hf_s3/_s4`; only the seed changes), then an evaluation of the
5-seed ensemble (s0–s4) against the 3-seed ensemble (s0–s2) per arm, paired over records, on source val, the 105 held-out
LEAP hands, Wuji and Shadow.
- **Use 5 seeds at inference** if, for an arm, the 5-seed ensemble has significantly lower contact error than the 3-seed one
  on the source pools and the 105 held-out hands, and is not significantly worse on Wuji or Shadow. Otherwise 3 seeds suffice
  (cost 3× instead of 5×).
- Also reported: the new seeds' single-seed results next to s0–s2 (do the 3/3 conclusions still hold with 5 seeds?), and the
  3-seed ensembles of other seed triples (how much the ensemble depends on which seeds).

## 4. Mixed ensemble D1-bv + CF2 (6 checkpoints) and inference timing (job 26214; readout cf2_readout/seed_ensemble_mix.txt)

| ensemble corr mm (wrist mm) | CF2 (3) | D1-bv (3) | D1-bv + CF2 (6) | no movement |
|---|---|---|---|---|
| Allegro | 5.56 | 5.60 | **5.47** (4.59) | 11.55 |
| Barrett | 5.44 | 5.46 | **5.31** | 14.45 |
| LEAP | 7.70 | 7.67 | **7.49** | 13.93 |
| held-out LEAP (105) | 8.49 | 8.24 | **8.16** (6.03) | 14.84 |
| Wuji | 8.12 | 8.02 | **7.82** (6.67) | 14.19 |
| Shadow | 8.55 (7.33) | 8.97 (7.66) | **8.13 (7.23)** | 13.97 (7.96) |

The 6-checkpoint mix is significantly better than **both** 3-seed ensembles on **every** pool (paired 95% CIs exclude 0; e.g.
held-out LEAP −0.07 vs D1-bv and −0.33 vs CF2; Shadow −0.42 vs CF2 and −0.84 vs D1-bv). It keeps D1-bv's LEAP strength and CF2's
Shadow strength at once. **Caveat:** it has twice as many members as each 3-seed ensemble, so part of the gain may come from
size rather than from mixing two model types; the 5-seed test (§3) measures the size effect for comparison.

**Timing** (RTX 2080 Ti, decoder only, ψ excluded, members run one after another): 1 checkpoint 20.9 ms per call at batch 1
(30.8 ms at batch 256); 3 checkpoints 62.3 ms (91.3); 6 checkpoints 121.2 ms (179.4). Cost is linear in members (~21 ms each at
batch 1); members could run in parallel or as one batched call to reduce latency.

### §3 result: five seeds vs three (2026-11-26 02:15 EST; job 32450; ${GETKITE_RUNS}/seed5_ensemble_v1/seed5_ensemble.json)

5-seed minus 3-seed (s0–s2) ensemble, contact error mm, paired bootstrap:
| Pool | CF2 | D1-bv |
|---|---|---|
| Allegro | −0.07 (sig) | −0.06 (sig) |
| Barrett | −0.07 (sig) | −0.09 (sig) |
| LEAP (source) | −0.06 (sig) | −0.05 (n.s., CI [−0.09, +0.00]) |
| held-out LEAP (105) | −0.10 (sig) | −0.06 (sig) |
| Wuji | +0.02 (n.s.) | **+0.10 (sig worse)** |
| Shadow | −0.08 (n.s.) | −0.24 (sig) |
**Predeclared rule: CF2 passes (use 5 seeds), D1-bv fails** (source LEAP not significant; Wuji significantly worse). Gains from 3 → 5
are ~0.05–0.10 mm on source and held-out LEAP, against 0.3–1.3 mm from 1 → 3: diminishing returns. Caveat: s0–s2 is the best of
the ten D1-bv triples on Wuji (triple range 8.02–8.43), so D1-bv's Wuji "loss" is partly comparing against a lucky triple.
Mixed ensemble vs size (unpaired, same records and start draws, no CI): mixed D1-bv+CF2 (6) Wuji 7.82 / Shadow 8.13 / held-out 8.16
vs best 5-seed Wuji 8.13 / Shadow 8.47 / held-out 8.18. The 3 → 5 gain is small, so most of the mix's Wuji/Shadow advantage is not
from member count; on held-out LEAP the mix ≈ D1-bv 5-seed.
