# CF-3g: split-trained teachers (proposal, 2026-11-28)

Status (2026-11-28 ~01:05): revised after Aryan's review (commit 783e926); 9 fold teachers relaunched from 783e926 (a first
version that also dropped LEAP-hand rows ran ~20 min and was cancelled; kept as `_superseded_cf3g_v0_leapcut__*`). Students not
launched. Go/no-go runs from commit eb43fc8 (adds the expected-biased label below). Matched result and the wider plan:
`docs/cf3f_proposal.md` ("Arm match result"), `docs/plan_2026-11-28.md`. Every launch needs the user's go-ahead (commit to
`getting-kite-universal`, clean merge into `main`, launch from that commit). The rules below were fixed with Aryan on
2026-11-28, before any fold teacher trained.

## Why

The matched-teacher arm (CF-3f, D1-bv-only targets on the 750 generated bodies CF2 never saw) trailed CF-3e on held-out
generated bodies by about 0.24-0.45 mm late in training. Two evaluation-only checks explain it: the CF-3e teachers memorised
the records they trained on, so on those records their targets are near-copies of the label and add little beyond the
ground-truth half of the loss.

| Check (corr_mm of the teacher-mean target vs ground truth) | Result |
|---|---|
| Generated bodies: D1-bv on its own records vs same bodies' unseen records | +0.42 [0.29, 0.55]; CF2 (never saw them) +0.05 [-0.08, 0.19]; difference in differences +0.37 [0.29, 0.44] |
| LEAP (48 CF-3e hands): CF2, rows 48-96 (trained on) vs rows 900-942 (no teacher) | +0.67 [0.48, 0.85]; D1-bv (saw neither) -0.05 [-0.24, 0.15]; difference in differences +0.71 [0.61, 0.81] |
| Barrett: 500 training vs 500 validation records (no teacher-level control) | +0.42 [0.13, 0.72] |
| Allegro: same | +0.13 [-0.21, 0.46], not significant |

Scripts: `scripts/decoder/benchmark/teacher_memorization_eval.py` and `teacher_memorization_source_eval.py` (results in
`$GETKITE_RUNS/teacher_memorization_v1/`, `teacher_memorization_source_v1/`).

What CF-3e's student sees today: Allegro, Barrett and LEAP rows 0-48 get no teacher that is fresh on the record; LEAP rows
48-417 get fresh D1-bv + memorised CF2; the 750 bodies get memorised D1-bv + fresh CF2.

## What changes

**Fold plan** (`configs/decoder/dlevels/cf3g_folds.json`, seeded, written once): the 800 generated training bodies and the 48
LEAP hands CF-3e trains on are each split into folds A / B / C (267 / 267 / 266 bodies, 16 / 16 / 16 hands).

**Stage 1 teachers, 9 runs** (`d1_bv_fold{A,B,C}_long_hf{,_s1,_s2}`): D1-bv with one change, the fold's generated bodies are
left out. Everything else as D1-bv (all 414 LEAP entries, 100k steps, seeds 0-2, selection on source validation). The fold plan's
LEAP folds are unused.

**Students, 3 seeds each**, CF-3e plus a per-hand teacher table (`distill_groups`; the original six stay the default):

| Arm | Generated body in fold k | LEAP rows 0-48 | LEAP rows 48-417 | Allegro, Barrett |
|---|---|---|---|---|
| `cf3g_split` | fold-k teachers (3) + CF2 (3) | original six | original six | original six |
| `cf3g_splitfresh` (Aryan's arm) | fold-k + CF2 | original six | original D1-bv only (record-fresh; memorised CF2 dropped) | original six |

Why not fold teachers on LEAP (Aryan's review): the original D1-bv trained only on rows 0-48 of the 48 hands, so it is already
record-fresh on rows 48-417; a per-hand fold teacher would never have seen the hand, a weaker target with no memorisation gain.

Everything else as CF-3e: data, alpha 0.5, model, 100k steps, seeds, selection on source validation. If the CF-3f long arm
passes its gate, the step count is revisited before the students launch.

## Rules fixed in advance

**Go/no-go on the teachers, before any student run** (`scripts/decoder/benchmark/cf3g_gonogo.py`; "teacher" = 3-seed mean
target; paired; bootstrap over bodies or hands). GO only if all hold:

1. On each fold's left-out generated bodies (excluding CF2's 50): fold teachers - CF2, 95 % CI upper bound < 0.
2. On the 50 held-out bodies P0800-P0849: mean over folds of (fold teachers - original D1-bv) <= +0.10 mm (point estimate;
   with 50 bodies the 95 % CI half-width is about 0.1-0.2 mm).

Amended 2026-11-28 with the user's approval, before any fold teacher finished: a LEAP check (fold teachers - original D1-bv on
rows 900-942) is reported, not decided on, on all 48 hands and on the 36 outside the teachers' checkpoint-selection set.

**Delivery:** a student replaces CF-3e seed 0 only if it passes the CF-3f gate (Q1-Q5) against CF-3e, same seed.

**Stage 2 decision** (9 CF2 fold teachers on LEAP): only if `cf3g_splitfresh` beats `cf3g_split` on the 105-hand LEAP pool,
lower in 3/3 seeds and significant in >= 2/3 (paired bootstrap, same seed). A failure of `cf3g_splitfresh` is weak evidence
(3 teachers instead of 6 on those rows).

## Checks before launch

`slurm/cf3g_checks.sbatch` (`scripts/decoder/dlevels/cf3g_checks.py`, then 300-step smoke runs of one fold teacher and one
student with stand-in teachers):

1. Each fold-teacher config drops exactly its fold, nothing else; folds cover all 800 bodies and 48 hands once.
2. / 3. The per-hand teacher table matches the design for both arms on CF-3e's bank.
4. Running each teacher only on its assigned examples gives the same targets as the dense weighted average (<= 1e-5).
5. Allegro / Barrett targets equal CF-3e's six-teacher targets (<= 1e-5).

With `distill_groups` unset (every existing config), training is unchanged.
