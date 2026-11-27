# Two evaluation-only checks: multi-pass inference (item 1) and base-origin shift (item 2) (2026-11-25)

Status: rules written 2026-11-25 before any result; user go-ahead ("Proceed") for these two evaluation-only checks.
No training, no new arms, no change to any model, no selection on Wuji / Shadow / held-out sets. CF-J is not touched.
Systems: frozen D1 (`d1_prim50_hf*`), D1-long (`d1_long_hf*`), CF2 (`cf2_long_hf*`), D1-bv (`d1_bv_long_hf*`), seeds 0/1/2,
paired by seed. CF-J can be added later with the same scripts once its own evaluation is complete (not part of this run).

## Item 1: multi-pass inference ("take a second look")

**Question.** Translation under-correction is still large (slope 0.55–0.62 on source Allegro, 0.37–0.44 on LEAP, 0.19–0.27 on
Shadow; transfer_benchmark_v1_bv). If the decoder is re-run from its own output, does the remaining error shrink?

**Procedure.** `scripts/decoder/benchmark/multipass_eval.py`. Pass 1 = the standard benchmark prediction. Pass k+1 uses pass
k's output (joints clamped to limits, wrist pose) as the new start; the request z and the robot are unchanged. Every pass is
scored with `transfer_benchmark.score` against the original goal and start, in the native bank. Pools: source val
(Allegro / Barrett / LEAP), held-out LEAP (all 105 val-split hands, as transfer_benchmark_v1_leapval105), Wuji, Shadow.
Passes 1, 2, 3. Pass 1 must reproduce transfer_benchmark_v1_leapval105 exactly.

**Predeclared rules.**
- **Number of passes K is chosen on source val only.** K = the largest k ∈ {1, 2, 3} such that pass k has lower corr_mm than
  pass k−1, significantly (paired bootstrap over records, 95% CI < 0), on all three source families in 3/3 seeds, for that
  system. Otherwise K = 1.
- **Multi-pass "helps" a system** if K ≥ 2.
- Then reported, not used for any choice: held-out LEAP, Wuji, Shadow at pass K vs pass 1 (corr_mm, compat_success, wrist_mm,
  wrist_deg, trans_R2_id), per seed; transfer beyond no movement at pass K.
- **Warning sign predeclared:** if pass K is significantly worse than pass 1 on Wuji or Shadow corr_mm in ≥ 2/3 seeds, report
  that multi-pass is harmful on unseen hands and must not be used for them, whatever it does on source.

## Item 2: base-origin shift ("does moving the reference point confuse it?")

**Question.** The wrist-frame decoder sees every hand relative to its base origin. CF canonicalises only the axes. Is accuracy
sensitive to where the origin sits, with the physical hand unchanged?

**Procedure.** `scripts/decoder/benchmark/origin_shift_eval.py`. For an offset o (in the base frame), every robot's
description is re-expressed with its origin moved to o: base-link template points p → p − o, origins of joints attached to the
base T → T − o; the start / goal wrist translations become t + R·o (world geometry identical; checked numerically as in the
relabelling test). The decoder's output translation is mapped back (t_out − R_out·o) and scored in the original bank. Offsets:
1, 2, 5 cm along ±x, ±y, ±z of the native base frame (18 conditions + no shift). Pools: source val, held-out LEAP (4 hands,
standard benchmark), Wuji, Shadow. The no-shift condition must reproduce the standard benchmark exactly. For CF2 the canonical
frame is recomputed from the shifted description (its rotation is translation-invariant; checked).

**Predeclared rules.**
- **"Sensitive"** if a 2 cm shift raises source corr_mm by ≥ 0.3 mm (mean over the 6 directions) in 3/3 seeds, for any
  system. Reported per family and per direction, with the 1 and 5 cm curve.
- If sensitive: item 3 (a canonical origin, code + retraining) becomes a candidate, **separate go-ahead needed**. If not
  sensitive: item 3 is dropped.
- Context only (no decision): how far each hand's palm centre sits from its native origin, so a reader can tell whether 2 cm
  is a realistic difference between hands.

## Cost

Two evaluation jobs on default_partition (standard constraint and exclude list), each well under 2 h.

## Results (2026-11-25, jobs 22722 / 22723; readout configs/decoder/dlevels/cf2_readout/multipass_origin.txt)

Reproduction: pass 1 and the no-shift condition reproduce the benchmarks exactly. Origin-shift checks: world points identical
to 6e-8 m, surface sampling identical, CF rotation unchanged (8e-16).

**Item 1, multi-pass: does not help (K = 1 for all four systems).** A second pass helps only Barrett (significant in 1–3/3
seeds per system), is neutral on Allegro, and is significantly worse on source LEAP, held-out LEAP (105 hands), Wuji and
Shadow in 3/3 seeds for every system; pass 3 is worse everywhere. Example D1 seed 0: Shadow 10.46 → 12.35 → 14.12 mm (no
movement 13.97); wrist error rises with each pass (Shadow 8.03 → 9.67 mm). Re-running from the model's own output pushes the
start outside the training start distribution and the decoder over-corrects. Not to be used, in particular not on unseen hands.

**Item 2, base-origin shift: SENSITIVE, far above the threshold.** Source contact error increase, mean of the three source
families, 2 cm shift: D1 +2.26 / +2.08 / +2.36 mm, D1-long +4.01 / +2.22 / +3.35, CF2 +1.87 / +2.10 / +2.05, D1-bv +2.36 /
+2.98 / +2.26 (threshold 0.3 mm, 3/3 seeds). A 1 cm shift already costs ~0.6–1.7 mm on source and held-out LEAP; 5 cm costs
3.4–9.4 mm. Wuji and Shadow are less sensitive (+0.1 to +0.6 mm at 2 cm), consistent with the decoder already using the wrist
less on them. CF2 is as sensitive as D1 (it canonicalises axes only).
Context: the palm-point centroid sits 3.8–4.1 cm from the native origin on Allegro, Wuji and Shadow and ~6.0 cm (median) on
LEAP variants, so origin placement differs between hand families by centimetres, the range where the effect is large.
Caveat: KITE's start noise is a rotation about the native origin, so with a shifted origin the required correction also has a
different statistical structure (an extra ~1–2 mm of translation coupled to the rotation); part of the measured cost is that
shift, which any origin convention would face. A canonical origin must therefore be judged on unseen hands, not on this test.
**Per the rule, item 3 (canonical origin, code + retraining) becomes a candidate; it needs its own design and go-ahead.**
