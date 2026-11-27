# A steadier canonical axis rule (2026-11-25; CPU-only check, no training)

Status: gate written 2026-11-25 before any candidate was computed; user go-ahead for the CPU check. Training with a new rule
is a later step and needs its own go-ahead.

## Problem

CF's rule E picks the distal axis d as the palm principal axis best aligned with the mean finger direction f. On LEAP palms
the two largest palm eigenvalues are close (ratio ~0.75), so when f lies between those two axes the choice flips by ~90°.
Relative to their own topology, 1.9% of train + val LEAP hands are flipped by > 30° and ~10% tilted by > 10° (measured in
the result below; an earlier "~7% flipped" estimate was wrong). CF is causally sensitive to frame error (Step 0, docs/cf_frame_jitter_proposal.md), and more
training bodies bring more flipped hands, which is the likely reason body variety did not help CF.

## Candidate (one, to avoid tuning)

**Rule G:** keep the part of rule E that is well determined, drop the ambiguous snap.
- n = the palm's **smallest**-variance principal axis (the palm plate's normal; that eigenvalue is well separated from the
  other two on LEAP, ratio ~0.35), signed by the curl direction (as rule E signs n).
- d = the mean finger direction f projected onto the palm plane (f − (f·n) n, normalised); no snapping to a palm axis.
- a = d × n; Q = [n a d].
Same inputs as rule E (description + template), same invariance argument (all vectors rotate with the description).

## Gate (predeclared; rule G is adopted only if all hold)

`scripts/decoder/dlevels/steady_axis_check.py`, on the 900 generated bodies, the 1,071 train + val LEAP hands (test splits
not read), Allegro, Barrett, Wuji, Shadow. Rule E is recomputed with the same code as the reference.
1. **Generated bodies:** ≥ 90% within 30° of their native frame (rule E: 91.7%). Report median / p90.
2. **LEAP flips:** hands > 30° from their topology's median frame ≤ 1% (rule E ~7%; recomputed here), and the median angle to
   the topology median not larger than rule E's.
3. **Named hands:** d points from palm to fingers and n to the palm side (checked as for rule E), reported per hand.
4. **Invariance:** rule G's frame of a relabelled description is the same physical frame (exact up to float error).

If the gate fails: report, do not tune further on these numbers, and come back to the user. If it passes: a training arm
(CF with rule G, then the combined run with canonical origin and B-scale body variety) is proposed separately.

## Result (2026-11-25, job 25022, CPU; configs/decoder/dlevels/steady_axis_check.json)

**Gate NOT passed (criterion 1 fails). Rule G is not adopted; no further tuning on these numbers.** Rule E recomputed here
matches the library exactly (max difference 0).

| | rule E (current) | rule G |
|---|---|---|
| generated bodies within 30° of native frame | **91.9%** (median 2.7°, p90 22.6°) | **88.6%** (median 13.4°, p90 31.0°) — fails ≥ 90% |
| LEAP hands > 30° from their topology's frame | 1.9% (20 of 1,071) | **0%** |
| LEAP median / p90 angle to topology frame | 3.7° / 9.9° | **1.0° / 1.7°** |
| LEAP median angle to the family-mean frame | 3.7° | 8.7° (9.8% > 30°) |
| named hands (d to fingers, n to palm side) | ok | ok, but Barrett differs from E by 90°, Allegro by 41° |
| relabelling invariance | exact | exact (1e-15) |

Reading: rule G removes the within-topology flips and makes a hand's frame much steadier across geometry draws, but the
projected finger direction varies with finger layout, so different topologies (and generated bodies with uneven finger
placement) get frames that disagree by more. Barrett's palm is nearly isotropic (eigenvalues 0.48 / 0.57 / 0.85 × 10⁻³), so
its thinnest axis is itself ambiguous.

**Correction to earlier statements:** the "~7% of LEAP hands flipped" figure (from the CF-O Step 1 origin check) was wrong.
That check flagged hands whose canonical origin coordinates moved > 1 cm, which happens for frame tilts of roughly ≥ 10°
(the origin sits ~6 cm from the native origin), not only for flips. Measured directly here, rule E puts **1.9%** of
train + val LEAP hands > 30° from their topology's frame; about 10% are > 10° off (p90 9.9°).

Back to the user: no candidate passed. Options are a different rule (would need a new predeclared gate) or dropping the
axis-rule line and relying on CF-O's result.

**User decision (2026-11-25):** axis-rule line dropped; rely on CF-O's result. The combined larger run, if proposed later, would
use rule E axes + canonical origin.

## Second candidate: knuckle-line rule K (2026-11-25; user go-ahead for a CPU check in parallel with CF-O)

Gate written before rule K was computed on any hand. To avoid tuning against numbers already seen, the generated-body
criterion uses **draw B** (`prim_b`, 900 bodies, generator seed 20261121), which no axis-rule check has used.

**Rule K** (description + template only):
- finger roots = the origins of each finger's first movable joint at q0 (the same fingers rule E uses);
- a (across) = the principal direction of the finger-root positions (for 2 fingers, the line through both roots);
- d (distal) = the mean finger direction f with its component along a removed, normalised;
- n = a × d, with the sign of a chosen so that n points to the curl side (as rule E signs n); Q = [n a d].
- Hands with a single finger have no knuckle line: they fall back to rule E (counted and reported).

**Gate (rule K adopted only if all hold):**
1. Draw-B generated bodies: ≥ 90% within 30° of their native frame (rule E recomputed on draw B as the reference).
2. LEAP (1,071 train + val hands): > 30° from their topology's frame ≤ 1%, and median angle to the topology frame ≤ rule E's.
3. LEAP agreement across topologies: median angle to the family-mean frame ≤ rule E's (the criterion rule G would have failed).
4. Named hands: d towards the fingers and n to the palm side.
5. Relabelling invariance exact.
Also reported (no decision): draw-A generated bodies, the single-finger fallback count. If it fails: report, no tuning, back to
the user.

### Rule K result (2026-11-25, job 25147, CPU; configs/decoder/dlevels/steady_axis_check_K.json)

**Gate NOT passed (criteria 1 and 3 fail badly). Rule K is not adopted; no tuning.**

| | rule E (current) | rule K |
|---|---|---|
| draw-B generated bodies within 30° (gate) | 91.4% (median 2.7°) | **19.3%** (median 47.5°) |
| draw-A generated bodies within 30° (reported) | 91.9% | 23.4% (median 45.6°) |
| LEAP > 30° from topology frame | 1.9% | 0% |
| LEAP median angle to topology frame | 3.7° | 0.8° |
| LEAP median angle to family-mean frame (gate) | 3.7° | **13.9°** (47% > 30°) |
| named hands vs rule E | — | Allegro 45°, Barrett 90°, Wuji 47°, Shadow 84° |

Reading: the finger roots do not lie on one knuckle line. The thumb (and, on generated bodies, side-mounted fingers) sits well
off the row, so the roots' principal direction is often dominated by the thumb offset and points nowhere consistent. Like
rule G, rule K makes each topology very self-consistent (0% flips) but different layouts disagree far more. Both candidates
show the same trade-off: rule E's palm-shape axes are the only ones that agree across layouts. **Recommendation: keep rule E;
close the axis-rule line.**
