#!/bin/bash
# Full readout for one extra CF-3f arm (docs/cf3f_extra_arms.md), as one job on one GPU, from the repo root:
#   bash scripts/decoder/benchmark/run_cf3f_extra_eval.sh <big300k|contact> [with360]
# 1. transfer benchmark: the arm AND CF-3e (plus CF2, D1-bv) scored in the same run (standard pools, 105 held-out LEAP;
#    the 360 pool too with "with360", reported only, needs >= 48 GB GPU memory);
# 2. external hands (7), arm and CF-3e;
# 3. the predeclared gate (cf3f_gate.py: Q1-Q5, plus Q6 for contact).
# Needs under $GETKITE_RUNS: best.pt of the arm, cf3e_long_hf, cf2_long_hf, d1_bv_long_hf (seeds 0-2), set_bridge_zworld_c3_20k_b;
# and log.jsonl + DONE of the arm and of cf3e_long_hf (gate Q4). Writes everything under $GETKITE_RUNS.
set -euo pipefail
ARM=${1:?arm: big300k or contact}; S=cf3f$ARM
for r in "cf3f_${ARM}_hf" cf3e_long_hf; do for sfx in "" _s1 _s2; do
  for f in best.pt log.jsonl DONE; do [ -f "$GETKITE_RUNS/$r$sfx/$f" ] || { echo "missing $r$sfx/$f" >&2; exit 2; }; done
done; done
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}
for c in "" _leapval105 ${2:+_leap360}; do
  echo "== benchmark transfer_v1_$S$c $(date -u +%FT%TZ)"
  python -u scripts/decoder/benchmark/transfer_benchmark.py --config "configs/decoder/benchmark/transfer_v1_$S$c.json"
done
echo "== external hands $(date -u +%FT%TZ)"
python -u scripts/decoder/benchmark/external_hands_eval.py --out "\${GETKITE_RUNS}/external_hands_$S" --arms cf3e "$S"
echo "== gate $(date -u +%FT%TZ)"
python -u scripts/decoder/benchmark/cf3f_gate.py "$ARM"
echo "== done $(date -u +%FT%TZ)"
