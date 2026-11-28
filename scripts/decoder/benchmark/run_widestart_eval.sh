#!/bin/bash
# Full readout for the wide-start arm (docs/wide_start_proposal.md), as one job on one GPU, from the repo root:
#   bash scripts/decoder/benchmark/run_widestart_eval.sh
# 1. transfer benchmark, the arm AND CF-3e scored in the same run: near-goal (KITE-noise) starts on the standard pools + Shadow
#    and the 105 held-out LEAP pool (gate Q1-Q5), then the same records with starts at 3x KITE noise (the predeclared gain W);
# 2. external hands (7), arm and CF-3e, near-goal starts (reported);
# 3. the predeclared gate (cf3f_gate.py widestart: Q1-Q5 with Q2 as non-inferiority, plus W).
# Needs under $GETKITE_RUNS: best.pt, log.jsonl and DONE of cf3f_widestart_hf{,_s1,_s2} and cf3e_long_hf{,_s1,_s2}.
set -euo pipefail
for r in cf3f_widestart_hf cf3e_long_hf; do for sfx in "" _s1 _s2; do
  for f in best.pt log.jsonl DONE; do [ -f "$GETKITE_RUNS/$r$sfx/$f" ] || { echo "missing $r$sfx/$f" >&2; exit 2; }; done
done; done
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}
for c in "" _leapval105 _wide3x _leapval105_wide3x; do
  echo "== benchmark transfer_v1_cf3fwidestart$c $(date -u +%FT%TZ)"
  python -u scripts/decoder/benchmark/transfer_benchmark.py --config "configs/decoder/benchmark/transfer_v1_cf3fwidestart$c.json" --skip_crossbody
done
echo "== external hands $(date -u +%FT%TZ)"
python -u scripts/decoder/benchmark/external_hands_eval.py --out "\${GETKITE_RUNS}/external_hands_cf3fwidestart" --arms cf3e cf3fwidestart
echo "== gate $(date -u +%FT%TZ)"
python -u scripts/decoder/benchmark/cf3f_gate.py widestart
echo "== done $(date -u +%FT%TZ)"
