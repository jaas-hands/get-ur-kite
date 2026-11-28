#!/bin/bash
# True-contact bound for frozen CF-3e (2026-11-28, evaluation only), as one job on one GPU, from the repo root:
#   bash scripts/decoder/benchmark/run_true_contact_bound.sh
# Scores CF-3e seeds 0-2 with their own predicted contacts AND with the GET decoder fed the true requested contacts
# (transfer_benchmark.TrueContactJ1), in the same run: standard pools + Shadow, the 105 held-out LEAP hands, the 7 external
# hands; then writes the summary. Needs under $GETKITE_RUNS: cf3e_long_hf{,_s1,_s2}/best.pt. No training, no gate.
set -euo pipefail
for sfx in "" _s1 _s2; do [ -f "$GETKITE_RUNS/cf3e_long_hf$sfx/best.pt" ] || { echo "missing cf3e_long_hf$sfx/best.pt" >&2; exit 2; }; done
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}
for c in "" _leapval105; do
  echo "== benchmark transfer_v1_cf3etrue$c $(date -u +%FT%TZ)"
  python -u scripts/decoder/benchmark/transfer_benchmark.py --config "configs/decoder/benchmark/transfer_v1_cf3etrue$c.json" --skip_crossbody
done
echo "== external hands $(date -u +%FT%TZ)"
python -u scripts/decoder/benchmark/external_hands_eval.py --out "\${GETKITE_RUNS}/external_hands_cf3etrue" --arms cf3e cf3e_true
echo "== summary $(date -u +%FT%TZ)"
python -u scripts/decoder/benchmark/true_contact_bound_summary.py
echo "== done $(date -u +%FT%TZ)"
