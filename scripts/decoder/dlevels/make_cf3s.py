"""CF-3s: recovery from lag, Step 2 (docs/lag_robustness_proposal.md; user-approved 2026-11-28). Arm `wide` = frozen CF-3e
(cf3e_long_hf*) + ONE change: half of the training starts are drawn with KITE's start noise scaled x3 (15 mm / 15 deg / 15 % of
joint range std) instead of x1; on those wide-start examples the distillation target is the ground-truth target (the teachers
were trained near the goal only). Everything else unchanged (teachers, alpha, data, model, steps, seeds, selection at x1).
Launched only if Step 1 (lag_robustness_v1) confirmed the lag gap, and only with the user's go-ahead.

GATE (fixed here, before launch; implemented in scripts/decoder/benchmark/cf3s_readout.py, committed before any result):
Per seed, arm - CF-3e of the same seed (CF-3e re-scored in the same jobs); seeds never pooled.
  (a) near-goal non-inferiority at x1 (the usual benchmark start): no seed in which the point estimate exceeds the anchor on any
      of: source val Allegro / Barrett / LEAP (0.20 mm), 105 pool (0.20), Wuji (0.54), Shadow (0.82), held-out bodies (0.14,
      training-log eval at the selected checkpoint). External hands reported per hand.
  (b) lag robustness: with the start noise x3 (lag_robustness_eval.py, same records and seed), arm corr_mm better than CF-3e's by
      more than 0.5 mm on the 105 pool AND on source-val LEAP in >= 2 of 3 seeds. x2 / x4 / x6 and repeated calls reported.
  PASS = (a) and (b). Janice's closed-loop diagnostic, when available, is reported alongside, not part of this gate.

    python scripts/decoder/dlevels/make_cf3s.py
"""
import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SC = REPO / "configs/decoder/scratch"
BM = REPO / "configs/decoder/benchmark"
SFX = ("", "_s1", "_s2")


def make(src, sfx):
    c = copy.deepcopy(src)
    assert "start_wide_frac" not in src and "start_wide_scale" not in src
    c["start_wide_frac"] = 0.5
    c["start_wide_scale"] = 3.0
    c["out_dir"] = f"${{GETKITE_RUNS}}/cf3s_wide_hf{sfx}"
    c["_note"] = (f"CF-3s-wide (2026-11-28): frozen CF-3e (cf3e_long_hf{sfx}) + one change (50 % of training starts at x3 start "
                  "noise, ground-truth targets on those); docs/lag_robustness_proposal.md. Made by scripts/decoder/dlevels/make_cf3s.py.")
    diff = {k for k in set(src) | set(c) if src.get(k) != c.get(k)} - {"_note", "out_dir"}
    assert diff == {"start_wide_frac", "start_wide_scale"}, diff
    return c


def bench():
    for suf in ("", "_leapval105"):
        b = json.loads((BM / f"transfer_v1_cf3flong{suf}.json").read_text())
        sy = {}
        for s in range(3):
            sx = "" if s == 0 else f"_s{s}"
            for k in ("cf2", "d1bv", "cf3e"):
                sy[f"{k}_s{s}"] = b["systems"][f"{k}_s{s}"]
            sy[f"cf3swide_s{s}"] = {"kind": "j1", "ckpt": f"${{GETKITE_RUNS}}/cf3s_wide_hf{sx}/best.pt"}
        b["systems"] = sy
        b["out_dir"] = f"${{GETKITE_RUNS}}/transfer_benchmark_v1_cf3swide{suf}"
        b["_note"] = "CF-3s-wide vs CF-3e (re-scored in the same job), CF2, D1-bv; docs/lag_robustness_proposal.md."
        (BM / f"transfer_v1_cf3swide{suf}.json").write_text(json.dumps(b, indent=1))


def main():
    for sfx in SFX:
        src = json.loads((SC / f"cf3e_long_hf{sfx}.json").read_text())
        frozen = REPO.parent / "getkite_runs/frozen_cf3e_2026-11-27/configs" / f"cf3e_long_hf{sfx}.json"
        if frozen.exists():
            assert json.loads(frozen.read_text()) == src
        (SC / f"cf3s_wide_hf{sfx}.json").write_text(json.dumps(make(src, sfx), indent=1))
        print(f"cf3s_wide_hf{sfx or '_s0'} written")
    bench()
    print("benchmark configs transfer_v1_cf3swide{,_leapval105}.json")


if __name__ == "__main__":
    main()
