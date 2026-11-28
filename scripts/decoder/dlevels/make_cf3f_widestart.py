"""Wide-start arm (docs/wide_start_proposal.md, 2026-11-28), run on Aryan's compute: CF-3e (cf3e_long_hf*) with ONE change,
the training start distribution. Each training example's start is drawn with KITE's noise (5 mm / 5 deg / 5 % of joint range
per axis) with prob 0.5, or with every scale x3 with prob 0.5; on wide examples the teacher term uses the ground-truth target
(the teachers only trained on KITE-noise starts). Validation, checkpoint selection and the near-goal benchmark keep KITE's
noise. 3 seeds. Also writes the benchmark configs: near-goal (standard + 105) and wide3x (same records, starts at 3x noise),
CF-3e scored in the same run.

    python scripts/decoder/dlevels/make_cf3f_widestart.py
"""
import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SC, BE = REPO / "configs/decoder/scratch", REPO / "configs/decoder/benchmark"
CHANGE = {"start_wide_mult": 3.0, "start_wide_frac": 0.5, "start_wide_teacher": "label"}
SFX = ("", "_s1", "_s2")


def main():
    for sfx in SFX:
        src = json.loads((SC / f"cf3e_long_hf{sfx}.json").read_text())
        assert not any(k in src for k in CHANGE)
        c = copy.deepcopy(src) | CHANGE
        c["out_dir"] = f"${{GETKITE_RUNS}}/cf3f_widestart_hf{sfx}"
        c["_note"] = (f"Wide-start arm (2026-11-28): frozen CF-3e (cf3e_long_hf{sfx}) + {CHANGE}. docs/wide_start_proposal.md. "
                      "Made by scripts/decoder/dlevels/make_cf3f_widestart.py.")
        diff = {k for k in set(src) | set(c) if src.get(k) != c.get(k)} - {"_note", "out_dir"}
        assert diff == set(CHANGE), diff
        (SC / f"cf3f_widestart_hf{sfx}.json").write_text(json.dumps(c, indent=1))
        print(f"cf3f_widestart_hf{sfx or '_s0'}: changed {sorted(diff)}")
    for pool in ("", "_leapval105"):
        ref = json.loads((BE / f"transfer_v1_cf3e{pool}.json").read_text())
        systems = {}
        for s, sfx in enumerate(SFX):
            systems[f"cf3e_s{s}"] = {"kind": "j1", "ckpt": f"${{GETKITE_RUNS}}/cf3e_long_hf{sfx}/best.pt"}
            systems[f"cf3fwidestart_s{s}"] = {"kind": "j1", "ckpt": f"${{GETKITE_RUNS}}/cf3f_widestart_hf{sfx}/best.pt"}
        for wide in ("", "_wide3x"):
            b = {"_note": ("Wide-start arm vs CF-3e (gate reference, scored in this same run); docs/wide_start_proposal.md. "
                           + ("Starts at 3x KITE noise (same records): the predeclared gain W. " if wide else "Near-goal (KITE-noise) starts: Q1-Q5. ")
                           + "Made by scripts/decoder/dlevels/make_cf3f_widestart.py."),
                 "base_config": ref["base_config"], "surface_sampling": ref["surface_sampling"],
                 "out_dir": f"${{GETKITE_RUNS}}/transfer_benchmark_v1_cf3fwidestart{pool}{wide}", "shadow_pool": ref["shadow_pool"],
                 **({"start_noise_mult": 3.0} if wide else {}), "systems": systems}
            (BE / f"transfer_v1_cf3fwidestart{pool}{wide}.json").write_text(json.dumps(b, indent=1))
            print("benchmark", f"transfer_v1_cf3fwidestart{pool}{wide}")


if __name__ == "__main__":
    main()
