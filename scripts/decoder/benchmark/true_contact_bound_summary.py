"""Summary of the true-contact bound for frozen CF-3e (2026-11-28, evaluation only; descriptive, no pass/fail).

Reads the per-example outputs of transfer_v1_cf3etrue{,_leapval105} (transfer_benchmark.py) and the external-hands run
(external_hands_eval.py --arms cf3e cf3e_true). For every pool / family and seed: CF-3e with its own predicted contacts,
CF-3e's GET decoder fed the TRUE requested contacts, and the paired difference (true - predicted; negative = the true
set helps), bootstrap over records within seed. Seeds are never pooled.

Reading: a large drop with true contacts = the contact-set predictor (psi -> contacts) is the bottleneck; a small drop =
the GET decoder is. The bound is optimistic in one way and pessimistic in another: the decoder never saw an exact
contact set in training (only its own predictions), so it may use true contacts slightly worse than a perfect predictor's.

    PYTHONPATH=src python scripts/decoder/benchmark/true_contact_bound_summary.py
Writes ${GETKITE_RUNS}/true_contact_bound_cf3e/{summary.json,summary.md}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from getkite.decoder import paths

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "diagnostics"))
from wuji_per_example import paired_boot  # noqa: E402

RUNS = Path(paths.get("GETKITE_RUNS"))
METRICS = ("corr_mm", "corr_success", "compat_success", "wrist_mm", "wrist_deg")


def main():
    out = RUNS / "true_contact_bound_cf3e"; out.mkdir(parents=True, exist_ok=True)
    S, md = {"benchmarks": {}, "external_hands": None}, ["# True-contact bound, frozen CF-3e (true - predicted, per seed)", ""]
    for sfx in ("", "_leapval105"):
        pe_path = RUNS / f"transfer_benchmark_v1_cf3etrue{sfx}" / "per_example.pt"
        PE = torch.load(pe_path, weights_only=False)
        B = {}
        for pn, d in PE.items():
            per, fam = d["per"], np.asarray(d["family"])
            for f in sorted(set(fam)):
                sel = fam == f
                row = {"n": int(sel.sum())}
                for s in range(3):
                    p, t = per[f"cf3e_s{s}"], per[f"cf3etrue_s{s}"]
                    row[f"s{s}"] = {m: {"predicted": round(float(p[m][sel].mean()), 4), "true": round(float(t[m][sel].mean()), 4),
                                        "true_minus_predicted": paired_boot(t[m][sel].astype(float), p[m][sel].astype(float))}
                                    for m in METRICS}
                    row[f"s{s}"]["no_movement_corr_mm"] = round(float(per["no_movement"]["corr_mm"][sel].mean()), 4)
                B[f"{pn}/{f}"] = row
                md.append(f"- {pn}/{f} (n={row['n']}): corr_mm predicted -> true, s0/s1/s2: " + "; ".join(
                    f"{row[f's{s}']['corr_mm']['predicted']:.2f} -> {row[f's{s}']['corr_mm']['true']:.2f} "
                    f"({row[f's{s}']['corr_mm']['true_minus_predicted']['mean_diff']:+.2f})" for s in range(3))
                    + "; corr_success " + " / ".join(f"{row[f's{s}']['corr_success']['predicted']:.2f}->{row[f's{s}']['corr_success']['true']:.2f}"
                                                     for s in range(3)))
        S["benchmarks"][f"transfer_v1_cf3etrue{sfx}"] = B
    ext = RUNS / "external_hands_cf3etrue" / "external_hands.json"
    if ext.exists():
        E = json.loads(ext.read_text())
        S["external_hands"] = {h: {"cf3e": F["cf3e"]["seeds"], "cf3e_true": F["cf3e_true"]["seeds"],
                                   "true_minus_predicted": F["cf3e_true"].get("minus_predicted")} for h, F in E["pools"].items()}
        md += ["", "External hands, corr_mm predicted -> true, s0/s1/s2:"]
        for h, F in S["external_hands"].items():
            md.append(f"- {h}: " + "; ".join(f"{F['cf3e']['corr_mm'][s]:.2f} -> {F['cf3e_true']['corr_mm'][s]:.2f}" for s in range(3)))
    (out / "summary.json").write_text(json.dumps(S, indent=1))
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print("wrote", out)


if __name__ == "__main__":
    main()
