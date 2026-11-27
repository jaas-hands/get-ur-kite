"""W1 seed summary (seeds 0/1/2): per-seed values, then mean ± SD (ddof=1, n=3) of the run-level means.
Evaluation examples are never pooled across runs; no seed is selected. Paired comparisons are within each run
(same records and starts), taken from each run's own eval output.

Inputs: runs/w1_wrist_transfer_eval{,_s1,_s2}/metrics.json and runs/w1_checks_eval{,_s1,_s2}/metrics.json.
    python scripts/decoder/benchmark/w1_seed_summary.py [--out runs/w1_seeds_summary.md]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())

R = Path(os.environ.get("GETKITE_RUNS", _REPO.parent / "getkite_runs"))
SUF = {0: "", 1: "_s1", 2: "_s2"}
POOLS = [("val", "allegro"), ("val", "barrett"), ("val", "leap"), ("leap_heldout", "leap"), ("wuji", "wuji")]
MET = ["corr_mm", "corr_success", "compat_success", "wrist_mm", "wrist_deg", "trans_slope", "trans_R2_id", "rot_slope", "rot_R2_id"]
PM = ["corr_mm", "corr_success", "compat_success", "wrist_mm"]


def ms(v):
    v = np.asarray(v, float)
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"


def cell(p):
    return f"{p['mean_diff']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(R / "w1_seeds_summary.md"))
    a = ap.parse_args()
    E = {s: json.load(open(R / f"w1_wrist_transfer_eval{SUF[s]}/metrics.json")) for s in SUF}
    C = {s: json.load(open(R / f"w1_checks_eval{SUF[s]}/metrics.json")) for s in SUF}
    L = ["# W1 seeds 0/1/2: fingers at goal, wrist-only starts, true contacts (and translator contacts)", "",
         "Per-seed run-level means; across seeds mean ± SD (ddof = 1, n = 3). Paired CIs are within each run (same records/starts). "
         "No seed selection; no pooling of examples. Selected checkpoints (source val): "
         + ", ".join(f"s{s} step {E[s]['_meta']['systems']['w1']['step']}" for s in SUF)
         + ". Control raw_ref (fingers-at-goal control, trained with finger noise) has seed 0 only; it is identical in every column.", ""]
    for pn, fam in POOLS:
        F = {s: E[s][pn][fam] for s in SUF}
        L += [f"## {pn}/{fam} (n = {F[0]['n']})", "",
              "| metric | no movement | raw_ref (s0) | W1 s0 | W1 s1 | W1 s2 | W1 mean ± SD |", "|---|---|---|---|---|---|---|"]
        for m in MET:
            if m not in F[0]["w1"]:
                continue
            w = [F[s]["w1"][m] for s in SUF]
            nm = F[0]["no_movement"].get(m, float("nan"))
            L.append(f"| {m} | {nm:.3f} | {F[0]['raw_ref'][m]:.3f} | " + " | ".join(f"{x:.3f}" for x in w) + f" | {ms(w)} |")
        for ref in ("no_movement", "raw_ref"):
            L += ["", f"W1 − {ref}, paired within each run:", "", "| metric | s0 | s1 | s2 | mean ± SD of diffs | runs with CI excluding 0 |", "|---|---|---|---|---|---|"]
            for m in PM:
                ps = [F[s]["_paired"][f"w1-{ref}/{m}"] for s in SUF]
                sig = "".join("+" if p["ci95"][0] > 0 else "−" for p in ps if p["ci95"][0] > 0 or p["ci95"][1] < 0)
                L.append(f"| {m} | " + " | ".join(cell(p) for p in ps) + f" | {ms([p['mean_diff'] for p in ps])} | {len(sig)}/3 {sig} |")
        if (pn, fam) in (("wuji", "wuji"), ("val", "allegro"), ("val", "barrett"), ("val", "leap"), ("leap_heldout", "leap")):
            G = {s: C[s][pn][fam] for s in SUF}
            L += ["", "Translator-predicted contacts (w1_checks_eval):", "",
                  "| metric | W1 translator s0 | s1 | s2 | mean ± SD | translator − true, s0 / s1 / s2 | translator − no movement, s0 / s1 / s2 |", "|---|---|---|---|---|---|---|"]
            for m in PM:
                v = [G[s]["full/translator"][m] for s in SUF]
                d1 = " / ".join(cell(G[s]["_paired"][f"full/translator-full/true/{m}"]) for s in SUF)
                d2 = " / ".join(cell(G[s]["_paired"][f"full/translator-no_movement/{m}"]) for s in SUF)
                L.append(f"| {m} | " + " | ".join(f"{x:.3f}" for x in v) + f" | {ms(v)} | {d1} | {d2} |")
            fm = [max(G[s][n].get("finger_move_rad", 0) for n in ("full/true", "full/translator")) for s in SUF]
            jh = [C[s]["_meta"]["w1"].get("joint_head_last_layer_max_abs") for s in SUF]
            L += ["", f"W1 finger movement (rad, max over conditions) per seed: {fm}; joint-head last layer max |w|: {jh}"]
        L.append("")
    txt = "\n".join(L)
    Path(a.out).write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
