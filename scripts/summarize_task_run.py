"""Check demo scores against KITE's results and write a short task report."""

import argparse
import json
from pathlib import Path

import numpy as np


def make_report(run, source_indices, successes):
    rows = run["episodes"]
    ids = run["requested_indices"]
    if run["exit_code"] != 0 or not rows or len(rows) != len(ids):
        raise ValueError("Run failed or is incomplete; do not report it as a finished evaluation.")
    if len(set(ids)) != len(ids) or list(source_indices) != ids:
        raise ValueError("KITE episode IDs do not match the requested episodes.")
    if len(successes) != len(rows):
        raise ValueError("KITE and the scorer saved different numbers of episodes.")
    for row, episode, success in zip(rows, ids, successes):
        if row["source_episode"] != episode or row["steps"] <= 0:
            raise ValueError("Missing steps or incorrect episode label.")
        if row["success"] != bool(success) or row["kite_success"] != bool(success):
            raise ValueError(f"Episode {episode}: scorer and KITE disagree about success.")
        if not 0 <= row["score"] <= 1 or not np.isfinite(row["max"]["agent_penetration"]):
            raise ValueError(f"Episode {episode}: invalid score or penetration value.")

    lines = [
        "# Task check", "",
        "KITE's released decoder with recorded Oracle Intents. This is a small setup check, "
        "not a model comparison or an estimate of overall success rate.", "",
        "| Episode | Hand | Object | Success | Peak score | Lowered into bin | Max hand penetration | Stages completed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        placed = row["flags"].get("placed")
        stages = ", ".join(k for k, t in row["stage_done_s"].items() if t is not None) or "none"
        lines.append(
            f"| {row['source_episode']} | {row['embodiment']} | {row.get('object_id', '—')} | "
            f"{'yes' if row['success'] else 'no'} | {100 * row['score']:.1f}% | "
            f"{'n/a' if placed is None else 'yes' if placed else 'no'} | "
            f"{1000 * row['max']['agent_penetration']:.2f} mm | {stages} |"
        )
    n = len(rows)
    lines += ["", f"Success: {sum(r['success'] for r in rows)}/{n}. "
              f"Mean peak score: {100 * sum(r['score'] for r in rows) / n:.1f}%.", "",
              "Checks: all requested episodes finished; episode IDs and success labels agree "
              "with KITE's separate results file.", "",
              "For Objects Placing, success means the object is over the bin; it need not be "
              "lowered or released. Peak score keeps credit for earlier progress even after a drop. "
              "Penetration is a diagnostic, not a success veto.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("kite_results", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads(args.scores.read_text())
    with np.load(args.kite_results, allow_pickle=False) as result:
        report = make_report(run, result["source_indices"].tolist(), result["success"].tolist())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
