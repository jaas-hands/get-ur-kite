"""Run a KITE rollout script with staged scoring attached, then print and save the scores.

Everything after ``--`` goes to the KITE script (Hydra overrides), unchanged:

    python scripts/rollout_scored.py --out outputs/scores/barrett.json -- \
        --config-name=rollout_gt_contacts/object_suite_barrett online_rollout.count=10

    python scripts/rollout_scored.py --script experiment/evaluate_object_suite.py -- \
        --config-name=evaluate_policy/object_suite_allegro

The KITE script runs in this process from the KITE root, so its relative paths work. Its
parallel workers would not be scored (they are separate processes), so this forces
``online_rollout.parallel.num_workers=1`` unless you set it. To watch episodes with the
score on screen, use ``scripts/demo_task.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from getkite.kite_bridge import ScoringSession, run_kite_script  # noqa: E402
from getkite.report import plain, summarize  # noqa: E402


def main() -> None:
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--script", default="experiment/rollout_gt_contacts.py", help="KITE script, relative to the KITE root")
    parser.add_argument("--out", type=Path, default=None, help="JSON file for the per-episode scores")
    args = parser.parse_args(argv[:split])
    kite_args = argv[split + 1 :]
    if not any(a.startswith("online_rollout.parallel.num_workers=") for a in kite_args):
        kite_args.append("online_rollout.parallel.num_workers=1")

    out = args.out.resolve() if args.out else None
    session = ScoringSession()
    try:
        with session:
            code = run_kite_script(args.script, kite_args)
        if code:
            print(f"KITE script exited with {code}", file=sys.stderr)
    finally:
        results = plain(session.results())
        print("\n== staged scores ==")
        print(summarize(results))
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"script": args.script, "args": kite_args, "episodes": results}, indent=1))
            print(f"saved scores: {out}")


if __name__ == "__main__":
    main()
