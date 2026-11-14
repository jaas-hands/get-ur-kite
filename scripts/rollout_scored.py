"""Run a KITE rollout script with staged scoring attached, then print and save the scores.

Everything after ``--`` goes to the KITE script (Hydra overrides), unchanged:

    python scripts/rollout_scored.py --out outputs/scores/barrett.json -- \
        --config-name=rollout_gt_contacts/object_suite_barrett online_rollout.count=10

    python scripts/rollout_scored.py --script experiment/evaluate_object_suite.py -- \
        --config-name=evaluate_policy/object_suite_allegro

The KITE script runs in this process from the KITE root, so its relative paths work. Its
parallel workers would not be scored (they are separate processes), so this forces
``online_rollout.parallel.num_workers=1`` unless you set it.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from getkite.kite_bridge import ScoringSession, kite_root  # noqa: E402


def _plain(x):
    """JSON-safe copy (numpy scalars and bools to Python)."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if hasattr(x, "item"):
        return x.item()
    return x


def summarize(results: list[dict]) -> str:
    if not results:
        return "no scored episodes"
    lines = [f"{'ep':>3} {'object':22} {'kite':>5} {'succ':>5} {'score':>6} {'quality':>7} {'pen mm':>6}  stages done / failure"]
    for r in results:
        done = [k for k, t in r["stage_done_s"].items() if t is not None]
        tail = f"FAILED: {r['failure']['name']} at {r['failure']['t']:.2f} s" if r["failure"] else ", ".join(done) or "-"
        lines.append(
            f"{r['episode']:>3} {str(r.get('object_id', '-'))[:22]:22} {('yes' if r['kite_success'] else 'no'):>5} "
            f"{('yes' if r['success'] else 'no'):>5} {r['score']:6.2f} {r['quality']:7.2f} "
            f"{1000 * r['max']['agent_penetration']:6.1f}  {tail}"
        )
    n = len(results)
    lines.append(
        f"{n} episodes: KITE success {sum(r['kite_success'] for r in results)}/{n}, "
        f"staged success {sum(r['success'] for r in results)}/{n}, "
        f"mean score {sum(r['score'] for r in results) / n:.3f}, "
        f"failures {sum(r['failure'] is not None for r in results)}/{n}"
    )
    return "\n".join(lines)


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
    root = kite_root()
    script = root / args.script
    import os

    os.chdir(root)
    sys.argv = [str(script), *kite_args]
    session = ScoringSession()
    try:
        with session:
            runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:  # Hydra exits after the job; keep the scores either way
        if exc.code not in (None, 0):
            print(f"KITE script exited with {exc.code}", file=sys.stderr)
    finally:
        results = _plain(session.results())
        print("\n== staged scores ==")
        print(summarize(results))
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"script": args.script, "args": kite_args, "episodes": results}, indent=1))
            print(f"saved scores: {out}")


if __name__ == "__main__":
    main()
