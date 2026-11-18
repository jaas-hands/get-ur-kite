"""Per-family table for the two C3 diagnostics (psi intent vs raw-contact diagnostic reference).

    python report_c3.py runs/scratch_c3_long50k runs/scratch_c3_raw50k [--step N]

Reads each run's log.jsonl evaluation rows (val = fresh goals on training hands, train_goals = the training
goals with a fixed evaluation start pose, leap_heldout = unseen LEAP hands) and prints model and no-movement
metrics on exactly those records. Default step = the last evaluation.
"""
import json
import sys
from pathlib import Path

FAMS = ["allegro", "barrett", "leap"]
METS = [("contact_mm", "contact mm", 2), ("success", "success", 3), ("wrist_mm", "wrist mm", 2),
        ("wrist_deg", "wrist deg", 2), ("res_loss", "res loss", 3)]
POOLS = [("train_goals:", "training goals"), ("", "validation goals"), ("leap_heldout:", "held-out LEAP hands")]


def evals(run):
    rows = [json.loads(l) for l in open(Path(run) / "log.jsonl")]
    return {r["step"]: r["eval"] for r in rows if "eval" in r}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    step = int(sys.argv[sys.argv.index("--step") + 1]) if "--step" in sys.argv else None
    runs = {Path(a).name: evals(a) for a in args if not a.isdigit()}
    for name, ev in runs.items():
        s = step if step is not None else max(ev)
        r = ev[s]
        print(f"\n## {name} @ step {s} (evaluations at {sorted(ev)[:3]}...{max(ev)})")
        for pre, label in POOLS:
            fams = ["leap"] if pre == "leap_heldout:" else FAMS
            if f"{pre}{fams[0]}/model/contact_mm" not in r:
                continue
            print(f"\n{label}")
            print("| family | who | " + " | ".join(m[1] for m in METS) + " |")
            print("|---|---|" + "---|" * len(METS))
            for f in fams:
                for who in ("model", "noop"):
                    vals = [r.get(f"{pre}{f}/{who}/{k}") for k, _, _ in METS]
                    print(f"| {f} | {who} | " + " | ".join("-" if v is None else f"{v:.{d}f}"
                                                         for v, (_, _, d) in zip(vals, METS)) + " |")
        curve = [(k, ev[k].get("all/model/contact_mm"), ev[k].get("train_goals:all/model/contact_mm"))
                 for k in sorted(ev)]
        print("\nmacro contact mm by step (val, train goals): " +
              ", ".join(f"{k}: {a:.2f}/{b:.2f}" for k, a, b in curve if a is not None and b is not None))


if __name__ == "__main__":
    main()
