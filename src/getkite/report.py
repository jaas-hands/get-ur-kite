"""Print and save per-episode scores (``EpisodeScore.result()`` dicts)."""

from __future__ import annotations


def plain(x):
    """JSON-safe copy (numpy scalars and bools to Python)."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if hasattr(x, "item"):
        return x.item()
    return x


def contact_flags(r: dict) -> list[str]:
    """CEBenchmark's contact failures that happened (reported, they do not block success)."""
    names = {"forced_through": "forced through", "excessive_force": "excessive force"}
    return [label for key, label in names.items() if r.get("flags", {}).get(key)]


def summarize(results: list[dict]) -> str:
    """One line per episode, then the totals. ``success`` is KITE's own success rule."""
    if not results:
        return "no scored episodes"
    lines = [f"{'ep':>3} {'object':22} {'succ':>5} {'score':>6} {'in bin':>6} {'quality':>7} {'pen mm':>6}  stages done / contact flags"]
    for r in results:
        if r["success"] != r["kite_success"]:  # cannot happen with the KITE task specs; say so if it does
            raise ValueError(f"episode {r['episode']}: success {r['success']} differs from KITE's {r['kite_success']}")
        done = [k for k, t in r["stage_done_s"].items() if t is not None]
        placed = r.get("flags", {}).get("placed")
        tail = ", ".join(done) or "-"
        if contact_flags(r):
            tail += "  [" + ", ".join(contact_flags(r)) + "]"
        lines.append(
            f"{r['episode']:>3} {str(r.get('object_id', '-'))[:22]:22} {('yes' if r['success'] else 'no'):>5} "
            f"{r['score']:6.2f} {('-' if placed is None else 'yes' if placed else 'no'):>6} {r['quality']:7.2f} "
            f"{1000 * r['max']['agent_penetration']:6.1f}  {tail}"
        )
    n = len(results)
    placed = [r for r in results if "placed" in r.get("flags", {})]
    lines.append(
        f"{n} episodes: success {sum(r['success'] for r in results)}/{n} (KITE's rule), "
        f"mean score {sum(r['score'] for r in results) / n:.3f}"
        + (f", lowered into bin {sum(r['flags']['placed'] for r in placed)}/{len(placed)}" if placed else "")
        + f", contact flags {sum(bool(contact_flags(r)) for r in results)}/{n}"
    )
    return "\n".join(lines)
