"""Keep failed runs and mismatched episode labels out of task reports."""

import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).parents[1] / "scripts" / "summarize_task_run.py"
spec = importlib.util.spec_from_file_location("summarize_task_run", path)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@pytest.fixture
def run():
    return {"exit_code": 0, "requested_indices": [55], "episodes": [{
        "source_episode": 55, "steps": 120, "success": True, "kite_success": True,
        "score": 1.0, "embodiment": "barrett", "object_id": "mug",
        "flags": {"placed": False}, "max": {"agent_penetration": 0.02},
        "stage_done_s": {"lift": 1.0, "carry": 2.0},
    }]}


def test_report_keeps_success_and_placement_separate(run):
    text = report.make_report(run, [55], [True])
    assert "| 55 | barrett | mug | yes | 100.0% | no | 20.00 mm |" in text


@pytest.mark.parametrize("change", ["failed", "missing", "wrong_id", "wrong_success"])
def test_rejects_unreliable_results(run, change):
    ids, successes = [55], [True]
    if change == "failed":
        run["exit_code"] = 1
    elif change == "missing":
        run["episodes"] = []
    elif change == "wrong_id":
        ids = [51]
    else:
        successes = [False]
    with pytest.raises(ValueError):
        report.make_report(run, ids, successes)
