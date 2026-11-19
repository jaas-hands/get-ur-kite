"""A failed KITE run must not look like a successful evaluation."""

import importlib.util
import json
import sys
from pathlib import Path


def test_failed_run_keeps_exit_status_and_partial_output(monkeypatch, tmp_path):
    path = Path(__file__).parents[1] / "scripts" / "rollout_scored.py"
    spec = importlib.util.spec_from_file_location("rollout_scored", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def results(self):
            return []

    out = tmp_path / "scores.json"
    monkeypatch.setattr(sys, "argv", [str(path), "--out", str(out)])
    monkeypatch.setattr(runner, "ScoringSession", Session)
    monkeypatch.setattr(runner, "run_kite_script", lambda *args: 3)
    assert runner.main() == 3
    saved = json.loads(out.read_text())
    assert saved["exit_code"] == 3
    assert saved["episodes"] == []
