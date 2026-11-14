"""Tests of the staged scoring core: no simulator needed."""

from types import SimpleNamespace

import pytest

from getkite.scoring import Failure, QualityTerm, ScoreTracker, Stage, TaskScore


def run(spec, frames, dt=0.05):
    """Feed feature dicts, one per control step, through a tracker; return it."""
    it = iter(frames)
    tracker = ScoreTracker(spec, lambda data: next(it))
    for k in range(len(frames)):
        tracker.update(SimpleNamespace(time=(k + 1) * dt))
    return tracker


SPEC = TaskScore(
    stages=(
        Stage("reach", 1.0, lambda f: f["a"]),
        Stage("finish", 1.0, lambda f: f["b"], gate=lambda f: f["a"] >= 1.0),
    ),
    success=lambda f: f["b"] >= 1.0,
    failures=(Failure("broken", lambda f: f.get("broken", False)),),
    quality=(QualityTerm("force", 1.0, lambda f: f.get("force", 0.0), 2.0),),
)


def test_later_stage_implies_earlier_and_peak_is_kept():
    t = run(SPEC, [{"a": 0.5, "b": 0.0}, {"a": 1.0, "b": 0.5}, {"a": 0.2, "b": 0.0}])
    r = t.result()
    assert r["progress_peak"] == pytest.approx(0.75)  # reach done (1) + half of finish, over 2
    assert r["progress_final"] == pytest.approx(0.1)  # back in the first stage
    assert r["score"] == pytest.approx(0.75)  # aggregate="peak"
    assert r["stage_done_s"]["reach"] == pytest.approx(0.10)
    assert not r["success"]


def test_success_latches_and_scores_one():
    r = run(SPEC, [{"a": 1.0, "b": 1.0}, {"a": 0.0, "b": 0.0}]).result()
    assert r["success"] and r["score"] == 1.0
    assert r["time_to_success_s"] == pytest.approx(0.05)


def test_failure_freezes_progress_and_blocks_success():
    r = run(SPEC, [{"a": 0.6, "b": 0.0}, {"a": 1.0, "b": 0.0, "broken": True}, {"a": 1.0, "b": 1.0}]).result()
    assert r["failure"]["name"] == "broken"
    assert not r["success"]
    assert r["score"] == pytest.approx(0.3)  # frozen at the step before the failure


def test_zero_score_failure():
    spec = TaskScore(stages=SPEC.stages, success=SPEC.success, failures=(Failure("cheat", lambda f: f.get("broken", False), zero_score=True),))
    r = run(spec, [{"a": 1.0, "b": 0.5}, {"a": 1.0, "b": 0.5, "broken": True}]).result()
    assert r["score"] == 0.0


def test_quality_integrates_rate_over_sim_time():
    # 4 N for 0.5 s = 2 N s = the full limit: quality 0; it never enters the score.
    frames = [{"a": 0.0, "b": 0.0, "force": 4.0} for _ in range(11)]
    r = run(SPEC, frames).result()
    assert r["quality"] == pytest.approx(0.0)
    assert r["score"] == 0.0
