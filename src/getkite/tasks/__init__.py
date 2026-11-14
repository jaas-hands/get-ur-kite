"""Score specs for KITE's tabletop tasks, one module per task.

Each module defines ``SPEC`` (a ``TaskScore``), ``make_features(model, data, scene, task)``
and ``make_score(...)`` returning an ``EpisodeScore``. ``make_score_for_task`` picks the
module from the KITE task object's class, so callers need not know which task runs.
"""

from __future__ import annotations

from getkite.tasks import keyboard_press, objects_placing, pump_bottle
from getkite.tasks.common import EpisodeScore

# KITE task class name -> score module
SCORES = {
    "ObjectSuitePickPlaceTask": objects_placing,
    "PumpBottleUseTask": pump_bottle,
    "KeyboardPressTask": keyboard_press,
}


def make_score_for_task(model, data, scene, task, meta: dict | None = None) -> EpisodeScore | None:
    """An ``EpisodeScore`` for a KITE task, or None for a task without a score (e.g. cube picking)."""
    module = SCORES.get(type(task).__name__)
    return None if module is None else module.make_score(model, data, scene, task, meta=meta)


__all__ = ["EpisodeScore", "SCORES", "make_score_for_task"]
