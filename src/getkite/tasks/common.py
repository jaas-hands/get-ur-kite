"""What every KITE task score shares: the episode wrapper and small feature helpers."""

from __future__ import annotations

from typing import Callable

from getkite.scoring import ScoreTracker, TaskScore

# ContactWatch settings for KITE scenes. CEBenchmark fails an attempt when the hand sinks
# more than 3 mm into a task part (calibrated on the Franka's pads). KITE's hands sink much
# deeper in ordinary grasps: with the shipped Oracle-Intent rollouts, Barrett's fingertips
# sit 2-2.6 cm inside the object for most of a successful episode. So for KITE tasks, hand
# penetration is not a failure but a diagnostic (``max["agent_penetration"]``); parts forced
# into each other (object into bin, 3 mm) still are.
KITE_CONTACTS = dict(agent_limit=float("inf"))

# Features whose episode maximum ``EpisodeScore.result`` reports under ``max``.
DIAGNOSTICS = ("agent_penetration", "penetration", "net_force")


class EpisodeScore:
    """One episode's ``ScoreTracker``, plus KITE's own success flag, latched.

    Every task's feature function returns ``kite_success``: the success rule KITE's rollout
    scripts use, at this control step. It is latched here and reported as ``kite_success``
    next to the staged score, so results stay comparable with the KITE paper's success rates
    while the staged score adds partial credit, failures and quality.

        score = EpisodeScore("objects_placing", SPEC, make_features(model, data, scene, task))
        score.update(data)    # every control step
        score.result()        # {"score": ..., "success": ..., "kite_success": ..., ...}
    """

    def __init__(self, task: str, spec: TaskScore, features: Callable[[object], dict], meta: dict | None = None):
        self.task = task
        self.meta = dict(meta or {})
        self.kite_success = False
        self.steps = 0
        self.max = {k: 0.0 for k in DIAGNOSTICS}

        def latched(data):
            f = features(data)
            self.kite_success = self.kite_success or bool(f["kite_success"])
            for k in DIAGNOSTICS:
                if k in f:
                    self.max[k] = max(self.max[k], float(f[k]))
            return f

        self.tracker = ScoreTracker(spec, latched)

    def update(self, data) -> dict:
        self.steps += 1
        return self.tracker.update(data)

    @property
    def done(self) -> bool:
        return self.tracker.done

    def result(self) -> dict:
        return {
            "task": self.task,
            **self.meta,
            "steps": self.steps,
            "kite_success": self.kite_success,
            **self.tracker.result(),
            "max": dict(self.max),
        }
