# Ported from CEBenchmark (cebench/tasks/_base/scoring.py @ ffad82e). Keep in sync: only import paths differ.
"""Staged, bounded task scores computed from object state only (embodiment-agnostic).

A task score is an ordered list of ``Stage``s. Each stage has an entry ``gate`` and a
``progress`` in [0, 1]. At every control step the active stage is the *last* one whose
gate holds; all stages before it count as complete:

    progress_score = (sum of weights before k + weight_k * progress_k) / sum of weights

so reaching a later stage implies the earlier ones, and the score is always in [0, 1].
The score is the episode's progress (peak, final or a blend, see ``aggregate``) and
nothing else. Three things are reported next to it, never mixed into it:

- **Quality**, how the task was done: ``quality = clip(1 - sum(weight_i * excess_i), 0, 1)``.
  Each ``QualityTerm`` integrates a non-negative rate (say, contact force above a soft
  limit, in N) over simulated time, or, with ``total=True``, takes a running total the
  task counts itself (say, wrong notes played); ``excess_i`` is that amount over the
  term's ``limit``, clipped to [0, 1].
- **Speed**, see below.
- **Failure**, if any (below).

A ``Failure`` ends the attempt: an irreversible outcome (a stripped screw), the task
done the unintended way (the block of the hook task touched by hand), or force that
tricks the simulator (parts forced into each other). From its first occurrence nothing
counts any more: progress keeps the value it had reached before (or drops to 0 with
``zero_score``), and success becomes impossible. Failures are checked until success; a
step that succeeds and fails at once is a failure, so a success reached by brute force
does not count. ``ScoreTracker.done`` (success or failure) is when an episode should end,
e.g. for reinforcement learning. Dropping an object is *not* a failure: progress that
was never made already costs it.

For training, ``RewardShaper`` turns a tracker into a per-step reward (``RewardWeights``):
the gain in *current* progress, so lost progress costs reward (the benchmark score keeps
the peak), minus a cost per unit of each quality term as it happens (a wrong note, force
above the soft limit), a bonus at success and a cost at a failure. A zero-score failure
takes back all progress, so the progress rewards always add up to the attempt's final
progress. The benchmark score itself never changes.

Speed is deliberately *not* part of the score; it is reported separately:

- ``stage_done_s``: simulated time at which each stage was completed (the last one at success),
- ``time_to_success_s``: the same for the whole task,
- ``speed``: optional normalisation of ``time_to_success_s`` by ``SpeedReference``, 1 at
  ``par_s`` falling linearly to 0 at ``budget_s`` (None without success).

Times are *simulated* seconds, not control steps, which differ between embodiments.

Gates, progress, failures and quality rates are plain functions of a feature dict,
which a task supplies through ``features(data)``. Keep features object-centric: sites
and bodies of the task objects, never robot joints or gripper contacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

Features = dict


@dataclass(frozen=True)
class Stage:
    name: str
    weight: float
    progress: Callable[[Features], float]
    gate: Callable[[Features], bool] = lambda f: True


@dataclass(frozen=True)
class Failure:
    name: str
    condition: Callable[[Features], bool]
    zero_score: bool = False  # True: the attempt scores 0; False: progress freezes where it was


@dataclass(frozen=True)
class QualityTerm:
    name: str
    weight: float  # quality lost at the full ``limit``
    rate: Callable[[Features], float]  # non-negative, integrated over simulated time (or a running total)
    limit: float  # amount that costs the full ``weight``
    total: bool = False  # ``rate`` already returns a running total: use it as is


@dataclass(frozen=True)
class SpeedReference:
    par_s: float = 0.0  # succeeding at or before this gives speed 1
    budget_s: float = 60.0  # succeeding at or after this gives speed 0 (also the episode time limit)


@dataclass(frozen=True)
class TaskScore:
    stages: tuple[Stage, ...]
    success: Callable[[Features], bool]
    failures: tuple[Failure, ...] = ()
    quality: tuple[QualityTerm, ...] = ()
    speed: SpeedReference = SpeedReference()
    # "peak": best progress at any time (a key that was seated then pulled out keeps its
    # credit); "final": progress at the end of the episode; a float in [0, 1] blends them
    # (0 = peak, 1 = final).
    aggregate: str | float = "peak"
    success_latched: bool = True  # success at any time counts, rather than only at the end


def clip01(x) -> float:
    return float(np.clip(x, 0.0, 1.0))


def linear(err, zero_at, one_at=0.0) -> float:
    """1 at ``err <= one_at``, falling linearly to 0 at ``err >= zero_at``."""
    return clip01((zero_at - err) / (zero_at - one_at))


@dataclass
class ScoreTracker:
    """Feed it ``update(data)`` every control step; read ``result()`` at the end."""

    spec: TaskScore
    features: Callable[[object], Features]
    history: list = field(default_factory=list)

    def __post_init__(self):
        self.total_weight = sum(s.weight for s in self.spec.stages)
        self.reset()

    def reset(self):
        self.history.clear()
        self.peak = 0.0
        self.excess = {q.name: 0.0 for q in self.spec.quality}  # time integrals of the quality rates
        self.stage_done = {s.name: None for s in self.spec.stages}
        self.success_time = None
        self.failure = None  # (name, time) of the first failure
        self.last = None

    def progress(self, f):
        """(active stage index, progress score in [0, 1]) for one feature snapshot."""
        k = max(i for i, s in enumerate(self.spec.stages) if i == 0 or s.gate(f))
        done = sum(s.weight for s in self.spec.stages[:k])
        stage = self.spec.stages[k]
        return k, (done + stage.weight * clip01(stage.progress(f))) / self.total_weight

    def update(self, data):
        t = float(data.time)
        f = self.features(data)
        dt = t - self.last["t"] if self.last else 0.0
        for q in self.spec.quality:
            value = max(0.0, float(q.rate(f)))
            self.excess[q.name] = value if q.total else self.excess[q.name] + value * dt
        if self.failure is None and self.success_time is None:  # failures count until success
            failed = next((x for x in self.spec.failures if x.condition(f)), None)
            if failed is not None:
                self.failure = (failed.name, t)
        if self.failure is not None:  # frozen: nothing after a failure counts
            frozen = self.last["score"] if self.last else 0.0
            self.last = dict(t=t, stage=f"failed: {self.failure[0]}", score=frozen, success=False)
            self.history.append(self.last)
            return self.last
        k, score = self.progress(f)
        success = bool(self.spec.success(f))
        if success:
            k, score = len(self.spec.stages) - 1, 1.0
        self.peak = max(self.peak, score)
        # A stage is done once a later one is active (or, for the last stage, on success).
        for i in range(len(self.spec.stages) if success else k):
            name = self.spec.stages[i].name
            if self.stage_done[name] is None:
                self.stage_done[name] = t
        if success and self.success_time is None:
            self.success_time = t
        self.last = dict(t=t, stage=self.spec.stages[k].name, score=score, success=success)
        self.history.append(self.last)
        return self.last

    @property
    def done(self) -> bool:
        """The attempt is over: success, or a failure. An episode can end here."""
        return self.success_time is not None or self.failure is not None

    def quality(self) -> tuple[float, dict]:
        """(quality in [0, 1], each term's excess as a fraction of its limit). Not part of the score."""
        excess = {q.name: clip01(self.excess[q.name] / q.limit) for q in self.spec.quality}
        lost = sum(q.weight * excess[q.name] for q in self.spec.quality)
        return clip01(1.0 - lost), excess

    def result(self) -> dict:
        spec = self.spec
        final = self.last["score"] if self.last else 0.0
        mix = {"peak": 0.0, "final": 1.0}.get(spec.aggregate, spec.aggregate)
        progress = (1.0 - float(mix)) * self.peak + float(mix) * final
        failure = None
        if self.failure is not None:
            failure = {"name": self.failure[0], "t": self.failure[1]}
            if next(x for x in spec.failures if x.name == self.failure[0]).zero_score:
                progress = 0.0
        quality, excess = self.quality()
        success = self.success_time is not None if spec.success_latched else bool(self.last and self.last["success"])
        speed = None
        if success and self.success_time is not None:
            speed = linear(self.success_time, spec.speed.budget_s, spec.speed.par_s)
        return {
            "score": clip01(progress),
            "success": success,
            "progress_peak": self.peak,
            "progress_final": final,
            "quality": quality,
            "quality_excess": excess,
            "failure": failure,
            "time_to_success_s": self.success_time,
            "speed": speed,
            "stage_done_s": dict(self.stage_done),
            "duration_s": self.last["t"] if self.last else 0.0,
        }


@dataclass(frozen=True)
class RewardWeights:
    """Training reward on top of a score (``RewardShaper``); the benchmark score is not changed."""

    progress: float = 1.0  # per unit of current progress gained (progress lost is negative)
    success: float = 0.0  # bonus, once, at success
    failure: float = 0.0  # extra cost, once, at a failure (a zero-score failure already takes back all progress)
    quality: dict = field(default_factory=dict)  # quality term -> cost per unit of its amount (per wrong note, per N s of force)


class RewardShaper:
    """Per-step training reward from a ``ScoreTracker``: call ``step()`` once after each ``tracker.update``.

        shaper = RewardShaper(tracker, REWARD)   # a task's suggested weights, or your own
        tracker.update(data); reward = shaper.step(); done = shaper.done
    """

    def __init__(self, tracker: ScoreTracker, weights: RewardWeights = RewardWeights()):
        unknown = set(weights.quality) - {q.name for q in tracker.spec.quality}
        if unknown:
            raise ValueError(f"no such quality terms: {sorted(unknown)}")
        self.tracker, self.weights = tracker, weights
        self.reset()

    def reset(self):
        self.prev_progress = 0.0
        self.prev_amount = {name: 0.0 for name in self.weights.quality}
        self.prev_success = self.prev_failed = False

    @property
    def done(self) -> bool:
        return self.tracker.done

    def current_progress(self) -> float:
        t = self.tracker
        if t.last is None:
            return 0.0
        if t.failure is not None and next(x for x in t.spec.failures if x.name == t.failure[0]).zero_score:
            return 0.0
        return float(t.last["score"])

    def step(self) -> float:
        t, w = self.tracker, self.weights
        progress = self.current_progress()
        reward = w.progress * (progress - self.prev_progress)
        for name, cost in w.quality.items():
            reward -= cost * (t.excess[name] - self.prev_amount[name])
            self.prev_amount[name] = t.excess[name]
        success, failed = t.success_time is not None, t.failure is not None
        if success and not self.prev_success:
            reward += w.success
        if failed and not self.prev_failed:
            reward -= w.failure
        self.prev_progress, self.prev_success, self.prev_failed = progress, success, failed
        return float(reward)


class ViewerScoreOverlay:
    """Live score panel (top right) and score-over-time plot (bottom right) in a passive viewer."""

    MAX_POINTS = 1000  # MjvFigure holds at most 1000 points per line

    def __init__(self, viewer, tracker: ScoreTracker, title="Score"):
        import mujoco

        self.mujoco, self.viewer, self.tracker = mujoco, viewer, tracker
        self.fig = mujoco.MjvFigure()
        mujoco.mjv_defaultFigure(self.fig)
        self.fig.title = title
        self.fig.xlabel = "sim time (s)"
        self.fig.flg_extend = 1  # the time axis grows with the curve, which fills the plot
        self.fig.range[1] = [0.0, 1.0]
        self.fig.gridsize = [5, 5]
        self.fig.figurergba[3] = 0.6
        for i, name in enumerate(("current", "peak")):
            self.fig.linename[i] = name

    def update(self, extra=()):
        """Refresh the panel and plot; ``extra``: more ``set_texts`` tuples to show alongside (e.g. a top-left panel)."""
        result, last = self.tracker.result(), self.tracker.last
        if last is None:
            return
        mj = self.mujoco
        labels = ["score", "progress now", "stage", "success", "sim time", "done at:", *(f"  {n}" for n in result["stage_done_s"])]
        done = [f"{t:.1f} s" if t is not None else "-" for t in result["stage_done_s"].values()]
        values = [
            f"{100 * result['score']:.1f} %",
            f"{100 * last['score']:.0f} %",
            last["stage"],
            f"yes (speed {result['speed']:.2f})" if result["success"] else "no",
            f"{last['t']:.1f} s",
            "",
            *done,
        ]
        if self.tracker.spec.quality:
            labels += ["quality", *(f"  {k}" for k in result["quality_excess"])]
            values += [f"{100 * result['quality']:.0f} %", *(f"{100 * v:.0f} % of limit" for v in result["quality_excess"].values())]
        if result["failure"]:
            labels.append("FAILED")
            values.append(f"{result['failure']['name']} at {result['failure']['t']:.1f} s")
        self.viewer.set_texts([(mj.mjtFontScale.mjFONTSCALE_150, mj.mjtGridPos.mjGRID_TOPRIGHT, "\n".join(labels), "\n".join(values)), *extra])

        # Thin long histories with a power-of-two stride, so drawn points stay put between frames.
        stride = 1
        while len(self.tracker.history) > stride * self.MAX_POINTS:
            stride *= 2
        hist = self.tracker.history[::stride]
        peak = np.maximum.accumulate([h["score"] for h in hist])
        for line, ys in enumerate(([h["score"] for h in hist], peak)):
            self.fig.linepnt[line] = len(hist)
            self.fig.linedata[line][: 2 * len(hist)] = np.column_stack([[h["t"] for h in hist], ys]).ravel()
        vp = self.viewer.viewport
        if vp is not None:
            w, h = vp.width // 3, vp.height // 3
            self.viewer.set_figures((mj.MjrRect(vp.width - w, 0, w, h), self.fig))
