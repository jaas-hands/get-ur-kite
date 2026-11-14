"""Staged, embodiment-agnostic task scores (ported from CEBenchmark).

- ``scoring``: ``Stage``, ``Failure``, ``QualityTerm``, ``TaskScore``, ``ScoreTracker``, ``RewardShaper``
- ``contacts``: ``ContactWatch`` (brute force, parts forced through each other, touches)
- ``lift``: ``LowestPoint`` (lift measured by an object's lowest point, not its origin)

See ``scoring.py`` for the semantics. Task-specific score specs live in ``getkite.tasks``.
"""

from getkite.scoring.contacts import ContactWatch, contact_failures, contact_quality
from getkite.scoring.lift import LowestPoint
from getkite.scoring.scoring import (
    Failure,
    QualityTerm,
    RewardShaper,
    RewardWeights,
    ScoreTracker,
    SpeedReference,
    Stage,
    TaskScore,
    clip01,
    linear,
)

__all__ = [
    "ContactWatch",
    "Failure",
    "LowestPoint",
    "QualityTerm",
    "RewardShaper",
    "RewardWeights",
    "ScoreTracker",
    "SpeedReference",
    "Stage",
    "TaskScore",
    "clip01",
    "contact_failures",
    "contact_quality",
    "linear",
]
