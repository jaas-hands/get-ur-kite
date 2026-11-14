"""Score for KITE's Bottle Pumping task (``PumpBottleUseTask``): lift the bottle and press its pump.

Not the CEBenchmark pump-bottle task (lock-down, plate, soap drops): KITE's bottle has a
spring-loaded button only. KITE's success rule, reported as ``kite_success`` and used as
the success here too: at some point the bottle's lowest point was ``lift_success_delta``
above its rest height, and at some point the button was down to ``button_success_slide``,
in either order (the rollout script latches the two separately). Stages:

1. lift   - the bottle's lowest point rises towards ``task.lift_success_delta``
2. press  - once lifted, the button travels from its rest slide to ``button_success_slide``

Read from the bottle and its button only (via ``task.observe``), the same for any hand.
"""

from __future__ import annotations

from getkite.scoring import ContactWatch, SpeedReference, Stage, TaskScore, clip01, contact_failures, contact_quality
from getkite.tasks.common import KITE_CONTACTS, EpisodeScore

SPEC = TaskScore(
    stages=(
        Stage("lift", 0.4, lambda f: f["lift"] / f["lift_target"]),
        Stage("press", 0.6, lambda f: f["press"], gate=lambda f: f["lifted_ever"]),
    ),
    success=lambda f: f["lifted_ever"] and f["pressed_ever"],
    failures=contact_failures(),
    quality=contact_quality(),
    speed=SpeedReference(par_s=3.0, budget_s=6.0),  # not calibrated yet
    aggregate="peak",
)


def make_features(model, data, scene, task):
    """Feature function over the pump bottle. ``scene``: KITE's ``SceneSpec``; ``task``: its ``PumpBottleUseTask``."""
    rest, down = float(task.button_reset_slide), float(task.button_success_slide)
    latch = {"lifted": False, "pressed": False}
    watch = ContactWatch(model, task_roots=[scene.pump_bottle_body_name], **KITE_CONTACTS)

    def features(d):
        obs = task.observe(model, d, scene)
        lift = float(obs["lift_clearance_delta"])
        slide = float(obs["button_slide"])
        latch["lifted"] |= lift >= float(task.lift_success_delta)
        latch["pressed"] |= slide <= down
        done = latch["lifted"] and latch["pressed"]
        return {
            **watch.features(d),
            "lift": lift,
            "lift_target": float(task.lift_success_delta),
            "press": clip01((rest - slide) / (rest - down)),
            "lifted_ever": latch["lifted"],
            "pressed_ever": latch["pressed"],
            "kite_success": done,
        }

    return features


def make_score(model, data, scene, task, spec: TaskScore = SPEC, meta: dict | None = None) -> EpisodeScore:
    return EpisodeScore("pump_bottle", spec, make_features(model, data, scene, task), {"object_id": task.object_id, **(meta or {})})
