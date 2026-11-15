"""Score for KITE's Objects Placing task (``ObjectSuitePickPlaceTask``): a YCB object into a bin.

Success is KITE's own rule (``task.is_success``), so success rates match KITE's: the
object's footprint is inside the bin's, with ``placement_margin`` to spare. Height does
not matter, so an object held in the air above the bin counts. Stages (each implies the
ones before it):

1. lift   - the object's lowest point rises towards ``task.lift_delta`` above its rest height
2. carry  - the object's footprint moves from where it started towards inside the bin;
            reaching it is success

Reported next to the score, not in it: ``placed``, whether the object was at some point
also lowered into the bin (its lowest point within ``task.bottom_tolerance`` of the bin
floor; KITE defines that tolerance but its success rule does not use it). KITE's
episodes (120 policy frames, 6 s) end with the object still in the hand, so letting go
is not asked for at all.

Everything is read from the object and the bin, so the score is the same for any hand.

    score = make_score(model, data, scene, task)   # after env.reset, and any pose overrides
    score.update(data)                              # every control step
    score.result()
"""

from __future__ import annotations

import numpy as np

from getkite.scoring import ContactWatch, LowestPoint, SpeedReference, Stage, TaskScore, clip01, contact_quality
from getkite.tasks.common import KITE_CONTACTS, EpisodeScore

SPEC = TaskScore(
    stages=(
        Stage("lift", 0.4, lambda f: f["lift"] / f["lift_target"]),
        Stage("carry", 0.6, lambda f: f["carry"], gate=lambda f: f["lift"] >= f["lift_target"] or f["over_bin"]),
    ),
    success=lambda f: f["kite_success"],
    # No failures: contact failures are flags (see common.FLAGS), so success stays KITE's rule.
    quality=contact_quality(),
    # Not calibrated yet: a KITE episode is 120 policy frames of 0.05 s, so 6 s is the time limit.
    speed=SpeedReference(par_s=3.0, budget_s=6.0),
    aggregate="peak",
)


def make_features(model, data, scene, task):
    """Feature function over the object and the bin. ``scene``: KITE's ``SceneSpec``; ``task``: its ``ObjectSuitePickPlaceTask``."""
    obj = model.body(scene.object_body_name).id
    qadr = int(model.jnt_qposadr[model.joint(scene.object_joint_name).id])
    lowest = LowestPoint(model, obj)
    lowest0 = lowest(data)
    floor_top = float(task.table_top_z) + float(task.bin_floor_thickness)

    def margin(d):
        return float(task.placement_xy_margin(task.world_bounds(np.asarray(d.qpos[qadr : qadr + 7], dtype=np.float64))))

    margin0 = margin(data)
    need = float(task.placement_margin)
    tolerance = float(task.bottom_tolerance)
    watch = ContactWatch(model, task_roots=[scene.object_body_name, scene.bin_body_name], watch=[scene.object_body_name], **KITE_CONTACTS)

    def features(d):
        m = margin(d)
        over = m >= need  # KITE's success rule (pick_place objective)
        height = lowest(d) - floor_top
        contact = watch.features(d)
        return {
            **contact,
            "lift": lowest(d) - lowest0,
            "lift_target": float(task.lift_delta),
            "carry": clip01((m - margin0) / (need - margin0)) if need > margin0 else 1.0,
            "over_bin": over,
            "height": height,
            "bottom_tolerance": tolerance,
            "placed": over and height <= tolerance,
            "held": scene.object_body_name in contact["touched"],
            "kite_success": over if task.objective == "pick_place" else float(d.qpos[qadr + 2]) >= float(task.pick_success_z),
        }

    return features


def make_score(model, data, scene, task, spec: TaskScore = SPEC, meta: dict | None = None) -> EpisodeScore:
    meta = {"object_id": task.object_spec.object_id, **(meta or {})}
    return EpisodeScore("objects_placing", spec, make_features(model, data, scene, task), meta)
