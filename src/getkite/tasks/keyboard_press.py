"""Score for KITE's Keyboard Pressing task (``KeyboardPressTask``): three target keys on a mini keyboard.

KITE's success rule, reported as ``kite_success`` and used as the success here too: all
three target keys down at the same control step (each past ``key_success_fraction`` of
its travel), i.e. a chord. Stages:

1. keys   - each target key's deepest press so far, towards the success depth (mean of the three)
2. chord  - once every target key has been down, how many are down together right now

Wrong keys (a non-target key pressed past the same depth) lower the quality, as wrong
notes do in CEBenchmark's MIDI-keyboard task; they are not a failure here. Read from the
keyboard's key joints only (via ``task.observe``), the same for any hand.
"""

from __future__ import annotations

import numpy as np

from getkite.scoring import ContactWatch, QualityTerm, SpeedReference, Stage, TaskScore, contact_failures, contact_quality
from getkite.tasks.common import KITE_CONTACTS, EpisodeScore

WRONG_KEYS = (0.2, 5)  # quality lost, and the number of wrong key presses that costs all of it
# Key depth (as a fraction of the success depth) that earns nothing: untouched keys sag
# to about 0.12 of it under gravity in KITE's scene.
KEY_DEADBAND = 0.25

# Keys take a few newtons: banging them is excessive long before it would be for a tool (as in CEBenchmark).
CONTACT_LIMITS = dict(force_limit=60.0, soft_force=10.0, **KITE_CONTACTS)

SPEC = TaskScore(
    stages=(
        Stage("keys", 0.6, lambda f: f["keys"]),
        Stage("chord", 0.4, lambda f: f["down_now"] / 3.0, gate=lambda f: f["all_down_ever"]),
    ),
    success=lambda f: f["kite_success"],
    failures=contact_failures(),
    quality=(QualityTerm("wrong keys", WRONG_KEYS[0], lambda f: f["wrong_keys"], WRONG_KEYS[1], total=True), *contact_quality()),
    speed=SpeedReference(par_s=3.0, budget_s=6.0),  # not calibrated yet
    aggregate="peak",
)


def make_features(model, data, scene, task):
    """Feature function over the keyboard's keys. ``scene``: KITE's ``SceneSpec``; ``task``: its ``KeyboardPressTask``."""
    obs0 = task.observe(model, data, scene)
    names = [str(n) for n in obs0["keyboard_key_joint_names"]]
    targets = [str(n) for n in obs0["target_key_joint_names"]]
    other = np.asarray([i for i in range(len(names)) if names[i] not in targets], dtype=int)
    frac = float(task.key_success_fraction)
    state = {"best": np.zeros(len(targets)), "ever": np.zeros(len(targets), dtype=bool), "wrong": 0, "other_down": np.zeros(len(other), dtype=bool)}
    watch = ContactWatch(model, task_roots=[scene.mini_keyboard_body_name], **CONTACT_LIMITS)

    def features(d):
        obs = task.observe(model, d, scene)
        target_depth = np.asarray(obs["target_key_slide"]) / np.asarray(obs["target_key_threshold"])  # 1 at the success depth
        down = np.asarray(obs["target_key_success"], dtype=bool)
        state["best"] = np.maximum(state["best"], np.clip((target_depth - KEY_DEADBAND) / (1.0 - KEY_DEADBAND), 0.0, 1.0))
        state["ever"] |= down
        if len(other):
            slide = np.asarray(obs["key_slide"])[other]
            lo = np.asarray(obs["key_slide_range"])[other, 0]
            other_down = slide <= lo * frac
            state["wrong"] += int(np.sum(other_down & ~state["other_down"]))  # count each new press once
            state["other_down"] = other_down
        return {
            **watch.features(d),
            "keys": float(np.mean(state["best"])) if len(targets) else 0.0,
            "down_now": int(np.sum(down)),
            "all_down_ever": bool(np.all(state["ever"])),
            "wrong_keys": state["wrong"],
            "kite_success": bool(obs["success"]),
        }

    return features


def make_score(model, data, scene, task, spec: TaskScore = SPEC, meta: dict | None = None) -> EpisodeScore:
    return EpisodeScore("keyboard_press", spec, make_features(model, data, scene, task), meta)
