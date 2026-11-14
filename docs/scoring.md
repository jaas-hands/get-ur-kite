# Staged scores for KITE's tasks

KITE reports one bit per episode (success or not). For comparing kinematic decoders,
especially on lesioned hands where most episodes will fail, we want to know *how far*
an episode got and *how* it got there. `getkite.scoring` is CEBenchmark's scoring core,
unchanged; `getkite.tasks` holds one score spec per KITE task.

## How a score works

A task score is an ordered list of **stages**, each with an entry gate and a progress in
[0, 1]. At every control step the active stage is the last one whose gate holds, and all
stages before it count as done:

    score = (weights of the stages before k + weight_k * progress_k) / total weight

The episode's score is its peak by default. Reported next to it, never mixed in:

- **failure**: an attempt that cheats or breaks something (here: parts forced into each
  other, or sustained brute force). Progress freezes from then on, and success becomes
  impossible.
- **quality**: how it was done, e.g. contact force above a soft limit integrated over
  time, or wrong keys pressed.
- **speed**: time to success, normalised (references not calibrated for KITE yet).
- **`kite_success`**: KITE's own success rule, latched, so every run stays comparable with
  the paper's numbers.

Everything is computed from object state only, never from robot joints, so the same score
applies to every hand, including lesioned ones.

## The tasks

| KITE task | Module | Stages | Success | `kite_success` | Status |
|---|---|---|---|---|---|
| Objects Placing (`ObjectSuitePickPlaceTask`) | [objects_placing.py](../src/getkite/tasks/objects_placing.py) | lift, carry, lower | lowered into the bin (lowest point within `bottom_tolerance` of the floor) | footprint inside the bin, at any height | checked on Oracle-Intent rollouts of Barrett, Allegro and Wuji |
| Keyboard Pressing (`KeyboardPressTask`) | [keyboard_press.py](../src/getkite/tasks/keyboard_press.py) | keys, chord | = `kite_success` | all three target keys down at once | smoke-tested on KITE's scene; KITE ships no rollout config for it |
| Bottle Pumping (`PumpBottleUseTask`) | [pump_bottle.py](../src/getkite/tasks/pump_bottle.py) | lift, press | = `kite_success` | lifted 8 cm and button pressed, each at some point, in either order | not run: KITE's release does not include the bottle assets |

Cube picking has no score yet: no released configuration uses it.

## Can CEBenchmark's scoring be used directly?

- **The framework** (`Stage`, `Failure`, `QualityTerm`, `TaskScore`, `ScoreTracker`,
  `RewardShaper`): yes, unchanged. It only needs a feature function per task.
- **`LowestPoint` and `ContactWatch`**: yes. KITE's scenes have a body called `table`,
  and the hand, the object and the bin are separate bodies under the world, which is
  what `ContactWatch` assumes.
- **CEBenchmark's task scores**: no. KITE's pump bottle and keyboard are different assets
  (no lock-down, plate or soap; a 3-key chord instead of two MIDI chords), so each KITE
  task needed its own spec.
- **Two calibration changes were needed** (both in `getkite/tasks/common.py` and the task
  modules):
  1. *Hand penetration is not a failure for KITE.* CEBenchmark fails an attempt when the
     hand sinks more than 3 mm into a task part (calibrated on the Franka's pads). In
     KITE's shipped rollouts, Barrett's fingertips sit 2–2.6 cm inside the object for most
     of a successful episode, and Wuji's up to 1.5 cm. So this is reported as a diagnostic
     (`max.agent_penetration`), and only parts forced into each other count as failures.
  2. *Success does not require letting go.* KITE's episodes (120 policy frames, 6 s) end
     with the object still in the hand, so "released and resting in the bin" is never
     reached. Success is "lowered into the bin" instead, which uses KITE's own
     `bottom_tolerance` (defined by its task, not used by its success rule).

## First numbers (Oracle Intent, 10 episodes covering all six objects)

`online_rollout.indices=[20,22,37,40,55,58,72,75,90,93]`, released decoder checkpoint:

| Hand | KITE success | Staged success (in bin) | Mean score | Failures | Hand penetration (max, mm) |
|---|---|---|---|---|---|
| Barrett | 9/10 | 6/10 | 0.88 | 0 | 4–26 |
| Wuji | 8/10 | 2/10 | 0.66 | 1 | 3–16 |
| Allegro | 1/10 | 1/10 | 0.31 | 1 | 0.0–0.4 |

Two things stand out. KITE's success rule is lenient: most Wuji "successes" hold the
object above the bin and never lower it. And Allegro's fingertips barely penetrate
objects (under 0.4 mm) while Barrett's and Wuji's sink centimetres in. That is a likely
lead on why Allegro lifts objects and then drops them: the other hands may be holding
on partly through soft-contact interpenetration. This needs checking before we build
lesion experiments on Allegro.

## Running it

```bash
python scripts/rollout_scored.py --out outputs/scores/barrett.json -- \
    --config-name=rollout_gt_contacts/object_suite_barrett \
    "online_rollout.indices=[20,22,37,40,55,58,72,75,90,93]"
```

Any KITE script that steps `TabletopManipulationEnv` can be scored the same way
(`--script experiment/evaluate_object_suite.py` for the trained policy). In your own code:

```python
from getkite.kite_bridge import ScoringSession

with ScoringSession() as session:
    ...  # build KITE envs, reset, step
print(session.results())
```

`ScoringSession` wraps `TabletopManipulationEnv.reset/step` in this process. The score
starts at the first step after each reset (KITE overrides the object pose after `reset`).
Parallel rollout workers are separate processes and are not scored, so the runner forces
`online_rollout.parallel.num_workers=1`.
