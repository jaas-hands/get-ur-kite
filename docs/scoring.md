# Staged scores for KITE's tasks

KITE reports one bit per episode (success or not). For comparing kinematic decoders,
especially on lesioned hands where most episodes will fail, we want to know *how far*
an episode got and *how* it got there. `getkite.scoring` is CEBenchmark's scoring core,
unchanged; `getkite.tasks` holds one score spec per KITE task.

**Success is KITE's own success rule**, for every task, so our success rates are KITE's
(`success == kite_success` in every result). The scores only evaluate rollouts: KITE's
intent encoder and policy never see them, so nothing has to be retrained for them.

## How a score works

A task score is an ordered list of **stages**, each with an entry gate and a progress in
[0, 1]. At every control step the active stage is the last one whose gate holds, and all
stages before it count as done:

    score = (weights of the stages before k + weight_k * progress_k) / total weight

Success makes the score 1. The episode's score is its peak, so a failed episode keeps
the credit for how far it got. Reported next to the score, never mixed in:

- **quality**: how it was done, e.g. contact force above a soft limit integrated over
  time, or wrong keys pressed.
- **flags** (whether each ever happened): `placed` (Objects Placing: lowered into the bin),
  and CEBenchmark's contact failures `forced_through` (parts forced into each other) and
  `excessive_force` (sustained brute force). In CEBenchmark these end an attempt; here they
  are only reported, because they would otherwise block a success KITE counts.
- **max**: the deepest the hand sank into a task part (`agent_penetration`), the deepest
  task parts sank into each other, and the largest net force.
- **speed**: time to success, normalised (references not calibrated for KITE yet).

Everything is computed from object state only, never from robot joints, so the same score
applies to every hand, including lesioned ones.

## The tasks

| KITE task | Module | Stages | Success (KITE's rule) | Also reported | Status |
|---|---|---|---|---|---|
| Objects Placing (`ObjectSuitePickPlaceTask`) | [objects_placing.py](../src/getkite/tasks/objects_placing.py) | lift, carry | the object's footprint is inside the bin's, at any height | `placed`: also lowered to within `bottom_tolerance` of the bin floor | checked on Oracle-Intent rollouts of Barrett, Allegro and Wuji |
| Keyboard Pressing (`KeyboardPressTask`) | [keyboard_press.py](../src/getkite/tasks/keyboard_press.py) | keys, chord | all three target keys down at once (each past 50 % of its travel) | wrong keys (quality) | checked on Oracle-Intent rollouts of Barrett, Allegro and Wuji, with a config assembled in `scripts/demo_task.py` |
| Bottle Pumping (`PumpBottleUseTask`) | [pump_bottle.py](../src/getkite/tasks/pump_bottle.py) | lift, press | lifted 8 cm and button pressed, each at some point, in either order | | not run: KITE's release does not include the bottle assets |

Cube picking (arXiv version of the paper only) has no score: the release has no data for it.

## Can CEBenchmark's scoring be used directly?

- **The framework** (`Stage`, `Failure`, `QualityTerm`, `TaskScore`, `ScoreTracker`,
  `RewardShaper`): yes, unchanged. It only needs a feature function per task.
- **`LowestPoint` and `ContactWatch`**: yes. KITE's scenes have a body called `table`,
  and the hand, the object and the bin are separate bodies under the world, which is
  what `ContactWatch` assumes.
- **CEBenchmark's task scores**: no. KITE's pump bottle and keyboard are different assets
  (no lock-down, plate or soap; a 3-key chord instead of two MIDI chords), so each KITE
  task needed its own spec.
- **Three adaptations** (in `getkite/tasks/common.py` and the task modules):
  1. *Success is KITE's rule*, so results compare with KITE's. For Objects Placing that rule
     is lenient (an object held in the air over the bin counts), so whether it was also
     lowered into the bin is reported as the `placed` flag.
  2. *Contact failures are flags, not failures*, for the same reason.
  3. *Hand penetration is not checked against CEBenchmark's 3 mm limit* (calibrated on the
     Franka's pads). In KITE's shipped rollouts Barrett's fingertips sit 2–2.6 cm inside
     the object for most of a successful episode, and Wuji's up to 1.5 cm, so it is
     reported under `max.agent_penetration` instead.

## Oracle-Intent numbers, against the paper

Released decoder checkpoint, KITE commit `1361595`, all assets verified by
`download_assets.py`, package versions as in KITE's `requirements.txt`. "Paper" is
KITE (Oracle Intent) in Table I of the ICRA 2027 version, 100 episodes per hand.

**Objects Placing** (KITE's shipped configs; 30 held-out episodes per hand, 5 per object,
indices `[1,4,7,10,13, 18,21,24,27,30, 35,38,41,44,47, 52,55,58,61,64, 69,72,75,78,81, 85,88,91,94,97]`):

| Hand | Paper | Ours | Pudding box | Bowl | Potted meat can | Mug | Sugar box | Mustard bottle | Mean score | Also lowered into bin |
|---|---|---|---|---|---|---|---|---|---|---|
| Barrett | 82 % | 20/30 (67 %) | 3/5 | 0/5 | 5/5 | 3/5 | 5/5 | 4/5 | 0.76 | 11/30 |
| Wuji | 85 % | 20/30 (67 %) | 3/5 | 2/5 | 4/5 | 3/5 | 4/5 | 4/5 | – | – |
| Allegro | 85 % | 5/30 (17 %) | 1/5 | 0/5 | 1/5 | 0/5 | 2/5 | 1/5 | 0.39 | 5/30 |

(Wuji's run was killed by the out-of-memory killer after 24 episodes when three hands ran
in parallel; its success counts come from KITE's log plus a re-run of the last 6, so its
score columns are missing. The first 27 episodes of the default order gave the same
picture: Barrett 16/27, Allegro 3/27.)

**Keyboard Pressing** (20 shipped episodes; config assembled by `scripts/demo_task.py`,
see below):

| Hand | Paper | Ours | Mean score |
|---|---|---|---|
| Barrett | 99 % | 1/20 | 0.55 |
| Allegro | 97 % | 15/20 | 0.83 |
| Wuji (the source hand, replaying its own demonstrations) | – | 14/20 | 0.86 |

What this says:

- **We do not reproduce the paper's Objects Placing numbers**, with KITE's own script and
  shipped configs: Barrett and Wuji are about 15 points low, Allegro about 70. This is an
  open problem to raise with the KITE authors before we build on Allegro. Ruled out so far:
  missing or corrupt assets (all verified), package versions (identical), GPU inference
  (CPU gives the same), a latent/checkpoint mismatch (the shipped latents were encoded with
  the released checkpoint), the start pose (Objects Placing uses the cube-pick start pose by
  design), and our scoring (success is KITE's own count). The release rejects the execution
  helpers in its configs (`grasp_hold`, `phase_finger_effort`) as "outside the paper method".
- **A lead on Allegro:** its fingertips barely penetrate objects (under 0.4 mm) while
  Barrett's and Wuji's sink centimetres in, and it lifts objects and then drops them. Its
  contact settings in the shipped config (`allegro_hand_collision_model: fingertip_mesh_only`,
  fingertip pads off) may differ from what produced the paper's numbers.
- **The keyboard numbers are not a reproduction attempt.** KITE ships the keyboard data and
  start poses but no rollout config, so ours reuses each hand's Objects Placing physics and
  controller settings. That Wuji replaying its own demonstrations reaches only 14/20 says
  this config is not the authors'. The mean scores show most failures are near misses.
- **KITE's Objects Placing success rule is lenient:** only about half of Barrett's
  successes also lower the object into the bin within the 6 s episode.

## What KITE's release contains

| Piece | In the release? |
|---|---|
| Intent encoder + decoder bank (`checkpoints/decoder/release.pt`) | yes: Wuji, Allegro, Barrett, Robotiq+Stick, MANO. No XHand (the nut-rotating source hand). |
| Interaction policies | no. Train one for Objects Placing with KITE's README steps (2,500 demos, then iDP3). |
| Objects Placing | everything: scene, 100 held-out Oracle episodes, rollout and evaluation configs. |
| Keyboard Pressing | scene, 20 Oracle episodes, start poses (Barrett, Allegro, Wuji, Robotiq+Stick). No configs. |
| Bottles Pumping | 20 Oracle episodes and start poses, but not the bottle meshes: cannot be simulated. |
| Nut Rotating | 20 Oracle episodes and alignments, but no simulation code in the repo. |
| Cube picking | only in the earlier arXiv version of the paper; no data in the release. |

So cross-embodiment can be watched today as *Oracle Intent* (recorded source intents through
the decoder) on Objects Placing and Keyboard Pressing. Closed-loop policy demos need a
trained policy first.

## Running it

To *watch* episodes with the score drawn live (viewer or MP4), see "Watching a task" in the
[README](../README.md) (`scripts/demo_task.py`). To score many episodes and save the numbers:

```bash
python scripts/rollout_scored.py --out outputs/scores/barrett.json -- \
    --config-name=rollout_gt_contacts/object_suite_barrett perception.include_camera_obs=false \
    "online_rollout.indices=[1,4,7,10,13,18,21,24,27,30,35,38,41,44,47,52,55,58,61,64,69,72,75,78,81,85,88,91,94,97]"
```

`perception.include_camera_obs=false` skips the camera images, which Oracle-Intent replay
does not use. Run one hand at a time on a laptop: each rollout process grows to several GB,
and three in parallel exhausted 15 GB.

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
