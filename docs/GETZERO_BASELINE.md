# GET-Zero baseline

GET-Zero is useful as an architectural reference. For our experiments, the useful
comparison is our selected decoder against a version trained without graph biases.
The released GET-Zero policy rotates a cube; it would need new training for placement.

## How it works

GET is the **Graph Embodiment Transformer** architecture. GET-Zero also includes
the training procedure: train separate hand-specific experts, collect their
actions, then train one shared network to imitate them. An extra training head
predicts each joint's 3D position, encouraging the network to understand the hand.
The paper studies cube rotation across LEAP variants; it does not establish
transfer to arbitrary tasks or completely different hand families.
([Paper](https://arxiv.org/html/2407.15002v2))

Each joint has a token containing its state and geometry. Attention shares
information between joints, with learned biases based on their distance and
parent/child relationships in the hand. All joints can still communicate.
([Attention implementation](https://github.com/real-stanford/get_zero/blob/279652da160823521e881952f67870b0d98674ce/get_zero/distill/models/embodiment_attention.py))

| | Released GET-Zero | Our decoder |
| --- | --- | --- |
| Request | Continue the learned cube-rotation behaviour | Realize a requested contact intent |
| Inputs | Joint/target history, phase, hand geometry and graph | Frozen KITE intent code, starting pose, hand description; exact representation depends on variant |
| Outputs | One control value per joint | Wrist pose and joint configuration |
| Training examples | Expert state/action trajectories | Synthetic intent/pose examples |
| Simulation | Isaac Gym, fixed hand base | KITE/MuJoCo task setup with moving wrist |
| Main evaluation | Cube rotation speed | Contact errors; separately, task success |

The released tokenizer uses six history steps. Its local observations contain
normalized joint angles and PD targets, plus joint origins/rotations. There is no
KITE intent input or wrist output. The deployment controller turns the predicted
joint actions into changes to joint targets.
([Observations](https://github.com/real-stanford/get_zero/blob/279652da160823521e881952f67870b0d98674ce/get_zero/rl/tasks/leap_hand_rot.py),
[controller](https://github.com/real-stanford/get_zero/blob/279652da160823521e881952f67870b0d98674ce/get_zero/deploy/leap/deploy.py))

## What is already in our code

- [kite_trunk.py](../src/getkite/decoder/kite_trunk.py): KITE's shared decoder with
  geometry-conditioned joint tokens and shortest-path/ancestor/descendant biases.
- [get_scratch.py](../src/getkite/decoder/get_scratch.py): a separate GET-style
  decoder with wrist, joint and intent representations. This is an adaptation,
  not the original GET-Zero checkpoint.
- Our analytical forward kinematics/contact loss is not GET-Zero's learned joint
  position prediction head. The scratch model's optional contact-set head is also
  a different auxiliary task.

So “add GET as another decoder” may duplicate our main model. We should first
choose which existing decoder is the main result and what question the baseline
answers. The shared-slot model is another useful comparison, but changes joint
representation as well as graph biases; it is not a pure graph ablation or the
official KITE model.

## Three possible comparisons

| Option | What it would take | Recommendation |
| --- | --- | --- |
| Released GET-Zero rotation policy | Reproduce its rotation task; for MuJoCo, port resets, observations, control timing, targets and physics | Separate reproduction only; no meaningful placement comparison |
| GET-style contact decoder | Same frozen encoder, inputs, synthetic data, wrist/joint outputs and evaluation as ours | Feasible, but our code already implements this idea; name it an adapted architecture |
| Full GET-Zero-style placement policy | Matched placement demonstrations or trained experts across hands, task observations, wrist action head, shared-policy training and validation | A new training project; defer unless the team explicitly owns the compute/data work |

Porting the simulator alone would not teach the rotation weights to place objects.
The official release includes pretrained GET and ET weights, so new expert training
is **not** needed just to try the original rotation policy. Its supplied rotation
data would not supply the missing placement behaviour.
([Official setup and checkpoints](https://github.com/real-stanford/get_zero))

## A fair first experiment

1. Choose one main decoder/config and a hand-level train/validation/test split.
   A completely held-out family means none of that family's variants enter
   training or checkpoint selection. LEAP variants alone cannot establish that claim.
2. Train a second model with the same data, encoder, inputs, trunk freezing,
   initialization policy, training budget, seeds and losses, changing only graph
   distance biases. Select both checkpoints on validation. Turning biases off in
   an already-trained model is only a sensitivity test, not this baseline.
3. Compare contact position/normal errors, contact success and joint-limit
   violations on identical examples. Report collision checks as unavailable where
   assets are missing. Keep the goal/no-motion checks as evaluator sanity checks.
4. Once rollout integration works, compare on Simon's fixed episodes and starts.
   Keep task success separate from contact accuracy; record whether success means
   over-bin, lowered, or released. Oracle-intent runs must be labelled separately
   from learned-policy runs.
5. Add a KITE reference with its adaptation status clearly labelled. Target-hand
   fine-tuning is an adapted reference, not another zero-shot method. Report its
   target-hand data/training budget. Do not predict the winner before measuring.

**Config issue found:** `kite_trunk_get_ft.json` fine-tunes the trunk (empty default
`kite_freeze`, learning-rate multiplier 0.1). The existing
`kite_trunk_get_graph_off.json` freezes unpacker, hand projection, scene and action
modules and uses a different evaluation split list. They are not a one-change
comparison. Copy the chosen resolved training config and change only `tree_bias`
and the output destination. For the scratch model, graph-off removes distance
information but retains token-type biases and geometry features; describe it that way.

Optional later experiment: add the learned joint-position auxiliary head, keeping
everything else fixed. This tests another GET-Zero component but is lower priority
than a clean graph-on/off pair.

## Mac check

- Downloaded the official source and GET/ET checkpoints into the ignored
  `outputs/getzero_review/` folder. Installed five small supporting packages there,
  without replacing the working MuJoCo environment's packages.
- Loaded both checkpoints with strict parameter matching. Used the official URDFs,
  joint mappings and tokenizer for hands 001 (16 joints) and 139 (11 joints).
- Both models produced finite, exactly repeatable CPU outputs. GET also returned
  its joint-position predictions. This used synthetic observations, not a rollout.
- The two upstream CPU architecture tests passed: graph/joint-order consistency
  and padding invariance. The full three-test suite's CUDA-only test failed because
  this PyTorch build has no CUDA. No original source or checkpoint was modified.

| Hand | Input shape | GET action shape | GET joint-position shape |
| --- | --- | --- | --- |
| 001 | 1 × 300 | 1 × 16 | 1 × 16 × 3 |
| 139 | 1 × 210 | 1 × 11 | 1 × 11 × 3 |

This establishes **network compatibility**, not simulation success, inference
throughput or a reproduced paper result. The full supplied Isaac Gym workflow
needs its Linux/NVIDIA environment; it is not a native Mac setup. NVIDIA now marks
Isaac Gym deprecated. Colab GPU access alone does not verify that legacy simulation
stack will work. ([NVIDIA](https://developer.nvidia.com/isaac-gym))

Local evidence: [CPU results](../outputs/getzero_review/cpu_check.json),
[check script](../outputs/getzero_review/check_cpu.py),
[CPU tests](../outputs/getzero_review/cpu_architecture_tests.log),
[full test log including CUDA failure](../outputs/getzero_review/architecture_tests.log).
These ignored files are local, not uploaded to GitHub.

Rerun the checkpoint check from this repo with the downloaded files still present:

```sh
OMP_NUM_THREADS=1 .venv-mac/bin/python outputs/getzero_review/check_cpu.py
```

## Next steps

- Choose the main decoder and test split with the team. LEAP variants and an unseen
  hand family answer different generalization questions.
- Ask the decoder owner for a matched graph-off training run, keeping the data,
  trunk freezing, training budget and checkpoint-selection rule the same.
- Compare contact metrics first, then use Simon's fixed episodes for task scoring.
  Agree whether task success requires the object to be lowered or released.
- Keep a full GET-Zero placement policy as optional work; it needs its own task
  data and training, beyond running the released network.
