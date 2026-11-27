# Track G: source-morphology manifest, splits and resource estimate (draft, 2026-11-19)

> **Status (2026-11-27): ran as G1** (48 LEAP training topologies; runs g1_leapdiv50k_hf*); results in docs/decoder_audit.md
> sections 45 and 46. G1 is the root of the D1 -> CF -> CF2 -> CF-3e config lineage.


Status: G0 written down. No data has been generated. The G1 run is not launched.

## Coverage choice

**Wuji stays excluded from source training (design 1).** Wuji remains a *development* target: unseen by our decoder, seen by
KITE's encoder (ψ). Every Wuji result stays "development transfer", and it is reported separately from the final test.
Reason: it keeps the J1 / two-stage / direct Wuji numbers comparable with G1. Adding Wuji (design 2) would also need a
reserved external family, and none exists here (below).

## What exists today (measured, not assumed)

| family | bodies in J1 source | procedural variation available | DOF | fingers | ψ saw it |
|---|---|---|---|---|---|
| Allegro (KITE URDF) | 1 nominal | **none**. A single URDF, KITE template. | 16 | 4 | yes |
| Barrett (KITE URDF, mimic ignored) | 1 nominal | **none** | 8 | 3 | yes |
| LEAP (kite_data) | 12 of 966 train hands | 243 topologies (2–4 fingers, 2–4 joints each, locked from the tip), geometry draws (link stretch 0.8–1.2 on long links, ±4 mm index/middle/ring mount shift) | 4–16 | 2–4 | **no** |
| Wuji (KITE URDF) | dev only | none | 20 | 5 | yes |

- **Extrapolation:** no source body has 5 fingers or more than 16 DOF. Wuji (20 DOF, 5 fingers) is therefore an
  intentional extrapolation in every G1 result. Padding to 24 joints does not mean it is covered.
- **Topology vs geometry.** A LEAP topology `I?-M?-R?-T?` (joints kept per finger) is a distinct graph. `_gN` is a
  geometry draw of the same graph. All LEAP variants share one palm and one 4-joint-per-finger template, so LEAP is one
  family, however many variants it has.

## Missing pieces (explicit)

1. **Allegro/Barrett morphology generator: does not exist.**
   - KITE's sampler (`gen_kite_family.py`, which uses `ContactPolicyProbeDataset`) is tied to KITE's robot registry.
   - kite_data's `Hand`/`HandSampler`/`Collider` are LEAP-specific: finger regions, box collider, LEAP link types.
   - Physically coherent variants need, per variant:
     - an edited `*_extended` URDF: per-finger uniform scale of joint origins *and* link meshes, palm mount offsets, and
       distal joints locked to fixed;
     - a regenerated 4096-point surface template, with collision primitives scaled to match;
     - registration with whichever sampler is used;
     - validation: FK vs the native model (MuJoCo), limits, coupling (Barrett mimic), and surface coverage.
   - Estimate: about 2–4 days of engineering before any data. The alternative, porting Allegro/Barrett into
     kite_data's generic-hand format, costs about the same.
2. **External final-test family: none ready.**
   - KITE ships only collision-primitive fits for Shadow, Inspire, Ability, Schunk, XHand, MetaHand, EZGripper and
     Robotiq-3F.
   - MANO `.pkl` is absent. The local menagerie checkout is sparse (leap_hand only).
   - One candidate needs a download of its official model and meshes (e.g. Shadow or Inspire from
     mujoco_menagerie / robot_descriptions), then an adapter, template, FK validation and 500 eval-only goals.
   - About 1–2 days. **Needs your OK to download.**
   - Neither Shadow nor Inspire was seen by ψ.

## Proposed manifest (design 1). Fixed before any generation.

**Train source bodies (target 46). Spread across families, no unphysical joints.**

| family | n | how |
|---|---|---|
| LEAP | 24 | The 12 J1 hands plus 12 more train-split hands. Chosen to cover every finger count (2/3/4) and DOF range 4–16, and at least 6 geometry draws ≥ 1. |
| Allegro | 12 | Nominal plus 11 variants. Per-finger uniform scale 0.85–1.15, index/middle/ring mount shift ±4 mm along the palm. Topologies: all 4×4 joints, and distal-joint locked on 1–2 fingers (3-joint fingers). One ring-removed graph. |
| Barrett | 10 | Nominal plus 9 variants. Per-finger link scale 0.85–1.15, spread-joint range kept. Topologies: all 3 fingers, and one 2-finger graph (finger 3 removed). |

**Sampling.**
- About 60k independent goals in total, balanced by family first (20k each), then by body.
- Per body: LEAP about 833, Allegro about 1,667, Barrett 2,000.
- In J1: LEAP 1,667, Allegro 20,000, Barrett 20,000.
- Same KITE noise/contact recipe, starts, 50k × 256 steps, J1 architecture, losses and optimiser.

**Splits** (no body in more than one; written to a JSON manifest with hashes before generation):

| role | content | used for |
|---|---|---|
| train | the 46 bodies above | training only |
| source val (unchanged) | existing J1 val: Allegro/Barrett nominal 500 each, 12 LEAP × 42 rows | checkpoint selection only |
| within-family dev | the 4 held-out LEAP (already inspected) | development |
| within-family final (reserved) | LEAP: 8 hands from `test_joint_count`/`test_combined` (never loaded so far). Allegro: 4 variants and Barrett: 3 variants from reserved generator seeds. Labelled separately as geometry / topology / joint-count changes. | final only, once |
| new-family dev | Wuji | development |
| external final (reserved) | Shadow or Inspire, if approved | final only, once, after design lock |

## Resource estimate

- **Records.** About 2,000 records/s on CPU (measured: the 10x pool was 48k records in 38 s; LEAP 12k records in about 4 s).
  60k records take minutes. Assets dominate.
- **Assets.** About 25 Allegro/Barrett variant templates, minutes each on CPU. The generator itself is the 2–4 days above.
- **Training.** One J1 run is about 1.5 GPU-h (50k × 256 at about 2,500 ex/s, 1.1 GB peak, like direct_worldaux_v1).
  Plus benchmark evaluation, about 5 min.
- **G2** (4× goals, same steps): the same GPU cost and minutes of generation, only after G1.

## Decision needed before step 4

Either:
- (a) build the Allegro/Barrett generators (2–4 days) and keep the cross-family manifest above; or
- (b) start G1 now with LEAP-only variation (Allegro/Barrett nominal). That departs from "spread variations across
  source families", so it would be labelled a LEAP-diversity run.

Separately: approve downloading one external hand model for the reserved final test.
