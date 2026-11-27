# GET Your KITE: decoder project handover (state as of 2026-11-25, ~15:40 EST)

> **Superseded (2026-11-27).** The delivered model is now **CF-3e seed 0**, a single model (no seed ensemble); see
> [cf3e_delivery.md](cf3e_delivery.md). Statements below about D1 / D1-bv as the delivery default or delivering a
> 3-seed ensemble describe the state on 2026-11-25. The rules, architecture notes and arm history below still apply.

This file is written for a new Claude session (or a collaborator) picking up the decoder work. It summarises what the
project is, the rules every session must follow, the architecture, every model arm we have trained, what each taught us,
and what is running now. Primary sources are linked throughout; when this file and a primary source disagree, **the
primary source wins** (especially `docs/decoder_audit.md`, which is the numbered lab notebook, §1–§63e).

---

## 0. Read this first: standing rules (non-negotiable unless the user changes them)

**Repository and git**
- Never push. Never `git add` `.md` files. Commit nothing unless the user explicitly says so.
- No Claude attribution and no `Co-Authored-By` lines in anything.
- Use only the official KITE repo (`helloqxwang/KITE` @ `1361595`), and never modify the KITE checkout.
- Don't name anything "g2".

**Experimental discipline**
- **ψ (KITE's intent encoder) stays frozen. Never retrain ψ.**
- Change one thing at a time. No extra variants, arms or seeds without the user's approval.
- Long or large GPU jobs (anything beyond a short smoke test or a ~30 min evaluation) need the user's go-ahead first.
  Report the ETA. If the queue slips, report the revised timeline; do not start extra experiments to fill time.
- **Select checkpoints on source validation only.** Never select or tune on Wuji, Shadow, held-out LEAP or held-out
  generated bodies.
- 3 seeds (0/1/2). Results are reported **per seed, paired by seed**. No seed is ever chosen as "best"; evaluation
  examples are never pooled across runs. Claims need 3/3 seeds. By convention, when one checkpoint must be named for
  delivery, it is seed 0 (convention, not selection).
- Predeclare the readout (primaries, thresholds) in a proposal doc **before** results exist. If the user adds a
  criterion, add it to the doc before results and say so.
- Preserve all existing runs and configs. Frozen reference runs are read-only with `SHA256SUMS` (section 8).

**Wording rules for reports**
- Raw contacts used as decoder input in diagnostics are called the **"diagnostic reference"**. Only deployment-time
  inputs (ψ's z, the robot description/template, the start state) may reach the decoder.
- Do **not** treat z as coordinates and do **not** apply Fourier features to z.
- Report **near-goal (same-body) Wuji separately** from cross-body results, and always label Wuji
  **"unseen by our decoder, seen by KITE's encoder (ψ)"**.
- **Shadow is the reserved final target**: never train, select or tune on it. Evaluating on it is allowed (user approved
  2026-11-20). Label: "unseen by decoder and ψ".
- Optimizer results are phrased as **"solutions found for X%"**, never as reachability proof.
- Don't describe any model as broadly "universal".
- Keep the **direct track** (z-only decoders) separate from the **W1/J1 track** in all reporting.
- Always separate "improvement over D1 (or the reference)" from "transfer beyond no movement".

**Cluster (Cornell Unicorn)**
- Partition: **`default_partition` only. Never use or propose the `dutta` partition** (not this lab's; user, 2026-11-25).
- GPU constraint: `--constraint="ampere|ada|2080ti"` (venv torch 2.15+cu132 has no Pascal/Volta kernels).
- GPU exclude list (broken/old drivers or very slow):
  set as `GETKITE_EXCLUDE` (training) and `GETKITE_EXCLUDE_EVAL` (benchmarks) in the gitignored `slurm/env.local.sh`;
  the chain scripts read them (cluster-specific; ask the team for the current list).
- CPU data jobs that use MuJoCo need `--constraint=avx2` (old CPUs SIGILL, which looks like a Pool hang).
- `squeue` only shows your own jobs; `sbatch --test-only` gives projected start times.

---

## 1. The project in one page

CS 6758 (Cornell, Fall 2026) course project. KITE (Wang et al., 2026) splits manipulation into a shared policy that
predicts a contact-based **latent intent z** (from KITE's frozen intent encoder **ψ**) and a **per-robot kinematic
decoder** that turns the intent into joint targets and a wrist pose. Every new hand needs its own decoder.

**Our goal:** one decoder, with weights shared across hands and conditioned on the robot's description (URDF graph,
GET-Zero style joint tokens with kinematic-tree attention bias), that turns ψ's intent into motion for hands it has never
been trained on, zero-shot, with ψ frozen.

**Task definition used throughout ("near-goal contact realisation"):** given a start hand state (joints + wrist pose,
perturbed from a goal grasp) and the intent z of a goal contact set, output joints + wrist pose so the hand's surface
realises the requested contacts. Scored against a **no-movement baseline** (stay at the start).

**Hand families**
- Training families: **Allegro** (one hand), **Barrett** (one hand), **LEAP** (procedural variants from
  get-ur-kite-dataset; 48 training hands in A, 414 in B), **generated primitive bodies "prim"** (procedural hands from
  `kite_data.prim_hand` / `build_prim`; P0000–P0049 in D1/A, P0000–P0799 in D3/B).
- Held-out (evaluation only): **held-out LEAP** (4 hands in the standard benchmark; all 105 dataset val-split hands in
  the widened evaluation), **held-out generated bodies** P0800–P0849.
- External hands: **Wuji** (unseen by our decoder, seen by KITE's ψ) and **Shadow** (unseen by both; reserved final
  target; MuJoCo Menagerie E3M5, Apache-2.0, adapted via `external_assets/shadow_adapter`).
- Not available: XHand decoder (not in KITE's release; XHand integration is explicitly out of scope for now).

---

## 2. Environment, paths and how to run things

- Workspace: `~/code/uni-kite/` containing `get-ur-kite/` (this repo, code in `src/getkite/decoder`),
  `get-ur-kite-dataset/` (data, `data/hands`, `data/examples/kite_code/{train,val,test_*}`, `data/splits.json`),
  `KITE/` (official KITE checkout), `getkite_runs/` (all run outputs), `getkite_data/`, `external_assets/`.
- Python: `~/envs/get_ur_kite/bin/python` (on the login node `GETKITE_PY` may resolve to a bare python; use the venv
  path explicitly). `PYTHONPATH=src:../get-ur-kite-dataset`.
- `source slurm/env.sh` from the repo root sets `GETKITE_RUNS` (default `../getkite_runs`), `GETKITE_DATASET`,
  `KITE_ROOT`, etc. (machine overrides in gitignored `slurm/env.local.sh`).
- MuJoCo hand checks need `GETKITE_LEAP_DATA=$GETKITE_DATASET`.
- **Training (J1 family, i.e. every model in sections 4–6):**
  `sbatch -p default_partition --constraint="ampere|ada|2080ti" --exclude=<list> -J kr-<name> -t 05:00:00 slurm/joint.sbatch configs/decoder/scratch/<cfg>.json [--set k=v ...]`
  (`slurm/joint.sbatch` → `python -m getkite.decoder.train_joint`). **Not `slurm/train.sbatch`** — that is the older
  LEAP-only trainer (`getkite.decoder.train`) and rejects J1 configs. Output dir = the config's `out_dir`; resumes from
  `last.pt`; writes `best.pt` (selected on source val: macro success, then macro contact mm), `log.jsonl`,
  `canonical_frames.json` (CF), `DONE` (best step).
- Smoke test pattern: `... -t 01:00:00 slurm/joint.sbatch <cfg> --set 'out_dir=${GETKITE_RUNS}/cf_dev/<name>_smoke' total_steps=200 log_every=50 eval_every=200 ckpt_every=200`.
- **Benchmark:** `sbatch ... -J kr-bench-<name> slurm/benchmark.sbatch configs/decoder/benchmark/<cfg>.json`
  (`scripts/decoder/benchmark/transfer_benchmark.py`; writes `metrics.json`, `per_example.pt`, `manifest.json` with
  SHA256 of every input). ~10 min for 15 systems.
- **Frame-relabelling test:** `slurm/frame_relabel_<arm>.sbatch` (`scripts/decoder/benchmark/frame_relabel_eval.py`):
  24 axis-aligned base-frame relabellings; CF models must give spread ≈ 0.
- **Auto-evaluation chains:** `slurm/<arm>_chain.sh` run with `nohup` from the repo root: wait for all seeds' `DONE`,
  submit benchmark(s) + relabel test, retry once, log to `slurm/logs/<arm>_chain.log`.
- Readout scripts (all in `scripts/decoder/benchmark/`): `cf2_summary.py` (Primaries 1–3, per-family, vs no movement),
  `cf_summary.py`, `seed_summary.py`, `followup_summary.py`, `leapval105_summary.py` (105-hand held-out, hand-level
  bootstrap, by finger count and frame angle), `leapval105_angle.py`, `frame_perturb_summary.py`.

---

## 3. Metrics (see `docs/scoring.md`, audit §33)

Per example, scored in the robot's original (native) bank with exact FK:
- `corr_mm`: mean distance between requested contacts and the corresponding hand-surface points after the move
  ("contact error"; the main number). Lower is better.
- `corr_success`, `compat_success`: success under the position/normal thresholds; `compat` uses the
  **normal-compatible** rule (nearest compatible surface point, 15° / 10 mm; approved 2026-11-17 because the original
  nearest-surface angle was degenerate).
- `wrist_mm`, `wrist_deg`: wrist translation / rotation error to the goal wrist.
- `trans_R2_id`, `rot_R2_id` (`transfer_benchmark.wrist_fit`): in the start-wrist frame, need = required correction,
  got = predicted correction; R²_id = 1 − Σ|got − need|² / Σ|need − mean need|². 1 = exact, 0 = no better than the
  mean required correction, < 0 = worse. Also `slope` = Σ⟨got, need⟩ / Σ|need|² (1 = full-size correction).
- **no movement**: the start state scored as the answer. "Transfer beyond no movement" = significantly better than it.
- Paired comparisons: within seed, paired bootstrap over the same records (95% CI in `metrics.json` `_paired`).
  "Significant" = CI excludes 0. For the 105-hand evaluation we also bootstrap over **hands**.

---

## 4. Architecture: the J1 decoder (current backbone of every model)

Introduced in audit §41–42 (2026-11-18), `src/getkite/decoder/joint_contact_get.py`, trainer `train_joint.py`.

```
goal contact set ──► frozen KITE ψ ──► z
z ──► ContactSetBridge (world-frame contact-set predictor, random init; z only)
   ──► pack slots (pos·v, normal·v, v), ordered by existence logit, hard mask
   ──► GET motion decoder (raw-contact GET, random init; exact analytic transform into the start-wrist frame;
        joint tokens from the URDF graph + surface tokens from the hand template; kinematic-tree attention bias)
   ──► Δjoints + wrist pose
```
- One module, one checkpoint, trained jointly: contact-set loss (weight 1.0) + motion loss. Motion gradients reach the
  contact predictor through positions/normals.
- Decoder config (D1 and successors): `d_model 128, n_layers 4, n_heads 4, mlp_ratio 4, pos_fourier 8, rot_fourier 8,
  contact_fourier 8 (wrist-relative contact positions, NOT z), intent_source raw_contacts, frame wrist`;
  ~0.99M decoder + ~0.76M contact-predictor params.
- Training: batch 256, lr 3e-4 (contacts lr 1e-3), warmup 500, cosine to 0 over `total_steps` (50k standard; 100k for
  the "-long" arms), weight decay 1e-4, normal noise 0.1 rad on ψ's input, start "resample".
  `BalancedSampler`: each family gets 25% of every batch (Allegro / Barrett / LEAP / prim).
- `surface_sampling: "hand_frame"` (audit §37–38 fix: the earlier link-local FPS put most surface tokens on the palm for
  URDF hands; every model since §38 uses `hand_frame`).
- Data budget (A recipe): Allegro 20k, Barrett 20k, LEAP 48 hands × 417, prim 50 × 400 records (rows chosen to never
  overlap val / fresh-eval rows).

**Canonical frame (CF), audit §63, `docs/cf_canonical_palm_frame_proposal.md`:** an optional wrapper
(`src/getkite/decoder/canonical.py`, `robots.canonical_frame` / `canonical_bank`). From each robot's description and
surface template only (deployment-time inputs), compute a palm frame by **rule E**: principal axes of the palm points;
d = the axis best aligned with the mean finger direction (signed by it), n = the remaining axis best aligned with the
finger-curl direction, a = d × n. The robot is re-expressed with `relabel_base(desc, Q)` (world geometry unchanged,
verified 6e-8 m); start/goal wrist rotations become R·Q and the output is mapped back R'·Qᵀ. Scoring stays in the native
bank. Exactly invariant to how a hand's base frame is labelled (relabelling spread = 0). Known weakness: on LEAP the two
palm eigenvalues are close (ratio ~0.75), so the frame wobbles a few degrees per hand (median 3–4°, p90 ~9°) and ~2% of
hands flip by ~90°.

**CF-J (running now):** CF + training-time frame jitter (`canonical_jitter` in `train_joint.py`,
`canonical_jitter_bank`): 32 fixed tilts (block 0 = identity, others random axis × U[0°,15°]); each training example
draws one. Evaluation/selection unjittered.

---

## 4b. Design choices at a glance (and the evidence behind each)

| Choice | Why (audit §) |
|---|---|
| ψ frozen, fed world-frame contacts exactly as KITE does | Project premise; keeps KITE's intent space and its other hands intact (§1–§3) |
| GET-style decoder: URDF joint tokens + surface tokens, kinematic-tree attention, no per-robot ids | One set of weights for any hand; tests enforce no id leaks (§6–§7) |
| Work in the **start-wrist frame**; analytic R0ᵀ(x−t0) outside the network | World-frame scratch decoders never learned the mm-scale correction (§5, §11–§15, §21); the wrist frame made C3 work (§19) |
| Wrist-relative contact Fourier (8 octaves); **no Fourier on z** and z not treated as coordinates | cf8 needed for translation (§18–§19); world-frame Fourier hashes goals (§13–§14) |
| Explicit internal **contact set** predicted in the world frame from z, then exact transform | Centroid was the wrong quantity (§26); learning the transform inside the net fails (§27); world target + analytic transform works (§30–§31) |
| **J1**: contact predictor + GET trained jointly, one checkpoint | Matches two-stage, simpler to ship (§41–§43) |
| `surface_sampling: hand_frame` | Fixes palm-heavy surface tokens (§37–§38) |
| Balanced family sampling, 25% per family | Family shares fixed so arms are comparable (§55, §58) |
| **Generated bodies** (50 is enough) as a 4th family | Body variety drives Wuji/Shadow transfer (§55–§59) |
| **Canonical palm frame** (rule E) | Decoder used each family's base-frame convention as a cue (§61 W2-a); random frames cost source accuracy (W2-b); a consistent geometry-derived frame keeps source and fixes Shadow (§63) |
| Frame jitter in training (CF-J, testing) | CF is causally sensitive to its own per-hand frame error (§63e) |
| Selection on source val; 3 seeds paired; predeclared readouts | Seed-to-seed spread is as large as many effects (e.g. §45 → §46, §55 → §56) |

---

## 5. History of approaches (summary; details in audit)

Naming collisions: "C1/C2/C3" in §11–§35 are **training gates** (C1 = 48 fixed examples with a fixed start, i.e.
memorisation; C2 = the same 48 goals with the start redrawn per batch; C3 = ~2k (later 20k) goals per family, scored on
fresh validation goals). "C1" in §58 is a different thing, the **family-share control**. "D1" in `direct_frozentrunk_d1`
(Track D) is unrelated to the morphology level D1.

### 5a. Foundations: KITE, ψ and what we copy from KITE (audit §1–§10)

- **KITE**: `helloqxwang/KITE` main @ `136159544aea…` (2026-11-08), unmodified. Decoder bank
  `checkpoints/decoder/release.pt` (epoch 550, step 4,297,150). KITE's released per-robot decoders are the in-family
  upper reference (e.g. on near-goal Wuji, which KITE trains on: 3.16 mm / corr success .954; audit §47).
- **ψ** = KITE's `_ContactOnlyEncoder`. Input: up to 6 world-frame contacts [xyz, outward normal] + a validity mask (no
  scaling or centering). Linear(6→256) → 2 self-attention blocks (Q/K also see raw xyz) → CLS cross-attention →
  **z ∈ ℝ²⁵⁶**. Co-trained with KITE's decoder bank on **wujihand_right, allegro_right, barrett, robotiq_gripper_stick,
  mano_right**. Not in ψ's training: LEAP, Shadow, Inspire, Ability, XHand, etc.
- **Task recipe copied from KITE (§3):** goal q2 (joints within limits, base U[−0.4, 0.4]³ m, uniform SO(3)); 2–6
  contacts from the 4,096-point surface template at q2 (`region_uniform`); start q1 = q2 + noise (translation
  N(0, 5 mm²), rotation N(0, 5°²) per Euler angle, joints N(0, (0.05·range)²), clamped); residual
  [Δxyz/5 mm, (6D−I)/5°, Δq/(0.05·range)], plain MSE. Normals jittered ~0.1 rad in training. Barrett uses
  `ignore_mimic=True` (8 independent coordinates) to match KITE. The 10 mm / 15° success threshold is **ours**; KITE has
  none.
- **Interface (§6–§7):** pad to `J_MAX = 24` joints; **no** canonical / joint / family / embodiment ids (embeddings only for
  graph-distance buckets and 4 token types); tests check padding has no effect, mixed batch = single robot, and joint
  permutation equivariance. FK matches KITE to < 0.1 mm. LEAP data comes from kite_data's `kite_code` regime
  (self-collision filter, own contact regions) — a deliberate deviation.
- **Deployment inputs only:** robot description, start pose, z = ψ(request). Goal pose, goal config, template
  correspondences, true count/validity are supervision or scoring only (enforced by tests).
- **Pilot data (§8):** Allegro/Barrett from KITE's own sampler (train seed 101, val seed 202); LEAP train rows [0,167),
  val rows [900,942); held-out LEAP 4 hands × 125 records. Selection = validation macro success, then macro contact mm.
- **Separate early LEAP-only line (pre-scope-change, 2026-11-14):** GET joint tokens vs a KITE-inspired shared-slot
  baseline on KITE's fine-tuned trunk: val 7.64 mm / 48.0% vs 7.80 / 46.3% (no movement 14.76 / 6.8%). LEAP-only
  reference, not part of the current line. Then the user broadened scope to one scratch decoder across families
  (`get_universal_scratch`).

### 5b. Making a cross-family scratch decoder learn at all (§5, §11–§31, 2026-11-14 → 11-17)

| § | What was tried | Result | Decision / lesson |
|---|---|---|---|
| §5, §11 | World-frame scratch GET (raw contacts or ψ) | C1 memorises; C2 needs pos+rot Fourier; C3 = no movement (11.5/14.8/14.5 vs 11.6/14.5/13.9) | World-frame fresh nets don't learn the mm-scale, start-dependent correction |
| §12–§13 | 50k steps, ψ vs raw world contacts | Both memorise train (~8.3–8.9 mm), val worse than no-op (17–18 mm) | Not ψ-specific. Fine world-frame contact Fourier hashes goals (shift sensitivity ~1 per mm) |
| §14–§15 | No / coarse contact Fourier; 10× goals (20k/family) in world frame | Smoother but still ≈ no-op | More data alone doesn't help in the world frame |
| §16–§18 | **Wrist frame** (express everything as R0ᵀ(x−t0)); fp32; translation-only gate | C2 passes only with **wrist-relative contact Fourier (8 octaves)**: ~1 mm / 100%; fp32 = bf16 | Wrist frame + cf8 is the working recipe |
| §19 | C3 `c3_raw50k_cf8_10x_wrist` (raw contacts, wrist frame, 10× pool) | **First working C3**: 5.80/5.66/7.99 mm (no-op 11.55/14.45/13.93), held-out LEAP 8.51; train ≈ val | Raw-contact decoder generalises; translation is the weakest part |
| §20–§21 | Frozen ψ in wrist frame (z standardised, **no Fourier on z**, world start pose via `world_pose_in`) | C2 passes but it is a 48-goal lookup; C3 = no movement. z linearly holds the contact centroid to < 1 mm | The net can't relate world z to the world start pose at mm precision |
| §22–§23 | ψ + KITE's frozen scene encoder (`psi_scene`) | Works on Allegro/Barrett (5.35/4.60) but memorises LEAP (10.92; held-out 13.02) | Comparison only; main model stays plain-ψ scratch GET |
| §24–§26 | Spatial bridge: z + pose → contact **centroid** in wrist frame | Bridge 1.1–1.5 mm, but decoder still = no-op; centroid explains R² 0.02 of translation, contact-set-vs-surface offset 0.49 | Centroid is the wrong quantity; need the contact **set** |
| §27–§29 | Contact-**set** bridge learning world→wrist internally | 26 mm (fails); raw control with analytic wrist mapping 2.4 mm (passes) | Learning the rigid transform inside the net is the failure |
| §30 | z → **world-frame** contact set, then **exact analytic** wrist transform | 2.7–2.9 mm fresh, held-out 3.0 mm | Predict in world, transform analytically |
| §31 | **Two-stage**: frozen z→set bridge + frozen raw-contact decoder | 6.44/6.48/8.39 mm, held-out 9.11 (keeps ~90% of the true-contact gain) | Basis for J1 |

Dead ends not to retry naively: world-frame scratch decoders; fine world-frame contact Fourier; Fourier on z; treating C2
passes as generalisation; centroid bridges; nets that must learn the world→wrist transform internally.

### 5c. External hands, J1, and the direct track (§32–§54, 2026-11-17 → 11-19)

- **§32–§33 Wuji + cross-body:** the frozen two-stage system does **not** transfer to near-goal Wuji (14.38 mm vs no
  movement 14.19). Cross-body success ≈ 1% (optimizer found solutions for 44.7%, but feasible cases need ~52 mm / 33° of
  motion; decoder moves ~9 mm) → cross-body is limited by the start distribution and same-body training, reported
  separately. **Normal-compatible scoring rule approved** (nearest template point with normal within 15°; success = all
  contacts ≤ 10 mm; unmatched = 1 m penalty, so success rate is primary).
- **§34–§37:** direct GET on ψ tokens + training-only aux contact head: gates pass, C3 = no movement. Probe: `intent_proj`
  already throws away world position (centroid 1.8 → ~42 mm). Independent review found the **surface-sampling bug**
  (link-local FPS → palm-heavy tokens; Wuji had two fingers with no tokens) and an aux-metric bug; conclusions qualified
  ("staged works better than *this* direct implementation").
- **§38:** `surface_sampling: hand_frame` fix → new two-stage reference (held-out 8.89); does not fix Wuji.
- **§39–§40:** Wuji failure is entirely the **wrist** (fingers-only ≈ no movement; wrist R² on Wuji ≈ 0 vs 0.44–0.75 on
  source). Moving Wuji's base origin changes predictions by up to 2.3 mm (first sign of frame dependence).
- **§41–§43 J1** (section 4): matches two-stage on source (6.31/6.31/8.30, held-out 8.85), internal contacts 2.19 mm,
  one checkpoint. Wuji 13.44 vs no movement 14.19 (within coordinate sensitivity, not transfer). **Frozen transfer
  benchmark v1** built (`transfer_benchmark.py`, sha256 manifest). **J1 becomes the reference.**
- **Direct track (§43–§44, §48, §51–§53; `docs/direct_decoder_status_2026-11-19.md`):** z-only trunk → intent tokens →
  GET (`direct_worldaux_v1`), Track D (frozen trunk, motion loss only), A1 (learned anchors), T1 (translation-only on one
  Allegro: z+pose MLP 0.10 mm vs direct GET 6.17 mm, no-op 7.91). All fail to move the wrist; the loss is localised at
  **intent-token formation** (additive z projection + start pose does not form the required translation). Untested
  hypotheses: H3 scale (KITE ~2.7B samples vs our ~13M), H4 mean-residual collapse. **User froze direct-model training**;
  keep it separate in reports.
- **§45–§46 G1:** J1 with 48 LEAP training hands (vs 12). Source ≈ J1; Wuji better than J1 in 3/3 seeds
  (corr 13.91 ± 1.23 vs 15.14 ± 1.51) but neither reliably beats no movement (§45's single-seed win was luck).
- **W1 track (§47, §49–§50, §54; `docs/w1_wrist_transfer_proposal.md`):** wrist-only training with fingers at goal and
  true contacts (diagnostic reference). Wuji contact realisation beats no movement (compat .382 ± .057 vs .266) but the
  wrist is worse than no movement 3/3 (+0.98 mm). Not comparable with J1/G1 numbers; reported separately.
- **Shadow** (Menagerie E3M5) adapted (`external_assets/shadow_adapter`, palm as root, FK to 2.3e-10 m) and **reserved**.

### 5d. Morphology diversity: generated bodies (§55–§59, 2026-11-20; `docs/morphology_diversity_proposal.md`)

- **Generated "prim" bodies** (`kite_data.prim_hand` / `build_prim`, seed 20261119): box palm + capsule phalanges, 2–5
  fingers, 2–4 joints/finger, 4–19 DOF, 900 bodies (P0000–P0899; P0800–P0849 held out). A 4th family at 25% of batches.
- **D0 = G1; D1 = + 50 bodies × 400 records; D3 = + 800 bodies × 25.** Wuji corr 13.91 (G1) → **9.42 ± 0.36 (D1)** /
  9.47 (D3), compat .233 → .565; **wrist beats no movement 3/3**; source unchanged. D3 ≈ D1 (50 bodies suffice). **D1
  became the reference** (and later the frozen delivery default).
- **Shadow first evaluated (§57, user-approved):** corr 14.48 (G1) → 10.45 (D1), compat .218 → .417; wrist does not beat
  no movement (8.23 vs 7.96).
- **Controls (§58–§59):** C1 (one 2-finger body × 20k) recovers only ~37% of the Wuji gain → the gain is mainly **body
  variety**. D1b (second body draw) replicates D1; C5 (one 5-finger body) ≈ C1 (resemblance of one body doesn't help);
  N5 (no 5-finger bodies) ≈ D1 (look-alikes unnecessary); R4 (4× records per body) adds nothing.

### 5e. The wrist and the base-frame convention (§60–§62, 2026-11-21)

- **Phase 0 (§60a, `wrist_diag.py`):** with the goal wrist given, fingers reach 2.4–2.9 mm on Wuji and Shadow → the
  remaining error is almost all wrist. Shadow translation fails in the palm plane (wrong direction, not step size).
  Not tied to coupled joints. → a decoder-side wrist problem.
- **Phase 1 (§60b, §62) D1c:** coupled-joint generated bodies (`prim_c`, 387/900 coupled). No effect on Shadow, small
  costs → not promoted.
- **W2-a (§61a, `frame_relabel_eval.py`):** re-express each robot's base frame by the 24 axis-aligned rotations. D1's
  source translation R²_id drops from +0.55 to ≈ −0.04 → **the decoder depends on each family's base-frame convention**
  (it partly uses the frame as a cue). Shadow's native frame follows a different convention.
- **W2-b (§61b–c):** D1 + random base frame per example (`base_frame_aug`, k = 32). Frame dependence gone; Shadow much
  better (wrist 8.23 → 7.21, beats no movement 3/3; corr 10.45 → 8.81) but source and held-out LEAP worse 3/3 (+1.4 to
  +1.9 mm) → trade-off, not promoted. Lesson: the decoder wants a **consistent** frame, not a random one → CF.

### 5f. Canonical palm frame and what followed (§63–§63e, 2026-11-22 → 11-25)

Covered in sections 4 and 6: CF (rule E) → CF ≈ D1 on source with large Wuji/Shadow gains but a small held-out LEAP cost
→ A (100k steps) and B (body variety) → the 105-hand evaluation localises CF's cost to hands whose canonical frame is
off by more than ~6° → Step 0 proves the frame error is causal → CF-J (frame jitter in training), running now.
Proposal docs: `docs/cf_canonical_palm_frame_proposal.md`, `docs/cf2_longer_training_proposal.md`,
`docs/cf_frame_jitter_proposal.md`.

---

## 6. Current model line-up and results (2026-11-25)

All 3 seeds each, `getkite_runs/<run>{,_s1,_s2}`. Contact error = corr_mm.

| Arm | Run dirs | Definition | Steps |
|---|---|---|---|
| D1 (frozen delivery default) | `d1_prim50_hf*` | J1 + 50 generated bodies, native frames | 50k |
| CF | `cf_canon_hf*` | D1 + canonical palm frame | 50k |
| D1-long | `d1_long_hf*` | D1 at 100k (control for length) | 100k |
| CF2 = CF-long | `cf2_long_hf*` | CF at 100k | 100k |
| D1-bv (D1-B) | `d1_bv_long_hf*` | D1-long + body variety (414 LEAP hands × 48 rec, 800 prims × 25 rec; same record budget) | 100k |
| CF-bv | `cf_bv_long_hf*` | CF2 + the same body variety | 100k |
| CF-J (done, not promoted) | `cfj_long_hf*` | CF2 + training-time canonical-frame jitter ≤ 15° | 100k |

**Key results (standard benchmark; readouts in `configs/decoder/dlevels/cf2_readout/`):**
- **CF vs D1 (50k), audit §63:** ≈ on source; Wuji corr −0.86 mm (3/3), Shadow −1.53 mm (3/3), Shadow wrist −0.80 mm
  (3/3, the only model whose Shadow wrist beats no movement); held-out LEAP +0.17 mm (significant in seed 0 only →
  Primary 1 failed; not promoted).
- **CF2 vs D1-long (A, §63c):** Primary 1 holds, Primary 2 holds (Shadow wrist −0.94 mm 3/3), Primary 3 (lower contact
  error on every family, 3/3) fails: held-out LEAP +0.12 (3/3 higher, none significant), generated bodies higher 2/3.
  CF2 vs frozen D1: better on every family 3/3 except held-out LEAP (2/3).
- **Longer training (A):** helps both (held-out LEAP −0.32 D1, −0.37 CF), but D1-long got worse on Shadow.
- **Body variety (B, §63d):** D1-bv beats frozen D1 on every family 3/3 (held-out LEAP −0.39, Wuji −0.79, held-out
  generated −0.84) except Shadow 2/3. CF-bv did **not** improve over CF2 and fails all primaries vs D1-bv.
- **D1-bv vs CF2:** tie on source and Wuji; D1-bv better on held-out LEAP (−0.19) and held-out generated (−0.64); CF2
  better on Shadow (corr −0.61, wrist −0.50 sig 3/3, trans R²_id 0.03–0.09 vs −0.11–0.01).
- **105-hand held-out LEAP (§63e, `transfer_benchmark_v1_leapval105`):** CF2 − D1-long +0.06/+0.14/+0.15 mm; ≈ 0 for
  hands whose canonical frame is < 6° from the mean LEAP frame (81/105), +0.55 at 10–15°, +1.3 above 15°. CF2 beats
  frozen D1 on ~90/105 hands in seeds 1–2. The original 4 held-out hands were not representative.
- **Step 0 dose-response (§63e, `cf2_frame_perturb`):** rotating CF2's frame by 2.5/5/10/15° raises held-out corr by
  +0.05/+0.21/+0.81–0.91/+1.56–1.83 mm, 3/3 → per-hand frame error causes CF's held-out gap. Motivates CF-J.

**Delivery note (user request, 2026-11-25): deliver and evaluate the chosen model as a 3-seed ensemble.** Run seeds s0/s1/s2
on the same input and average the outputs (joints, wrist translation; wrist rotation = chordal mean). Inference-time only.
It beat the individual seeds in 102/108 comparisons (−0.3 to −1.3 mm; every ensemble's Shadow wrist beats no movement);
see `docs/variety_and_ensemble_checks.md` §2. In the final evaluation report both per-seed results (3/3 rule) and the
ensemble. Inference cost ≈ 3× (not yet timed).

**Delivery status:** no CF model has met Primaries 1–3 against its matched D1, so the delivery choice is with the user.
Recommendation given to the user (2026-11-25): **D1-bv seed 0** as default (best on LEAP-like and generated hands, ties on
Wuji), **CF2** as the alternative for hands with a different base-frame convention (Shadow-like). The user has not
chosen yet; D1-bv and CF2 run dirs are **not yet frozen** (freeze on the user's pick).

---

## 7. What is running right now

**Update 2026-11-25 19:45:** CF-J finished and did NOT meet its readout (see `docs/cf_frame_jitter_proposal.md`, Step 1 result):
jitter reduced frame-angle sensitivity but a held-out LEAP gap to D1-long remains in seeds 1–2, and Shadow became more
seed-variable. Items 1–2 (`docs/multipass_origin_checks_proposal.md`): multi-pass doesn't help; all models are strongly
sensitive to the base origin → item 3 canonical origin (CF-O, base CF2) drafted in `docs/canonical_origin_proposal.md`,
not launched. Nothing is running. The text below describes the state before CF-J finished.

- **CF-J training**, jobs 20044–20046 (`cfj_long_hf`, `_s1`, `_s2`), started ~15:35 EST 2026-11-25, 5 h limit,
  ~2–3.5 h expected. Plan, predeclared readout and launch details: `docs/cf_frame_jitter_proposal.md`.
- **Auto-evaluation** `slurm/cfj_chain.sh` (log `slurm/logs/cfj_chain.log`): on all three `DONE`, submits
  `transfer_v1_cfj.json` (4-hand held-out + Shadow; D1, D1-long, CF2, D1-bv, CF-J), `transfer_v1_cfj_leapval105.json`
  (105 held-out hands) and `slurm/frame_relabel_cfj.sbatch`.
- **CF-J success criteria (predeclared):** on the 105 hands, CF-J − D1-long hand-bootstrap CI includes 0 or below in
  3/3 seeds, and the 6–15° bins' gap shrinks ≥ 50% vs CF2 in 3/3; Primaries 1–3 vs D1-long on transfer_v1 still hold,
  incl. Shadow wrist significantly better 3/3; relabelling spread ≈ 0.
- **Readout commands once done:**
  ```
  source slurm/env.sh; export PYTHONPATH=src:../get-ur-kite-dataset; P=~/envs/get_ur_kite/bin/python
  $P scripts/decoder/benchmark/cf2_summary.py --metrics $GETKITE_RUNS/transfer_benchmark_v1_cfj/metrics.json --cand cfj --ref d1long --cand_runs cfj_long_hf --ref_runs d1_long_hf
  $P scripts/decoder/benchmark/cf2_summary.py --metrics $GETKITE_RUNS/transfer_benchmark_v1_cfj/metrics.json --cand cfj --ref cf2 --cand_runs cfj_long_hf --ref_runs cf2_long_hf
  $P scripts/decoder/benchmark/leapval105_summary.py --pe $GETKITE_RUNS/transfer_benchmark_v1_cfj_leapval105/per_example.pt --a cfj --b d1long
  $P scripts/decoder/benchmark/leapval105_angle.py $GETKITE_RUNS/transfer_benchmark_v1_cfj_leapval105/per_example.pt cfj-d1long cf2-d1long cfj-cf2
  ```
- **Open decisions for the user:** delivery model (D1-bv vs CF2 vs CF-J if it passes); whether to use the reserved
  `test_geometry` dataset split (40 hands) as an untouched check for CF-J (the 105 val hands informed the CF-J design).
- **Out of scope until the user says otherwise:** XHand integration, further direct-model training, frame-rule changes,
  combining jitter with B's data, larger models, LEAP up-weighting.

---

## 8. Frozen references and important artifacts

- `getkite_runs/fallback_d1_reference_2026-11-24/`: D1 s0/s1/s2 `best.pt`, configs, provenance, checks, DONE, benchmark
  metrics; read-only with `SHA256SUMS`. **The delivery fallback.**
- `getkite_runs/cf_freeze_2026-11-24/`: code + config snapshot at CF launch.
- `getkite_runs/frozen_cf_2026-11-25/`: CF s0–s2 `best.pt`, configs, `canonical_frames.json`, benchmark, relabel,
  `cf_summary.txt`.
- Benchmarks: `transfer_benchmark_v1{,_seeds,_g1,_dlevels,_dlevels_seeds,_shadow,_c1,_followups,_w2b,_d1c,_direct,_cf,_cf2,_bv,_leapval105}` (plus `_cfj`, `_cfj_leapval105` once CF-J's evaluation runs).
- Readouts: `configs/decoder/dlevels/*_summary.txt`, `configs/decoder/dlevels/cf2_readout/` (A, B, 105-hand, Step 0).
- Config generators: `scripts/decoder/dlevels/make_{dlevels,followups,d1c,w2b,cf,cf2,bodyvar,cfj}.py` (each asserts that only the
  intended keys change).

---

## 9. Proposal and status documents (read in this order for context)

1. `docs/decoder.md` – decoder overview and conventions.
2. `docs/scoring.md` – scoring rules.
3. `docs/decoder_audit.md` – the lab notebook, §1–§63e (authoritative numbers).
4. `docs/direct_decoder_status_2026-11-19.md`, `docs/next_steps_2026-11-19.md` – the direct track and why it stopped.
5. `docs/w1_wrist_transfer_proposal.md`, `docs/t1_translation_diagnostic_proposal.md`.
6. `docs/morphology_diversity_proposal.md`, `docs/track_g_morphology_manifest.md` – generated bodies (D-levels).
7. `docs/coupling_and_wrist_proposal.md`, `docs/w2_wrist_frame_proposal.md` – wrist diagnosis, W2.
8. `docs/cf_canonical_palm_frame_proposal.md` – CF.
9. `docs/cf2_longer_training_proposal.md` – A (length) and B (body variety), Primary 3, results.
10. `docs/cf_frame_jitter_proposal.md` – 105-hand finding, Step 0, CF-J (running).
