"""All knobs in one dataclass. JSON files in configs/ override defaults; --set key=value overrides those."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields


@dataclass
class Config:
    # ---------------- data
    data_dir: str = ""  # kite_data's data/ (holds hands/ and examples/); "" = kite_data.config.DATA_DIR
    regime: str = "kite"  # examples/<regime>/<split>/
    train_split: str = "train"  # "a+b" concatenates splits (e.g. the non-zero-shot oracle on test hands)
    train_part: str = "fit"  # "all" | "fit" | "holdout": examples of each hand used for training (see holdout_frac)
    train_max_examples: int = 0  # 0 = all; else a fixed random subset (same seed as eval, e.g. for an overfit test)
    holdout_frac: float = 0.0  # last ceil(frac * n) examples of every hand form its "holdout" part; 0 = no holdout
    eval_splits: list = field(default_factory=lambda: ["val", "test_geometry", "test_joint_count",
                                                       "test_finger_subset", "test_combined"])
    # an eval entry "split@part" evaluates one part only, e.g. "train@holdout" (unseen contact sets on training hands)
    frame: str = "wrist_init"  # "world" | "wrist_init" (canonicalise into q_init's wrist frame)
    normal_noise_rad: float = 0.1  # train-time rotation noise on the normals psi sees (KITE: 0.1 rad)
    resample_q_init: bool = True  # fresh start-pose noise per training batch (KITE re-draws it every epoch)
    joint_noise_frac: float = 0.05  # KITE's start-pose noise model, used when resampling
    wrist_pos_noise: float = 0.005
    wrist_rot_noise_deg: float = 5.0
    wrist_rot_noise: str = "euler_xyz"  # "euler_xyz" (KITE: per Euler angle) | "rotvec" (our stored data)

    # ---------------- intent interface
    intent: str = "own_psi"  # "raw" | "own_psi" | "kite_psi"
    z_dim: int = 256
    n_intent_tokens: int = 4
    psi_width: int = 256
    psi_layers: int = 4
    psi_checkpoint: str = ""
    freeze_psi: bool = False
    kite_root: str = ""  # KITE checkout (or env KITE_ROOT); needed for any kite_* option
    kite_ckpt: str = ""  # KITE's checkpoints/decoder/release.pt
    freeze_unpacker: bool = False
    z_in_tokens: bool = True  # also concatenate z into every token's input MLP (PROJECT_OVERVIEW)

    # ---------------- decoder
    arch: str = "get"  # "get" (our transformer, any intent) | "kite_trunk" (KITE's pretrained trunk)
    # kite_trunk only
    joint_encoder: str = "get"  # "get" (from the URDF graph) | "slot" (KITE recipe: one embedding per LEAP joint id)
    tree_bias: bool = True  # GET-Zero kinematic-tree attention biases (both archs; false = graph-off ablation)
    hand_points: str = "kite"  # "kite" (KITE's link-local FPS indices) | "fps" (assembled-hand FPS)
    kite_init_head: str = "allegro_right"  # KITE head whose I/O layers initialise ours ("" = random)
    kite_freeze: list = field(default_factory=list)  # any of unpacker, hand_proj, scene, action (psi is always frozen)
    kite_trunk_lr_mult: float = 0.1  # learning-rate multiplier for KITE's pretrained shared modules
    # get only
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.0
    surface_tokens: bool = True  # KITE-style 32 surface tokens at q_init
    finger_id: bool = False  # one-hot finger identity in joint tokens (may hurt unseen finger subsets)
    max_spd: int = 12
    max_anc: int = 6
    pos_scale: float = 0.0  # 0 = auto: 2.5 for world (cube is +-0.4 m), 10 for wrist_init
    wrist_step: float = 0.01  # metres per unit of the wrist translation head
    fk_step: float = 0.05  # metres per unit of the self-model head

    # ---------------- loss
    w_residual: float = 1.0  # KITE's loss: MSE of the residual normalised by the start-pose noise
    w_wrist: float = 0.0  # our earlier wrist/joint terms (superseded by w_residual; kept for ablations)
    w_joint: float = 0.0
    w_contact: float = 0.1  # FK contact loss (not in KITE)
    w_normal: float = 0.1  # lambda_n inside the contact loss
    w_fk: float = 0.1  # get arch only (self-model head)
    w_limit: float = 1.0
    w_mask: float = 0.1  # get arch only: predict which joints should move
    joint_loss_units: str = "range"  # "range" (normalised by joint range) | "rad"
    sigma_pos: float = 0.01  # metres that count as one unit of position loss
    sigma_rot_deg: float = 5.0

    # ---------------- optimisation
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 2000
    total_steps: int = 100_000
    grad_clip: float = 1.0
    amp: str = "bf16"  # "bf16" | "none"; FK and losses always run in fp32
    seed: int = 0

    # ---------------- logging / eval
    out_dir: str = "runs/default"
    log_every: int = 100
    eval_every: int = 5000
    ckpt_every: int = 1000  # last.pt (atomic) every N steps, so a preempted run loses at most N steps
    eval_max_examples: int = 20000  # per split; 0 = all
    success_pos_mm: float = 10.0
    success_normal_deg: float = 15.0
    wandb: bool = False

    def resolved_pos_scale(self) -> float:
        return self.pos_scale or (2.5 if self.frame == "world" else 10.0)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path: str | None = None, overrides: list[str] | None = None) -> "Config":
        d = json.loads(open(path).read()) if path else {}
        types = {f.name: f.type for f in fields(cls)}
        for item in overrides or []:
            k, v = item.split("=", 1)
            if k not in types:
                raise KeyError(f"unknown config key {k}")
            d[k] = json.loads(v) if v[:1] in "[{" or v in ("true", "false") or _is_number(v) else v
        from . import paths
        d = paths.expand(d)
        for k, default in (("kite_root", lambda: paths.get("KITE_ROOT")), ("kite_ckpt", paths.kite_ckpt)):
            if str(d.get(k, "")).startswith("SET_ME"):  # placeholder in the shipped configs
                d[k] = default()
        unknown = set(d) - set(types)
        if unknown:
            raise KeyError(f"unknown config keys {sorted(unknown)}")
        return cls(**d)


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
