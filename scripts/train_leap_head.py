"""Baseline B0: KITE's decoder head for the LEAP hand, started from KITE's released trunk.

KITE's release has no LEAP decoder. This trains one with KITE's own training script
(`third_party/KITE/train_kinematic_decoder.py`: its data generator, normalizers, loss, EMA
and checkpoint format), changed in one place only: after the model is built, every module
that KITE shares across hands (ψ, hand/scene encoders, action decoder, ...) is loaded from
`checkpoints/decoder/release.pt` and frozen. What remains trainable is exactly what KITE
learns per hand (joint table, joint/pose projections and heads, action queries).

KITE is not modified. LEAP's assets must be in KITE's assets folder first
(`python scripts/install_kite_leap.py`).

Usage (Hydra overrides go straight to KITE's script):

    python scripts/train_leap_head.py                                   # full run, KITE's settings
    python scripts/train_leap_head.py synthetic_data.train_samples=20000 \
        synthetic_data.test_samples=500 training.epochs=3 training.compile_heads=false \
        training.save_dir=../../outputs/leap_head/pilot                  # pilot

Options as environment variables (Hydra owns the command line):
    GETKITE_B0_SOURCE_HEAD  release head the shared modules are read from (default allegro_right;
                            all heads share the same tensors, this only picks the key prefix)
    GETKITE_B0_INIT         "random" (default, as KITE trains a new head) or "source": start the
                            per-hand parameters from the source head where shapes match
    GETKITE_B0_UNFREEZE     "trunk": also train the shared modules except ψ (contact_only_encoder), so z
                            stays KITE's; "all": train ψ too. Either changes the trunk the other hands use.
    GETKITE_B0_LEAP_REGIONS "0": keep KITE's own contact regions for LEAP (thumb + "palm" only); default
                            "1" gives index, middle and ring their own regions (the dataset's recipe)
    GETKITE_KITE_CHECKPOINT path to release.pt (default: KITE's checkpoints/decoder/release.pt)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KITE = Path(os.environ.get("KITE_ROOT", REPO / "third_party" / "KITE")).resolve()
LEAP_NAME = "leaphand_right"


def _patch_trimesh_closest_point(chunk: int = 1024) -> None:
    """Run trimesh's closest-point queries in chunks (same result, bounded memory).

    KITE builds a hand's surface template once and caches it. For LEAP's high-resolution
    meshes the unchunked query needs more than 14 GB of RAM; chunked it needs ~10 GB.
    """
    import numpy as np
    import trimesh.proximity as P

    original = P.closest_point

    def closest_point(mesh, points):
        points = np.asarray(points)
        if len(points) <= chunk:
            return original(mesh, points)
        parts = [original(mesh, points[i:i + chunk]) for i in range(0, len(points), chunk)]
        return tuple(np.concatenate(z, axis=0) for z in zip(*parts))

    P.closest_point = closest_point


_LEAP_FINGER_LINKS = {"mcp_joint": 1, "pip": 1, "dip": 1, "fingertip": 1}  # index; _2 = middle, _3 = ring


def _patch_leap_regions() -> None:
    """Give LEAP's finger links KITE's finger regions (index, middle, ring) for region-uniform contacts.

    KITE labels template points by matching link and joint names against Shadow/Wuji-style names
    (`data_process/contact_policy_probe_dataset.py::_region_index_from_text`). For LEAP only the thumb
    matches; index, middle and ring fall back to "palm", so half of all sampled contacts land on the
    thumb and the three fingers get ~26 % together. With this patch LEAP has five regions (palm, thumb,
    index, middle, ring), sampled uniformly: the team dataset's recipe. Other hands are unchanged.
    Off with GETKITE_B0_LEAP_REGIONS=0.
    """
    if os.environ.get("GETKITE_B0_LEAP_REGIONS", "1") == "0":
        return
    from data_process.contact_policy_probe_dataset import ContactPolicyProbeDataset as D

    original = D._region_index_from_text

    def region(text: str):
        link = str(text).split(" ", 1)[0]  # the text starts with the point's link name
        stem, _, suffix = link.rpartition("_") if link.rsplit("_", 1)[-1] in {"2", "3"} else (link, "", "")
        if stem in _LEAP_FINGER_LINKS:
            return {"": 1, "2": 2, "3": 3}[suffix]
        return original(text)

    D._region_index_from_text = staticmethod(region)


def _frozen_trunk_class(base, shared_names):
    import torch

    ckpt_path = Path(os.environ.get("GETKITE_KITE_CHECKPOINT", KITE / "checkpoints" / "decoder" / "release.pt"))
    source = os.environ.get("GETKITE_B0_SOURCE_HEAD", "allegro_right")
    init = os.environ.get("GETKITE_B0_INIT", "random")
    unfreeze = os.environ.get("GETKITE_B0_UNFREEZE", "none")
    if unfreeze not in {"none", "trunk", "all"}:
        raise ValueError(f"GETKITE_B0_UNFREEZE must be none, trunk or all, got {unfreeze!r}")
    if init not in {"random", "source"}:
        raise ValueError(f"GETKITE_B0_INIT must be 'random' or 'source', got {init!r}")

    class FrozenTrunkCoTrain(base):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            released = ckpt["ema_model"]  # KITE evaluates and ships the EMA weights
            prefix = f"heads.{source}."
            if not any(k.startswith(prefix) for k in released):
                raise KeyError(f"head {source!r} not in {ckpt_path} (has {ckpt['manipulators']})")

            shared, per_hand_init = 0, 0
            for name in self.manipulator_names:
                head = self.head_for(name)
                for sub in shared_names:
                    module = getattr(head, sub, None)
                    if module is None:
                        continue
                    state = {k: released[f"{prefix}{sub}.{k}"] for k in module.state_dict()}
                    module.load_state_dict(state, strict=True)
                    train_it = unfreeze == "all" or (unfreeze == "trunk" and sub != "contact_only_encoder")
                    for p in module.parameters():
                        p.requires_grad_(train_it)
                        shared += p.numel()
                if init == "source":
                    own = head.state_dict()
                    for k, v in own.items():
                        top = k.split(".")[0]
                        src = released.get(prefix + k)
                        if top in shared_names or "normalizer" in k or src is None or src.shape != v.shape:
                            continue
                        own[k] = src.clone()
                        per_hand_init += v.numel()
                    head.load_state_dict(own, strict=True)

            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            print(
                f"[B0] shared modules from {ckpt_path.name}:{source} ({shared:,} params, "
                f"{'frozen' if unfreeze == 'none' else 'trainable: ' + unfreeze}); per-hand init={init}"
                + (f" ({per_hand_init:,} params copied)" if init == "source" else "")
                + f"; trainable {trainable:,} of {total:,}",
                flush=True,
            )

    return FrozenTrunkCoTrain


def _log_metrics_to_file(argv, defaults) -> None:
    """Also write every `wandb.log` call of KITE's script to <save_dir>/metrics.jsonl.

    KITE prints only the action MSE per epoch; its other test metrics go to W&B, which is
    off by default. The file gets them all, with or without W&B.
    """
    import json
    import time

    import wandb

    save_dir = None
    for a in list(defaults) + list(argv[1:]):
        if a.lstrip("+~").startswith("training.save_dir="):
            save_dir = a.split("=", 1)[1]
    if save_dir is None:
        return
    path = Path(save_dir)
    if not path.is_absolute():
        path = KITE / path
    path.mkdir(parents=True, exist_ok=True)
    path = path / "metrics.jsonl"

    def wrap(original):
        def log(data, *args, **kwargs):
            row = {"time": time.time(), "step": kwargs.get("step")}
            for k, v in dict(data).items():
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    continue
            if any(k.startswith(("test/", "train_eval/")) or "epoch" in k for k in row):
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            return original(data, *args, **kwargs)

        return log

    original_init = wandb.init

    def init(*args, **kwargs):  # wandb.init rebinds wandb.log to the new run's method
        run = original_init(*args, **kwargs)
        wandb.log = wrap(wandb.log)
        return run

    wandb.init = init


def main() -> None:
    if not (KITE / "assets" / "dro_urdf_robot" / "leaphand" / "leap_hand_right_extended.urdf").is_file():
        sys.exit("LEAP is not in KITE's assets yet: run `python scripts/install_kite_leap.py` first.")
    os.chdir(KITE)  # KITE's configs use paths relative to its root
    sys.path.insert(0, str(KITE))
    _patch_trimesh_closest_point()

    import runpy

    _patch_leap_regions()

    import networks.contact_probe_policy_paired_cotrain as cotrain

    # KITE's script does `from networks.contact_probe_policy_paired_cotrain import
    # ContactProbePolicyPairedCoTrain` when it runs, so it picks up the patched class.
    cotrain.ContactProbePolicyPairedCoTrain = _frozen_trunk_class(
        cotrain.ContactProbePolicyPairedCoTrain, cotrain._ALWAYS_SHARED_SUBMODULE_NAMES
    )
    defaults = [
        f"decoder_bank.embodiments=[{LEAP_NAME}]",
        f"+surface_template.per_embodiment.{LEAP_NAME}=4096",
        "run_tag=leap_head_b0",
        "wandb.name=leap_head_b0",
        f"training.save_dir={REPO / 'outputs' / 'leap_head' / 'b0'}",
        "hydra.run.dir=" + str(REPO / "outputs" / "leap_head" / "hydra"),
    ]
    user = sys.argv[1:]
    user_keys = {a.split("=", 1)[0].lstrip("+~") for a in user if "=" in a}
    _log_metrics_to_file(sys.argv, defaults)
    script = str(KITE / "train_kinematic_decoder.py")
    sys.argv = [script] + [d for d in defaults if d.split("=", 1)[0].lstrip("+~") not in user_keys] + user
    runpy.run_path(script, run_name="__main__")  # Hydra finds conf/ next to the script


if __name__ == "__main__":
    main()
