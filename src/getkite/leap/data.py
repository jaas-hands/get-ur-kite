"""Training batches for kinematic decoders on the LEAP hand family (get-ur-kite-dataset).

One example (the dataset's KITE recipe): an intent c (2-6 world-frame contacts), a start pose
q_init (wrist pose + finger joints) and the label q_goal = m * q_contact + (1 - m) * q_init.
This module turns a split into batches with everything a decoder consumes, computed on the
device:

    split = LeapSplit.load("train", device="cuda")      # all examples of the split in memory
    psi = IntentEncoder.load(device="cuda")
    for batch in split.batches(256, shuffle=True, psi=psi, normal_noise=0.1):
        ...

Batch keys (B = batch size, D = MAX_DOF = 16, N = MAX_NODES = 17, wrist = node 0):

| key | shape | contents |
|---|---|---|
| ``hand`` | (B,) | index into ``split.bank.hand_ids`` |
| ``z`` | (B, 256) | psi(contacts), if ``psi`` is given |
| ``contact_pos``, ``contact_normal``, ``contact_valid`` | (B, 6, 3), (B, 6, 3), (B, 6) | the intent |
| ``template_idx`` | (B, 6) | template points that make the contacts (-1 = unused) |
| ``wrist_init``, ``wrist_goal`` | (B, 4, 4) | palm pose, start and label |
| ``joints_init``, ``joints_goal`` | (B, D) | finger joints, padded with 0 |
| ``dof_valid``, ``mask`` | (B, D) | real joints; joints the label moves (upstream of a contact) |
| ``lower``, ``upper`` | (B, D) | joint limits |
| ``joint_features`` | (B, D, 14) | URDF features per movable joint (``hands.JOINT_FEATURES``) |
| ``joint_pos_init``, ``joint_axis_init`` | (B, D, 3) | each movable joint's frame origin and axis in the world at q_init |
| ``hand_points``, ``hand_normals`` | (B, 32, 3) | KITE's hand tokens: the template's 32 FPS points at q_init |
| ``spd``, ``anc``, ``desc`` | (B, N, N) | graph distances (GET-Zero's attention biases); spd -1 = padding |
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from getkite.leap.hands import MAX_DOF, HandBank, pose_matrix
from getkite.leap.intent import MAX_CONTACTS
from getkite.leap.paths import examples_dir, splits_file

# Arrays kept per example (everything else in the files is metadata)
EXAMPLE_KEYS = (
    "k", "template_idx", "contact_pos", "contact_normal",
    "joints_contact", "joints_init", "joints_goal", "mask",
    "wrist_pos_contact", "wrist_quat_contact", "wrist_pos_init", "wrist_quat_init",
    "wrist_pos_goal", "wrist_quat_goal", "wide",
)
JOINT_KEYS = ("joints_contact", "joints_init", "joints_goal", "mask")


def split_hand_ids(split: str) -> list[str]:
    """Hand ids of a split, as listed in the dataset's splits.json."""
    hands = json.loads(splits_file().read_text())["hands"]
    if split not in hands:
        raise KeyError(f"no split {split!r}; splits: {', '.join(hands)}")
    return list(hands[split])


def load_examples(split: str, regime: str = "kite", root: Path | None = None, hands: list[str] | None = None):
    """All examples of a split as numpy arrays (joint arrays padded to MAX_DOF), plus hand ids.

    The same layout as ``kite_data.dataset.load_split``, restricted to the training arrays and
    optionally to some hands.
    """
    root = Path(root) if root is not None else examples_dir(regime) / split
    files = sorted(root.glob("*.npz"))
    if hands is not None:
        files = [f for f in files if f.stem in set(hands)]
    if not files:
        raise FileNotFoundError(f"no examples in {root}: generate the dataset first (see the README)")
    parts: dict[str, list] = {k: [] for k in EXAMPLE_KEYS}
    parts["hand"] = []
    hand_ids = []
    for hi, f in enumerate(files):
        z = np.load(f)
        hand_ids.append(str(z["hand_id"]))
        n, nd = z["joints_goal"].shape
        for key in EXAMPLE_KEYS:
            a = z[key]
            if key in JOINT_KEYS:
                p = np.zeros((n, MAX_DOF), dtype=a.dtype)
                p[:, :nd] = a
                a = p
            parts[key].append(a)
        parts["hand"].append(np.full(n, hi, dtype=np.int64))
    return {k: np.concatenate(v) for k, v in parts.items()}, hand_ids


@dataclass
class LeapSplit:
    name: str
    bank: HandBank
    examples: dict[str, torch.Tensor]  # EXAMPLE_KEYS + "hand", on ``device``

    @classmethod
    def load(cls, split: str, device="cpu", regime: str = "kite", hands: list[str] | None = None,
             full_template: bool = False, max_examples: int | None = None, seed: int = 0,
             root: Path | None = None) -> "LeapSplit":
        """``root``: a folder of ``<hand_id>.npz`` files to read instead of the dataset's split folder."""
        arrays, hand_ids = load_examples(split, regime=regime, root=root, hands=hands)
        if max_examples is not None and max_examples < len(arrays["k"]):
            keep = np.sort(np.random.default_rng(seed).choice(len(arrays["k"]), max_examples, replace=False))
            arrays = {k: v[keep] for k, v in arrays.items()}
        bank = HandBank.load(hand_ids, device=device, full_template=full_template)
        ex = {k: torch.as_tensor(v, device=device) for k, v in arrays.items()}
        return cls(split, bank, ex)

    def __len__(self) -> int:
        return int(self.examples["k"].shape[0])

    @property
    def device(self) -> torch.device:
        return self.examples["k"].device

    def batch(self, idx: torch.Tensor, psi=None, normal_noise: float = 0.0, generator: torch.Generator | None = None,
              dtype=torch.float32) -> dict[str, torch.Tensor]:
        """One batch for example indices ``idx`` (a 1-D tensor on any device)."""
        idx = idx.to(self.device)
        e = {k: v[idx] for k, v in self.examples.items()}
        h = e["hand"]
        bank = self.bank
        out: dict[str, torch.Tensor] = {"hand": h}
        k = e["k"].long()
        valid = torch.arange(MAX_CONTACTS, device=self.device)[None] < k[:, None]
        out["contact_pos"], out["contact_normal"] = e["contact_pos"].to(dtype), e["contact_normal"].to(dtype)
        out["contact_valid"], out["template_idx"] = valid, e["template_idx"].long()
        if psi is not None:
            out["z"] = psi(out["contact_pos"], out["contact_normal"], valid=valid, normal_noise=normal_noise, generator=generator)
        out["wrist_init"] = pose_matrix(e["wrist_pos_init"].to(dtype), e["wrist_quat_init"].to(dtype))
        out["wrist_goal"] = pose_matrix(e["wrist_pos_goal"].to(dtype), e["wrist_quat_goal"].to(dtype))
        out["joints_init"], out["joints_goal"] = e["joints_init"].to(dtype), e["joints_goal"].to(dtype)
        out["dof_valid"], out["mask"] = bank.dof_valid[h], e["mask"].bool() & bank.dof_valid[h]
        out["lower"], out["upper"] = bank.lower[h].to(dtype), bank.upper[h].to(dtype)
        out["joint_features"] = bank.joint_features[h].to(dtype)
        T = bank.fk(h, out["joints_init"], out["wrist_init"])
        Tj = torch.gather(T, 1, bank.dof_link[h][:, :, None, None].expand(-1, -1, 4, 4))  # each joint's frame
        keep = out["dof_valid"][..., None]
        out["joint_pos_init"] = Tj[..., :3, 3] * keep
        out["joint_axis_init"] = (Tj[..., :3, :3] @ bank.dof_axis[h].to(dtype)[..., None])[..., 0] * keep
        out["hand_points"], out["hand_normals"] = bank.hand_points(h, T)
        out["spd"], out["anc"], out["desc"] = bank.spd[h], bank.anc[h], bank.desc[h]
        return out

    def batches(self, batch_size: int, shuffle: bool = True, seed: int = 0, drop_last: bool = False, **batch_kwargs):
        """Iterate over the split once in batches (``batch_kwargs`` go to ``batch``)."""
        n = len(self)
        g = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.randperm(n, generator=g) if shuffle else torch.arange(n)
        stop = n - n % batch_size if drop_last else n
        for s in range(0, stop, batch_size):
            yield self.batch(order[s : s + batch_size], **batch_kwargs)
