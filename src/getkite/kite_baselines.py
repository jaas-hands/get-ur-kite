"""The zero-shot KITE baselines on LEAP variants: B0 clamped, B1 sliced, B2 rows reassigned.

All three reuse one trained LEAP head (B0, `scripts/train_leap_head.py`) without training:

- **clamped** (the "KITE" column for jammed joints): the full 16-joint head; locked joints are fed
  and returned at their locked angle (0 rad in the dataset). Undefined when a finger's links are gone.
- **B1 sliced**: a head with one joint token per movable joint, using B0's row of
  `joint_identity_embed` (and normalizer entries) for that joint's (finger, position in finger).
  In KITE's per-joint mode these rows are the only joint-specific weights; everything else per hand
  (`qref_joint_proj`, `head_joints`, wrist projections) is shared by all joints.
- **B2 rows reassigned**: as B1, but when whole non-thumb fingers are missing, the remaining ones take
  the rows of the fingers towards the index (index missing: middle <- index rows, ring <- middle rows;
  {thumb, ring}: ring <- index rows). The thumb keeps its rows. Fixed before any test result.

Hand tokens for every variant come from KITE's own surface template (DRO's LEAP meshes), restricted to
the links the hand has, deformed with the dataset's per-link geometry (`kite_data` deformations) and
farthest-point sampled exactly as KITE does; for the nominal hand this reproduces KITE's 32 tokens.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from getkite.kite_head import KITE_ROOT, LEAP_URDF, N_WRIST, KiteHead, _in_kite
from getkite.leap.paths import hands_dir

FINGERS = ("index", "middle", "ring", "thumb")
MAIN_FINGERS = ("index", "middle", "ring")
KITE_TO_DATASET_LINK = {"thumb_temp_base": "pip_4"}
DATASET_TO_KITE_LINK = {v: k for k, v in KITE_TO_DATASET_LINK.items()}


def hand_desc(hand_id: str) -> dict:
    return json.loads((hands_dir() / hand_id / "hand.json").read_text())


def finger_map(desc: dict, mode: str) -> dict[str, str]:
    """Which nominal finger's rows each finger of the hand uses ('slice': its own; 'shift': B2)."""
    present = [f for f in FINGERS if any(j["finger"] == f and not j["locked"] for j in desc["joints"])]
    if mode == "slice":
        return {f: f for f in present}
    if mode != "shift":
        raise ValueError(mode)
    main = [f for f in MAIN_FINGERS if f in present]
    out = {f: MAIN_FINGERS[i] for i, f in enumerate(main)}
    if "thumb" in present:
        out["thumb"] = "thumb"
    return out


def joint_rows(desc: dict, mode: str = "slice") -> list[int]:
    """B0 rows (0..15, KITE's LEAP order) for the hand's movable joints, in the hand's dof order."""
    fmap = finger_map(desc, mode)
    by_name = {j["name"]: j for j in desc["joints"]}
    return [4 * FINGERS.index(fmap[by_name[n]["finger"]]) + int(by_name[n]["k"]) for n in desc["dof_names"]]


# ---------------------------------------------------------------------------------------------
# Hand tokens
# ---------------------------------------------------------------------------------------------

def kite_template() -> dict:
    """KITE's LEAP surface template (link-local points, normals, link index, link names)."""
    path = sorted(LEAP_URDF.parent.glob("surface_template_graph_n4096_*.pt"))
    if not path:
        raise FileNotFoundError("LEAP's surface template is not built yet (scripts/install_kite_leap.py)")
    a = torch.load(path[0], map_location="cpu", weights_only=False)
    return {
        "pos": a["surface_template_points_local"].float(),
        "normal": a["surface_template_normals_local"].float(),
        "link": a["surface_template_link_indices"].long(),
        "link_names": [KITE_TO_DATASET_LINK.get(n, n) for n in a["link_names"]],  # dataset names
    }


def _fps_kite(points: torch.Tensor, n: int) -> torch.Tensor:
    with _in_kite():
        from robot_model.robot_model import _farthest_point_sample_indices_local
    return _farthest_point_sample_indices_local(points, n)


def variant_tokens(hand_id: str, n: int = 32, template: dict | None = None) -> dict:
    """KITE-style hand tokens of one dataset hand, in link frames: link names, link (n,), pos, normal."""
    from kite_data.qa import load_hand

    t = template or kite_template()
    hand, _ = load_hand(hands_dir() / hand_id)
    names = t["link_names"]
    keep = torch.tensor([names[i] in hand.deformation for i in t["link"].tolist()])
    pos, nrm, link = t["pos"][keep].numpy().astype(np.float64), t["normal"][keep].numpy().astype(np.float64), t["link"][keep]
    for li in link.unique().tolist():
        rows = (link == li).numpy()
        d = hand.deformation[names[li]]
        P = pos[rows]
        pos[rows] = d.map_points(P)
        J = d.jacobian(P)  # normals transform with the inverse transpose
        n_ = np.einsum("nij,nj->ni", np.linalg.inv(J).transpose(0, 2, 1), nrm[rows])
        nrm[rows] = n_ / np.linalg.norm(n_, axis=1, keepdims=True).clip(1e-12)
    pos_t = torch.tensor(pos, dtype=torch.float32)
    idx = _fps_kite(pos_t, n)
    return {"link_names": [names[i] for i in link[idx].tolist()], "pos": pos_t[idx],
            "normal": torch.tensor(nrm, dtype=torch.float32)[idx]}


def tokens_world(tokens: dict, T: torch.Tensor, link_names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokens at link frames T (B, L, 4, 4) of a hand whose links are `link_names` (hand.json order)."""
    li = torch.tensor([link_names.index(n) for n in tokens["link_names"]], device=T.device)
    Tl = T[:, li]  # (B, n, 4, 4)
    R, t = Tl[..., :3, :3], Tl[..., :3, 3]
    p = (R @ tokens["pos"].to(T)[None, :, :, None])[..., 0] + t
    nrm = (R @ tokens["normal"].to(T)[None, :, :, None])[..., 0]
    return p, nrm


# ---------------------------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------------------------

def sliced_head(b0: KiteHead, rows: list[int]) -> KiteHead:
    """A KITE head with len(rows) joint tokens, built by KITE's own constructor, B0's weights copied
    and its per-joint rows (identity embedding, normalizer entries) selected by `rows`."""
    with _in_kite():
        from networks.contact_probe_policy_paired_cotrain import ContactProbePolicyPairedCoTrain
        from omegaconf import OmegaConf
        from paper_config import runtime_config
        from train_kinematic_decoder import _build_cotrain_shared_model_kwargs, _resolve_per_manipulator_action_io

        ckpt_cfg = torch.load(b0.checkpoint, map_location="cpu", weights_only=False)["config"]
        cfg = runtime_config(OmegaConf.create(ckpt_cfg))
        src = b0.head
        slot = src._slot
        j0, j1 = (int(x) for x in slot["joints_slice"])
        assert j1 - j0 == 16, slot["joints_slice"]
        r = torch.tensor(rows, dtype=torch.long)
        act = torch.cat([torch.arange(N_WRIST), N_WRIST + r])
        res = torch.cat([torch.arange(j0), j0 + r])
        spec = {
            "name": b0.name,
            "action_dim": N_WRIST + len(rows),
            "action_normalizer_scale": src.action_normalizer_scale[0, act].cpu(),
            "action_normalizer_offset": src.action_normalizer_offset[0, act].cpu(),
            "residual_normalizer_scale": src.residual_normalizer_scale[0, res].cpu(),
            "residual_normalizer_offset": src.residual_normalizer_offset[0, res].cpu(),
            "base_translation_indices": [0, 1, 2],
            "base_rotation_indices": [3, 4, 5],
            "pose_groups": [{"name": "base", "translation_indices": [0, 1, 2], "rotation_indices": [3, 4, 5]}],
        }
        bank = ContactProbePolicyPairedCoTrain(
            manipulators=[spec], shared_kwargs=_build_cotrain_shared_model_kwargs(cfg),
            per_manipulator_action_io=_resolve_per_manipulator_action_io(cfg),
        )
    state = {}
    for k, v in src.state_dict().items():
        if k == "joint_identity_embed":
            v = v[r.to(v.device)]
        elif k in ("action_normalizer_scale", "action_normalizer_offset"):
            v = v[:, act.to(v.device)]
        elif k in ("residual_normalizer_scale", "residual_normalizer_offset"):
            v = v[:, res.to(v.device)]
        state[k] = v
    bank.head_for(b0.name).load_state_dict(state, strict=True)
    bank = bank.to(b0.device).eval()
    out = copy.copy(b0)
    out.bank = bank
    return out


@dataclass
class VariantRunner:
    """Runs one baseline on one dataset hand. mode: 'b0' (nominal), 'clamped', 'b1', 'b2'."""
    b0: KiteHead
    hand_id: str
    mode: str
    template: dict | None = None

    def __post_init__(self):
        self.desc = hand_desc(self.hand_id)
        self.links = [l["name"] if isinstance(l, dict) else l for l in self.desc["links"]]
        self.tokens = variant_tokens(self.hand_id, template=self.template)
        self.n_dof = int(self.desc["n_dof"])
        if self.mode in ("b0", "clamped"):
            if self.mode == "clamped" and len(self.links) < 17:
                raise ValueError(f"{self.hand_id}: a finger's links are gone; clamped KITE is undefined")
            self.head = self.b0
            # where each movable joint sits among the 16 nominal joints (same finger and position)
            self.rows = joint_rows(self.desc, "slice")
        else:
            self.rows = joint_rows(self.desc, "slice" if self.mode == "b1" else "shift")
            self.head = sliced_head(self.b0, self.rows)

    @torch.no_grad()
    def predict(self, batch: dict, bank) -> tuple[torch.Tensor, torch.Tensor]:
        """A single-hand LeapSplit batch -> predicted palm pose (B, 4, 4) and joints (B, 16, padded)."""
        from getkite.kite_head import from_kite_q, to_kite_q

        dev = self.b0.device
        T_palm = self.b0.T_palm.to(dev)
        wrist0 = batch["wrist_init"].double().to(dev)
        j0 = batch["joints_init"].double().to(dev)[:, : self.n_dof]
        B = j0.shape[0]
        if self.mode in ("b0", "clamped"):
            full = torch.zeros(B, 16, dtype=j0.dtype, device=dev)  # locked joints at 0 rad
            full[:, self.rows] = j0
            q0 = to_kite_q(wrist0, full, T_palm)
        else:
            q0 = torch.cat([to_kite_q(wrist0, torch.zeros(B, 16, dtype=j0.dtype, device=dev), T_palm)[:, :N_WRIST], j0], 1)
        T = bank.fk(batch["hand"], batch["joints_init"].double(), batch["wrist_init"].double())
        pts, nrm = tokens_world(self.tokens, T.to(dev), self.links)
        z = self.head.latent(batch["contact_pos"], batch["contact_normal"], batch["contact_valid"])
        q = self.head.predict_kite(z, q0, pts, nrm).double()
        wrist, joints = from_kite_q(torch.cat([q[:, :N_WRIST], torch.zeros(B, 16, dtype=q.dtype, device=dev)], 1), T_palm)
        out = torch.zeros(B, 16, dtype=q.dtype, device=dev)
        out[:, : self.n_dof] = q[:, N_WRIST:][:, self.rows] if self.mode in ("b0", "clamped") else q[:, N_WRIST:]
        return wrist, out
