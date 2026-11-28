"""Use CF-3e in KITE's existing Oracle-Intent loop (Barrett smoke test).

Keep KITE's episode, state feedback, controller and success rule. Replace only
the saved-latent-to-action call; never modify the vendored KITE checkout.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def wrist_matrix(q):
    # The URDF's serial virtual joints multiply Rx @ Ry @ Rz (intrinsic XYZ).
    return Rotation.from_euler("XYZ", np.asarray(q)[:, 3:6]).as_matrix()


def nearby_euler(matrix, previous):
    """Equivalent XYZ angles nearest the previous command, avoiding 2*pi jumps."""
    a = Rotation.from_matrix(matrix).as_euler("XYZ")
    b = np.stack((a[:, 0] + np.pi, np.pi - a[:, 1], a[:, 2] + np.pi), axis=-1)
    choices = np.stack((a, b), axis=1)
    choices += 2 * np.pi * np.round((previous[:, None] - choices) / (2 * np.pi))
    best = ((choices - previous[:, None]) ** 2).sum(-1).argmin(1)
    return choices[np.arange(len(a)), best]


class CF3eAdapter:
    def __init__(self, runtime, checkpoint, kite_root):
        from getkite.decoder.canonical import CanonicalFrame
        from getkite.decoder.get_scratch import ScratchConfig
        from getkite.decoder.joint_contact_get import JointContactGET
        from getkite.decoder.robots import J_MAX, RobotBank, urdf_desc
        from getkite.decoder.spatial_bridge import SetBridgeConfig

        if runtime.low_level_manipulator != "barrett":
            raise ValueError("This first rollout adapter is verified for Barrett only")
        if runtime.non_contact_enabled:
            raise ValueError("CF-3e does not implement KITE's non-contact stream")
        self.device = runtime.device
        ck = torch.load(checkpoint, map_location=self.device, weights_only=False)
        if ck["kind"] != "joint_contact_get" or ck["config"].get("surface_sampling") != "hand_frame":
            raise ValueError("Expected a J1 checkpoint with hand_frame sampling")
        self.model = JointContactGET(ScratchConfig.from_dict(ck["model_config"]),
                                     SetBridgeConfig(**ck["set_config"])).to(self.device)
        self.model.load_state_dict(ck["model"])
        self.model.decoder.amp = False
        self.model.eval().requires_grad_(False)
        if ck["config"].get("canonical_frame"):
            self.model = CanonicalFrame(self.model, origin=bool(ck["config"].get("canonical_origin"))).eval()

        names = runtime.robot_model.pk_chain.get_joint_parameter_names()
        expected = ["virtual_joint_" + s for s in ("x", "y", "z", "roll", "pitch", "yaw")]
        if names[:6] != expected or len(names) != runtime.local_dof:
            raise ValueError(f"Unexpected KITE action order: {names}")
        assets = Path(kite_root) / "assets/dro_urdf_robot/barrett"
        desc = urdf_desc(assets / "model_extended.urdf",
                         assets / "surface_template_graph_n4096_3e0ba32e061c.pt",
                         "barrett", "barrett", coords=names[6:], ignore_mimic=True)
        self.bank = RobotBank([desc], device=self.device, surface_sampling="hand_frame")
        self.width, self.nj = J_MAX, len(desc.coords)
        self.rows = []
        self.meta = {
            "decoder": "CF-3e", "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            "checkpoint_step": ck.get("step"), "joint_order": names,
            "intent": "KITE saved oracle latent, unmodified",
            "frame": "KITE dataset-shifted frame; rollout restores world translation",
            "canonical_frame": bool(ck["config"].get("canonical_frame")),
            "finger_clamping": "none in adapter; existing KITE controller unchanged",
            "fk_check": self.check_fk(runtime.robot_model),
        }

    def inputs(self, q):
        q = np.asarray(q, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != 6 + self.nj or not np.isfinite(q).all():
            raise ValueError(f"Invalid robot state: shape {q.shape}")
        qc = torch.zeros((len(q), self.width), device=self.device)
        qc[:, :self.nj] = torch.as_tensor(q[:, 6:], device=self.device)
        r = torch.zeros(len(q), dtype=torch.long, device=self.device)
        R = torch.as_tensor(wrist_matrix(q), dtype=torch.float32, device=self.device)
        t = torch.as_tensor(q[:, :3], device=self.device)
        return r, qc, R, t

    @torch.no_grad()
    def check_fk(self, robot):
        """Compare all 4096 surface points/normals against KITE's independent FK."""
        rng = np.random.default_rng(27)
        lo, hi = map(np.asarray, robot.pk_chain.get_joint_limits())
        q = rng.uniform(lo, hi, size=(8, len(lo))).astype(np.float32)
        q[:, :3] = rng.uniform(-0.3, 0.3, size=(8, 3))
        q[:, 3:6] = rng.uniform(-2.8, 2.8, size=(8, 3))
        r, qc, R, t = self.inputs(q)
        RR, tt = self.bank.fk(r, self.bank.expand(r, qc), R, t)
        desc = self.bank.descs[0]
        idx = torch.arange(len(desc.tpl_pos), device=self.device).expand(len(q), -1)
        p, n = self.bank.points(r, RR, tt, idx)
        frames = robot.pk_chain.forward_kinematics(torch.as_tensor(q, device=self.device))
        expected_p, expected_n = np.zeros(p.shape), np.zeros(n.shape)
        for i, name in enumerate(desc.tpl_link_names):
            mask = desc.tpl_link == i
            if not mask.any():
                continue
            T = frames[name].get_matrix().cpu().numpy()
            expected_p[:, mask] = np.einsum("bij,pj->bpi", T[:, :3, :3], desc.tpl_pos[mask]) + T[:, None, :3, 3]
            expected_n[:, mask] = np.einsum("bij,pj->bpi", T[:, :3, :3], desc.tpl_normal[mask])
        ep = float(np.abs(p.cpu().numpy() - expected_p).max())
        en = float(np.abs(n.cpu().numpy() - expected_n).max())
        if ep > 2e-5 or en > 2e-5:
            raise ValueError(f"KITE/CF-3e geometry mismatch: position={ep}, normal={en}")
        return {"poses": len(q), "points_per_pose": len(desc.tpl_pos),
                "max_position_component_error_m": ep, "max_normal_component_error": en}

    @torch.inference_mode()
    def predict(self, *, runtime, latent_action, q_prev_shifted, seed):
        q = np.asarray(q_prev_shifted, dtype=np.float32)
        z = np.asarray(latent_action, dtype=np.float32).reshape(1, -1)
        if z.shape != (1, 256) or len(q) != 1 or not np.isfinite(z).all():
            raise ValueError("Expected one finite 256-dimensional oracle latent")
        r, qc, R, t = self.inputs(q)
        out = self.model(self.bank, r, qc, R, t, torch.as_tensor(z, device=self.device))
        rot = out["R"].cpu().numpy()
        result = np.concatenate((out["t"].cpu().numpy(), nearby_euler(rot, q[:, 3:6]),
                                 out["qc"][:, :self.nj].cpu().numpy()), axis=1).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("CF-3e returned a non-finite action")
        err = float(np.abs(wrist_matrix(result) - rot).max())
        if err > 2e-5:
            raise ValueError(f"Output rotation conversion error: {err}")
        self.rows.append((q.copy(), result.copy(), z.copy(), err))
        return result

    def save(self, out):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        self.meta["decoder_calls"] = len(self.rows)
        if self.rows:
            np.savez_compressed(out / "cf3e_actions.npz",
                                q_before=np.concatenate([x[0] for x in self.rows]),
                                q_pred=np.concatenate([x[1] for x in self.rows]),
                                intent=np.concatenate([x[2] for x in self.rows]),
                                rotation_error=np.array([x[3] for x in self.rows]))
        (out / "cf3e_adapter.json").write_text(json.dumps(self.meta, indent=2) + "\n")


def run_cf3e_rollout(args, checkpoint, out):
    """Run the original Hydra entry point with a scoped prediction replacement."""
    from getkite.kite_bridge import add_kite_to_path
    root = add_kite_to_path()
    from experiment import rollout_gt_contacts as rollout

    original_load = rollout._load_online_low_level_runtime
    original_predict = rollout._predict_low_level_q_from_saved_latent
    original_cloud = rollout._predict_low_level_q_batch
    adapter = None

    def load(**kwargs):
        nonlocal adapter
        runtime = original_load(**kwargs)
        adapter = CF3eAdapter(runtime, checkpoint, root)
        return runtime

    def predict(**kwargs):
        if adapter is None:
            raise RuntimeError("CF-3e adapter was not initialized")
        return adapter.predict(**kwargs)

    def unsupported_cloud(**kwargs):
        raise ValueError("Use online_rollout.contact_input=latent for this CF-3e adapter")

    old_cwd, old_argv = Path.cwd(), sys.argv
    rollout._load_online_low_level_runtime = load
    rollout._predict_low_level_q_from_saved_latent = predict
    rollout._predict_low_level_q_batch = unsupported_cloud
    try:
        os.chdir(root)
        sys.argv = [str(root / "experiment/rollout_gt_contacts.py"), *args]
        try:
            # KITE's relative conf directory is not an importable Python package.
            import hydra
            entry = hydra.main(version_base="1.2", config_path=str(root / "experiment/conf"),
                               config_name="rollout_gt_contacts/object_suite_barrett")(rollout.main.__wrapped__)
            entry()
        except SystemExit as exc:
            return int(exc.code or 0)
        return 0
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv
        rollout._load_online_low_level_runtime = original_load
        rollout._predict_low_level_q_from_saved_latent = original_predict
        rollout._predict_low_level_q_batch = original_cloud
        if adapter is not None:
            adapter.save(out)
