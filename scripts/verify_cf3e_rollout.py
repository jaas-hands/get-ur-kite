"""Check a recorded Barrett rollout against the team's existing J1 inference path.

Example: python scripts/verify_cf3e_rollout.py --run outputs/cf3e_oracle_smoke
Requires GETKITE_KITE_ROOT and the sibling kite_data package on PYTHONPATH.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/decoder"))
import predict_j1 as j1
from getkite.cf3e_rollout import wrist_matrix
from getkite.kite_bridge import kite_root
from getkite.decoder.train_scratch import build_bank, FrozenPsi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    a = ap.parse_args()
    root = kite_root()
    cf = a.run / "cf3e"
    meta = json.loads((cf / "cf3e_adapter.json").read_text())
    trace = np.load(cf / "cf3e_actions.npz")
    result = np.load(cf / "kite_results_placing_barrett.npz", allow_pickle=False)
    baseline = np.load(a.run / "kite/kite_results_placing_barrett.npz", allow_pickle=False)
    assert len(result["source_indices"]) == 1, "Verifier currently handles one episode"
    assert str(result["action_decoder"]) == "cf3e"
    for key in ("source_indices", "contact_episode_indices", "object_id", "dt", "frame_skip",
                "dataset_xyz_translation", "initial_pose_path", "initial_robot_model_q_world",
                "proprio_source", "controller_steps_per_prediction"):
        np.testing.assert_array_equal(result[key], baseline[key], err_msg=key)
    def settings(folder):
        scores = json.loads((folder / "scores_placing_barrett.json").read_text())
        assert scores["exit_code"] == 0
        return [x for x in scores["args"] if not x.startswith(("online_rollout.save_results_npz=", "hydra.run.dir="))]
    assert settings(cf) == settings(a.run / "kite"), "Rollout configuration overrides differ"
    for values in (result, baseline):
        assert not any(str(e) for e in values["error_messages"])
        np.testing.assert_array_equal(values["completed_steps"], values["action_frame_count"])
    assert len(trace["q_before"]) == int(result["completed_steps"].sum()) == meta["decoder_calls"]

    source = torch.load(str(result["contact_source_pt_path"]), map_location="cpu", weights_only=False)
    ep = int(result["contact_episode_indices"][0])
    ends = source["episode_ends"]
    rows = slice(int(ends[ep-1]) if ep else 0, int(ends[ep]))
    data = source["data"]
    saved = data["latent_action"][rows].numpy()
    np.testing.assert_array_equal(saved, trace["intent"])
    psi = FrozenPsi(str(root / "checkpoints/decoder/release.pt"), str(root))
    cloud = data["contact_action"][rows]
    z = psi(cloud[..., :3], cloud[..., 3:6], data["contact_action_valid_mask"][rows]).numpy()
    np.testing.assert_allclose(z, saved, atol=1e-4, rtol=1e-4)

    assets = root / "assets/dro_urdf_robot/barrett"
    bank = build_bank({"surface_sampling": "hand_frame", "robots": [{
        "kind": "kite", "robot": "barrett", "family": "barrett",
        "urdf": str(assets / "model_extended.urdf"),
        "template": str(assets / "surface_template_graph_n4096_3e0ba32e061c.pt"),
        "coords": meta["joint_order"][6:], "ignore_mimic": True,
    }]}, torch.device("cpu"))
    system, _ = j1.load_system("cf3e", {"kind": "j1", "ckpt": meta["checkpoint"]}, torch.device("cpu"), "hand_frame")
    q = trace["q_before"]
    qc = torch.zeros(len(q), bank.coord_valid.shape[1]); qc[:, :8] = torch.from_numpy(q[:, 6:])
    g = SimpleNamespace(qc1=qc, R1=torch.from_numpy(wrist_matrix(q)).float(), t1=torch.from_numpy(q[:, :3]))
    with torch.no_grad():
        predicted = system.predict(bank, torch.zeros(len(q), dtype=torch.long), g, torch.from_numpy(saved))
    errors = {}
    for key, expected in (("t", trace["q_pred"][:, :3]), ("R", wrist_matrix(trace["q_pred"])),
                          ("qc", trace["q_pred"][:, 6:])):
        actual = predicted[key].numpy()
        if key == "qc": actual = actual[:, :8]
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        errors[key] = float(np.abs(actual - expected).max())
    report = {"passed": True, "control_steps": len(q), "same_episode_settings": True,
              "saved_intents_match_exactly": True, "reencoded_latent_max_error": float(np.abs(z-saved).max()),
              "existing_J1_inference_max_errors": errors, "geometry": meta["fk_check"],
              "cf3e_success": bool(result["success"][0]), "kite_success": bool(baseline["success"][0])}
    (a.run / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
