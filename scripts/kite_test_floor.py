"""Contact error of the start pose ("no movement") and of the label on a KITE probe test buffer.

KITE's trainer reports `test/<hand>/contact_point_mse` for the trained head; this gives the floor to
compare it with, on the same test examples (and checks the buffer: the label must score ~0).

    python scripts/kite_test_floor.py outputs/leap_head/local/b0-trunk           # LEAP run directory
    python scripts/kite_test_floor.py RUN_DIR --hand allegro_right
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from getkite.kite_head import KITE_ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--hand", default="leaphand_right")
    args = ap.parse_args()
    import pytorch_kinematics as pk

    assets = KITE_ROOT / "assets"
    urdf, tmpl_dir = {
        "leaphand_right": (assets / "dro_urdf_robot/leaphand/leap_hand_right_extended.urdf", assets / "dro_urdf_robot/leaphand"),
        "allegro_right": (assets / "ohra_urdf_robot/allegro/allegro_hand_right_extended.urdf", assets / "ohra_urdf_robot/allegro"),
    }[args.hand]
    tmpl = sorted(tmpl_dir.glob("surface_template_graph_n4096_*.pt"))[0]
    b = torch.load(args.run_dir / "buffers" / args.hand / "probe_test_buffer.pt", map_location="cpu", weights_only=False)["buffers"]
    links = torch.load(tmpl, map_location="cpu", weights_only=False)["link_names"]
    chain = pk.build_chain_from_urdf(urdf.read_bytes()).to(dtype=torch.float64)

    def contact_error(q):
        fr = chain.forward_kinematics(q.double())
        li = b["contact_link_indices"].long().clamp_min(0)
        valid = b["contact_valid_mask"]
        M = torch.stack([fr[n].get_matrix() for n in links], 1)
        Ml = torch.gather(M, 1, li[..., None, None].expand(-1, -1, 4, 4))
        p = (Ml[..., :3, :3] @ b["contact_local_points"].double()[..., None])[..., 0] + Ml[..., :3, 3]
        diff = p - b["contact_cloud"][..., :3].double()
        n = valid.sum(1).clamp_min(1)
        mm = ((diff.norm(dim=-1) * 1000 * valid).sum(1) / n).mean()
        mse = (((diff ** 2).sum(-1) * valid).sum(1) / (3 * n)).mean()
        return float(mm), float(mse)

    for name, key in (("label", "action"), ("no movement (start pose)", "q1_padded")):
        mm, mse = contact_error(b[key])
        print(f"{name:26s} contact {mm:7.3f} mm   contact_point_mse {mse:.3e}")


if __name__ == "__main__":
    main()
