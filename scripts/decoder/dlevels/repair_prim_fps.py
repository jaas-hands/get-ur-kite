"""Apply kite_data.prim_hand.ensure_finger_tokens to the existing primitive-body templates (2026-11-19).

Only template.npz["fps"] (the decoder's 32 surface-token centres) can change; template points, hands and records are
untouched (records index the 4096 template points, not fps). Bodies whose fps already gives every finger a token are left
byte-identical. The original fps of a changed body is kept as template.npz["fps_plain"]. Writes a log of changed bodies
to <hands>/fps_repair.json.

    python scripts/decoder/dlevels/repair_prim_fps.py --hands $GETKITE_DATASET/prim/hands
"""
import argparse
import json
from pathlib import Path

import numpy as np
from kite_data.prim_hand import ensure_finger_tokens, load_prim


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--hands", required=True); a = ap.parse_args()
    root, changed = Path(a.hands), {}
    for d in sorted(p for p in root.iterdir() if (p / "template.npz").exists()):
        tpl = dict(np.load(d / "template.npz"))
        if "fps_plain" in tpl:  # already repaired
            changed[d.name] = "already"; continue
        hand = load_prim(d); T0 = hand.fk(np.zeros(hand.n_dof))
        P0 = np.empty_like(tpl["pos"], dtype=np.float64)
        for li in np.unique(tpl["link"]):
            m = tpl["link"] == li; T = T0[hand.links[int(li)]]
            P0[m] = tpl["pos"][m] @ T[:3, :3].T + T[:3, 3]
        new = ensure_finger_tokens(P0, tpl["region"], tpl["fps"].astype(np.int64)).astype(np.int16)
        if np.array_equal(new, tpl["fps"]):
            continue
        tpl["fps_plain"], tpl["fps"] = tpl["fps"], new
        np.savez(d / "template.npz", **tpl)
        changed[d.name] = {"replaced_slots": np.flatnonzero(new != tpl["fps_plain"]).tolist(),
                           "regions_added": sorted(set(tpl["region"][new].tolist()) - set(tpl["region"][tpl["fps_plain"]].tolist()))}
    (root / "fps_repair.json").write_text(json.dumps(changed, indent=1) + "\n")
    print(len(changed), "bodies changed:", sorted(changed))


if __name__ == "__main__":
    main()
