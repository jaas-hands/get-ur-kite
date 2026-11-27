"""Shadow Hand E3M5 (MuJoCo Menagerie, reserved external final target) -> KITE-style *_extended URDF. 2026-11-19.

Adapter decisions (docs/next_steps_2026-11-19.md section 1; no tuning on Shadow performance):
  * floating base = rh_palm. Forearm and wrist bodies removed; WRJ2/WRJ1 fixed at 0 (as KITE's own Shadow handling,
    robot_model.py probe_palm_root). Six virtual_joint_{x,y,z,roll,pitch,yaw} ahead of the palm, as KITE's URDFs.
  * kinematics from the compiled MJCF (body pos/quat, joint axis/range: every joint sits at its body origin, one hinge
    per body; checked here).
  * coupled distal joints: the MJCF fixed tendons {FF,MF,RF,LF}J0 = J1 + J2 (actuated 0..pi; split set by dynamics).
    Adapter: J1 = <mimic> of J2 (multiplier 1, offset 0) -> 18 independent hand coordinates, 22 physical joints.
  * geometry: the visual meshes (geom pos/quat as written in the MJCF, mesh scale 0.001) as both <visual> and
    <collision>, so KITE's template / primitive tools can sample the surface.

    python scripts/decoder/shadow/make_shadow_urdf.py [--menagerie DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
EXT_ROOT = Path(os.environ.get("GETKITE_EXTERNAL", _REPO.parent / "external_assets"))  # third-party hand assets (docs/external_hands.md)

MENAGERIE = EXT_ROOT / "mujoco_menagerie/shadow_hand"
OUT = EXT_ROOT / "shadow_adapter"
ROOT = "rh_palm"
COUPLED = {f"rh_{f}J1": f"rh_{f}J2" for f in ("FF", "MF", "RF", "LF")}
VIRTUAL = [("x", "prismatic", "1 0 0", -2, 2), ("y", "prismatic", "0 1 0", -2, 2), ("z", "prismatic", "0 0 1", -2, 2),
           ("roll", "revolute", "1 0 0", -3.14159265, 3.14159265), ("pitch", "revolute", "0 1 0", -3.14159265, 3.14159265),
           ("yaw", "revolute", "0 0 1", -3.14159265, 3.14159265)]


def f(x):
    return " ".join(f"{v:.10g}" for v in x)


def rpy(quat_wxyz):
    return Rot.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_euler("xyz")


def xml_geoms(xml_path):
    """body name -> [(mesh name, pos, quat)] for the visual mesh geoms, as written (MJCF defaults: pos 0, quat 1 0 0 0)."""
    root = ET.parse(xml_path).getroot()
    out = {}
    for b in root.iter("body"):
        g = []
        for e in b.findall("geom"):
            if e.get("class") == "plastic_visual" and e.get("mesh"):
                g.append((e.get("mesh"), [float(v) for v in e.get("pos", "0 0 0").split()],
                          [float(v) for v in e.get("quat", "1 0 0 0").split()]))
        out[b.get("name")] = g
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--menagerie", default=str(MENAGERIE))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    src, out = Path(a.menagerie), Path(a.out)
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    xml = src / "right_hand.xml"
    m = mujoco.MjModel.from_xml_path(str(xml))
    geoms = xml_geoms(xml)
    root_id = m.body(ROOT).id
    sub = [b for b in range(m.nbody) if b == root_id or any(p == root_id for p in _ancestors(m, b))]
    meshes_used = set()
    L, J = [], []
    for b in sub:
        name = m.body(b).name
        vis = []
        for mesh, pos, quat in geoms.get(name, []):
            fn = f"meshes/{mesh}.obj"
            meshes_used.add(mesh)
            o = f'<origin xyz="{f(pos)}" rpy="{f(rpy(np.array(quat) / np.linalg.norm(quat)))}" />'
            g = f'<geometry><mesh filename="{fn}" scale="0.001 0.001 0.001" /></geometry>'
            vis.append(f"    <visual>{o}{g}</visual>\n    <collision>{o}{g}</collision>")
        L.append(f'  <link name="{name}">\n' + "\n".join(vis) + "\n  </link>" if vis else f'  <link name="{name}" />')
        if b == root_id:
            continue
        js = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b]
        assert len(js) == 1, (name, js)
        j = js[0]
        assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and np.allclose(m.jnt_pos[j], 0), name
        jn = m.joint(j).name
        mim = f'<mimic joint="{COUPLED[jn]}" multiplier="1" offset="0" />' if jn in COUPLED else ""
        J.append(f'  <joint name="{jn}" type="revolute"><origin xyz="{f(m.body_pos[b])}" rpy="{f(rpy(m.body_quat[b]))}" />'
                 f'<parent link="{m.body(m.body_parentid[b]).name}" /><child link="{name}" /><axis xyz="{f(m.jnt_axis[j])}" />'
                 f'<limit lower="{m.jnt_range[j][0]:.10g}" upper="{m.jnt_range[j][1]:.10g}" effort="10" velocity="10" />{mim}</joint>')
    V = ['  <link name="virtual_base_link" />'] + [f'  <link name="virtual_joint_{n}_link" />' for n, *_ in VIRTUAL[:-1]]
    prev = "virtual_base_link"
    for n, typ, ax, lo, hi in VIRTUAL:
        child = ROOT if n == "yaw" else f"virtual_joint_{n}_link"
        V.append(f'  <joint name="virtual_joint_{n}" type="{typ}"><origin xyz="0 0 0" rpy="0 0 0" /><parent link="{prev}" />'
                 f'<child link="{child}" /><axis xyz="{ax}" /><limit lower="{lo}" upper="{hi}" effort="1000" velocity="1000" /></joint>')
        prev = child
    urdf = '<?xml version="1.0" encoding="utf-8"?>\n<robot name="shadow_hand_right">\n' + "\n".join(V + L + J) + "\n</robot>\n"
    path = out / "shadow_hand_right_extended.urdf"
    path.write_text(urdf)
    for mesh in sorted(meshes_used):
        shutil.copy2(src / "assets" / f"{mesh}.obj", out / "meshes" / f"{mesh}.obj")
    shutil.copy2(src / "LICENSE", out / "LICENSE.menagerie_shadow_hand")
    info = {"source": str(xml), "menagerie_commit": "370c439", "license": "Apache-2.0 (LICENSE.menagerie_shadow_hand)",
            "root": ROOT, "removed": ["rh_forearm", "rh_wrist (WRJ2)", "WRJ1 (palm joint; fixed at 0)"],
            "coupling": {k: {"mimic_of": v, "multiplier": 1.0, "offset": 0.0} for k, v in COUPLED.items()},
            "n_physical_joints": len(J), "n_independent": len(J) - len(COUPLED), "meshes": sorted(meshes_used),
            "status": "reserved external final target; not used for training, selection or tuning"}
    (out / "adapter_info.json").write_text(json.dumps(info, indent=1))
    print(path, json.dumps({k: info[k] for k in ("n_physical_joints", "n_independent", "meshes")}))


def _ancestors(m, b):
    while b != 0:
        b = m.body_parentid[b]
        yield b


if __name__ == "__main__":
    main()
