"""Floating, actuated MJCF for a LEAP hand of the dataset, in KITE's simulator format.

The dataset writes one collision MJCF per hand (``data/hands/<id>/hand.xml``): MuJoCo
Menagerie's hand-tuned collision boxes and fingertip pads, deformed with the hand's geometry,
meshes inline, locked joints already rigid, palm fixed to the world, no actuators. KITE's
floating hands (e.g. ``allegro_hand_right_extended_floating_actuated.xml``) instead hang the palm
from six virtual joints (x, y, z slides, then roll, pitch, yaw hinges) and position-control every
joint. ``floating_leap_xml`` converts the first into the second, keeping the dataset's joint
names (LEAP motor IDs, the same names as in its ``hand.urdf``), so the simulator, the URDF and
the training data index joints the same way.

The wrist pose maps to the virtual joints as R = Rx(roll) Ry(pitch) Rz(yaw) (``wrist_to_virtual``),
applied in the palm frame; the palm's own frame is the dataset's wrist frame.

    python scripts/make_leap_mjcf.py I4-M4-R4-T4_g0          # -> assets/leap/I4-M4-R4-T4_g0_floating.xml
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from getkite.leap.paths import hands_dir

PALM = "palm_lower"
VIRTUAL_JOINTS = ("virtual_joint_x", "virtual_joint_y", "virtual_joint_z",
                  "virtual_joint_roll", "virtual_joint_pitch", "virtual_joint_yaw")
# Gains as in KITE's floating Allegro (its controller config can rescale them at run time).
BASE_ACTUATOR = dict(gainprm="300", biasprm="0 -300 -25", forcerange="-30 30")
FINGER_ACTUATOR = dict(gainprm="720", biasprm="0 -720 -2", forcerange="-0.5 0.5")
BASE_JOINT = dict(armature="1", damping="0.1", actuatorfrcrange="-1000 1000")
FINGER_JOINT = dict(armature="0.001", damping="0.02", actuatorfrcrange="-10 10")
SLIDE_RANGE = "-2 2"
HINGE_RANGE = "-6.28 6.28"


def floating_leap_xml(hand_xml: Path | str, model_name: str | None = None) -> str:
    """The floating, actuated MJCF (as text) for one dataset hand's ``hand.xml``."""
    root = ET.parse(hand_xml).getroot()
    if model_name:
        root.set("model", model_name)
    world = root.find("worldbody")
    palm = world.find(f"body[@name='{PALM}']")
    if palm is None:
        raise ValueError(f"{hand_xml}: no top-level body {PALM!r}")
    world.remove(palm)

    # virtual_base_link > x > y > z > roll > pitch > base_link (yaw joint) > palm, as KITE's Allegro
    parent = ET.SubElement(world, "body", name="virtual_base_link", gravcomp="1")
    chain = [("virtual_joint_x_link", "virtual_joint_x", "slide", "1 0 0"),
             ("virtual_joint_y_link", "virtual_joint_y", "slide", "0 1 0"),
             ("virtual_joint_z_link", "virtual_joint_z", "slide", "0 0 1"),
             ("virtual_joint_roll_link", "virtual_joint_roll", "hinge", "1 0 0"),
             ("virtual_joint_pitch_link", "virtual_joint_pitch", "hinge", "0 1 0"),
             ("base_link", "virtual_joint_yaw", "hinge", "0 0 1")]
    for body, joint, jtype, axis in chain:
        parent = ET.SubElement(parent, "body", name=body, gravcomp="1")
        ET.SubElement(parent, "inertial", pos="0 0 0", mass="0.001", diaginertia="1e-6 1e-6 1e-6")
        ET.SubElement(parent, "joint", name=joint, type=jtype, axis=axis, pos="0 0 0", limited="true",
                      range=SLIDE_RANGE if jtype == "slide" else HINGE_RANGE, **BASE_JOINT)
    parent.append(palm)
    ET.SubElement(palm, "site", name="right_palm", pos="0 0 0", size="0.005", rgba="1 1 1 0")

    finger_joints = []
    for body in palm.iter("body"):
        body.set("gravcomp", "1")
        for j in body.findall("joint"):
            for key, value in FINGER_JOINT.items():
                j.set(key, value)
            j.set("limited", "true")
            finger_joints.append((j.get("name"), j.get("range")))
    palm.set("gravcomp", "1")

    act = root.find("actuator")
    if act is None:
        act = ET.SubElement(root, "actuator")
    for name in VIRTUAL_JOINTS:
        rng = SLIDE_RANGE if name in VIRTUAL_JOINTS[:3] else HINGE_RANGE
        ET.SubElement(act, "general", name=name, joint=name, ctrllimited="true", ctrlrange=rng, forcelimited="true",
                      biastype="affine", **BASE_ACTUATOR)
    for name, rng in finger_joints:
        ET.SubElement(act, "general", name=name, joint=name, ctrllimited="true", ctrlrange=rng, forcelimited="true",
                      biastype="affine", **FINGER_ACTUATOR)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def write_floating_leap(hand_id: str, out_dir: Path, root: Path | None = None) -> Path:
    src = (Path(root) if root is not None else hands_dir()) / hand_id / "hand.xml"
    out = Path(out_dir) / f"{hand_id}_floating.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(floating_leap_xml(src, model_name=f"leap_{hand_id}_floating") + "\n")
    return out


def wrist_to_virtual(pos, R) -> np.ndarray:
    """Virtual-joint values (x, y, z, roll, pitch, yaw) for a palm pose: position (3,), rotation (3, 3).

    The chain applies Rx(roll) @ Ry(pitch) @ Rz(yaw), i.e. intrinsic x-y-z Euler angles.
    """
    from scipy.spatial.transform import Rotation

    return np.concatenate([np.asarray(pos, float), Rotation.from_matrix(np.asarray(R, float)).as_euler("XYZ")])
