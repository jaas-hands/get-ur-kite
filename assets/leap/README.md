# LEAP hand for KITE's simulator

`I4-M4-R4-T4_g0_floating.xml` is the nominal LEAP Hand (all four fingers, four joints each,
nominal geometry) as a floating, actuated MJCF in the format of KITE's floating hands
(`third_party/KITE/assets/ohra_urdf_robot/allegro/allegro_hand_right_extended_floating_actuated.xml`):

- six virtual wrist joints (`virtual_joint_x/y/z` slides, `virtual_joint_roll/pitch/yaw`
  hinges; R = Rx(roll) Ry(pitch) Rz(yaw)), then the palm `palm_lower` with a `right_palm` site;
- the 16 finger joints, named by LEAP motor ID (`0`–`15`) exactly as in the dataset's
  `hand.json` and `hand.urdf`, with LEAP's official joint limits;
- one position actuator per joint (22), with the gains of KITE's floating Allegro;
- collision geometry only: MuJoCo Menagerie's hand-tuned collision boxes and fingertip pads,
  meshes inline. There are no visual meshes; KITE's viewer shows collision geoms.

It is generated, not hand-edited:

```bash
python scripts/make_leap_mjcf.py I4-M4-R4-T4_g0                     # this file
python scripts/make_leap_mjcf.py I3-M0-R2-T4_g2 --out outputs/leap   # any hand of the family
```

`scripts/make_leap_mjcf.py` wraps the dataset's collision model of a hand
(`third_party/get-ur-kite-dataset/data/hands/<id>/hand.xml`). Locked joints stay rigid, so a
lesioned or stretched hand gets a model as easily as the nominal one. `tests/test_leap.py`
checks that its link frames match the dataset's kinematics to 1e-6 m and 1e-6 rad.

**Not done yet:** KITE's simulator does not know this hand. Using it in KITE's tasks needs an
`EmbodimentSpec` and scene hook-up, start poses and a decoder head; see
[docs/training_plan.md](../../docs/training_plan.md).

**Provenance and licenses.** Collision boxes and fingertip pads: MuJoCo Menagerie `leap_hand`
@ `b065505` (MIT, Copyright 2023 Ananye Agarwal), deformed by get-ur-kite-dataset. Kinematics
(joint origins, axes, limits): LEAP Hand's URDF as published in GET-Zero @ `279652d`, via
get-ur-kite-dataset.
