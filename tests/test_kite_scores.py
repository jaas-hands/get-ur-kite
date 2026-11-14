"""Tests of the contact checks, the lift measure and the Objects Placing score on a tiny MuJoCo scene.

The KITE task is replaced by a stand-in with the same attributes and methods the score
uses, so these run without a KITE checkout. Scenes mirror KITE's naming (``table``,
``task_object``, ``pick_place_bin``).
"""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from getkite.scoring import ContactWatch, LowestPoint
from getkite.tasks import objects_placing

TABLE_Z = 0.0
HALF = 0.02  # object half size (m)
BIN_XY = (0.30, 0.0)
BIN_HALF = 0.08

XML = f"""
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="table" pos="0 0 -0.025"><geom name="table_top" type="box" size="1 1 0.025"/></body>
    <body name="task_object" pos="0 0 {HALF}">
      <freejoint name="task_object_free"/>
      <geom name="task_object_collision" type="box" size="{HALF} {HALF} {HALF}" mass="0.05"/>
    </body>
    <body name="pick_place_bin" pos="{BIN_XY[0]} {BIN_XY[1]} 0">
      <geom name="pick_place_bin_floor" type="box" size="{BIN_HALF} {BIN_HALF} 0.003" pos="0 0 0.003"/>
    </body>
    <body name="hand" pos="0 0 0.2">
      <freejoint/>
      <geom name="finger" type="sphere" size="0.01" mass="0.2"/>
    </body>
  </worldbody>
</mujoco>
"""


class StandInTask:
    """The parts of KITE's ``ObjectSuitePickPlaceTask`` that the score reads."""

    objective = "pick_place"
    table_top_z = TABLE_Z
    bin_floor_thickness = 0.006
    placement_margin = 0.005
    bottom_tolerance = 0.03
    lift_delta = 0.03
    pick_success_z = HALF + 0.03
    object_spec = SimpleNamespace(object_id="test_box")

    def world_bounds(self, pose):
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, np.asarray(pose[3:7], dtype=np.float64))
        corners = np.array([[x, y, z] for x in (-HALF, HALF) for y in (-HALF, HALF) for z in (-HALF, HALF)])
        world = corners @ rot.reshape(3, 3).T + pose[:3]
        return np.stack([world.min(0), world.max(0)])

    def placement_xy_margin(self, bounds):
        lo, hi = np.subtract(BIN_XY, BIN_HALF), np.add(BIN_XY, BIN_HALF)
        return float(min(bounds[0, 0] - lo[0], hi[0] - bounds[1, 0], bounds[0, 1] - lo[1], hi[1] - bounds[1, 1]))


SCENE = SimpleNamespace(object_body_name="task_object", object_joint_name="task_object_free", bin_body_name="pick_place_bin")


@pytest.fixture
def sim():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def place(model, data, body, xyz, quat=(1, 0, 0, 0)):
    adr = model.jnt_qposadr[model.body_jntadr[model.body(body).id]]
    data.qpos[adr : adr + 7] = [*xyz, *quat]
    data.qvel[:] = 0.0
    data.time += 0.05
    mujoco.mj_forward(model, data)


def test_lowest_point_ignores_tipping_origin(sim):
    model, data = sim
    low = LowestPoint(model, model.body("task_object").id)
    assert low(data) == pytest.approx(0.0, abs=1e-6)
    # Rotated 45 degrees about x and raised so an edge touches the table: the origin rises, the lowest point does not.
    c = np.cos(np.pi / 8)
    place(model, data, "task_object", (0, 0, HALF * np.sqrt(2)), (c, np.sin(np.pi / 8), 0, 0))
    assert low(data) == pytest.approx(0.0, abs=1e-6)


def test_contact_watch_sorts_bodies_and_measures_agent_force(sim):
    model, data = sim
    watch = ContactWatch(model, task_roots=["task_object", "pick_place_bin"], watch=["task_object"], agent_limit=float("inf"))
    assert model.body("hand").id in watch.agent and model.body("task_object").id in watch.task
    assert model.body("table").id not in watch.agent | watch.task
    place(model, data, "hand", (0, 0, 2 * HALF + 0.0095))  # sphere resting on the box, slightly sunk in
    for _ in range(200):
        mujoco.mj_step(model, data)
    f = watch.features(data)
    assert "task_object" in f["touched"]
    assert f["net_force"] == pytest.approx(0.2 * 9.81, rel=0.2)  # the hand's weight pushes the box
    assert not f["forced_through"]


def test_objects_placing_stages_and_success(sim):
    model, data = sim
    task = StandInTask()
    score = objects_placing.make_score(model, data, SCENE, task)
    path = [
        (0.0, 0.0, HALF + 0.015),  # half lifted
        (0.0, 0.0, HALF + 0.04),  # lifted
        (0.15, 0.0, HALF + 0.08),  # carried halfway
        (BIN_XY[0], 0.0, HALF + 0.10),  # over the bin, high: KITE success already
        (BIN_XY[0], 0.0, HALF + 0.006 + 0.01),  # lowered into the bin: placed
    ]
    scores = []
    for xyz in path:
        place(model, data, "task_object", xyz)
        scores.append(score.update(data)["score"])
    assert scores == sorted(scores)
    assert scores[0] == pytest.approx(0.3 * 0.5, abs=0.02)
    r = score.result()
    assert r["kite_success"] and r["success"] and r["score"] == 1.0
    assert all(t is not None for t in r["stage_done_s"].values())


def test_objects_placing_held_above_bin_is_kite_success_only(sim):
    model, data = sim
    score = objects_placing.make_score(model, data, SCENE, StandInTask())
    for xyz in [(0.0, 0.0, HALF + 0.05), (BIN_XY[0], 0.0, HALF + 0.12)]:
        place(model, data, "task_object", xyz)
        score.update(data)
    r = score.result()
    assert r["kite_success"] and not r["success"]
    assert 0.7 <= r["score"] < 1.0  # lift and carry done, lowering not yet
