"""The offscreen score view draws CEBenchmark's overlay into video frames (needs offscreen OpenGL)."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from getkite.scoring.scoring import ViewerScoreOverlay  # noqa: E402
from getkite.tasks import objects_placing  # noqa: E402
from getkite.viz import OffscreenScoreView, set_view  # noqa: E402
from test_kite_scores import SCENE, XML, StandInTask, place  # noqa: E402


def test_overlay_is_drawn_into_the_frame():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    try:
        view = OffscreenScoreView(model, 320, 240)
    except Exception as exc:  # no offscreen OpenGL on this machine
        pytest.skip(f"no offscreen OpenGL: {exc}")
    set_view(view.cam, model, data)
    score = objects_placing.make_score(model, data, SCENE, StandInTask())
    overlay = ViewerScoreOverlay(view, score.tracker, "test score")
    plain = view.render(data)
    for z in (0.03, 0.05):
        place(model, data, "task_object", (0.0, 0.0, z))
        score.update(data)
    overlay.update()
    assert view._texts and view._figures  # the overlay asked for its panel and plot
    frame = view.render(data)
    assert frame.shape == (240, 320, 3) and frame.dtype == np.uint8
    top_right = (slice(0, 80), slice(200, 320))
    assert np.abs(frame[top_right].astype(int) - plain[top_right].astype(int)).sum() > 0  # the panel is there
    view.close()
