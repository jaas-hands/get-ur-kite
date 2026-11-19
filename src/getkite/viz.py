"""Watch KITE episodes with their staged score on screen, as CEBenchmark's demos show it.

Two displays, both drawing CEBenchmark's ``ViewerScoreOverlay`` unchanged (live score panel
top right, score-over-time plot bottom right) plus the episode's numbers top left:

- ``ViewerDisplay``: MuJoCo's passive viewer. Space pauses, closing the window stops the run,
  Tab and Shift+Tab show the side panels. Playback is paced to sim time times ``speed``.
- ``VideoDisplay``: one MP4 per episode, drawn offscreen with the same MuJoCo overlay calls
  the viewer uses (``OffscreenScoreView``), so videos look like the viewer.

A display hooks into KITE's physics stepping (``substep_hook``) and draws between the
substeps of a control step, so slow motion stays smooth. It reads the env and its score
from a ``ScoringSession`` (``session.active``).
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path

import mujoco
import numpy as np

from getkite.scoring.scoring import ViewerScoreOverlay

# Camera: looking over the table from the front left, at the point between the object and
# the bin (the task's centre), a little above the table.
VIEW = dict(distance=0.8, elevation=-28.0, azimuth=200.0, lift=0.08)
HOLD_S = 1.0  # real (viewer) or video seconds the end of an episode stays in view
SPACE = 32


@contextlib.contextmanager
def substep_hook(callback):
    """Call ``callback(backend)`` after every physics substep of KITE's simulator backends.

    KITE's ``RawMujocoBackend.step`` (also used by ``MjlabBackend``) runs ``frame_skip``
    substeps per control step. Running them one call at a time gives the same physics:
    each substep starts by resetting ``qfrc_applied`` itself.
    """
    from experiment.tabletop_sim.backend import RawMujocoBackend

    orig = RawMujocoBackend.step

    def step(backend, n_steps=None):
        for _ in range(int(backend.frame_skip if n_steps is None else n_steps)):
            orig(backend, 1)
            callback(backend)

    RawMujocoBackend.step = step
    try:
        yield
    finally:
        RawMujocoBackend.step = orig


def task_lookat(model, data) -> np.ndarray:
    """Midpoint of the task's object and its bin (or the object alone), a little above them."""
    points = []
    for name in ("task_object", "pick_place_bin", "pump_bottle_bottle_base", "mini_keyboard"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            points.append(np.array(data.xpos[bid]))
    centre = np.mean(points, axis=0) if points else np.array(model.stat.center)
    return centre + np.array([0.0, 0.0, VIEW["lift"]])


def set_view(cam, model, data) -> None:
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = task_lookat(model, data)
    cam.distance, cam.elevation, cam.azimuth = VIEW["distance"], VIEW["elevation"], VIEW["azimuth"]


class OffscreenScoreView:
    """The parts of ``mujoco.viewer.Handle`` that ``ViewerScoreOverlay`` uses, drawn offscreen.

    ``set_texts`` and ``set_figures`` store what the overlay asks for; ``render(data)``
    renders the scene from ``cam`` and draws those texts and figures with ``mjr_overlay`` and
    ``mjr_figure``, as the viewer does, and returns an RGB frame.
    """

    def __init__(self, model, width: int, height: int):
        self.model, self.width, self.height = model, int(width), int(height)
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), self.width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), self.height)
        self.gl = mujoco.GLContext(self.width, self.height)
        self.gl.make_current()
        scale = mujoco.mjtFontScale.mjFONTSCALE_150 if self.height <= 800 else mujoco.mjtFontScale.mjFONTSCALE_200
        self.ctx = mujoco.MjrContext(model, scale)
        self.scene = mujoco.MjvScene(model, maxgeom=20000)
        self.opt, self.pert, self.cam = mujoco.MjvOption(), mujoco.MjvPerturb(), mujoco.MjvCamera()
        self.viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        self._texts, self._figures = [], []
        self._rgb = np.empty((self.height, self.width, 3), dtype=np.uint8)

    def set_texts(self, texts) -> None:
        self._texts = [texts] if isinstance(texts, tuple) else list(texts)

    def set_figures(self, figures) -> None:
        self._figures = [figures] if isinstance(figures, tuple) else list(figures)

    def render(self, data) -> np.ndarray:
        self.gl.make_current()
        mujoco.mjv_updateScene(self.model, data, self.opt, self.pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.ctx)
        mujoco.mjr_render(self.viewport, self.scene, self.ctx)
        for _, grid, left, right in self._texts:  # the font scale is the context's
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, grid, self.viewport, left or "", right or "", self.ctx)
        for rect, fig in self._figures:
            mujoco.mjr_figure(rect, fig, self.ctx)
        mujoco.mjr_readPixels(self._rgb, None, self.viewport, self.ctx)
        return np.flipud(self._rgb).copy()

    def close(self) -> None:
        self.ctx.free()
        self.gl.free()


class Display:
    """What both displays share: following episodes, pacing draws by sim time, the episode panel."""

    def __init__(self, session, *, title: str, labels=None, hand: str = "", frame_period_s: float, say=print):
        self.session, self.title, self.hand, self.say = session, title, hand, say
        self.labels = list(labels or [])  # display label of episode k (e.g. its KITE episode index)
        self.period = float(frame_period_s)  # sim seconds between draws
        self.score = self.model = None
        self.next_t = 0.0
        self._reported = set()

    def episode_label(self, k: int) -> str:
        return self.labels[k] if k < len(self.labels) else str(k)

    def report(self, score) -> None:
        """One line when an episode ends: success (KITE's rule), score, stages done, flags."""
        from getkite.report import contact_flags

        if score is None or id(score) in self._reported:
            return
        self._reported.add(id(score))
        r = score.result()
        k = score.meta.get("episode", 0)
        done = [name for name, t in r["stage_done_s"].items() if t is not None]
        extra = []
        if "placed" in r["flags"]:
            extra.append("lowered into bin" if r["flags"]["placed"] else "not lowered into bin")
        extra += contact_flags(r)
        self.say(
            f"  episode {self.episode_label(k):>3}  {str(score.meta.get('object_id') or score.task):20}  "
            f"{'success   ' if r['success'] else 'no success'}  score {100 * r['score']:5.1f} %  "
            f"stages done: {', '.join(done) or '-':12}  {', '.join(extra)}"
        )

    def panel(self, env, score):
        """The top-left texts: which episode this is, and what the score reports next to it."""
        from getkite.report import contact_flags

        r = score.result()
        labels = ["hand", "task" if not score.meta.get("object_id") else "object", "episode", "control step"]
        values = [
            self.hand or str(getattr(env, "embodiment", "")),
            str(score.meta.get("object_id") or score.task),
            f"{self.episode_label(score.meta.get('episode', 0))}  ({score.meta.get('episode', 0) + 1} of {max(len(self.labels), 1)})",
            str(score.steps),
        ]
        if "placed" in r["flags"]:
            labels.append("lowered into bin")
            values.append("yes" if r["flags"]["placed"] else "no")
        labels += ["finger depth max", "contact flags"]
        values += [f"{1000 * r['max']['agent_penetration']:.1f} mm", ", ".join(contact_flags(r)) or "-"]
        return (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, "\n".join(labels), "\n".join(values))

    def on_substep(self, backend) -> None:
        active = self.session.active
        if active is None or active[1] is None:
            return
        env, score = active
        if score is not self.score:
            self.report(self.score)
            self.start_episode(env, score)
            self.score = score
            self.next_t = float(env.data.time)
        if float(env.data.time) + 1e-9 >= self.next_t:
            self.next_t += self.period
            self.draw(env, score)

    def start_episode(self, env, score) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def draw(self, env, score) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def close(self) -> None:
        self.report(self.score)


class ViewerDisplay(Display):
    """The passive viewer, paced to sim time times ``speed`` (at most 60 draws per real second)."""

    def __init__(self, session, *, speed: float = 1.0, panels: bool = False, **kwargs):
        super().__init__(session, frame_period_s=float(speed) / 60.0, **kwargs)
        self.speed, self.panels = float(speed), bool(panels)
        self.viewer = self.overlay = None
        self._threads = []
        self._overlay_step = None
        self.paused = False
        self.last = None

    def _key(self, keycode: int) -> None:
        if keycode == SPACE:
            self.paused = not self.paused

    def _close_viewer(self) -> None:
        """Close the window and wait for the viewer's thread to finish.

        ``Handle.close`` only asks the viewer's render thread to stop. If Python exits while
        that thread still runs, GLFW is terminated under it and the process crashes
        (GLXBadDrawable, segfault or heap corruption), so wait for it here.
        """
        if self.viewer is None:
            return
        import threading

        self.viewer.close()
        for thread in self._threads:
            # mjpython exposes native threads as dummy threads; Python cannot join them.
            if not isinstance(thread, threading._DummyThread):
                thread.join(timeout=5.0)
        self.viewer, self._threads = None, []

    def _launch(self, env) -> None:
        import threading

        import mujoco.viewer

        self._close_viewer()
        before = set(threading.enumerate())
        self.viewer = mujoco.viewer.launch_passive(
            env.model, env.data, key_callback=self._key, show_left_ui=self.panels, show_right_ui=self.panels
        )
        self._threads = [t for t in threading.enumerate() if t not in before]  # the viewer's render thread
        set_view(self.viewer.cam, env.model, env.data)
        self.model = env.model

    def start_episode(self, env, score) -> None:
        if self.score is not None and self.viewer is not None:
            time.sleep(HOLD_S)  # the end of the previous episode stays in view
        if self.viewer is None or env.model is not self.model:
            self._launch(env)  # KITE builds a new model when the object changes
        if self.overlay is None:
            self.overlay = ViewerScoreOverlay(self.viewer, score.tracker, self.title)
        self.overlay.viewer, self.overlay.tracker = self.viewer, score.tracker
        self._overlay_step = None
        self.last = time.monotonic()

    def _check_open(self) -> None:
        if not self.viewer.is_running():
            self.say("viewer closed: stopping")
            raise SystemExit(0)  # Hydra re-raises it; run_kite_script returns

    def draw(self, env, score) -> None:
        self._check_open()
        # ``set_texts`` / ``set_figures`` hand the overlay to the render thread and wait until it
        # has taken the previous one, which it does under the viewer's lock: so never call
        # them holding ``viewer.lock()`` (that deadlocks), and only when the score changed,
        # i.e. once per control step, not on every drawn frame.
        if score.steps != self._overlay_step:
            self._overlay_step = score.steps
            self.overlay.update(extra=[self.panel(env, score)])
        self.viewer.sync()
        while self.paused:
            self._check_open()
            self.viewer.sync()
            time.sleep(0.05)
            self.last = time.monotonic()
        now = time.monotonic()
        self.last = max(self.last + self.period / self.speed, now - 0.2)  # catch up only a little after a slow frame
        time.sleep(max(0.0, self.last - now))

    def close(self) -> None:
        super().close()
        if self.viewer is not None and self.viewer.is_running():
            time.sleep(HOLD_S)
        self._close_viewer()


class VideoDisplay(Display):
    """One MP4 per episode in ``out_dir``, ``fps`` frames per second, played at ``speed``."""

    def __init__(self, session, *, out_dir: Path, size=(1280, 720), fps: int = 30, speed: float = 1.0, prefix: str = "", **kwargs):
        super().__init__(session, frame_period_s=float(speed) / float(fps), **kwargs)
        self.out_dir, self.size, self.fps, self.speed, self.prefix = Path(out_dir), tuple(size), int(fps), float(speed), prefix
        self.view = self.overlay = self.writer = None
        self.last_frame = None
        self.paths = []

    def _finish(self) -> None:
        if self.writer is not None:
            for _ in range(int(HOLD_S * self.fps)):
                self.writer.append_data(self.last_frame)
            self.writer.close()
            self.writer = None
            self.say(f"    video: {self.paths[-1]}")

    def start_episode(self, env, score) -> None:
        import imageio.v2 as imageio

        self._finish()  # the previous episode's video (reported just before)
        if self.view is None or env.model is not self.model:
            if self.view is not None:
                self.view.close()
            self.view = OffscreenScoreView(env.model, *self.size)
            self.model = env.model
        set_view(self.view.cam, env.model, env.data)
        self.view.set_texts([])  # nothing from the previous episode until the first control step
        self.view.set_figures([])
        self.overlay = ViewerScoreOverlay(self.view, score.tracker, self.title)
        label = self.episode_label(score.meta.get("episode", 0))
        speed = "" if self.speed == 1.0 else f"_{self.speed:g}x"
        name = f"{score.meta['object_id']}_" if score.meta.get("object_id") else ""
        path = self.out_dir / f"{self.prefix}{name}ep{label}{speed}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(str(path), fps=self.fps, codec="libx264", quality=8, macro_block_size=8)
        self.paths.append(path)

    def draw(self, env, score) -> None:
        self.overlay.update(extra=[self.panel(env, score)])
        self.last_frame = self.view.render(env.data)
        self.writer.append_data(self.last_frame)

    def close(self) -> None:
        super().close()
        self._finish()
        if self.view is not None:
            self.view.close()
