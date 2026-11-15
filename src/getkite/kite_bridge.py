"""Bridge to the KITE checkout: find it, import from it, and score its rollouts without editing it.

KITE lives in ``third_party/KITE`` (a git submodule); set ``GETKITE_KITE_ROOT`` to use a
checkout elsewhere. KITE's scripts import ``experiment.*``, ``networks.*`` etc. relative to
its root, so that root must be on ``sys.path`` (``add_kite_to_path``) and, for its relative
asset paths, the working directory.

``ScoringSession`` wraps ``TabletopManipulationEnv.reset`` and ``.step``: the first step
after each reset starts an ``EpisodeScore`` for the env's task (KITE's rollout scripts
override the object pose after ``reset``, so the score starts from the state the first
step sees), and every step updates it. Any KITE rollout or evaluation script that runs in
this process is scored, unchanged:

    with ScoringSession() as session:
        ...run a KITE rollout...
    session.results()    # one dict per episode

``run_kite_script`` runs one of KITE's Hydra scripts in this process, so a session sees it.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
KITE_ROOT_ENV = "GETKITE_KITE_ROOT"


def kite_root() -> Path:
    root = Path(os.environ.get(KITE_ROOT_ENV, REPO_ROOT / "third_party" / "KITE")).expanduser().resolve()
    if not (root / "experiment" / "tabletop_sim").is_dir():
        raise FileNotFoundError(
            f"No KITE checkout at {root}. Run `git submodule update --init` or set {KITE_ROOT_ENV}."
        )
    return root


def add_kite_to_path() -> Path:
    root = kite_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


class ScoringSession:
    """Scores every episode of every ``TabletopManipulationEnv`` in this process while installed."""

    def __init__(self):
        self.scores = []
        self.active = None  # (env, EpisodeScore | None) of the step in progress or the last one
        self._current = {}  # id(env) -> EpisodeScore | None; a missing key means "start one at the next step"
        self._cls = None
        self._orig = None

    def install(self) -> "ScoringSession":
        add_kite_to_path()
        from experiment.tabletop_sim.env import TabletopManipulationEnv

        from getkite.tasks import make_score_for_task

        session, cls = self, TabletopManipulationEnv
        orig_reset, orig_step = cls.reset, cls.step

        def reset(env, *args, **kwargs):
            out = orig_reset(env, *args, **kwargs)
            session._current.pop(id(env), None)
            return out

        def step(env, *args, **kwargs):
            if id(env) not in session._current:
                meta = {"episode": len(session.scores), "embodiment": str(getattr(env, "embodiment", ""))}
                score = make_score_for_task(env.model, env.data, env.scene, env.task, meta=meta)
                session._current[id(env)] = score
                if score is not None:
                    session.scores.append(score)
            session.active = (env, session._current[id(env)])
            out = orig_step(env, *args, **kwargs)
            score = session._current[id(env)]
            if score is not None:
                score.update(env.data)
            return out

        self._cls, self._orig = cls, (orig_reset, orig_step)
        cls.reset, cls.step = reset, step
        return self

    def uninstall(self) -> None:
        if self._cls is not None:
            self._cls.reset, self._cls.step = self._orig
            self._cls = None

    def __enter__(self) -> "ScoringSession":
        return self.install()

    def __exit__(self, *exc) -> None:
        self.uninstall()

    def results(self) -> list[dict]:
        return [s.result() for s in self.scores]


def run_kite_script(script: str, args: list[str]) -> int:
    """Run a KITE script (path relative to the KITE root) with Hydra ``args``, in this process.

    The working directory is the KITE root while it runs, for KITE's relative paths. Returns
    the script's exit code; Hydra exits through ``SystemExit`` even on success.
    """
    root = add_kite_to_path()
    path = root / script
    cwd, argv = os.getcwd(), sys.argv
    os.chdir(root)
    sys.argv = [str(path), *args]
    try:
        runpy.run_path(str(path), run_name="__main__")
        return 0
    except SystemExit as exc:
        return 0 if exc.code in (None, 0) else (exc.code if isinstance(exc.code, int) else 1)
    finally:
        os.chdir(cwd)
        sys.argv = argv
