"""Watch KITE episodes of a task with the staged score on screen, as in CEBenchmark's demos.

Replays held-out Oracle-Intent episodes (the latent intents encoded from the source
demonstrations, through the released kinematic decoder) on one hand, in MuJoCo's viewer
or into MP4 videos. Top right: the live score, stage, success and stage times; bottom
right: the score over time (current and peak); top left: the episode, whether the object
was lowered into the bin, the deepest a finger has sunk into the object, and contact flags.
Success is KITE's own success rule.

    python scripts/demo_task.py --task keyboard --hand allegro --episodes 3 # keyboard pressing
    python scripts/demo_task.py --hand barrett --object mug                 # objects placing, 1 mug episode
    python scripts/demo_task.py --hand wuji --object sugar --episodes 3 --speed 0.5
    python scripts/demo_task.py --hand allegro --episodes 2 --record        # MP4s, no window
    python scripts/demo_task.py --hand barrett --indices 37,75 --record --video-speed 0.5

Viewer keys: Space pauses, Tab / Shift+Tab show MuJoCo's side panels, mouse drag moves the
camera, closing the window stops the run. Everything after ``--`` goes to KITE's rollout
script as Hydra overrides.

The terminal shows one line per episode. KITE's own output goes to ``<out>/kite_log.txt``
(``--verbose`` prints it instead).

Tasks (``--task``), the two whose simulation assets and Oracle episodes are in KITE's release:

- ``placing``: Objects Placing, Robotiq demonstrations -> Barrett / Allegro / Wuji. 100
  held-out episodes over six YCB objects; KITE ships the rollout configs.
- ``keyboard``: Keyboard Pressing (three keys at once), Wuji demonstrations -> Barrett /
  Allegro (Wuji replays its own demonstrations). 20 episodes. KITE ships the data and start
  poses but no rollout config, so the overrides in ``TASKS`` below supply it; the keyboard
  values are the ones stored in the data's metadata.

Bottles Pumping (no bottle meshes) and Nut Rotating (no simulation code, no XHand decoder)
cannot be run from the release.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

HANDS = {"barrett": "Barrett (floating)", "allegro": "Allegro (floating)", "wuji": "Wuji (floating)"}
OBJECTS = ("008_pudding_box", "024_bowl", "010_potted_meat_can", "025_mug", "004_sugar_box", "006_mustard_bottle")
DATA = "./experiment/data/high_level_policy"
START_POSE_STEM = {"barrett": "barrett", "allegro": "allegro_right", "wuji": "wuji"}
# Per task: display name, score-panel title, number of Oracle episodes, and the Hydra overrides
# that turn KITE's shipped Objects Placing rollout config into this task's.
TASKS = {
    "placing": dict(name="Objects Placing", title="Objects placing score", episodes=100, overrides=lambda hand: []),
    "keyboard": dict(
        name="Keyboard Pressing",
        title="Keyboard pressing score",
        episodes=20,
        overrides=lambda hand: [
            "task.name=keyboard_pressing",
            "+keyboard.x_range=[0.145,0.245]",
            "+keyboard.y_range=[-0.2,-0.1]",
            "+keyboard.yaw_range_deg=[0,0]",
            "+keyboard.key_success_fraction=0.5",
            "+keyboard.target_key_joint_names=[white_key_02_E2_slide,white_key_04_G2_slide,white_key_06_B2_slide]",
            f"online_rollout.contact_source_pt={DATA}/keyboard_main_paper_contacts_v1_20.pt",
            "online_rollout.object_reset_source_pt=null",
            "online_rollout.initial_pose.task=keyboard_press",
            f"online_rollout.initial_pose.pose_pt={DATA}/{START_POSE_STEM[hand]}_keyboard_press_start_pose.pt",
        ],
    ),
}


def parse_args():
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=sorted(TASKS), default="placing")
    p.add_argument("--hand", choices=sorted(HANDS), default="barrett")
    p.add_argument("--object", default=None, help=f"placing only: episodes with this object (a part of its name is enough): {', '.join(OBJECTS)}")
    p.add_argument("--episodes", type=int, default=1, help="how many episodes to show (default 1)")
    p.add_argument("--start", type=int, default=0, help="skip this many of the matching episodes")
    p.add_argument("--indices", default=None, help="exact episode indices (placing 0-99, keyboard 0-19), comma separated; overrides --object")
    p.add_argument("--speed", type=float, default=1.0, help="viewer playback speed, e.g. 0.25 for slow motion")
    p.add_argument("--panels", action="store_true", help="show MuJoCo's side panels from the start")
    p.add_argument("--record", action="store_true", help="write MP4s instead of opening the viewer")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "demos", help="folder for videos and run files")
    p.add_argument("--video-speed", type=float, default=1.0, help="playback speed of the videos, e.g. 0.5 for slow motion")
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--video-size", default="1280x720", metavar="WxH")
    p.add_argument("--verbose", action="store_true", help="print KITE's own output instead of writing it to <out>/kite_log.txt")
    args = p.parse_args(argv[:split])
    return args, argv[split + 1 :]


@contextlib.contextmanager
def output_to(log_path: Path | None):
    """Send everything written to stdout and stderr, by Python or by C libraries, to ``log_path``.

    Yields a ``say`` function that still prints to the terminal. With ``log_path=None``
    nothing is redirected.
    """
    if log_path is None:
        yield lambda text: print(text, flush=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    terminal = os.fdopen(os.dup(1), "w", buffering=1)
    with open(log_path, "w") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            yield lambda text: print(text, file=terminal, flush=True)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])
            terminal.close()


def log_tail(path: Path, lines: int = 15) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def episode_indices(args, config_name: str) -> list[int]:
    """Held-out episode indices for --indices, or for --object / --start / --episodes."""
    total = TASKS[args.task]["episodes"]
    if args.indices:
        chosen = [int(x) for x in args.indices.split(",") if x.strip()]
        if any(k < 0 or k >= total for k in chosen):
            raise SystemExit(f"--indices must be in 0-{total - 1} for --task {args.task}")
        return chosen
    if args.task != "placing":
        if args.object:
            raise SystemExit("--object only applies to --task placing")
        return list(range(total))[args.start : args.start + args.episodes]
    from getkite.kite_bridge import kite_root

    import torch
    from omegaconf import OmegaConf

    root = kite_root()
    cfg = OmegaConf.load(root / "experiment" / "conf" / f"{config_name}.yaml")
    source = root / str(cfg.online_rollout.object_reset_source_pt)
    meta = torch.load(source, map_location="cpu", weights_only=False)["episode_meta"]
    ids = [str(m["object_id"]) for m in meta]
    if args.object:
        matches = sorted({i for i in ids if args.object.lower() in i.lower()})
        if len(matches) != 1:
            raise SystemExit(f"--object {args.object!r} matches {matches or 'nothing'}; objects: {', '.join(OBJECTS)}")
        chosen = [k for k, i in enumerate(ids) if i == matches[0]]
    else:  # spread over the objects: the k-th episode of each object in turn
        by_object = {}
        for k, i in enumerate(ids):
            by_object.setdefault(i, []).append(k)
        chosen = [ks[j] for j in range(max(map(len, by_object.values()))) for ks in by_object.values() if j < len(ks)]
    return chosen[args.start : args.start + args.episodes]


def main() -> None:
    args, extra = parse_args()
    # The viewer needs an on-screen OpenGL context (GLFW); videos render offscreen (EGL).
    os.environ["MUJOCO_GL"] = "glfw" if sys.platform == "darwin" or not args.record else "egl"
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    from getkite.kite_bridge import ScoringSession, add_kite_to_path, run_kite_script
    from getkite.report import plain, summarize

    add_kite_to_path()
    from getkite.viz import VideoDisplay, ViewerDisplay, substep_hook

    task = TASKS[args.task]
    config_name = f"rollout_gt_contacts/object_suite_{args.hand}"
    indices = episode_indices(args, config_name)
    if not indices:
        raise SystemExit("no episodes selected")
    out = args.out.resolve()
    kite_args = [
        f"--config-name={config_name}",
        f"online_rollout.indices=[{','.join(map(str, indices))}]",
        "online_rollout.parallel.num_workers=1",
        "online_rollout.render=false",
        # The Oracle-Intent replay does not use camera images; skipping them is faster.
        "perception.include_camera_obs=false",
        f"online_rollout.save_results_npz={out / f'kite_results_{args.task}_{args.hand}.npz'}",
        f"hydra.run.dir={out / 'hydra'}",
        *task["overrides"](args.hand),
        *extra,
    ]
    log = None if args.verbose else out / "kite_log.txt"
    mode = "recording videos" if args.record else ("viewer" + (f", {args.speed:g}x speed" if args.speed != 1.0 else ""))
    print(f"{HANDS[args.hand]}, {task['name']}, episode{'s' if len(indices) > 1 else ''} "
          f"{', '.join(map(str, indices))} ({mode})", flush=True)
    if not args.record:
        print("  Space pauses, Tab shows MuJoCo's panels, closing the window stops.", flush=True)

    session = ScoringSession()
    code = 0
    with output_to(log) as say:
        common = dict(title=task["title"], labels=[str(k) for k in indices], hand=HANDS[args.hand], say=say)
        if args.record:
            w, h = (int(x) for x in args.video_size.lower().split("x"))
            display = VideoDisplay(session, out_dir=out, size=(w, h), fps=args.video_fps, speed=args.video_speed, prefix=f"{args.task}_{args.hand}_", **common)
        else:
            display = ViewerDisplay(session, speed=args.speed, panels=args.panels, **common)
        try:
            with session, substep_hook(display.on_substep):
                code = run_kite_script("experiment/rollout_gt_contacts.py", kite_args)
        except Exception as exc:  # report briefly; the details are in the log
            say(f"error: {type(exc).__name__}: {exc}")
            code = 1
        finally:
            display.close()

    results = plain(session.results())
    for result, source_index in zip(results, indices):
        result["source_episode"] = source_index
    out.mkdir(parents=True, exist_ok=True)
    scores_path = out / f"scores_{args.task}_{args.hand}.json"
    scores_path.write_text(json.dumps({
        "script": "experiment/rollout_gt_contacts.py", "args": kite_args,
        "exit_code": code, "requested_indices": indices, "episodes": results,
    }, indent=2) + "\n")
    print(f"Scores: {scores_path}")
    if args.verbose:
        print("\n== staged scores ==\n" + summarize(results))
    elif results:
        n = len(results)
        print(f"{n} episode{'s' if n > 1 else ''}: mean score {100 * sum(r['score'] for r in results) / n:.0f} %, "
              f"success {sum(r['success'] for r in results)}/{n}")
    if code and log is not None:
        print(f"KITE stopped with an error; the end of {log}:\n{log_tail(log)}", file=sys.stderr)
    elif log is not None:
        print(f"KITE's output: {log}")
    # Everything is closed and written. Skip the interpreter's teardown: there GLFW (used by
    # the viewer) is terminated from an atexit handler while other GL users are torn down
    # in arbitrary order, which can crash the process after a successful run.
    sys.stdout.flush()
    sys.stderr.flush()
    if sys.platform == "darwin":
        raise SystemExit(code)  # let Python release its multiprocessing resources on macOS
    os._exit(int(code))


if __name__ == "__main__":
    main()
