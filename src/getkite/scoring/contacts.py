# Ported from CEBenchmark (cebench/tasks/_base/contacts.py @ ffad82e). Keep in sync: only import paths differ.
"""Contact-based checks shared by the task scores: brute force, forced-through parts, touches.

``ContactWatch`` sorts the model's bodies into the task's (the subtrees of the root bodies
a score names), the agent's (every other body except the world and the table: any
robot, hand or gripper) and the scene's. Every control step it measures, from contacts
only, so the same for every embodiment:

- **net agent force** on each task body: the vector sum of the agent's contact forces on
  it. A balanced grasp squeeze cancels out; what remains pushes the object. Above
  ``force_limit`` for ``force_time`` seconds is *excessive force* (a failure). The amount
  above ``soft_force`` is reported as a quality rate (``force_excess``, in N), counting
  only force held over two control steps: forces are sampled once per step, and a
  millisecond impact spike would otherwise weigh as much as a 50 ms push.
- **penetration**: parts forced into each other, the way brute force tricks a simulator.
  Between task parts (not the table) beyond ``penetration_limit``, or ``precise_limit``
  for the precision pairs a score names (a peg in its hole); or the agent sunk into a
  task part beyond ``agent_limit`` (a rigid hand driven through a wall, which soft
  contacts let happen at modest force). Any of these, held over two control steps, is
  *forced through* (a failure): forcing and tunnelling last, while an impact (a cube
  dropped a few millimetres onto a soft palm) sinks in for a single step. Bodies in
  ``soft`` (soap drops) never count.
- **touches**: which of the ``watch`` bodies (with their subtrees) or ``watch_geoms``
  groups ({label: geom names}) the agent touches right now, for scores that forbid
  touching a part directly (the block of the hook task, the knife's blade).

The floating hand of the scripted demos holds tools by a weld, not by contact; its
forces through the weld are not measured.
"""

from __future__ import annotations

import mujoco
import numpy as np

from getkite.scoring.scoring import Failure, QualityTerm

FORCE_LIMIT = 250.0  # N of sustained net agent force on one task body that counts as excessive
FORCE_TIME = 0.15  # s it has to last (single-step solver spikes are not force)
# N above which force use lowers the quality. The Franka experts' stiff position control
# pushes objects with up to 46-91 N sustained; at 75 N their quality stays at 0.88-1.00.
SOFT_FORCE = 75.0
PENETRATION_LIMIT = 0.003  # m between task parts: forced through
PRECISE_LIMIT = 0.0001  # m between the parts of a precision fit (clean fits stay below 0.03 mm)
AGENT_LIMIT = 0.003  # m the agent may sink into a task part (the Franka's pads stay below 0.4 mm)


def subtree(model, name):
    """Body ids of ``name`` and everything below it."""
    root = model.body(name).id
    out, frontier = set(), [root]
    while frontier:
        b = frontier.pop()
        out.add(b)
        frontier += [c for c in range(model.nbody) if model.body_parentid[c] == b and c != b]
    return out


class ContactWatch:
    def __init__(self, model, task_roots, precise=(), soft=(), watch=(), watch_geoms=None, force_limit=FORCE_LIMIT, force_time=FORCE_TIME,
                 soft_force=SOFT_FORCE, penetration_limit=PENETRATION_LIMIT, precise_limit=PRECISE_LIMIT, agent_limit=AGENT_LIMIT):
        self.model = model
        roots = {model.body_rootid[model.body(n).id] for n in task_roots}
        table = model.body_rootid[model.body("table").id]
        self.task = {b for b in range(model.nbody) if model.body_rootid[b] in roots}
        self.agent = {b for b in range(1, model.nbody) if model.body_rootid[b] not in roots and model.body_rootid[b] != table}
        self.precise = [(subtree(model, a), subtree(model, b)) for a, b in precise]
        self.soft = set().union(*(subtree(model, n) for n in soft)) if soft else set()
        self.watch = {n: subtree(model, n) for n in watch}
        self.watch_geoms = {label: {model.geom(g).id for g in geoms} for label, geoms in (watch_geoms or {}).items()}
        self.force_limit, self.force_time, self.soft_force = force_limit, force_time, soft_force
        self.penetration_limit, self.precise_limit, self.agent_limit = penetration_limit, precise_limit, agent_limit
        self.over_force_s = 0.0
        self.last_force = 0.0
        self.was_through = False  # penetration beyond a limit at the last control step
        self.last_time = None
        self._f = np.zeros(6)

    def _is_precise(self, b1, b2):
        return any((b1 in a and b2 in b) or (b1 in b and b2 in a) for a, b in self.precise)

    def features(self, data) -> dict:
        """Measure this control step; the keys feed ``contact_failures`` and ``contact_quality``."""
        m = self.model
        dt = 0.0 if self.last_time is None else max(0.0, data.time - self.last_time)
        self.last_time = data.time
        net, penetration, precise_pen, agent_pen, touched = {}, 0.0, 0.0, 0.0, set()
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 in self.task and b2 in self.task and not (b1 in self.soft or b2 in self.soft):
                if self._is_precise(b1, b2):
                    precise_pen = max(precise_pen, -c.dist)
                else:
                    penetration = max(penetration, -c.dist)
            elif (b1 in self.agent) != (b2 in self.agent) and (b1 in self.task or b2 in self.task):
                obj = b2 if b1 in self.agent else b1
                if obj not in self.soft:
                    agent_pen = max(agent_pen, -c.dist)
                mujoco.mj_contactForce(m, data, i, self._f)
                force = c.frame.reshape(3, 3).T @ self._f[:3]  # world frame, acting on geom2
                net[obj] = net.get(obj, np.zeros(3)) + (force if obj == b2 else -force)
                touched.update(name for name, bodies in self.watch.items() if obj in bodies)
                geom = c.geom2 if b1 in self.agent else c.geom1
                touched.update(label for label, geoms in self.watch_geoms.items() if geom in geoms)
        net_force = max((float(np.linalg.norm(v)) for v in net.values()), default=0.0)
        through = penetration > self.penetration_limit or precise_pen > self.precise_limit or agent_pen > self.agent_limit
        forced_through, self.was_through = through and self.was_through, through
        self.over_force_s = self.over_force_s + dt if net_force > self.force_limit else 0.0
        held = min(net_force, self.last_force)  # force held over this and the last control step
        self.last_force = net_force
        return {
            "net_force": net_force,
            "force_excess": max(0.0, held - self.soft_force),
            "excessive_force": self.over_force_s >= self.force_time,
            "penetration": penetration,
            "precise_penetration": precise_pen,
            "agent_penetration": agent_pen,
            "forced_through": forced_through,
            "touched": touched,
        }


def contact_failures():
    """The failures every task shares: parts forced through each other, and brute force."""
    return (
        Failure("forced through", lambda f: f["forced_through"]),
        Failure("excessive force", lambda f: f["excessive_force"]),
    )


def contact_quality(limit=20.0):
    """Force use above the soft limit, integrated (N s); ``limit`` of it makes quality 0."""
    return (QualityTerm("force", 1.0, lambda f: f["force_excess"], limit),)
