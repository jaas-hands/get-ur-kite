# Ported from CEBenchmark (cebench/tasks/_base/lift.py @ ffad82e). Keep in sync: only import paths differ.
"""How far an object is off the table: the height of its lowest point.

A score that measures lift by the object's origin can be fooled: tipping an object over
raises its origin by about its half-width (a toppled drill "lifts" its origin by 3.8 cm,
past the 3 cm lift stage), although it never left the table. The lowest point of the
object's collision geometry stays on the table when it tips.

``LowestPoint(model, body)`` collects the collision geoms of ``body`` and its subtree and
returns, for a ``data``, the lowest z of their bounding boxes (``geom_aabb``, the eight
corners in world). For boxes that is exact; for round shapes it is a little lower than the
true lowest point once tilted, so a tipped object counts as even less lifted.
"""

from __future__ import annotations

import itertools

import numpy as np


class LowestPoint:
    def __init__(self, model, body):
        subtree, frontier = set(), [body]
        while frontier:
            b = frontier.pop()
            subtree.add(b)
            frontier += [c for c in range(model.nbody) if model.body_parentid[c] == b and c != b]
        self.geoms = [g for g in range(model.ngeom)
                      if model.geom_bodyid[g] in subtree and (model.geom_contype[g] or model.geom_conaffinity[g])]
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
        self.corners = [model.geom_aabb[g][:3] + signs * model.geom_aabb[g][3:] for g in self.geoms]  # (8, 3) local each

    def __call__(self, data) -> float:
        return float(min(np.min(data.geom_xpos[g][2] + self.corners[i] @ data.geom_xmat[g].reshape(3, 3)[2]) for i, g in enumerate(self.geoms)))
