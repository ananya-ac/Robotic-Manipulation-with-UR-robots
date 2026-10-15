"""Parallel-jaw gripper model for grasp collision checks, as axis-aligned boxes in the TIP frame.

Tip frame convention (matches GraspCandidate.msg): fingers close along x, +z is the approach
direction (from the palm toward the object), y = z cross x. The palm and fingers therefore sit at
negative z. Boxes are given as [[xmin, ymin, zmin], [xmax, ymax, zmax]] with the gripper OPEN;
`mirror_x: true` adds the x-mirrored copy (left/right finger).
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Box:
    lo: np.ndarray
    hi: np.ndarray

    def contains(self, pts):
        return np.all((pts >= self.lo) & (pts <= self.hi), axis=1)


@dataclass
class GripperModel:
    stroke: float                       # max opening between finger pads (m)
    contact_depth: float                # contacts sit this far behind the tip along -z (m)
    finger_reach: float                 # lowest finger point when closed, along +z of the tip (m)
    closing_region: Box                 # volume swept between the pads while closing
    bodies: list = field(default_factory=list)   # [Box] palm, fingers, linkage (open)

    @classmethod
    def from_dict(cls, d):
        def boxes(spec):
            out = []
            for b in spec:
                lo, hi = np.array(b["box"][0], float), np.array(b["box"][1], float)
                out.append(Box(lo, hi))
                if b.get("mirror_x", False):
                    out.append(Box(np.array([-hi[0], lo[1], lo[2]]), np.array([-lo[0], hi[1], hi[2]])))
            return out
        cr = d["closing_region"]
        return cls(stroke=float(d["stroke"]), contact_depth=float(d.get("contact_depth", 0.0)),
                   finger_reach=float(d.get("finger_reach", 0.0)),
                   closing_region=Box(np.array(cr[0], float), np.array(cr[1], float)),
                   bodies=boxes(d["bodies"]))

    def collides(self, pts_tip):
        """True if any point (tip frame) lies inside a gripper body."""
        return any(b.contains(pts_tip).any() for b in self.bodies)
