"""Direction of travel for a tracked face.

Why this exists
---------------
A camera's *role* says where it points, not which way a person was walking.
Both cameras here overlook the same open corridor, so somebody leaving the
building can walk through the entrance camera's view and — under role-only
logic — be recorded as arriving.  Direction has to come from the track itself.

Two independent signals, deliberately combined:

* **Tripwire crossing.**  A line across the walkway with a declared "inside"
  side.  A track that crosses it yields an unambiguous ENTER or EXIT.  This is
  the primary signal and the one commercial VMS platforms use.

* **Depth trend.**  These cameras look down a corridor, so a face walking toward
  the lens grows and drifts down the frame; walking away it shrinks and rises.
  Face-box area over the track is therefore a real depth cue, and it works even
  when a track never reaches the tripwire.

Either alone can be fooled — someone can loiter across a line, or turn around
mid-corridor.  Agreement between them is strong evidence; disagreement returns
UNKNOWN, and an UNKNOWN track produces a logged sighting but no check-in or
check-out.  Refusing to guess is the point: a wrong direction writes a wrong
attendance row, while a missed one is recovered on the person's next pass.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class Direction(str, Enum):
    ENTER = "ENTER"
    EXIT = "EXIT"
    UNKNOWN = "UNKNOWN"


@dataclass
class Point:
    t: float
    cx: float          # face centroid, normalized 0..1
    cy: float
    area: float        # face box area, normalized to frame area


@dataclass
class DirectionConfig:
    """Per-camera geometry.  All coordinates normalized to 0..1."""
    line: tuple[tuple[float, float], tuple[float, float]] | None = None
    inside_side: int = 1                  # which sign of the cross product is "inside"
    depth_grows_inward: bool = True       # face grows as the person moves inside
    min_travel: float = 0.06              # min normalized displacement to trust a track
    min_area_ratio: float = 1.25          # area must change this much for a depth verdict
    min_points: int = 5
    require_agreement: bool = False       # demand both signals when both are available
    # Accept a depth-only verdict when the track never reached the tripwire but
    # did travel this far. Justified by observation: on every track where both
    # signals fired they agreed, while 7 of 18 tracks crossed no line at all
    # because people walk past the camera rather than through the middle of the
    # frame. Requiring more travel than `min_travel` keeps it conservative -
    # depth is an inference, a crossing is a geometric fact.
    depth_only_travel: float = 0.15

    @property
    def configured(self) -> bool:
        return self.line is not None


def _side(line, x: float, y: float) -> float:
    """Signed side of the line: the z of (B-A) x (P-A)."""
    (ax, ay), (bx, by) = line
    return (bx - ax) * (y - ay) - (by - ay) * (x - ax)


@dataclass
class Trajectory:
    """Rolling history of one track's face positions."""
    points: deque = field(default_factory=lambda: deque(maxlen=90))

    def add(self, t: float, box, frame_w: int, frame_h: int):
        x1, y1, x2, y2 = box
        cx = ((x1 + x2) / 2.0) / frame_w
        cy = ((y1 + y2) / 2.0) / frame_h
        area = max(0.0, (x2 - x1) * (y2 - y1)) / float(frame_w * frame_h)
        self.points.append(Point(t, cx, cy, area))

    # -- signal 1: tripwire ------------------------------------------------
    def crossing(self, cfg: DirectionConfig) -> Direction:
        if not cfg.configured or len(self.points) < 2:
            return Direction.UNKNOWN
        pts = list(self.points)
        sides = [_side(cfg.line, p.cx, p.cy) for p in pts]

        # Last clean sign change, ignoring points sitting on the line.
        last = Direction.UNKNOWN
        prev_sign = 0
        for s in sides:
            sign = 1 if s > 1e-4 else (-1 if s < -1e-4 else 0)
            if sign == 0:
                continue
            if prev_sign and sign != prev_sign:
                inward = (sign == cfg.inside_side)
                last = Direction.ENTER if inward else Direction.EXIT
            prev_sign = sign
        return last

    # -- signal 2: depth trend --------------------------------------------
    def depth(self, cfg: DirectionConfig) -> Direction:
        if len(self.points) < cfg.min_points:
            return Direction.UNKNOWN
        pts = list(self.points)
        n = max(2, len(pts) // 3)
        first = sum(p.area for p in pts[:n]) / n
        last = sum(p.area for p in pts[-n:]) / n
        if first <= 0 or last <= 0:
            return Direction.UNKNOWN
        ratio = last / first
        if ratio >= cfg.min_area_ratio:
            growing = True
        elif ratio <= 1.0 / cfg.min_area_ratio:
            growing = False
        else:
            return Direction.UNKNOWN
        inward = growing if cfg.depth_grows_inward else (not growing)
        return Direction.ENTER if inward else Direction.EXIT

    # -- how far the person actually moved ---------------------------------
    def travel(self) -> float:
        if len(self.points) < 2:
            return 0.0
        a, b = self.points[0], self.points[-1]
        return math.hypot(b.cx - a.cx, b.cy - a.cy)

    def duration(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return self.points[-1].t - self.points[0].t

    # -- combined verdict --------------------------------------------------
    def direction(self, cfg: DirectionConfig) -> tuple[Direction, str]:
        """Returns (direction, human-readable reason)."""
        if len(self.points) < cfg.min_points:
            return Direction.UNKNOWN, f"only {len(self.points)} points"
        if self.travel() < cfg.min_travel:
            return Direction.UNKNOWN, f"stationary ({self.travel():.3f} < {cfg.min_travel})"

        cross = self.crossing(cfg)
        dep = self.depth(cfg)

        if cross is not Direction.UNKNOWN and dep is not Direction.UNKNOWN:
            if cross == dep:
                return cross, f"line+depth agree ({cross.value})"
            if cfg.require_agreement:
                return Direction.UNKNOWN, f"line={cross.value} depth={dep.value} disagree"
            # The tripwire is the stronger evidence: it is a geometric fact,
            # while depth is an inference from box area.
            return cross, f"line={cross.value} (depth said {dep.value})"

        if cross is not Direction.UNKNOWN:
            return cross, f"line only ({cross.value})"
        if dep is not Direction.UNKNOWN:
            if not cfg.configured:
                return dep, f"depth only ({dep.value})"
            travelled = self.travel()
            if travelled >= cfg.depth_only_travel:
                return dep, f"depth only ({dep.value}, travel {travelled:.2f})"
            # Barely moved and never crossed: not enough to call.
            return Direction.UNKNOWN, (f"no crossing, travel {travelled:.2f} "
                                       f"< {cfg.depth_only_travel} (depth said {dep.value})")
        return Direction.UNKNOWN, "no usable signal"


def config_from_camera(cam) -> DirectionConfig:
    """Build a DirectionConfig from the Camera row's stored geometry."""
    line = None
    if cam.line_x1 is not None and cam.line_x2 is not None:
        line = ((cam.line_x1, cam.line_y1), (cam.line_x2, cam.line_y2))
    return DirectionConfig(
        line=line,
        inside_side=cam.inside_side if cam.inside_side else 1,
        depth_grows_inward=bool(cam.depth_grows_inward),
        min_travel=cam.min_travel if cam.min_travel else 0.06,
    )
