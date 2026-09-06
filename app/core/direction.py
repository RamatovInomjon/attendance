"""Direction of travel for a tracked person.

Why this exists
---------------
A camera's *role* says where it points, not which way a person was walking.
Both cameras here overlook the same lobby from opposite ends, so somebody
leaving the building walks through the entrance camera's view and - under
role-only logic - would be recorded as arriving.  Direction has to come from
the track itself.

Two independent signals, deliberately combined:

* **Tripwire crossing.**  A line across the walkway with a declared "inside"
  side.  A track that starts on one side and ends on the other has crossed it,
  and the side it ended on says which way.  This is the primary signal and the
  one commercial VMS platforms use.

* **Depth trend.**  These cameras look down a corridor, so a head walking
  toward the lens grows and drifts down the frame; walking away it shrinks and
  rises.  Head-box area over the track is therefore a real depth cue, and it
  works even when a track never reaches the tripwire.

The verdict is a property of the WHOLE pass
-------------------------------------------
Direction used to be computed over a rolling window of the last 90 points -
4.5 s at 20 fps - as "the last time the head changed sides".  Two things went
wrong with that in production, both visible in the 2026-09-02..04 export:

* **Pausing lost the verdict.**  Someone who crossed the line and then stood
  still for a few seconds - at the door, at the desk, talking - had the
  crossing fall out of the window.  A latch kept the old verdict for 5 s more
  and then declared it "stale": 27 of the 41 recognised passes that produced
  no attendance in three days were exactly this, with reasons like
  ``stale(6s, was ENTER)``.  The person had crossed; the system had forgotten.

* **A U-turn read as a crossing.**  Somebody who walks to the door area and
  comes straight back ends where they began, but the last sign change in the
  window says ENTER, and the other camera - which only saw the outbound half -
  says EXIT.  Both were "line+depth agree".  One walk, two confident opposite
  answers, and whichever camera had the stronger face won.  That booked
  check-outs for people who never left.

So the track keeps EVERY point from its first frame to its last, and the
question asked at the end is the only one that matters for attendance: *where
did this person end up, relative to where they started?*

* start side != end side  -> a net crossing, ENTER or EXIT
* both sides visited, same side at both ends -> "crossed and returned": the
  person demonstrably went across and is back on the side they started on, so
  the verdict names that side (ENTER for inside, EXIT for outside) - the
  state machine reads a verdict as "this is where they are now", and the
  cross-camera arbiter lets this whole view of a U-turn outrank the other
  camera's half of it
* never crossed but moved far and grew or shrank -> depth verdict
* never moved -> UNKNOWN

There is no staleness any more: a crossing followed by ten minutes standing
still is still a crossing, because the person is still on the other side.
Refusing to guess is preserved - an UNKNOWN track logs a sighting and changes
no attendance state, since a missed direction is recovered on the next pass
while a wrong one writes a wrong row that nothing reveals.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Direction(str, Enum):
    ENTER = "ENTER"
    EXIT = "EXIT"
    UNKNOWN = "UNKNOWN"


@dataclass
class Point:
    t: float
    cx: float          # head centroid, normalized 0..1
    cy: float
    area: float        # head box area, normalized to frame area
    synth: bool = False    # head position synthesised from the person box
    edge: bool = False     # box clipped by the frame border


@dataclass
class DirectionConfig:
    """Per-camera geometry.  All coordinates normalized to 0..1."""
    line: tuple[tuple[float, float], tuple[float, float]] | None = None
    inside_side: int = 1                  # which sign of the cross product is "inside"
    depth_grows_inward: bool = True       # head grows as the person moves inside
    min_travel: float = 0.06              # a track that never got this far from its start is standing still
    min_area_ratio: float = 1.25          # area must change this much for a depth verdict
    min_points: int = 5
    require_agreement: bool = False       # demand both signals when both are available
    # Accept a depth-only verdict when the track never reached the tripwire but
    # did travel this far. Justified by observation: on every track where both
    # signals fired they agreed, while 7 of 18 tracks crossed no line at all
    # because people walk past the camera rather than through the middle of
    # the frame. Requiring more travel than `min_travel` keeps it conservative -
    # depth is an inference, a crossing is a geometric fact.
    depth_only_travel: float = 0.15
    # Both cameras look DOWN the corridor, so somebody approaching the lens
    # moves down the frame as they grow. A depth-only verdict has to show that
    # vertical travel, in the direction the size change implies. Without it, a
    # person walking sideways along the far end - toward the side doors, say -
    # grew 3x through perspective alone and was called ENTER on depth, when
    # they never came a step nearer the line.
    approach_is_down: bool = True
    # A point closer to the line than this (normalized, perpendicular) is "on
    # the line" and counts for neither side. Detection jitter on a head box is
    # a few pixels at 4K; 0.012 of the frame is ~26 px, comfortably above it.
    dead_band: float = 0.012
    # How many clean points at each end of the track define where it started
    # and where it ended. Half a second at 20 fps: long enough to outvote a
    # flicker, short enough that a pause at the far end still reads as the end.
    edge_points: int = 10
    # Memory bound. A track that lives an hour at 20 fps is 72k points; beyond
    # this the history is decimated 2:1, which keeps both ends and costs
    # nothing the verdict depends on.
    max_points: int = 100_000

    @property
    def configured(self) -> bool:
        return self.line is not None


def _side(line, x: float, y: float) -> float:
    """Signed side of the line: the z of (B-A) x (P-A)."""
    (ax, ay), (bx, by) = line
    return (bx - ax) * (y - ay) - (by - ay) * (x - ax)


def _majority(signs: np.ndarray) -> int:
    s = float(np.sum(signs))
    return 1 if s > 0 else (-1 if s < 0 else 0)


def _end_side(signs: np.ndarray, k: int) -> int:
    """Which side a run of clean points begins on.

    `signs` is ordered from the end in question inward (the first element is
    the very first, or very last, clean point). The outermost point decides
    when at least three of the outermost five agree with it; otherwise the
    majority of the first `k` does. The outermost point matters because a
    person who appears half a second before crossing has only a handful of
    points on their starting side, and a plain majority over ten would put
    the start on the far side of the line and lose the crossing. The
    three-of-five check keeps one wild detection from deciding on its own.
    """
    if len(signs) == 0:
        return 0
    first = int(signs[0])
    near = signs[:5]
    if first != 0 and int(np.count_nonzero(near == first)) >= min(3, len(near)):
        return first
    return _majority(signs[:k])


# Column layout of the point buffer.
_T, _CX, _CY, _AREA, _SYNTH, _EDGE = range(6)


class Trajectory:
    """Every position of one track, first frame to last.

    Stored as a growing numpy array rather than a list of objects so that the
    whole-track verdict stays a handful of vectorised operations however long
    the person stays in view: a 12 000-point track (ten minutes) is decided in
    well under a millisecond, which is what lets the live overlay ask every
    frame.
    """

    __slots__ = ("_buf", "n", "max_points")

    def __init__(self, capacity: int = 256, max_points: int = 100_000):
        self._buf = np.zeros((max(8, capacity), 6), np.float64)
        self.n = 0
        self.max_points = max_points

    def add(self, t: float, box, frame_w: int, frame_h: int, synth: bool = False,
            edge: bool | None = None):
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        if self.n == self._buf.shape[0]:
            if self.n >= self.max_points:
                kept = self._buf[:self.n:2].copy()
                self.n = kept.shape[0]
                self._buf[:self.n] = kept
            else:
                self._buf = np.concatenate([self._buf, np.zeros_like(self._buf)])
        if edge is None:
            edge = x1 <= 1.0 or y1 <= 1.0 or x2 >= frame_w - 1.0 or y2 >= frame_h - 1.0
        self._buf[self.n] = (
            t,
            ((x1 + x2) * 0.5) / frame_w,
            ((y1 + y2) * 0.5) / frame_h,
            max(0.0, (x2 - x1) * (y2 - y1)) / float(frame_w * frame_h),
            1.0 if synth else 0.0,
            1.0 if edge else 0.0,
        )
        self.n += 1

    def __len__(self) -> int:
        return self.n

    # -- column views ------------------------------------------------------
    @property
    def t(self) -> np.ndarray:
        return self._buf[:self.n, _T]

    @property
    def cx(self) -> np.ndarray:
        return self._buf[:self.n, _CX]

    @property
    def cy(self) -> np.ndarray:
        return self._buf[:self.n, _CY]

    @property
    def area(self) -> np.ndarray:
        return self._buf[:self.n, _AREA]

    @property
    def synth(self) -> np.ndarray:
        return self._buf[:self.n, _SYNTH] > 0.5

    @property
    def edge(self) -> np.ndarray:
        return self._buf[:self.n, _EDGE] > 0.5

    @property
    def points(self) -> list[Point]:
        """Object view, for tests and tools. The verdicts never build this."""
        return [Point(r[_T], r[_CX], r[_CY], r[_AREA], r[_SYNTH] > 0.5, r[_EDGE] > 0.5)
                for r in self._buf[:self.n]]

    # -- signal 1: tripwire ------------------------------------------------
    def _distances(self, cfg: DirectionConfig) -> np.ndarray:
        """Signed perpendicular distance of every point from the line, in
        normalized frame units. Same sign convention as `_side`."""
        (ax, ay), (bx, by) = cfg.line
        length = math.hypot(bx - ax, by - ay) or 1.0
        return ((bx - ax) * (self.cy - ay) - (by - ay) * (self.cx - ax)) / length

    def ends(self, cfg: DirectionConfig) -> tuple[int, int, int, int]:
        """(start_side, end_side, points_positive, points_negative).

        Sides come from the first and last clean points - clean meaning
        further than `dead_band` from the line, so a head hovering on the
        line votes for neither side - see `_end_side`. 0 means undecided.
        """
        if not cfg.configured or self.n == 0:
            return 0, 0, 0, 0
        d = self._distances(cfg)
        idx = np.flatnonzero(np.abs(d) > cfg.dead_band)
        if len(idx) == 0:
            return 0, 0, 0, 0
        k = max(1, cfg.edge_points)
        signs = np.sign(d[idx]).astype(int)
        start = _end_side(signs, k)
        end = _end_side(signs[::-1], k)
        pos = int(np.count_nonzero(d[idx] > 0))
        return start, end, pos, len(idx) - pos

    def crossing(self, cfg: DirectionConfig) -> Direction:
        """Net crossing over the whole track: the side it ended on, if that
        differs from the side it started on."""
        if not cfg.configured or self.n < 2:
            return Direction.UNKNOWN
        start, end, _, _ = self.ends(cfg)
        if start == 0 or end == 0 or start == end:
            return Direction.UNKNOWN
        return Direction.ENTER if end == cfg.inside_side else Direction.EXIT

    def excursion(self, cfg: DirectionConfig) -> bool:
        """Crossed and came back: both sides were genuinely visited, yet the
        track ends on the side it began. The person is where they started,
        and that side is now a fact observed twice over."""
        if not cfg.configured:
            return False
        start, end, pos, neg = self.ends(cfg)
        return start != 0 and start == end and min(pos, neg) >= max(1, cfg.edge_points)

    def last_crossing(self, cfg: DirectionConfig) -> Direction:
        """The most recent sign change - what the rolling-window design used.
        Kept for diagnosis; attendance reads `crossing()`."""
        if not cfg.configured or self.n < 2:
            return Direction.UNKNOWN
        d = self._distances(cfg)
        signs = np.sign(d[np.abs(d) > cfg.dead_band])
        if len(signs) < 2:
            return Direction.UNKNOWN
        changes = np.flatnonzero(signs[1:] != signs[:-1])
        if len(changes) == 0:
            return Direction.UNKNOWN
        final = int(signs[changes[-1] + 1])
        return Direction.ENTER if final == cfg.inside_side else Direction.EXIT

    # -- signal 2: depth trend --------------------------------------------
    def depth(self, cfg: DirectionConfig) -> Direction:
        """Did the head end up larger or smaller than it started?

        Compares the median area over the first and last `edge_points` usable
        points. Usable means a DETECTED head that is not clipped by the frame
        border: a synthesised box has a different size from a real one, and a
        box cut off by the edge of the frame shrinks for a reason that has
        nothing to do with distance - somebody walking under the camera loses
        half their head box in the last frames and would read as walking away.
        Synthesised points are used only when the track never showed enough
        real heads to judge from.
        """
        if self.n < cfg.min_points:
            return Direction.UNKNOWN
        area = self.area
        usable = area[~self.synth & ~self.edge]
        if len(usable) < 2 * cfg.min_points:
            usable = area[~self.edge]
        if len(usable) < 2 * cfg.min_points:
            usable = area
        if len(usable) < cfg.min_points:
            return Direction.UNKNOWN
        k = max(1, min(cfg.edge_points, len(usable) // 2))
        first = float(np.median(usable[:k]))
        last = float(np.median(usable[-k:]))
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
        """Net displacement, first point to last."""
        if self.n < 2:
            return 0.0
        return math.hypot(self.cx[-1] - self.cx[0], self.cy[-1] - self.cy[0])

    def descent(self) -> float:
        """Net vertical displacement, positive when the head moved DOWN the
        frame - toward a camera that looks down the corridor."""
        if self.n < 2:
            return 0.0
        return float(self.cy[-1] - self.cy[0])

    def extent(self) -> float:
        """The farthest the track ever got from where it began. Unlike
        `travel()` this does not collapse when the person comes back."""
        if self.n < 2:
            return 0.0
        return float(np.hypot(self.cx - self.cx[0], self.cy - self.cy[0]).max())

    def duration(self) -> float:
        if self.n < 2:
            return 0.0
        return float(self.t[-1] - self.t[0])

    # -- combined verdict --------------------------------------------------
    def direction(self, cfg: DirectionConfig) -> tuple[Direction, str]:
        """Returns (direction, human-readable reason) for the whole track."""
        if self.n < cfg.min_points:
            return Direction.UNKNOWN, f"only {self.n} points"
        ext = self.extent()
        if ext < cfg.min_travel:
            return Direction.UNKNOWN, f"stationary ({ext:.3f} < {cfg.min_travel})"

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

        if self.excursion(cfg):
            # Went across and came back. Whatever the depth trend says, the
            # person is on the side they started on - and having watched
            # them cross twice, that side is certain. Report it: a U-turn
            # to the door and back leaves somebody INSIDE, and saying so is
            # what lets this view overrule the other camera, which saw only
            # the outbound half and called it EXIT.
            end = self.ends(cfg)[1]
            inside = end == cfg.inside_side
            return (Direction.ENTER if inside else Direction.EXIT,
                    f"crossed and returned ({'inside' if inside else 'outside'})")

        if dep is not Direction.UNKNOWN:
            if not cfg.configured:
                return dep, f"depth only ({dep.value})"
            travelled = self.travel()
            if travelled < cfg.depth_only_travel:
                # Barely moved and never crossed: not enough to call.
                return Direction.UNKNOWN, (f"no crossing, travel {travelled:.2f} "
                                           f"< {cfg.depth_only_travel} (depth said {dep.value})")
            if cfg.approach_is_down:
                # The size change must be borne out by motion along the
                # corridor: approaching means moving down the frame.
                grew = (dep is Direction.ENTER) == cfg.depth_grows_inward
                dy = self.descent()
                if abs(dy) < cfg.depth_only_travel * 0.5 or (dy > 0) != grew:
                    return Direction.UNKNOWN, (f"no crossing, sideways travel "
                                               f"(dy {dy:+.2f}, depth said {dep.value})")
            return dep, f"depth only ({dep.value}, travel {travelled:.2f})"
        return Direction.UNKNOWN, "no usable signal"


def config_from_camera(cam) -> DirectionConfig:
    """Build a DirectionConfig from the Camera row's stored geometry."""
    # ALL FOUR, not just the x pair. A row with x1/x2 set and y1 NULL built a
    # line containing None, and `_side()` then raised TypeError for every track
    # on every frame - recognition stops dead while frames keep flowing and the
    # stream stats stay green, which is the hardest failure here to notice.
    line = None
    coords = (cam.line_x1, cam.line_y1, cam.line_x2, cam.line_y2)
    if all(c is not None for c in coords):
        line = ((cam.line_x1, cam.line_y1), (cam.line_x2, cam.line_y2))
    return DirectionConfig(
        line=line,
        inside_side=cam.inside_side if cam.inside_side else 1,
        depth_grows_inward=bool(cam.depth_grows_inward),
        min_travel=cam.min_travel if cam.min_travel else 0.06,
    )
