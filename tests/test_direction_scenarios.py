"""Direction of travel, scenario by scenario.

Every walk here is one that the 2026-09-02..04 production export showed the
rolling-window design getting wrong, or one it must keep getting right. The
geometry is the real site's: both cameras share the tripwire
(0,0.42)-(1,0.50); the Entrance camera's inside is BELOW the line (the
foreground) and a head GROWS walking inward; the Exit camera is the mirror.

The synthetic walks are deliberately simple - straight lines, plateaus, a
turn - because the questions are about the verdict logic, not the tracker.
Real trajectories are covered by bench/direction_eval.py on dumped footage.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.core.direction import Direction, DirectionConfig, Trajectory

W, H = 3840, 2160
FPS = 20.0
LINE = ((0.0, 0.42), (1.0, 0.50))

ENTRANCE = DirectionConfig(line=LINE, inside_side=1, depth_grows_inward=True)
EXIT_CAM = DirectionConfig(line=LINE, inside_side=-1, depth_grows_inward=False)

# Normalised head positions/areas. "far" is the top of the frame (small head),
# "near" is the bottom (large head). The line sits at y~0.46 mid-frame.
FAR = (0.5, 0.15, 0.0004)        # ~44 px head at 4K
NEAR = (0.5, 0.85, 0.0030)       # ~120 px head
ABOVE = (0.5, 0.30, 0.0009)      # outside for Entrance, inside for Exit cam
BELOW = (0.5, 0.65, 0.0018)      # inside for Entrance, outside for Exit cam


def _box(cx, cy, area):
    side = math.sqrt(max(area, 1e-9) * W * H)
    x, y = cx * W, cy * H
    return [x - side / 2, y - side / 2, x + side / 2, y + side / 2]


def segment(p0, p1, seconds, t0=0.0, synth=False, edge=False, area_scale=1.0):
    """Linear walk from p0 to p1 lasting `seconds`, one point per frame."""
    n = max(1, int(round(seconds * FPS)))
    rows = []
    for i in range(n):
        f = i / max(1, n - 1) if n > 1 else 0.0
        cx = p0[0] + (p1[0] - p0[0]) * f
        cy = p0[1] + (p1[1] - p0[1]) * f
        ar = (p0[2] + (p1[2] - p0[2]) * f) * area_scale
        rows.append((t0 + i / FPS, cx, cy, ar, synth, edge))
    return rows


def pause(p, seconds, t0=0.0, jitter=0.0, seed=0):
    """Standing still at p, with optional detection jitter."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(int(round(seconds * FPS))):
        dx, dy = (rng.normal(0, jitter), rng.normal(0, jitter)) if jitter else (0.0, 0.0)
        rows.append((t0 + i / FPS, p[0] + dx, p[1] + dy, p[2], False, False))
    return rows


def chain(*parts):
    """Concatenate segments, re-timing each to follow the previous one."""
    rows, t = [], 0.0
    for part in parts:
        for (_t, cx, cy, ar, synth, edge) in part:
            rows.append((t, cx, cy, ar, synth, edge))
            t += 1.0 / FPS
    return rows


def build(rows) -> Trajectory:
    tr = Trajectory()
    for (t, cx, cy, ar, synth, edge) in rows:
        b = _box(cx, cy, ar)
        if edge:
            # push the box off the bottom of the frame, as a head passing
            # under the camera does
            b = [b[0], H - 40, b[2], H + 60]
        tr.add(t, b, W, H, synth=synth)
    return tr


# --- the straightforward cases must keep working -------------------------

def test_a_straight_walk_in_is_enter_on_both_signals():
    d, why = build(segment(FAR, NEAR, 3.0)).direction(ENTRANCE)
    assert d is Direction.ENTER
    assert why == "line+depth agree (ENTER)"


def test_a_straight_walk_out_is_exit_on_both_signals():
    d, why = build(segment(NEAR, FAR, 3.0)).direction(ENTRANCE)
    assert d is Direction.EXIT
    assert why == "line+depth agree (EXIT)"


def test_the_exit_camera_sees_the_same_walk_mirrored():
    """Walking IN means walking AWAY from the Exit camera: the head shrinks and
    rises, and the far side is that camera's inside. Both cameras must call
    one walk the same way - that is the whole premise of the arbiter."""
    d, why = build(segment(NEAR, FAR, 3.0)).direction(EXIT_CAM)
    assert d is Direction.ENTER
    assert why == "line+depth agree (ENTER)"
    d, _ = build(segment(FAR, NEAR, 3.0)).direction(EXIT_CAM)
    assert d is Direction.EXIT


def test_too_short_a_track_is_unknown():
    d, why = build(segment(FAR, NEAR, 0.15)).direction(ENTRANCE)
    assert d is Direction.UNKNOWN
    assert why.startswith("only ")


def test_standing_still_is_unknown_however_long():
    """The 2026-08-27 case the staleness rule was written for: a 592-second
    track of somebody who never moved must not produce a verdict, and
    detection jitter must not be mistaken for travel."""
    d, why = build(pause(BELOW, 592.0, jitter=0.004)).direction(ENTRANCE)
    assert d is Direction.UNKNOWN
    assert why.startswith("stationary")


# --- what the rolling window got wrong ------------------------------------

def test_a_pause_after_crossing_keeps_the_verdict():
    """27 of the 41 recognised passes that produced no attendance in the
    export were 'stale(Ns, was ENTER/EXIT)': the person had crossed, stopped
    for longer than the window plus the latch, and the verdict was thrown
    away. A crossing followed by standing still is still a crossing."""
    walk = chain(segment(FAR, NEAR, 3.0), pause(NEAR, 45.0, jitter=0.002))
    d, why = build(walk).direction(ENTRANCE)
    assert d is Direction.ENTER
    assert "agree" in why


def test_ten_minutes_at_the_desk_after_walking_in_is_still_an_entry():
    walk = chain(segment(FAR, BELOW, 4.0), pause(BELOW, 600.0, jitter=0.002))
    d, _ = build(walk).direction(ENTRANCE)
    assert d is Direction.ENTER


def test_a_u_turn_seen_whole_names_the_side_they_came_back_to():
    """The contradiction pattern: somebody inside walks to the door area,
    crossing the line outward, and comes straight back. The rolling window's
    'last sign change' said ENTER; the other camera, which saw only the
    outbound half, said EXIT. The truth is that the person ended where they
    began - inside - and the verdict says so, with a reason that lets the
    arbiter rank it as a whole view over the other camera's half."""
    walk = chain(segment(NEAR, ABOVE, 2.5), pause(ABOVE, 3.0), segment(ABOVE, NEAR, 2.5))
    tr = build(walk)
    d, why = tr.direction(ENTRANCE)
    assert d is Direction.ENTER                      # "they are inside", twice observed
    assert why == "crossed and returned (inside)"
    # ...and the same image path under the mirror camera's geometry ends on
    # that camera's outside.
    d2, why2 = tr.direction(EXIT_CAM)
    assert d2 is Direction.EXIT
    assert why2 == "crossed and returned (outside)"
    # Diagnostics still expose what the old rule would have said.
    assert tr.last_crossing(ENTRANCE) is Direction.ENTER


def test_a_u_turn_that_never_reaches_the_line_is_not_a_crossing_either():
    walk = chain(segment(NEAR, BELOW, 2.0), segment(BELOW, NEAR, 2.0))
    d, why = build(walk).direction(ENTRANCE)
    assert d is Direction.UNKNOWN
    assert "returned" not in why           # never crossed at all


def test_extent_and_travel_tell_an_excursion_from_a_stroll():
    walk = chain(segment(NEAR, ABOVE, 2.5), segment(ABOVE, NEAR, 2.5))
    tr = build(walk)
    assert tr.extent() > 0.5
    assert tr.travel() < 0.02


def test_crossing_back_and_forth_ends_where_the_last_crossing_left_it():
    """Two crossings out, one back: the person is outside. The verdict is the
    NET result, not a count."""
    walk = chain(segment(NEAR, ABOVE, 2.0), segment(ABOVE, NEAR, 2.0),
                 segment(NEAR, ABOVE, 2.0), pause(ABOVE, 2.0))
    d, why = build(walk).direction(ENTRANCE)
    assert d is Direction.EXIT
    assert why.startswith("line")


# --- the depth signal must not be fooled by the tracker ------------------

def test_synthesised_heads_of_the_wrong_size_do_not_fake_a_depth_change():
    """When the head detector misses, the pipeline synthesises a head from
    the person box. Before the fix that box was a fixed fraction of the
    shoulder width and came out ~1.6x the real head's area, so a walk whose
    second half was synthesised looked like somebody approaching. Depth now
    reads detected heads only whenever there are enough of them."""
    real = segment(NEAR, BELOW, 2.0)                      # walking away, shrinking
    fake = segment(BELOW, ABOVE, 2.0, synth=True, area_scale=1.9)   # inflated stand-ins
    tr = build(chain(real, fake))
    assert tr.depth(ENTRANCE) is Direction.EXIT
    d, why = tr.direction(ENTRANCE)
    assert d is Direction.EXIT
    assert why == "line+depth agree (EXIT)"


def test_a_head_clipped_by_the_frame_edge_does_not_read_as_walking_away():
    """Somebody walking under the camera loses half their head box in the
    last frames. Its area collapses for a reason that has nothing to do with
    distance, and those frames are excluded from the depth comparison."""
    walk = chain(segment(FAR, NEAR, 3.0), segment(NEAR, NEAR, 0.6, edge=True))
    tr = build(walk)
    assert tr.depth(ENTRANCE) is Direction.ENTER
    d, why = tr.direction(ENTRANCE)
    assert d is Direction.ENTER
    assert why == "line+depth agree (ENTER)"


def test_depth_only_needs_real_travel():
    """Never crossed, but walked far toward the camera and doubled in size:
    accepted, because the observation on this site is that depth and line
    agree whenever both fire. Barely moved: refused."""
    far_right = (0.9, 0.55, 0.0008)
    near_right = (0.7, 0.85, 0.0030)
    d, why = build(segment(far_right, near_right, 3.0)).direction(ENTRANCE)
    assert d is Direction.ENTER
    assert why.startswith("depth only (ENTER")
    short = segment((0.9, 0.55, 0.0008), (0.86, 0.63, 0.0016), 3.0)
    d, why = build(short).direction(ENTRANCE)
    assert d is Direction.UNKNOWN
    assert why.startswith("no crossing")


def test_without_a_line_depth_decides_alone():
    cfg = DirectionConfig(line=None, depth_grows_inward=True)
    d, why = build(segment(FAR, NEAR, 3.0)).direction(cfg)
    assert d is Direction.ENTER
    assert why == "depth only (ENTER)"


# --- robustness ----------------------------------------------------------

def test_a_flicker_at_the_start_does_not_move_the_start_side():
    """One or two wild detections when a track is born must not decide which
    side it began on; the start is the majority of the first clean points."""
    rows = [(0.0, 0.5, 0.80, 0.0030, False, False), (0.05, 0.5, 0.78, 0.0030, False, False)]
    rows += segment(FAR, NEAR, 3.0, t0=0.1)
    d, _ = build(rows).direction(ENTRANCE)
    assert d is Direction.ENTER


def test_hovering_on_the_line_counts_for_neither_side():
    """A head sitting inside the dead band votes for nobody, so a person who
    appears ON the line and walks inside has no start side above it - that
    is not a crossing, and it must not become an EXIT by a jitter either."""
    on_line = (0.5, 0.46, 0.0012)
    walk = chain(pause(on_line, 2.0, jitter=0.003), segment(on_line, NEAR, 2.0))
    d, why = build(walk).direction(ENTRANCE)
    assert d is not Direction.EXIT


def test_the_history_is_bounded_and_the_verdict_survives_decimation():
    cfg = DirectionConfig(line=LINE, inside_side=1, depth_grows_inward=True, max_points=4000)
    tr = Trajectory(capacity=64, max_points=cfg.max_points)
    rows = chain(segment(FAR, NEAR, 3.0), pause(NEAR, 400.0, jitter=0.002))
    for (t, cx, cy, ar, synth, edge) in rows:
        tr.add(t, _box(cx, cy, ar), W, H, synth=synth)
    assert len(rows) > cfg.max_points
    assert len(tr) <= cfg.max_points
    assert tr.t[0] == pytest.approx(0.0)
    assert tr.duration() == pytest.approx(rows[-1][0], abs=0.1)
    d, _ = tr.direction(cfg)
    assert d is Direction.ENTER


def test_points_view_matches_what_was_added():
    tr = build(segment(FAR, NEAR, 0.5))
    pts = tr.points
    assert len(pts) == len(tr) == 10
    assert pts[0].cy < pts[-1].cy
    assert not pts[0].synth and not pts[0].edge


def test_a_sideways_walk_along_the_far_end_is_not_a_depth_verdict():
    """Seen in 20260831_114043_Entrance: a person stood at the far end for
    45 s, then walked sideways toward the side doors. The head grew 3x through
    perspective and the rolling window called it on depth. They never came
    nearer the line; nothing about their state is known from this track."""
    far_left = (0.40, 0.03, 0.0002)
    far_right = (0.72, 0.07, 0.0007)
    walk = chain(pause(far_left, 45.0, jitter=0.002), segment(far_left, far_right, 10.0))
    d, why = build(walk).direction(ENTRANCE)
    assert d is Direction.UNKNOWN
    assert why.startswith("no crossing, sideways")


def test_a_size_change_against_the_motion_is_refused():
    """Growing while moving UP the frame is a detector artefact, not depth."""
    walk = segment((0.5, 0.80, 0.0006), (0.5, 0.55, 0.0016), 3.0)     # grows but recedes
    d, why = build(walk).direction(ENTRANCE)
    assert d is Direction.UNKNOWN
    assert "sideways" in why


def test_a_track_born_half_a_second_before_the_line_still_starts_on_its_side():
    """20260826_131255_Entrance: somebody appears near the camera, crosses
    out within the first ten frames, spends ten seconds by the door and comes
    back. A majority over the first ten clean points put the start on the
    far side and no crossing was seen; the first clean points decide."""
    walk = chain(segment((0.9, 0.62, 0.0020), (0.9, 0.35, 0.0009), 0.6),
                 pause((0.9, 0.30, 0.0009), 10.0, jitter=0.002),
                 segment((0.9, 0.35, 0.0009), (0.8, 0.75, 0.0026), 2.0))
    tr = build(walk)
    assert tr.ends(ENTRANCE)[0] == 1
    d, why = tr.direction(ENTRANCE)
    assert d is Direction.ENTER
    assert why == "crossed and returned (inside)"


def test_a_track_that_ends_right_after_crossing_ends_on_the_new_side():
    walk = chain(segment(FAR, (0.5, 0.40, 0.0012), 3.0), segment((0.5, 0.40, 0.0012), (0.5, 0.55, 0.0016), 0.3))
    tr = build(walk)
    assert tr.ends(ENTRANCE)[1] == 1
    assert tr.direction(ENTRANCE)[0] is Direction.ENTER


def test_a_walk_that_never_reaches_the_line_says_nothing():
    """20260826_162805_Exit: from the far end to mid-frame and back, never
    across. The rolling window answered ENTER; there is nothing to answer."""
    walk = chain(pause(FAR, 5.0), segment(FAR, (0.3, 0.31, 0.0012), 6.0),
                 segment((0.3, 0.31, 0.0012), FAR, 6.0), pause(FAR, 3.0))
    d, why = build(walk).direction(EXIT_CAM)
    assert d is Direction.UNKNOWN
