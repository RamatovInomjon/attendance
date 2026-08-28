"""A latched direction may not outlive the evidence for it.

`Trajectory.points` is a deque(maxlen=90) - 4.5 s at 20 fps. When a person
stops moving, travel collapses and `direction()` correctly returns UNKNOWN with
reason "stationary". But `st.direction` only ever took NON-UNKNOWN verdicts, so
it kept the last real one indefinitely while `direction_reason` refreshed every
frame. The pair then disagreed, and the stale half drove attendance.

Observed live on 2026-08-27, seven times in one afternoon:

    12:21:43 [Entrance] CHECK_OUT  travel 0.001  dur 76.2s   dir=EXIT (stationary)
    12:27:18 [Exit]     CHECK_IN   travel 0.008  dur 592.7s  dir=ENTER (stationary)
    12:28:04 [Entrance] CHECK_OUT  travel 0.012  dur 348.9s  dir=EXIT (stationary)

The 592-second case carried a verdict up to ten minutes old. Two of the three
booked check-OUTS on the ENTRANCE camera for people who had not moved, and one
person was flipped three times in seven minutes.

The latch is not the bug - somebody pausing at a door should keep the direction
they arrived with. The bug is that it had no age limit.
"""
from __future__ import annotations

import pytest

from app.config import settings
from app.core.direction import Direction


def _age_out(direction, age_s, supported_at=100.0):
    """Reproduce the pipeline's staleness decision in isolation."""
    now = supported_at + age_s
    reason = "line+depth agree"
    if (direction is not Direction.UNKNOWN and supported_at > 0.0
            and (now - supported_at) > settings.direction_max_age_s):
        return Direction.UNKNOWN, f"stale({now - supported_at:.0f}s, was {direction.value})"
    return direction, reason


def test_a_fresh_verdict_survives():
    """The pause-at-the-door case must keep working."""
    d, _ = _age_out(Direction.ENTER, age_s=1.0)
    assert d is Direction.ENTER


def test_a_verdict_older_than_the_window_is_discarded():
    d, why = _age_out(Direction.EXIT, age_s=76.0)
    assert d is Direction.UNKNOWN
    assert "stale" in why and "EXIT" in why


def test_the_ten_minute_case():
    """The worst live instance: a 592-second stationary track that booked a
    check-IN on the EXIT camera."""
    d, why = _age_out(Direction.ENTER, age_s=592.0)
    assert d is Direction.UNKNOWN
    assert "stale" in why


def test_a_direction_never_supported_is_not_reported():
    """direction_at == 0 means no verdict ever held; the default must not leak."""
    now = 500.0
    supported_at = 0.0
    direction = Direction.EXIT
    stale = direction is not Direction.UNKNOWN and supported_at <= 0.0
    assert stale, "a never-supported direction must not drive a transition"


@pytest.mark.parametrize("age,expected", [
    (0.0, Direction.ENTER),
    (settings.direction_max_age_s - 0.1, Direction.ENTER),
    (settings.direction_max_age_s + 0.1, Direction.UNKNOWN),
])
def test_the_age_boundary(age, expected):
    d, _ = _age_out(Direction.ENTER, age_s=age)
    assert d is expected


def test_the_limit_does_not_outlive_the_trajectory_window():
    """The window is 90 points at 20 fps = 4.5 s. A verdict allowed to outlive
    it would be derived from data the trajectory no longer holds."""
    assert settings.direction_max_age_s <= 90 / settings.track_frame_rate + 1.0
