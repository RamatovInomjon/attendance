"""IN/OUT attendance state machine.

Replaces the previous elapsed-time heuristic ("first sighting is IN, anything
300 s later is OUT"), under which standing near the entrance camera for six
minutes checked you out without moving.

Direction comes from the **camera's role**; whether an event counts comes from
the employee's **presence state**.  Both are required — that is the whole point.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from sqlalchemy import select

from app.config import settings
from app.core.direction import Direction
from app.db.models import (
    CameraRole, DailyAttendance, PresenceStatus, RecognitionEvent,
)

log = logging.getLogger(__name__)


def business_date(ts: datetime) -> date:
    """Local calendar date, with the day starting at DAY_BOUNDARY_HOUR.

    04:00 rather than midnight so a late shift ending at 01:00 files against the
    day it started.  The old code used `datetime.utcnow().date()`, which in
    Asia/Tashkent (UTC+5) rolled the day over at 05:00 local and filed every
    early-morning arrival under the previous day.
    """
    local = ts.astimezone(settings.tz)
    if local.hour < settings.day_boundary_hour:
        local -= timedelta(days=1)
    return local.date()


@dataclass
class Decision:
    transition: str          # "" | CHECK_IN | CHECK_OUT | RE_SIGHTING | DEBOUNCED
    daily_id: int | None = None

    @property
    def is_transition(self) -> bool:
        return self.transition in ("CHECK_IN", "CHECK_OUT")


class AttendanceService:
    """Applies one recognition event to the durable attendance state."""

    def __init__(self, cooldown_s: int | None = None):
        self.cooldown = timedelta(seconds=cooldown_s or settings.event_cooldown_s)

    def _daily(self, s, employee_id: int, bdate: date) -> DailyAttendance:
        row = s.execute(
            select(DailyAttendance).where(
                DailyAttendance.employee_id == employee_id,
                DailyAttendance.business_date == bdate,
            )
        ).scalar_one_or_none()
        if row is None:
            row = DailyAttendance(
                employee_id=employee_id, business_date=bdate,
                presence=PresenceStatus.OUTSIDE, worked_seconds=0, status="PRESENT",
            )
            s.add(row)
            s.flush()
        return row

    def _debounced(self, s, employee_id: int, camera_id: int | None, ts: datetime) -> bool:
        """Same person, same camera, inside the cooldown -> ignore."""
        last = s.execute(
            select(RecognitionEvent.ts)
            .where(
                RecognitionEvent.employee_id == employee_id,
                RecognitionEvent.camera_id == camera_id,
            )
            .order_by(RecognitionEvent.ts.desc())
            .limit(1)
        ).scalar_one_or_none()
        if last is None:
            return False
        return (ts - last) < self.cooldown

    @staticmethod
    def _effective_role(role: CameraRole, direction: str) -> CameraRole | None:
        """What this sighting means, given how the person was actually moving.

        Direction wins over the camera's role.  The role only says where the
        camera points; both cameras here overlook the same corridor, so a
        person leaving can pass through the entrance camera's view.  Trusting
        the role there writes a check-in for somebody who is walking out.

        A track whose direction could not be determined yields None, and the
        caller logs the sighting without moving the attendance state.  Guessing
        would put a wrong time in the record; skipping costs nothing, because
        the person is seen again on their next pass.
        """
        if direction == Direction.ENTER.value:
            return CameraRole.IN
        if direction == Direction.EXIT.value:
            return CameraRole.OUT
        if role == CameraRole.BOTH:
            return None          # a dual-purpose camera tells us nothing on its own
        return None

    def record(
        self, s, *, employee_id: int, camera_id: int | None, role: CameraRole,
        ts: datetime, score: float, margin: float = 0.0, track_id: int = -1,
        face_px: int = 0, votes: str = "", snapshot: str | None = None,
        direction: str = "UNKNOWN", direction_reason: str = "",
        require_direction: bool = True,
    ) -> Decision:
        bdate = business_date(ts)

        if self._debounced(s, employee_id, camera_id, ts):
            return Decision("DEBOUNCED")

        eff = self._effective_role(role, direction)
        if eff is None and not require_direction:
            eff = role          # fallback for cameras with no geometry configured

        daily = self._daily(s, employee_id, bdate)
        transition = "RE_SIGHTING" if eff is not None else "NO_DIRECTION"

        if eff in (CameraRole.IN, CameraRole.BOTH) and daily.presence == PresenceStatus.OUTSIDE:
            daily.presence = PresenceStatus.INSIDE
            daily.entered_at = ts
            if daily.check_in_time is None or ts < daily.check_in_time:
                daily.check_in_time = ts
                if snapshot:
                    daily.check_in_snapshot = snapshot
            transition = "CHECK_IN"

        elif eff in (CameraRole.OUT, CameraRole.BOTH) and daily.presence == PresenceStatus.OUTSIDE \
                and daily.check_in_time is None:
            # First sighting of the day is somebody LEAVING. They were in the
            # building before the system saw them - arrived early, or the
            # service was restarted mid-day. Dropping this loses real evidence
            # of presence, so the check-out is recorded and the row is flagged
            # NO_CHECKIN rather than inventing an arrival time.
            daily.check_out_time = ts
            if snapshot:
                daily.check_out_snapshot = snapshot
            daily.status = "NO_CHECKIN"
            transition = "CHECK_OUT"

        elif eff in (CameraRole.OUT, CameraRole.BOTH) and daily.presence == PresenceStatus.INSIDE:
            daily.presence = PresenceStatus.OUTSIDE
            if daily.entered_at is not None:
                worked = (ts - daily.entered_at).total_seconds()
                if worked > 0:
                    # Accumulate, so a lunch break subtracts instead of vanishing.
                    daily.worked_seconds = int((daily.worked_seconds or 0) + worked)
                daily.entered_at = None
            daily.check_out_time = ts          # last OUT wins
            if snapshot:
                daily.check_out_snapshot = snapshot
            transition = "CHECK_OUT"

        daily.event_count = (daily.event_count or 0) + 1

        # First sighting of the day on the OUT camera: the person is leaving a
        # building we never saw them enter -- they came in through an uncovered
        # door, or the IN camera missed them.  Flag it for review rather than
        # inventing a check-in time out of a departure.
        if daily.check_in_time is None and transition in ("RE_SIGHTING", "CHECK_OUT"):
            daily.status = "NO_CHECKIN"
        elif daily.status != "NO_CHECKIN":
            daily.status = "PRESENT"

        s.add(RecognitionEvent(
            employee_id=employee_id, camera_id=camera_id, role=role, ts=ts,
            business_date=bdate, score=score, margin=margin, track_id=track_id,
            face_px=face_px, votes=votes, snapshot=snapshot, accepted=True,
            transition=transition, direction=direction,
            direction_reason=direction_reason[:96],
        ))
        return Decision(transition, daily.id)

    def close_open_intervals(self, s, bdate: date) -> int:
        """End-of-day sweep.

        An open interval is **flagged**, never closed with an invented time.
        Writing a plausible-looking check-out is the one option that destroys
        trust in the whole report.
        """
        rows = s.execute(
            select(DailyAttendance).where(
                DailyAttendance.business_date == bdate,
                DailyAttendance.presence == PresenceStatus.INSIDE,
            )
        ).scalars().all()
        for r in rows:
            if settings.open_interval_policy == "close_at_eod":
                r.presence = PresenceStatus.OUTSIDE
                r.entered_at = None
            r.status = "NO_CHECKOUT"
        return len(rows)
