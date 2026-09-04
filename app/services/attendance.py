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

    def _debounced(self, s, employee_id: int, camera_id: int | None,
                   ts: datetime, direction: str = "") -> bool:
        """Same person, same camera, SAME direction, inside the cooldown.

        The direction check is load-bearing and was missing. The rule used to
        suppress ANY event from a camera within the cooldown, which throws away
        genuine state changes: somebody who steps out and comes back a minute
        later has their return silently dropped, because their departure was
        recorded less than 90 s earlier on the same camera.

        Found by inspecting `20260831_114309_Entrance.mp4`, which holds a clean
        entrance - recognised at 0.689 with 45 agreeing frames and ENTER
        confirmed by both the tripwire and the depth trend - whose CHECK_IN was
        discarded because the same camera had seen that person leaving shortly
        before. The recognition was never the problem; the event was thrown
        away after the fact.

        A REPEAT of the same direction is a re-sighting and is still
        suppressed - that is what the cooldown is for. A CHANGE of direction is
        never a re-sighting.
        """
        row = s.execute(
            select(RecognitionEvent.ts, RecognitionEvent.direction)
            .where(
                RecognitionEvent.employee_id == employee_id,
                RecognitionEvent.camera_id == camera_id,
            )
            .order_by(RecognitionEvent.ts.desc())
            .limit(1)
        ).first()
        if row is None:
            return False
        last_ts, last_dir = row
        if (ts - last_ts) >= self.cooldown:
            return False
        if direction and last_dir and direction != last_dir:
            return False          # a different direction is a real transition
        return True

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
        require_direction: bool = True, apply_state: bool = True,
    ) -> Decision:
        """Apply one completed pass.

        `apply_state=False` records the sighting WITHOUT moving attendance
        state. That is what the losing half of a cross-camera pair gets: both
        cameras really did see the person, so the evidence is kept, but one walk
        may only produce one transition. See app/services/arbiter.py.
        """
        bdate = business_date(ts)

        if self._debounced(s, employee_id, camera_id, ts, direction):
            return Decision("DEBOUNCED")

        if not apply_state:
            daily = self._daily(s, employee_id, bdate)
            daily.event_count = (daily.event_count or 0) + 1
            s.add(RecognitionEvent(
                employee_id=employee_id, camera_id=camera_id, role=role, ts=ts,
                business_date=bdate, score=score, margin=margin, track_id=track_id,
                face_px=face_px, votes=votes, snapshot=snapshot, accepted=True,
                transition="DUPLICATE_VIEW", direction=direction,
                direction_reason=direction_reason[:96],
            ))
            return Decision("DUPLICATE_VIEW", daily.id)

        eff = self._effective_role(role, direction)
        if eff is None and not require_direction:
            eff = role          # fallback for cameras with no geometry configured

        daily = self._daily(s, employee_id, bdate)
        transition = self._apply(daily, eff, ts, snapshot)
        daily.event_count = (daily.event_count or 0) + 1

        s.add(RecognitionEvent(
            employee_id=employee_id, camera_id=camera_id, role=role, ts=ts,
            business_date=bdate, score=score, margin=margin, track_id=track_id,
            face_px=face_px, votes=votes, snapshot=snapshot, accepted=True,
            transition=transition, direction=direction,
            direction_reason=direction_reason[:96],
        ))
        return Decision(transition, daily.id)

    def _apply(self, daily, eff: CameraRole | None, ts: datetime,
               snapshot: str | None) -> str:
        """Move one daily row's presence state for one sighting.

        Separated from `record` so the SAME state machine can be replayed over
        the surviving events when an admin voids one - see `rebuild`. Derived
        state cannot be patched by hand: `worked_seconds` accumulates and
        `presence` is a latch, so subtracting one event's effect is guesswork
        that goes wrong the moment two events interleave. Replaying the real
        rules is the only version that stays correct as the rules change.

        Returns the transition; the caller owns `event_count` and the row.
        """
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

        # First sighting of the day on the OUT camera: the person is leaving a
        # building we never saw them enter -- they came in through an uncovered
        # door, or the IN camera missed them.  Flag it for review rather than
        # inventing a check-in time out of a departure.
        #
        # RECOMPUTED from state, never guarded on its own previous value. The
        # old form was `elif daily.status != "NO_CHECKIN": status = "PRESENT"`,
        # which could set the flag but never clear it: once raised, the guard
        # was permanently false, so a genuine check-in arriving later left the
        # row flagged anyway. 21 of the 30 flagged rows in the live database
        # HAVE a check-in time - 70% false positives on the one flag operators
        # are asked to review.
        #
        # NO_CHECKOUT is not touched here. It is written by the end-of-day
        # sweep, for a business date that is already over, so no live event can
        # legitimately overwrite it.
        if daily.status != "NO_CHECKOUT":
            daily.status = "PRESENT" if daily.check_in_time is not None else "NO_CHECKIN"
        return transition

    def rebuild(self, s, employee_id: int, bdate: date) -> int:
        """Recompute one person's day from the events that still stand.

        Called after an admin voids a recognition. The daily row is DERIVED
        state - `worked_seconds` accumulates, `presence` is a latch, and
        `check_in_time` keeps the earliest - so it cannot be corrected by
        subtracting the voided event's effect. Nor can the stored transitions
        simply be replayed: a transition is a function of the state at the time,
        so voiding the event that produced CHECK_IN means the next event's
        RE_SIGHTING must BECOME the check-in. Both are recomputed here, and the
        events' `transition` column is rewritten to match, so the log and the
        summary never disagree.

        Only `transition` is rewritten. Nothing else on an event is touched: the
        score, the snapshot and the time are what the camera saw, and no
        correction makes them untrue.

        Returns the number of events replayed.
        """
        from app.core.direction import config_from_camera
        from app.db.models import Camera

        row = self._daily(s, employee_id, bdate)
        row.presence = PresenceStatus.OUTSIDE
        row.entered_at = None
        row.check_in_time = row.check_out_time = None
        row.check_in_snapshot = row.check_out_snapshot = None
        row.worked_seconds = 0
        row.event_count = 0
        row.status = "PRESENT"

        # Whether a camera can fall back to its ROLE when direction is unknown
        # is a property of its geometry, exactly as it is on the capture thread
        # (worker.py passes `direction_cfg.configured`). Read it from the same
        # place rather than storing a second copy on every event.
        geom = {c.id: config_from_camera(c).configured
                for c in s.execute(select(Camera)).scalars().all()}

        events = s.execute(
            select(RecognitionEvent)
            .where(RecognitionEvent.employee_id == employee_id,
                   RecognitionEvent.business_date == bdate,
                   RecognitionEvent.voided_at.is_(None))
            .order_by(RecognitionEvent.ts, RecognitionEvent.id)
        ).scalars().all()

        for e in events:
            row.event_count = (row.event_count or 0) + 1
            if e.transition == "DUPLICATE_VIEW":
                # The losing half of a cross-camera pair. It was recorded as
                # evidence and deliberately moved no state; replaying it as a
                # transition would invent the second check-in the arbiter
                # exists to prevent.
                continue
            eff = self._effective_role(e.role, e.direction or "")
            if eff is None and not geom.get(e.camera_id, True):
                eff = e.role
            e.transition = self._apply(row, eff, e.ts, e.snapshot)

        # A finished day with nobody having checked out is flagged, never closed
        # with an invented time - the same rule the end-of-day sweep applies,
        # and it has to be re-applied here because the reset above cleared it.
        if row.presence == PresenceStatus.INSIDE and bdate < business_date(
                datetime.now(settings.tz)):
            self._close_row(row)
        return len(events)

    @staticmethod
    def _close_row(r) -> None:
        """Flag one unfinished day. Shared by the sweep and by `rebuild`."""
        if settings.open_interval_policy == "close_at_eod":
            r.presence = PresenceStatus.OUTSIDE
            r.entered_at = None
        if r.status != "NO_CHECKIN":
            r.status = "NO_CHECKOUT"

    def close_open_intervals(self, s, bdate: date, *, catch_up: bool = True) -> int:
        """End-of-day sweep.

        An open interval is **flagged**, never closed with an invented time.
        Writing a plausible-looking check-out is the one option that destroys
        trust in the whole report.

        `catch_up` sweeps every business date up to and including `bdate`, not
        just that one day. Sweeping a single day means any day the service was
        down is never revisited, and its rows claim those people are still in
        the building forever. 21 such rows had accumulated, the oldest five days
        stale, because the timer that would have run this was never installed.

        A row that was already flagged NO_CHECKIN keeps that flag. It is the
        more specific fact - we never saw them arrive - and overwriting it with
        NO_CHECKOUT loses it for exactly the rows where both are true.
        """
        q = select(DailyAttendance).where(
            DailyAttendance.presence == PresenceStatus.INSIDE,
            DailyAttendance.business_date <= bdate if catch_up
            else DailyAttendance.business_date == bdate,
        )
        rows = s.execute(q).scalars().all()
        for r in rows:
            self._close_row(r)
        return len(rows)
