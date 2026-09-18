"""Attendance aggregated the way HR reads it: hours, pass counts, evidence.

The attendance tables answer "what happened to this person on this day".  HR
asks something else — "how long did this person work this month, how many times
did they come and go, and show me the face behind each one" — and that question
spans days and people.  Aggregating it in the page function would put the rules
below inside a template loop, where they cannot be tested and are copied wrong
by the next reader; so they live here, and the pages render what this returns.

FOUR RULES, EACH OF WHICH THE LIVE DATA ALREADY BREAKS
------------------------------------------------------

**Hours come only from closed pairs, and an unclosed day is not a zero.**
`worked_seconds` accumulates when a check-out closes an interval, so a day that
never saw the person leave carries 0 — and on the production database that is
142 of the last 492 day-rows (29%), 66 of them with a check-in recorded.  A
timesheet that sums the column and prints "0.0" for those is not approximately
right, it is confidently wrong about a third of the month, and nothing on the
page would say so.  `PersonTotals.incomplete_days` counts them and the pages
show it beside the total, so the number HR reads is "hours we can prove" next
to "days still to fix" — never one silently standing in for the other.

**An open day is reported separately, not folded in.** 168 rows sit at
`presence=INSIDE`.  Someone still in the building has worked time that is real
but not yet observed to have ended, and adding a running `now - entered_at` into
a monthly total makes that total change every time the page is refreshed and
disagree with the CSV exported a minute earlier.  It is returned as
`open_seconds`, shown as "davom etmoqda", and left out of the sum.

**Only CHECK_IN and CHECK_OUT are passes.** `recognition_event` also holds
RE_SIGHTING, DEBOUNCED, DUPLICATE_VIEW and NO_DIRECTION — real sightings that
deliberately moved no state (the losing half of a cross-camera pair, a repeat
inside the cooldown, a track whose direction could not be read).  Counting them
as arrivals would tell HR somebody entered the building eleven times before
lunch.  They stay visible in the day timeline, labelled, because an operator
checking a disputed day needs to see everything the cameras had; they are not
counted.

**A voided event is not evidence.** An admin saying "that is not that person"
leaves the row in place on purpose — the record is the table — so every query
here filters `voided_at IS NULL`.  Forgetting it would let a known false accept
inflate somebody's hours, which is precisely the error voiding exists to undo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from sqlalchemy import func, or_, select

from app.config import settings
from app.db.models import (
    Camera, DailyAttendance, Employee, PresenceStatus, RecognitionEvent,
)

# The transitions that are an arrival or a departure. Everything else in
# `recognition_event.transition` is a sighting that moved no attendance state -
# see the module docstring.
CHECK_IN, CHECK_OUT = "CHECK_IN", "CHECK_OUT"
PASS_TRANSITIONS = (CHECK_IN, CHECK_OUT)

# A day whose presence never closed. Its `worked_seconds` is not a measurement,
# so it is excluded from hours and counted instead.
INCOMPLETE_STATUSES = ("NO_CHECKOUT", "NO_CHECKIN")


def _local(dt: datetime | None) -> datetime | None:
    return dt.astimezone(settings.tz) if dt else None


def _hours(seconds: int | None) -> float:
    return round((seconds or 0) / 3600.0, 2)


@dataclass
class PassVM:
    """One pass, with the face the camera kept and the time it happened."""
    id: int
    ts: datetime                  # local
    kind: str                     # IN | OUT | SEEN
    transition: str
    camera: str
    direction: str
    score: float
    snapshot: str | None          # already a prefixed media URL
    manual: bool
    voided: bool
    void_reason: str = ""

    @property
    def time(self) -> str:
        return self.ts.strftime("%H:%M:%S")


@dataclass
class DayVM:
    """One person, one business date."""
    date: date
    check_in: datetime | None
    check_out: datetime | None
    worked_seconds: int
    open_seconds: int
    status: str
    presence: str
    n_in: int
    n_out: int
    passes: list[PassVM] = field(default_factory=list)

    @property
    def worked_hours(self) -> float:
        return _hours(self.worked_seconds)

    @property
    def open_hours(self) -> float:
        return _hours(self.open_seconds)

    @property
    def is_open(self) -> bool:
        return self.presence == PresenceStatus.INSIDE.value

    @property
    def is_incomplete(self) -> bool:
        """The day cannot be costed: presence never closed.

        Deliberately NOT "worked_seconds == 0". Somebody who walked in and out
        within the same minute genuinely worked ~0 hours and their day IS
        complete; somebody whose check-out was never seen worked an unknown
        amount. Conflating them is the whole error this module exists to avoid.
        """
        return self.status in INCOMPLETE_STATUSES or self.is_open


@dataclass
class PersonTotals:
    """One person over the whole requested range."""
    employee_id: int
    name: str
    department: str
    external_id: str
    days_attended: int
    worked_seconds: int
    open_seconds: int
    n_in: int
    n_out: int
    incomplete_days: int

    @property
    def worked_hours(self) -> float:
        return _hours(self.worked_seconds)

    @property
    def open_hours(self) -> float:
        return _hours(self.open_seconds)

    @property
    def avg_hours(self) -> float:
        """Average over the days that CAN be costed.

        Dividing by `days_attended` would drag the average down by every
        unclosed day, reporting a 4-hour average for a team that works 8.
        """
        costed = self.days_attended - self.incomplete_days
        if costed <= 0:
            return 0.0
        return round(self.worked_seconds / 3600.0 / costed, 2)


def _open_seconds(row, now: datetime) -> int:
    """Time since the person was last seen entering, if they never left.

    Returned separately from `worked_seconds` and never added to it: see the
    module docstring on why a running total must not enter a monthly figure.
    """
    if row.presence != PresenceStatus.INSIDE or row.entered_at is None:
        return 0
    return max(0, int((now - row.entered_at).total_seconds()))


def _pass_counts(s, start: date, end: date, employee_ids=None) -> dict:
    """{(employee_id, business_date): [n_in, n_out]} over the range.

    One grouped query for the whole page rather than two per rendered row: the
    attendance list already materialises a month of rows, and a per-row count
    turns that into a few hundred round trips against a 300 MB database.
    """
    q = (
        select(RecognitionEvent.employee_id, RecognitionEvent.business_date,
               RecognitionEvent.transition, func.count(RecognitionEvent.id))
        .where(RecognitionEvent.business_date.between(start, end),
               RecognitionEvent.employee_id.isnot(None),
               RecognitionEvent.voided_at.is_(None),
               RecognitionEvent.transition.in_(PASS_TRANSITIONS))
        .group_by(RecognitionEvent.employee_id, RecognitionEvent.business_date,
                  RecognitionEvent.transition)
    )
    if employee_ids is not None:
        q = q.where(RecognitionEvent.employee_id.in_(list(employee_ids)))
    out: dict = {}
    for emp_id, bdate, transition, n in s.execute(q):
        slot = out.setdefault((emp_id, bdate), [0, 0])
        slot[0 if transition == CHECK_IN else 1] += n
    return out


def _employee_filter(statement, query: str | None, department: str | None):
    if query:
        statement = statement.where(or_(
            Employee.full_name.ilike(f"%{query}%"),
            Employee.external_id.ilike(f"%{query}%"),
        ))
    if department:
        statement = statement.where(Employee.department == department)
    return statement


def totals(s, start: date, end: date, *, query: str | None = None,
           department: str | None = None, now: datetime | None = None
           ) -> list[PersonTotals]:
    """One row per employee who has any attendance in the range.

    Absent people are not invented here. A person with no row in the range has
    no measured attendance, and rendering them as "0 hours" would state a fact
    the cameras never established - they may have been on leave, on a site
    visit, or simply enrolled last week. HR filters the roster; this reports
    what was observed.
    """
    now = now or datetime.now(settings.tz)
    statement = _employee_filter(
        select(DailyAttendance, Employee)
        .join(Employee, Employee.id == DailyAttendance.employee_id)
        .where(DailyAttendance.business_date.between(start, end)),
        query, department,
    )
    rows = s.execute(statement).all()
    counts = _pass_counts(s, start, end,
                          {d.employee_id for d, _ in rows} or None)

    acc: dict[int, PersonTotals] = {}
    for d, emp in rows:
        t = acc.get(emp.id)
        if t is None:
            t = acc[emp.id] = PersonTotals(
                employee_id=emp.id, name=emp.full_name,
                department=emp.department or "", external_id=emp.external_id or "",
                days_attended=0, worked_seconds=0, open_seconds=0,
                n_in=0, n_out=0, incomplete_days=0,
            )
        opened = _open_seconds(d, now)
        incomplete = d.status in INCOMPLETE_STATUSES or d.presence == PresenceStatus.INSIDE
        t.days_attended += 1
        t.worked_seconds += int(d.worked_seconds or 0)
        t.open_seconds += opened
        t.incomplete_days += 1 if incomplete else 0
        n_in, n_out = counts.get((emp.id, d.business_date), (0, 0))
        t.n_in += n_in
        t.n_out += n_out

    return sorted(acc.values(), key=lambda t: (-t.worked_seconds, t.name))


def day_rows(s, employee_id: int, start: date, end: date, *,
             now: datetime | None = None) -> list[DayVM]:
    """Every attended day for one person, newest first, without the passes."""
    now = now or datetime.now(settings.tz)
    rows = s.execute(
        select(DailyAttendance)
        .where(DailyAttendance.employee_id == employee_id,
               DailyAttendance.business_date.between(start, end))
        .order_by(DailyAttendance.business_date.desc())
    ).scalars().all()
    counts = _pass_counts(s, start, end, [employee_id])
    out = []
    for d in rows:
        n_in, n_out = counts.get((employee_id, d.business_date), (0, 0))
        out.append(DayVM(
            date=d.business_date,
            check_in=_local(d.check_in_time), check_out=_local(d.check_out_time),
            worked_seconds=int(d.worked_seconds or 0),
            open_seconds=_open_seconds(d, now),
            status=d.status or "PRESENT",
            presence=d.presence.value if hasattr(d.presence, "value") else str(d.presence),
            n_in=n_in, n_out=n_out,
        ))
    return out


def passes(s, employee_id: int, start: date, end: date) -> dict:
    """{business_date: [PassVM, ...]} — the evidence strip behind each day.

    Voided rows ARE returned, flagged, because an operator looking at a
    contested day must be able to see that a correction landed; they are simply
    not counted anywhere. Non-transition sightings come back labelled "SEEN"
    for the same reason.

    The snapshot is turned into a prefixed media URL here rather than in the
    template: under /faceid a bare `/media/...` resolves against the domain
    root, which on the shared host belongs to another project.
    """
    from app.web.django_compat import media_path

    rows = s.execute(
        select(RecognitionEvent, Camera.name)
        .outerjoin(Camera, Camera.id == RecognitionEvent.camera_id)
        .where(RecognitionEvent.employee_id == employee_id,
               RecognitionEvent.business_date.between(start, end))
        .order_by(RecognitionEvent.business_date.desc(), RecognitionEvent.ts)
    ).all()

    out: dict = {}
    for e, cam in rows:
        kind = {CHECK_IN: "IN", CHECK_OUT: "OUT"}.get(e.transition or "", "SEEN")
        out.setdefault(e.business_date, []).append(PassVM(
            id=e.id, ts=_local(e.ts), kind=kind, transition=e.transition or "",
            camera=cam or "—", direction=e.direction or "UNKNOWN",
            score=float(e.score or 0.0),
            snapshot=media_path(e.snapshot) if e.snapshot else None,
            manual=(e.source or "live") == "manual",
            voided=e.voided_at is not None,
            void_reason=e.void_reason or "",
        ))
    return out


def day_rows_with_passes(s, employee_id: int, start: date, end: date, *,
                         now: datetime | None = None) -> list[DayVM]:
    """`day_rows`, each carrying its evidence strip."""
    days = day_rows(s, employee_id, start, end, now=now)
    by_date = passes(s, employee_id, start, end)
    for d in days:
        d.passes = by_date.get(d.date, [])
    return days


@dataclass
class RangeSummary:
    """The totals row: what the whole filtered range came to."""
    people: int
    days_attended: int
    worked_seconds: int
    n_in: int
    n_out: int
    incomplete_days: int

    @property
    def worked_hours(self) -> float:
        return _hours(self.worked_seconds)

    @classmethod
    def of(cls, rows: list[PersonTotals]) -> "RangeSummary":
        return cls(
            people=len(rows),
            days_attended=sum(r.days_attended for r in rows),
            worked_seconds=sum(r.worked_seconds for r in rows),
            n_in=sum(r.n_in for r in rows),
            n_out=sum(r.n_out for r in rows),
            incomplete_days=sum(r.incomplete_days for r in rows),
        )
