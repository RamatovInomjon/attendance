"""The HR timesheet's aggregation rules.

Each test here pins a rule that the PRODUCTION database already breaks, so
these are regressions against real data rather than hypotheticals:

* 142 of the last 492 day-rows are NO_CHECKOUT and carry `worked_seconds = 0`;
* 66 of those have a check-in recorded, so they are demonstrably worked days;
* 168 rows sit at `presence=INSIDE`;
* `recognition_event` holds four transitions that are sightings, not passes.

A timesheet that sums the column and counts every row would report a third of
the month as zero hours and tell HR people entered the building a dozen times
before lunch. The point of each test is that the number HR reads is either
correct or explicitly marked unknown - never quietly wrong.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import delete, select

from app.config import settings
from app.db.models import (
    CameraRole, DailyAttendance, Employee, PresenceStatus, RecognitionEvent,
)
from app.db.session import session_scope
from app.services import timesheet

DAY = date(2026, 5, 11)
OTHER = date(2026, 5, 12)


def _utc(d: date, hh: int, mm: int = 0) -> datetime:
    """A local wall-clock time on `d`, stored as the aware UTC the schema uses."""
    local = datetime(d.year, d.month, d.day, hh, mm, tzinfo=settings.tz)
    return local.astimezone(timezone.utc)


@pytest.fixture
def person():
    with session_scope() as s:
        e = Employee(full_name="Timesheet Tester", department="HR",
                     external_id="TS-1")
        s.add(e)
        s.flush()
        eid = e.id
    yield eid
    with session_scope() as s:
        s.execute(delete(RecognitionEvent).where(
            RecognitionEvent.employee_id == eid))
        s.execute(delete(DailyAttendance).where(
            DailyAttendance.employee_id == eid))
        s.execute(delete(Employee).where(Employee.id == eid))


def _daily(s, eid, d, **kw):
    row = DailyAttendance(employee_id=eid, business_date=d, **kw)
    s.add(row)
    return row


def _event(s, eid, ts, transition, **kw):
    row = RecognitionEvent(
        employee_id=eid, camera_id=kw.pop("camera_id", 1),
        role=kw.pop("role", CameraRole.IN), ts=ts,
        business_date=kw.pop("business_date", timesheet_date(ts)),
        transition=transition, snapshot=kw.pop("snapshot", "snapshots/x.jpg"),
        **kw)
    s.add(row)
    return row


def timesheet_date(ts: datetime) -> date:
    from app.services.attendance import business_date
    return business_date(ts)


class TestHoursAreNotFakedByAnUnclosedDay:
    def test_a_day_that_never_closed_is_incomplete_not_zero_hours(self, person):
        """NO_CHECKOUT is 29% of the live table. It must not read as 0 worked."""
        with session_scope() as s:
            _daily(s, person, DAY, check_in_time=_utc(DAY, 9),
                   worked_seconds=0, status="NO_CHECKOUT",
                   presence=PresenceStatus.OUTSIDE)
        with session_scope() as s:
            days = timesheet.day_rows(s, person, DAY, DAY)
            assert len(days) == 1
            assert days[0].is_incomplete is True
            (t,) = timesheet.totals(s, DAY, DAY)
            assert t.incomplete_days == 1
            assert t.days_attended == 1

    def test_a_short_but_closed_day_is_complete(self, person):
        """Zero hours honestly measured is NOT the same as hours unknown."""
        with session_scope() as s:
            _daily(s, person, DAY, check_in_time=_utc(DAY, 9),
                   check_out_time=_utc(DAY, 9), worked_seconds=0,
                   status="PRESENT", presence=PresenceStatus.OUTSIDE)
        with session_scope() as s:
            days = timesheet.day_rows(s, person, DAY, DAY)
            assert days[0].is_incomplete is False
            (t,) = timesheet.totals(s, DAY, DAY)
            assert t.incomplete_days == 0

    def test_average_divides_by_costable_days_only(self, person):
        """One 8-hour day and one unclosed day averages 8, not 4."""
        with session_scope() as s:
            _daily(s, person, DAY, check_in_time=_utc(DAY, 9),
                   check_out_time=_utc(DAY, 17), worked_seconds=8 * 3600,
                   status="PRESENT", presence=PresenceStatus.OUTSIDE)
            _daily(s, person, OTHER, check_in_time=_utc(OTHER, 9),
                   worked_seconds=0, status="NO_CHECKOUT",
                   presence=PresenceStatus.OUTSIDE)
        with session_scope() as s:
            (t,) = timesheet.totals(s, DAY, OTHER)
            assert t.days_attended == 2
            assert t.incomplete_days == 1
            assert t.worked_hours == 8.0
            assert t.avg_hours == 8.0


class TestAnOpenDayIsReportedSeparately:
    def test_open_interval_is_not_added_to_worked_hours(self, person):
        """168 live rows are INSIDE. A running total must not enter the sum,
        or the page disagrees with the CSV exported a minute earlier."""
        now = datetime.now(timezone.utc)
        entered = now - timedelta(hours=3)
        with session_scope() as s:
            _daily(s, person, DAY, check_in_time=entered, entered_at=entered,
                   worked_seconds=0, status="PRESENT",
                   presence=PresenceStatus.INSIDE)
        with session_scope() as s:
            (t,) = timesheet.totals(s, DAY, DAY, now=now)
            assert t.worked_seconds == 0
            assert t.open_seconds == pytest.approx(3 * 3600, abs=5)
            assert t.incomplete_days == 1
            days = timesheet.day_rows(s, person, DAY, DAY, now=now)
            assert days[0].is_open is True
            assert days[0].open_hours == pytest.approx(3.0, abs=0.01)


class TestOnlyRealPassesAreCounted:
    def test_resightings_and_duplicates_do_not_inflate_the_count(self, person):
        """Four transitions move no attendance state. Counting them would say
        somebody entered the building eleven times before lunch."""
        with session_scope() as s:
            _daily(s, person, DAY, check_in_time=_utc(DAY, 9),
                   check_out_time=_utc(DAY, 18), worked_seconds=9 * 3600,
                   status="PRESENT", presence=PresenceStatus.OUTSIDE)
            _event(s, person, _utc(DAY, 9), "CHECK_IN", business_date=DAY)
            _event(s, person, _utc(DAY, 12), "CHECK_OUT", business_date=DAY)
            _event(s, person, _utc(DAY, 13), "CHECK_IN", business_date=DAY)
            _event(s, person, _utc(DAY, 18), "CHECK_OUT", business_date=DAY)
            for noise in ("RE_SIGHTING", "DEBOUNCED", "DUPLICATE_VIEW",
                          "NO_DIRECTION"):
                _event(s, person, _utc(DAY, 14), noise, business_date=DAY)
        with session_scope() as s:
            (t,) = timesheet.totals(s, DAY, DAY)
            assert (t.n_in, t.n_out) == (2, 2)
            days = timesheet.day_rows(s, person, DAY, DAY)
            assert (days[0].n_in, days[0].n_out) == (2, 2)

    def test_non_transitions_are_still_visible_as_evidence(self, person):
        """Counted: no. Shown: yes - a disputed day needs everything seen."""
        with session_scope() as s:
            _daily(s, person, DAY, worked_seconds=0, status="PRESENT",
                   presence=PresenceStatus.OUTSIDE)
            _event(s, person, _utc(DAY, 9), "CHECK_IN", business_date=DAY)
            _event(s, person, _utc(DAY, 14), "RE_SIGHTING", business_date=DAY)
        with session_scope() as s:
            strip = timesheet.passes(s, person, DAY, DAY)[DAY]
            assert [p.kind for p in strip] == ["IN", "SEEN"]
            assert all(p.snapshot for p in strip)


class TestAVoidedEventIsNotEvidence:
    def test_voided_passes_are_not_counted_but_stay_visible(self, person):
        """Voiding exists to undo a false accept; counting one would re-apply
        the very error the admin corrected."""
        with session_scope() as s:
            _daily(s, person, DAY, worked_seconds=0, status="PRESENT",
                   presence=PresenceStatus.OUTSIDE)
            _event(s, person, _utc(DAY, 9), "CHECK_IN", business_date=DAY)
            _event(s, person, _utc(DAY, 10), "CHECK_IN", business_date=DAY,
                   voided_at=_utc(DAY, 11), voided_by="admin",
                   void_reason="not this person")
        with session_scope() as s:
            (t,) = timesheet.totals(s, DAY, DAY)
            assert t.n_in == 1
            strip = timesheet.passes(s, person, DAY, DAY)[DAY]
            assert len(strip) == 2
            assert [p.voided for p in strip] == [False, True]


class TestEveryPassCarriesItsTimeAndItsImage:
    def test_passes_expose_local_time_and_a_prefixed_media_url(self, person):
        """This is HR's third ask, and the prefix is the bug that has bitten
        every media URL in this project before."""
        with session_scope() as s:
            _daily(s, person, DAY, worked_seconds=0, status="PRESENT",
                   presence=PresenceStatus.OUTSIDE)
            _event(s, person, _utc(DAY, 9, 5), "CHECK_IN", business_date=DAY,
                   snapshot="snapshots/a.jpg")
        with session_scope() as s:
            (p,) = timesheet.passes(s, person, DAY, DAY)[DAY]
            assert p.time == "09:05:00"
            assert p.kind == "IN"
            assert p.snapshot.endswith("/media/snapshots/a.jpg")
            prefix = settings.url_prefix.rstrip("/")
            if prefix:
                assert p.snapshot.startswith(prefix)

    def test_days_with_passes_attaches_the_strip_to_its_day(self, person):
        with session_scope() as s:
            _daily(s, person, DAY, worked_seconds=3600, status="PRESENT",
                   presence=PresenceStatus.OUTSIDE)
            _daily(s, person, OTHER, worked_seconds=3600, status="PRESENT",
                   presence=PresenceStatus.OUTSIDE)
            _event(s, person, _utc(DAY, 9), "CHECK_IN", business_date=DAY)
            _event(s, person, _utc(OTHER, 9), "CHECK_IN", business_date=OTHER)
            _event(s, person, _utc(OTHER, 17), "CHECK_OUT", business_date=OTHER)
        with session_scope() as s:
            days = timesheet.day_rows_with_passes(s, person, DAY, OTHER)
            assert [d.date for d in days] == [OTHER, DAY]      # newest first
            assert [len(d.passes) for d in days] == [2, 1]


class TestTheRangeSummary:
    def test_summary_adds_up_what_the_rows_hold(self, person):
        with session_scope() as s:
            _daily(s, person, DAY, worked_seconds=4 * 3600, status="PRESENT",
                   presence=PresenceStatus.OUTSIDE)
            _daily(s, person, OTHER, worked_seconds=0, status="NO_CHECKOUT",
                   presence=PresenceStatus.OUTSIDE)
            _event(s, person, _utc(DAY, 9), "CHECK_IN", business_date=DAY)
        with session_scope() as s:
            rows = timesheet.totals(s, DAY, OTHER)
            summary = timesheet.RangeSummary.of(rows)
            assert summary.people == 1
            assert summary.days_attended == 2
            assert summary.worked_hours == 4.0
            assert summary.incomplete_days == 1
            assert summary.n_in == 1


class TestFilters:
    def test_department_and_query_narrow_the_roster(self, person):
        with session_scope() as s:
            _daily(s, person, DAY, worked_seconds=3600, status="PRESENT",
                   presence=PresenceStatus.OUTSIDE)
        with session_scope() as s:
            assert len(timesheet.totals(s, DAY, DAY, department="HR")) == 1
            assert timesheet.totals(s, DAY, DAY, department="Nope") == []
            assert len(timesheet.totals(s, DAY, DAY, query="Timesheet")) == 1
            assert timesheet.totals(s, DAY, DAY, query="zzzz") == []

    def test_a_person_with_no_row_in_the_range_is_not_invented(self, person):
        """Absence is not 0 hours: the cameras never established anything."""
        with session_scope() as s:
            _daily(s, person, DAY, worked_seconds=3600, status="PRESENT",
                   presence=PresenceStatus.OUTSIDE)
        with session_scope() as s:
            assert timesheet.totals(s, OTHER, OTHER) == []
