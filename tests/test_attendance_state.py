"""The attendance state machine, and the three bugs that corrupted it.

Every assertion here corresponds to damage found in the live database on
2026-08-26 (54 people, 179 recognised passes, 73 daily rows). None of it was
caught before, because `app/services/attendance.py` had no test at all - it was
exercised only by `bench/test_attendance.py`, an ad-hoc script that writes to
the REAL database and that pytest never collects.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.db.models import CameraRole, DailyAttendance, PresenceStatus
from app.db.session import session_scope
from app.services.attendance import AttendanceService, business_date

ENTER, EXIT = "ENTER", "EXIT"
CAM_IN, CAM_OUT = 1, 2


@pytest.fixture(autouse=True)
def remove_test_employees():
    """One in-memory database is shared by the whole suite, so employees
    invented here would show up in another test's roster counts.

    Dependent rows are deleted EXPLICITLY. The test engine does not enable
    `PRAGMA foreign_keys`, so ON DELETE CASCADE never fires, and SQLite reuses
    the freed employee id - which handed the next test a previous test's
    daily_attendance row under its own "new" employee.
    """
    from app.db.models import DailyAttendance, Employee, RecognitionEvent
    from app.db.session import session_scope
    from sqlalchemy import delete, select
    with session_scope() as s:
        before = set(s.execute(select(Employee.id)).scalars())
    yield
    with session_scope() as s:
        made = [i for i in s.execute(select(Employee.id)).scalars()
                if i not in before]
        if made:
            s.execute(delete(RecognitionEvent).where(
                RecognitionEvent.employee_id.in_(made)))
            s.execute(delete(DailyAttendance).where(
                DailyAttendance.employee_id.in_(made)))
            s.execute(delete(Employee).where(Employee.id.in_(made)))


def _emp(s, name="Test Person"):
    from app.db.models import Employee
    e = Employee(full_name=name, is_active=True)
    s.add(e)
    s.flush()
    return e.id


def _at(h, m, sec=0):
    """A UTC instant on a fixed day."""
    return datetime(2026, 8, 26, h, m, sec, tzinfo=timezone.utc)


def _daily(s, emp, ts) -> DailyAttendance:
    """The ORM row, refreshed - the same object production reads."""
    from sqlalchemy import select
    s.flush()
    row = s.execute(
        select(DailyAttendance).where(
            DailyAttendance.employee_id == emp,
            DailyAttendance.business_date == business_date(ts))
    ).scalar_one()
    s.refresh(row)
    return row


# --- B1: NO_CHECKIN was a one-way latch ------------------------------------

def test_no_checkin_clears_when_the_check_in_finally_arrives():
    """The exact shape of employee 17's row: seen leaving before we ever saw
    them arrive, then a genuine check-in hours later.

    The old rule guarded the clear on the flag's own value
    (`elif daily.status != "NO_CHECKIN"`), so once raised it was permanent.
    21 of the 30 flagged rows in the live database have a check-in time.
    """
    svc = AttendanceService(cooldown_s=0)
    with session_scope() as s:
        emp = _emp(s, "Latch Clears")
        svc.record(s, employee_id=emp, camera_id=CAM_OUT, role=CameraRole.OUT,
                   ts=_at(6, 44), score=0.3, direction=EXIT)
        assert _daily(s, emp, _at(6, 44)).status == "NO_CHECKIN"

        svc.record(s, employee_id=emp, camera_id=CAM_IN, role=CameraRole.IN,
                   ts=_at(10, 59), score=0.3, direction=ENTER)
        row = _daily(s, emp, _at(10, 59))
        assert row.check_in_time is not None
        assert row.status == "PRESENT", (
            "a row with a check-in time may not stay flagged NO_CHECKIN")


def test_no_checkin_stands_while_there_is_still_no_check_in():
    svc = AttendanceService(cooldown_s=0)
    with session_scope() as s:
        emp = _emp(s, "Latch Stands")
        svc.record(s, employee_id=emp, camera_id=CAM_OUT, role=CameraRole.OUT,
                   ts=_at(6, 0), score=0.3, direction=EXIT)
        svc.record(s, employee_id=emp, camera_id=CAM_OUT, role=CameraRole.OUT,
                   ts=_at(9, 0), score=0.3, direction=EXIT)
        assert _daily(s, emp, _at(9, 0)).status == "NO_CHECKIN"


# --- B3: the end-of-day sweep ----------------------------------------------

def test_sweep_reaches_every_unfinished_day_not_just_yesterday():
    """It swept one date, so any day the service was down was never revisited.
    21 rows sat at INSIDE, the oldest five days stale."""
    svc = AttendanceService(cooldown_s=0)
    with session_scope() as s:
        emp = _emp(s, "Stale Days")
        for day in (20, 22, 24):
            ts = datetime(2026, 8, day, 9, 0, tzinfo=timezone.utc)
            svc.record(s, employee_id=emp, camera_id=CAM_IN, role=CameraRole.IN,
                       ts=ts, score=0.3, direction=ENTER)
        s.flush()
        # Scoped to THIS employee: the suite shares one in-memory database, so
        # a global count would also pick up rows other tests left open.
        from sqlalchemy import select
        svc.close_open_intervals(s, business_date(_at(9, 0)))
        s.flush()
        mine = s.execute(select(DailyAttendance).where(
            DailyAttendance.employee_id == emp)).scalars().all()
        for r in mine:
            s.refresh(r)
        assert len(mine) == 3
        assert all(r.status == "NO_CHECKOUT" for r in mine), (
            "every unfinished day must be swept, not only yesterday; "
            f"got {[(str(r.business_date), r.status) for r in mine]}")


def test_the_sweep_does_not_destroy_a_no_checkin_flag():
    """A row can be both: never seen arriving, and still marked inside. The
    sweep overwrote the more specific fact with the vaguer one."""
    svc = AttendanceService(cooldown_s=0)
    with session_scope() as s:
        emp = _emp(s, "Both Flags")
        svc.record(s, employee_id=emp, camera_id=CAM_OUT, role=CameraRole.OUT,
                   ts=_at(6, 0), score=0.3, direction=EXIT)
        row = _daily(s, emp, _at(6, 0))
        row.presence = PresenceStatus.INSIDE
        s.flush()
        svc.close_open_intervals(s, business_date(_at(6, 0)) + timedelta(days=1))
        s.refresh(row)
        assert row.status == "NO_CHECKIN", (
            "NO_CHECKIN is the more specific fact and must survive the sweep")


# --- the transitions that were already right, pinned so they stay right ----

def test_a_normal_day_is_one_in_one_out():
    svc = AttendanceService(cooldown_s=0)
    with session_scope() as s:
        emp = _emp(s, "Normal Day")
        svc.record(s, employee_id=emp, camera_id=CAM_IN, role=CameraRole.IN,
                   ts=_at(8, 0), score=0.4, direction=ENTER)
        svc.record(s, employee_id=emp, camera_id=CAM_OUT, role=CameraRole.OUT,
                   ts=_at(17, 0), score=0.4, direction=EXIT)
        row = _daily(s, emp, _at(17, 0))
        assert row.status == "PRESENT"
        assert row.worked_seconds == 9 * 3600
        assert row.check_out_time > row.check_in_time


def test_direction_overrides_the_camera_role():
    """Both cameras see the whole corridor, so the role alone is not evidence."""
    svc = AttendanceService(cooldown_s=0)
    with session_scope() as s:
        emp = _emp(s, "Direction Wins")
        # On the ENTRANCE camera, but walking out.
        svc.record(s, employee_id=emp, camera_id=CAM_IN, role=CameraRole.IN,
                   ts=_at(8, 0), score=0.4, direction=EXIT)
        assert _daily(s, emp, _at(8, 0)).check_in_time is None


def test_an_undetermined_direction_moves_nothing():
    svc = AttendanceService(cooldown_s=0)
    with session_scope() as s:
        emp = _emp(s, "No Direction")
        d = svc.record(s, employee_id=emp, camera_id=CAM_IN, role=CameraRole.IN,
                       ts=_at(8, 0), score=0.4, direction="UNKNOWN")
        assert d.transition == "NO_DIRECTION"
        assert _daily(s, emp, _at(8, 0)).check_in_time is None


# --- apply_state=False: the losing half of a cross-camera pair -------------

def test_a_duplicate_view_is_recorded_but_moves_no_state():
    svc = AttendanceService(cooldown_s=0)
    with session_scope() as s:
        emp = _emp(s, "Duplicate View")
        svc.record(s, employee_id=emp, camera_id=CAM_IN, role=CameraRole.IN,
                   ts=_at(8, 0), score=0.4, direction=ENTER)
        b = _daily(s, emp, _at(8, 0))
        before = (b.presence, b.check_out_time, b.worked_seconds, b.event_count)
        d = svc.record(s, employee_id=emp, camera_id=CAM_OUT, role=CameraRole.OUT,
                       ts=_at(8, 0, 4), score=0.2, direction=EXIT,
                       apply_state=False)
        a = _daily(s, emp, _at(8, 0))
        after = (a.presence, a.check_out_time, a.worked_seconds, a.event_count)
        assert d.transition == "DUPLICATE_VIEW"
        assert after[:3] == before[:3], (
            "the losing half of a cross-camera pair must not move state")
        assert after[3] == before[3] + 1, (
            "the sighting still happened and should still be counted")


# --- the debounce must not swallow a real transition ----------------------

def test_a_return_is_not_debounced_by_the_departure_before_it():
    """`20260831_114309_Entrance.mp4` holds a clean entrance: recognised at
    0.689 with 45 agreeing frames, ENTER confirmed by tripwire AND depth. Its
    CHECK_IN was discarded because the same camera had recorded that person
    leaving shortly before, and the debounce only looked at person+camera+time.

    The recognition was never the problem. The event was thrown away after it.
    """
    svc = AttendanceService(cooldown_s=90)
    with session_scope() as s:
        emp = _emp(s, "Steps Out And Back")
        svc.record(s, employee_id=emp, camera_id=CAM_IN, role=CameraRole.IN,
                   ts=_at(11, 40, 43), score=0.484, direction=EXIT)
        d = svc.record(s, employee_id=emp, camera_id=CAM_IN, role=CameraRole.IN,
                       ts=_at(11, 41, 9), score=0.689, direction=ENTER)
        assert d.transition == "CHECK_IN", (
            f"a genuine entrance 26s after a departure must not be debounced; "
            f"got {d.transition}")
        assert _daily(s, emp, _at(11, 41, 9)).check_in_time is not None


def test_the_same_direction_repeated_is_still_debounced():
    """The cooldown still has a job: one person lingering in view produces
    several completed tracks, and they are re-sightings, not transitions."""
    svc = AttendanceService(cooldown_s=90)
    with session_scope() as s:
        emp = _emp(s, "Lingering")
        svc.record(s, employee_id=emp, camera_id=CAM_IN, role=CameraRole.IN,
                   ts=_at(9, 0, 0), score=0.5, direction=ENTER)
        d = svc.record(s, employee_id=emp, camera_id=CAM_IN, role=CameraRole.IN,
                       ts=_at(9, 0, 30), score=0.5, direction=ENTER)
        assert d.transition == "DEBOUNCED"


def test_the_cooldown_still_expires():
    svc = AttendanceService(cooldown_s=90)
    with session_scope() as s:
        emp = _emp(s, "After Cooldown")
        svc.record(s, employee_id=emp, camera_id=CAM_IN, role=CameraRole.IN,
                   ts=_at(9, 0, 0), score=0.5, direction=ENTER)
        d = svc.record(s, employee_id=emp, camera_id=CAM_IN, role=CameraRole.IN,
                       ts=_at(9, 2, 0), score=0.5, direction=ENTER)
        assert d.transition != "DEBOUNCED"


# --- a later EXIT corrects an earlier check-out time ---------------------

def test_an_exit_while_already_out_moves_the_check_out_time_forward():
    """Baxtiyor, 2026-09-03: a U-turn at the door was recorded as a check-out
    at 11:01, and the real departure at 11:50 was then only a re-sighting, so
    the day showed him leaving 49 minutes early. Somebody seen LEAVING was
    inside a moment before, so the later time is the true one. No state
    moves and nothing is worked for the unobserved interval."""
    svc = AttendanceService(cooldown_s=90)
    with session_scope() as s:
        emp = _emp(s, "Later Exit")
        svc.record(s, employee_id=emp, camera_id=CAM_IN, role=CameraRole.IN,
                   ts=_at(5, 8), score=0.5, direction=ENTER)
        d1 = svc.record(s, employee_id=emp, camera_id=CAM_OUT, role=CameraRole.OUT,
                        ts=_at(6, 1), score=0.5, direction=EXIT)
        assert d1.transition == "CHECK_OUT"
        d2 = svc.record(s, employee_id=emp, camera_id=CAM_OUT, role=CameraRole.OUT,
                        ts=_at(6, 50), score=0.5, direction=EXIT, snapshot="s2.jpg")
        assert d2.transition == "RE_SIGHTING"
        row = _daily(s, emp, _at(6, 50))
        assert row.presence == PresenceStatus.OUTSIDE
        assert row.check_out_time == _at(6, 50)
        assert row.check_out_snapshot == "s2.jpg"
        assert row.worked_seconds == 53 * 60          # only the observed interval
        assert row.status == "PRESENT"


def test_an_exit_before_the_recorded_check_out_does_not_move_it_back():
    svc = AttendanceService(cooldown_s=90)
    with session_scope() as s:
        emp = _emp(s, "Out Of Order")
        svc.record(s, employee_id=emp, camera_id=CAM_IN, role=CameraRole.IN,
                   ts=_at(5, 0), score=0.5, direction=ENTER)
        svc.record(s, employee_id=emp, camera_id=CAM_OUT, role=CameraRole.OUT,
                   ts=_at(7, 0), score=0.5, direction=EXIT)
        svc.record(s, employee_id=emp, camera_id=CAM_OUT, role=CameraRole.OUT,
                   ts=_at(6, 0), score=0.5, direction=EXIT)      # arrives late, earlier ts
        row = _daily(s, emp, _at(7, 0))
        assert row.check_out_time == _at(7, 0)


def test_a_debounced_pass_is_written_and_moves_nothing():
    """Anvarxodja, 2026-09-04 09:55: in, back toward the door, and in again
    forty seconds later. The second entry was inside the cooldown and was
    dropped without a trace; it is now a row that changes no state and that
    a rebuild ignores."""
    from sqlalchemy import select
    from app.db.models import RecognitionEvent
    svc = AttendanceService(cooldown_s=90)
    with session_scope() as s:
        emp = _emp(s, "Debounced Row")
        svc.record(s, employee_id=emp, camera_id=CAM_IN, role=CameraRole.IN,
                   ts=_at(5, 0, 0), score=0.5, direction=ENTER)
        d = svc.record(s, employee_id=emp, camera_id=CAM_IN, role=CameraRole.IN,
                       ts=_at(5, 0, 40), score=0.4, direction=ENTER, snapshot="again.jpg")
        assert d.transition == "DEBOUNCED"
        rows = s.execute(select(RecognitionEvent).where(
            RecognitionEvent.employee_id == emp).order_by(RecognitionEvent.ts)).scalars().all()
        assert [r.transition for r in rows] == ["CHECK_IN", "DEBOUNCED"]
        assert rows[1].snapshot == "again.jpg"
        row = _daily(s, emp, _at(5, 0))
        assert row.presence == PresenceStatus.INSIDE
        assert row.check_in_time == _at(5, 0, 0)
        # ...and a rebuild treats it as evidence only.
        n = svc.rebuild(s, emp, business_date(_at(5, 0)))
        assert n == 2
        row = _daily(s, emp, _at(5, 0))
        assert row.check_in_time == _at(5, 0, 0) and row.presence == PresenceStatus.INSIDE
