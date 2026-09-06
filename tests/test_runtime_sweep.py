"""A process started after the 04:10 sweep must still close out what it missed.

The end-of-day timer in app/api/main.py only fires while the process is up. A
service down across it - or started later in the day - never revisited those
days, so their rows kept claiming people were inside the building until the
next 04:10 the process happened to see.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select

from app.db.models import CameraRole, DailyAttendance, Employee, RecognitionEvent
from app.db.session import session_scope
from app.services.attendance import AttendanceService


def test_the_start_up_sweep_flags_a_day_left_inside():
    from app.runtime import Runtime

    with session_scope() as s:
        e = Employee(full_name="Left Inside", is_active=True)
        s.add(e); s.flush()
        emp = e.id
    try:
        old = datetime.now(timezone.utc) - timedelta(days=3)
        with session_scope() as s:
            AttendanceService(cooldown_s=0).record(
                s, employee_id=emp, camera_id=1, role=CameraRole.IN, ts=old,
                score=0.5, direction="ENTER")
        with session_scope() as s:
            row = s.execute(select(DailyAttendance).where(
                DailyAttendance.employee_id == emp)).scalar_one()
            assert row.status == "PRESENT", "three days on, nobody has flagged it"

        assert Runtime._sweep_open_days() >= 1

        with session_scope() as s:
            row = s.execute(select(DailyAttendance).where(
                DailyAttendance.employee_id == emp)).scalar_one()
            assert row.status == "NO_CHECKOUT", "flagged, never closed with an invented time"
            assert row.check_out_time is None
    finally:
        with session_scope() as s:
            for model in (RecognitionEvent, DailyAttendance):
                s.execute(delete(model).where(model.employee_id == emp))
            s.execute(delete(Employee).where(Employee.id == emp))
