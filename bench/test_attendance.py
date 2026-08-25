"""Drive the state machine through a realistic day. No cameras involved."""
import sys
sys.path.insert(0, "/home/inomjon/projectAI/face_rec/face_recognition_airi")
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, delete
from app.config import settings
from app.db.models import CameraRole, DailyAttendance, Employee, RecognitionEvent
from app.db.session import init_db, session_scope
from app.services.attendance import AttendanceService, business_date

init_db()
svc = AttendanceService()
TZ = settings.tz

def L(h, m):  # local Tashkent time -> aware UTC
    return datetime(2026, 8, 19, h, m, tzinfo=TZ).astimezone(timezone.utc)

with session_scope() as s:
    emp = s.execute(select(Employee)).scalars().first()
    eid = emp.id
    bd = business_date(L(9, 0))
    s.execute(delete(RecognitionEvent).where(RecognitionEvent.employee_id == eid))
    s.execute(delete(DailyAttendance).where(DailyAttendance.employee_id == eid,
                                            DailyAttendance.business_date == bd))

script = [
    (L(9, 2),   CameraRole.IN,  1, "arrives"),
    (L(9, 3),   CameraRole.IN,  1, "still in view -> debounced (<90s)"),
    (L(9, 6),   CameraRole.IN,  1, "lingers -> re-sighting, no change"),
    (L(13, 5),  CameraRole.OUT, 2, "leaves for lunch"),
    (L(13, 50), CameraRole.IN,  1, "back from lunch"),
    (L(15, 0),  CameraRole.OUT, 2, "steps out briefly"),
    (L(15, 20), CameraRole.IN,  1, "returns"),
    (L(18, 10), CameraRole.OUT, 2, "goes home"),
    (L(18, 12), CameraRole.OUT, 2, "still near exit -> debounced"),
]

print(f"employee: {emp.full_name}  (business_date {bd})\n")
print(f"{'local':>7}  {'role':<4} {'transition':<12} {'presence':<9} {'in':>6} {'out':>6} {'worked':>8}")
print("-" * 64)
with session_scope() as s:
    for ts, role, cam, note in script:
        d = svc.record(s, employee_id=eid, camera_id=cam, role=role, ts=ts, score=0.92)
        s.flush()
        row = s.execute(select(DailyAttendance).where(
            DailyAttendance.employee_id == eid, DailyAttendance.business_date == bd)).scalar_one()
        ci = row.check_in_time.astimezone(TZ).strftime("%H:%M") if row.check_in_time else "-"
        co = row.check_out_time.astimezone(TZ).strftime("%H:%M") if row.check_out_time else "-"
        hrs = f"{(row.worked_seconds or 0)/3600:.2f}h"
        print(f"{ts.astimezone(TZ).strftime('%H:%M'):>7}  {role.value:<4} {d.transition:<12} "
              f"{row.presence.value:<9} {ci:>6} {co:>6} {hrs:>8}   {note}")

with session_scope() as s:
    row = s.execute(select(DailyAttendance).where(
        DailyAttendance.employee_id == eid, DailyAttendance.business_date == bd)).scalar_one()
    n = s.execute(select(RecognitionEvent).where(
        RecognitionEvent.employee_id == eid, RecognitionEvent.business_date == bd)).scalars().all()
    print("-" * 64)
    print(f"\nfinal: in={row.check_in_time.astimezone(TZ):%H:%M}  "
          f"out={row.check_out_time.astimezone(TZ):%H:%M}  "
          f"worked={(row.worked_seconds or 0)/3600:.2f}h  presence={row.presence.value}  "
          f"status={row.status}")
    print(f"events logged: {len(n)} (2 debounced calls wrote nothing)")
    gross = (row.check_out_time - row.check_in_time).total_seconds()/3600
    print(f"\ngross span {gross:.2f}h  minus {gross-(row.worked_seconds/3600):.2f}h outside = "
          f"{row.worked_seconds/3600:.2f}h worked")
