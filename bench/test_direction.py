"""The scenario the user raised: a person LEAVING walks through the ENTRANCE
camera's view.  Under role-only logic that wrote a false check-in."""
import sys
sys.path.insert(0, "/home/inomjon/projectAI/face_rec/face_recognition_airi")
from datetime import datetime, timezone
from sqlalchemy import select, delete

from app.config import settings
from app.core.direction import Direction
from app.db.models import CameraRole, DailyAttendance, Employee, RecognitionEvent
from app.db.session import init_db, session_scope
from app.services.attendance import AttendanceService, business_date

init_db(); svc = AttendanceService(); TZ = settings.tz
def L(h, m): return datetime(2026, 8, 19, h, m, tzinfo=TZ).astimezone(timezone.utc)

with session_scope() as s:
    emp = s.execute(select(Employee)).scalars().first(); eid = emp.id
    bd = business_date(L(9, 0))
    s.execute(delete(RecognitionEvent).where(RecognitionEvent.employee_id == eid))
    s.execute(delete(DailyAttendance).where(DailyAttendance.employee_id == eid,
                                            DailyAttendance.business_date == bd))

script = [
    (L(9, 0),  CameraRole.IN,  1, Direction.ENTER,   "arrives, walks in past entrance cam"),
    (L(12, 0), CameraRole.IN,  1, Direction.EXIT,    "LEAVES for lunch, passes ENTRANCE cam walking out"),
    (L(13, 0), CameraRole.OUT, 2, Direction.ENTER,   "returns, passes EXIT cam walking in"),
    (L(15, 0), CameraRole.IN,  1, Direction.UNKNOWN, "loiters near entrance cam, direction unclear"),
    (L(18, 0), CameraRole.OUT, 2, Direction.EXIT,    "goes home past exit cam"),
]

print(f"{'local':>6} {'cam role':>8} {'direction':>9} | {'transition':<13} {'presence':<8} "
      f"{'in':>6} {'out':>6} {'worked':>7}  note")
print("-" * 118)
with session_scope() as s:
    for ts, role, cam, d, note in script:
        dec = svc.record(s, employee_id=eid, camera_id=cam, role=role, ts=ts,
                         score=0.4, direction=d.value, direction_reason="test",
                         require_direction=True)
        s.flush()
        r = s.execute(select(DailyAttendance).where(
            DailyAttendance.employee_id == eid,
            DailyAttendance.business_date == bd)).scalar_one()
        ci = r.check_in_time.astimezone(TZ).strftime("%H:%M") if r.check_in_time else "-"
        co = r.check_out_time.astimezone(TZ).strftime("%H:%M") if r.check_out_time else "-"
        print(f"{ts.astimezone(TZ):%H:%M} {role.value:>8} {d.value:>9} | {dec.transition:<13} "
              f"{r.presence.value:<8} {ci:>6} {co:>6} {(r.worked_seconds or 0)/3600:6.2f}h  {note}")

print("-" * 118)
print("\nThe 12:00 row is the case that was broken: the ENTRANCE camera saw someone")
print("walking OUT.  Direction overrides the camera's role, so it is a check-out,")
print("not a second check-in.  The 15:00 row shows an undetermined direction")
print("logging a sighting without touching the attendance state.")
