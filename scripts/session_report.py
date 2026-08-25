#!/usr/bin/env python3
"""What happened over a run: people passed, recognized, checked in/out.

    python scripts/session_report.py            # since the server started
    python scripts/session_report.py --hours 3
"""
import argparse, sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func, select

from app.config import settings
from app.db.models import (Camera, DailyAttendance, Employee, PresenceStatus,
                           RecognitionEvent, UnknownSighting)
from app.db.session import session_scope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=None)
    a = ap.parse_args()
    since = datetime.now(timezone.utc) - timedelta(hours=a.hours) if a.hours else None
    TZ = settings.tz

    with session_scope() as s:
        eq = select(RecognitionEvent)
        uq = select(UnknownSighting)
        if since is not None:
            eq = eq.where(RecognitionEvent.ts >= since)
            uq = uq.where(UnknownSighting.first_seen >= since)
        events = s.execute(eq.order_by(RecognitionEvent.ts)).scalars().all()
        unknowns = s.execute(uq.order_by(UnknownSighting.first_seen)).scalars().all()
        cams = {c.id: c.name for c in s.execute(select(Camera)).scalars()}
        names = {e.id: e.full_name for e in s.execute(select(Employee)).scalars()}
        daily = s.execute(
            select(DailyAttendance, Employee)
            .join(Employee, Employee.id == DailyAttendance.employee_id)
            .order_by(DailyAttendance.check_in_time)
        ).all()

        ev = [dict(ts=e.ts, emp=e.employee_id, cam=cams.get(e.camera_id, "?"),
                   role=e.role.value, direction=e.direction, transition=e.transition,
                   score=e.score) for e in events]
        unk = [dict(ts=u.first_seen, cam=cams.get(u.camera_id, "?"), frames=u.frames,
                    best=u.best_score, near=names.get(u.nearest_employee_id, "-")) for u in unknowns]
        rows = [dict(name=e.full_name, ci=d.check_in_time, co=d.check_out_time,
                     worked=(d.worked_seconds or 0)/3600.0,
                     presence=d.presence.value, status=d.status) for d, e in daily]

    span = ""
    allts = [e["ts"] for e in ev] + [u["ts"] for u in unk]
    if allts:
        lo, hi = min(allts).astimezone(TZ), max(allts).astimezone(TZ)
        span = f"{lo:%H:%M} - {hi:%H:%M}  ({(hi-lo).total_seconds()/3600:.1f} h)"

    rec_passes = len({(e["emp"], e["cam"], e["ts"].replace(second=0, microsecond=0)) for e in ev})
    total = rec_passes + len(unk)

    print("=" * 72)
    print(f"  SESSION REPORT   {span}")
    print("=" * 72)
    print(f"\n  PEOPLE PASSING")
    print(f"    total passes observed   {total}")
    if total:
        print(f"    recognized              {rec_passes}   ({rec_passes/total*100:.0f}%)")
        print(f"    not recognized          {len(unk)}   ({len(unk)/total*100:.0f}%)")

    print(f"\n  RECOGNITION EVENTS        {len(ev)}")
    tr = Counter(e["transition"] for e in ev)
    for k in ("CHECK_IN", "CHECK_OUT", "RE_SIGHTING", "NO_DIRECTION", "DEBOUNCED"):
        if tr.get(k): print(f"    {k:<16} {tr[k]}")
    di = Counter(e["direction"] for e in ev)
    print(f"\n  DIRECTION RESOLVED")
    for k in ("ENTER", "EXIT", "UNKNOWN"):
        n = di.get(k, 0)
        print(f"    {k:<16} {n}" + (f"   ({n/len(ev)*100:.0f}%)" if ev else ""))
    cc = Counter(e["cam"] for e in ev)
    print(f"\n  BY CAMERA")
    for c, n in cc.most_common(): print(f"    {c:<16} {n}")

    print(f"\n  ATTENDANCE ({len(rows)} people)")
    if rows:
        print(f"    {'name':<26} {'in':>6} {'out':>6} {'worked':>8}  status")
        for r in rows:
            ci = r["ci"].astimezone(TZ).strftime("%H:%M") if r["ci"] else "-"
            co = r["co"].astimezone(TZ).strftime("%H:%M") if r["co"] else "-"
            print(f"    {r['name'][:26]:<26} {ci:>6} {co:>6} {r['worked']:>7.2f}h  "
                  f"{r['presence']}/{r['status']}")
    else:
        print("    nobody recorded")

    if unk:
        print(f"\n  UNRECOGNIZED PASSES ({len(unk)})   nearest match shown - a high")
        print(f"  best score means the threshold or the gallery is the problem,")
        print(f"  a low one means it was genuinely someone not enrolled.")
        print(f"    {'time':>6} {'camera':<12} {'frames':>7} {'best':>7}  nearest")
        for u in unk[:25]:
            print(f"    {u['ts'].astimezone(TZ):%H:%M} {u['cam']:<12} {u['frames']:>7} "
                  f"{u['best']:>7.3f}  {u['near'][:26]}")
        if len(unk) > 25: print(f"    ... and {len(unk)-25} more")
    print()


if __name__ == "__main__":
    main()
