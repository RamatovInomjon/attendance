"""HTML routes rendering the project's original Bootstrap templates.

These serve `templates/` — the UI the project shipped with — fed from the v3
recognition core.  `app/web/django_compat.py` supplies the Django constructs the
markup needs; `app/web/viewmodels.py` maps the v3 schema into the field names it
reads.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import func, select

from app.config import settings
from app.db.models import (
    Camera, DailyAttendance, Employee, PresenceStatus, RecognitionEvent, UnknownSighting,
)
from app.db.session import session_scope
from app.runtime import runtime
from app.services.attendance import business_date
from app.web.django_compat import build_env
from app.web.viewmodels import DailyVM, EmployeeVM, EventVM, Page

log = logging.getLogger(__name__)
router = APIRouter(tags=["pages"])
env = build_env()


def render(name: str, **ctx) -> HTMLResponse:
    ctx.setdefault("request", None)
    return HTMLResponse(env.get_template(name).render(**ctx))


def today() -> date:
    return business_date(datetime.now(settings.tz))


def _daily_rows(s, day: date):
    rows = s.execute(
        select(DailyAttendance, Employee)
        .join(Employee, Employee.id == DailyAttendance.employee_id)
        .where(DailyAttendance.business_date == day)
        .order_by(DailyAttendance.check_in_time)
    ).all()
    return [DailyVM.of(d, e) for d, e in rows]


def _events(s, limit: int = 12):
    rows = s.execute(
        select(RecognitionEvent, Employee.full_name, Employee.department, Camera.name)
        .join(Employee, Employee.id == RecognitionEvent.employee_id)
        .outerjoin(Camera, Camera.id == RecognitionEvent.camera_id)
        .order_by(RecognitionEvent.ts.desc()).limit(limit)
    ).all()
    return [EventVM.of(e, n, d, c) for e, n, d, c in rows]


# ---------------------------------------------------------------- dashboard --
@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    day = today()
    with session_scope() as s:
        total = s.execute(select(func.count(Employee.id))
                          .where(Employee.is_active.is_(True))).scalar() or 0
        records = _daily_rows(s, day)
        events = _events(s, 12)

    present = len(records)
    checked_out = sum(1 for r in records if r.check_out_time)
    inside = sum(1 for r in records if r.presence == PresenceStatus.INSIDE.value)
    late = sum(1 for r in records
               if r.check_in_time and r.check_in_time.hour >= 9 and
               not (r.check_in_time.hour == 9 and r.check_in_time.minute == 0))
    early = sum(1 for r in records if r.check_out_time and r.check_out_time.hour < 18)

    return render(
        "dashboard/index.html", request=request,
        total_employees=total, present_today=present, absent_today=total - present,
        checked_out_today=checked_out, inside_now=inside,
        late_arrivals=late, early_checkouts=early,
        present_ratio=round(present / total * 100) if total else 0,
        checkout_ratio=round(checked_out / present * 100) if present else 0,
        today_attendance=records, latest_records=records[:8], recent_events=events,
        attendance_summary=f"{present} of {total} employees seen today, {inside} currently inside",
        today=day,
    )


# ---------------------------------------------------------------- employees --
@router.get("/employees", response_class=HTMLResponse)
def employees_list(request: Request, query: str | None = None, department: str | None = None):
    with session_scope() as s:
        q = select(Employee).where(Employee.is_active.is_(True))
        if query:
            q = q.where(Employee.full_name.ilike(f"%{query}%"))
        if department:
            q = q.where(Employee.department == department)
        emps = [EmployeeVM.of(e) for e in
                s.execute(q.order_by(Employee.full_name)).scalars()]
        depts = [d for (d,) in s.execute(
            select(Employee.department).distinct().order_by(Employee.department)) if d]
    return render("employees/list.html", request=request, employees=emps,
                  page_obj=Page(emps), is_paginated=False, query=query or "",
                  department=department or "", departments=depts)


# Declared before /employees/{employee_id}: a path param would otherwise
# swallow "add" and fail int conversion with a 422.
@router.get("/employees/add", response_class=HTMLResponse)
def employee_add(request: Request):
    with session_scope() as s:
        cams = [{"id": c.id, "label": c.name, "url": c.rtsp_url, "role": c.role.value}
                for c in s.execute(select(Camera)).scalars()]
    return render("employees/register.html", request=request, employee=None, action="add",
                  cameras=cams, available_cameras=cams, use_ip_camera=bool(cams),
                  default_camera_url=cams[0]["url"] if cams else "")


@router.get("/employees/{employee_id}", response_class=HTMLResponse)
def employee_detail(request: Request, employee_id: int):
    with session_scope() as s:
        e = s.get(Employee, employee_id)
        if not e:
            raise HTTPException(404, "Employee not found")
        vm = EmployeeVM.of(e)
        rows = s.execute(
            select(DailyAttendance).where(DailyAttendance.employee_id == employee_id)
            .order_by(DailyAttendance.business_date.desc()).limit(30)).scalars().all()
        hist = [DailyVM.of(d, e) for d in rows]
        n_emb = s.execute(select(func.count()).select_from(
            select(Employee).where(Employee.id == employee_id).subquery())).scalar()
    return render("employees/detail.html", request=request, employee=vm,
                  attendance_history=hist, records=hist, embedding_count=n_emb)


# --------------------------------------------------------------- attendance --
@router.get("/attendance", response_class=HTMLResponse)
def attendance_list(request: Request, target_date: str | None = None,
                    start_date: str | None = None, end_date: str | None = None):
    day = date.fromisoformat(target_date) if target_date else today()
    with session_scope() as s:
        records = _daily_rows(s, day)
        total = s.execute(select(func.count(Employee.id))
                          .where(Employee.is_active.is_(True))).scalar() or 0
        # last 7 days, for the chart the template draws
        chart = []
        for i in range(6, -1, -1):
            d = day - timedelta(days=i)
            n = s.execute(select(func.count(DailyAttendance.id)).where(
                DailyAttendance.business_date == d,
                DailyAttendance.check_in_time.isnot(None))).scalar() or 0
            chart.append({"date": d.strftime("%d.%m"), "count": n})

    present = len(records)
    return render("attendance/list.html", request=request, records=records,
                  daily_records=records, page_obj=Page(records), is_paginated=False,
                  selected_date=day, start_date=start_date or "", end_date=end_date or "",
                  department="", employee="", status="",
                  weekly_chart_data=chart,
                  stats={"total_employees": total, "present": present,
                         "absent": total - present,
                         "late": sum(1 for r in records
                                     if r.check_in_time and r.check_in_time.hour >= 9)})


@router.get("/attendance/history", response_class=HTMLResponse)
def attendance_history(request: Request):
    return attendance_list(request)


@router.get("/attendance/unknown", response_class=HTMLResponse)
def attendance_unknown(request: Request):
    with session_scope() as s:
        rows = s.execute(select(UnknownSighting)
                         .order_by(UnknownSighting.last_seen.desc()).limit(100)).scalars().all()
        attempts = [{
            "id": u.id, "camera_id": u.camera_id, "attempt_count": u.frames,
            "first_seen": u.first_seen.astimezone(settings.tz),
            "last_seen": u.last_seen.astimezone(settings.tz),
            "latest_record": {"snapshot": {"url": f"/media/{u.snapshot}" if u.snapshot else None}},
        } for u in rows]
    return render("attendance/unknown.html", request=request,
                  unknown_attempts=attempts, page_obj=Page(attempts), is_paginated=False)


# ------------------------------------------------------------------ cameras --
@router.get("/cameras", response_class=HTMLResponse)
def cameras(request: Request):
    with session_scope() as s:
        cams = s.execute(select(Camera)).scalars().all()
        rows = [{
            "id": c.id, "name": c.name, "role": c.role.value, "rtsp_url": c.rtsp_url,
            "camera_type": "ip", "ip": c.ip, "enabled": c.enabled,
            "width": 3840, "height": 2160, "fps": 12,
        } for c in cams]
    live = {w.camera_id: w.stats() for w in runtime.workers.values()}
    for r in rows:
        r["live"] = live.get(r["id"])
    return render("camera/settings.html", request=request, cameras=rows,
                  camera=rows[0] if rows else None, active_cameras=live)


@router.get("/cameras/rtsp", response_class=HTMLResponse)
@router.get("/cameras/add", response_class=HTMLResponse)
def cameras_add(request: Request):
    with session_scope() as s:
        cams = s.execute(select(Camera)).scalars().all()
    return render("camera/rtsp_register.html", request=request, camera=None,
                  cameras=[{"id": c.id, "name": c.name} for c in cams], action="add")


# -------------------------------------------------------------- recognition --
@router.get("/recognition", response_class=HTMLResponse)
def recognition_live(request: Request):
    day = today()
    with session_scope() as s:
        records = _daily_rows(s, day)
        total = s.execute(select(func.count(Employee.id))
                          .where(Employee.is_active.is_(True))).scalar() or 0
        n_unknown = s.execute(select(func.count(UnknownSighting.id))
                              .where(UnknownSighting.business_date == day)).scalar() or 0
        events = _events(s, 10)
        cams = s.execute(select(Camera).where(Camera.enabled.is_(True))).scalars().all()
        cam_rows = [{"id": c.id, "name": c.name, "role": c.role.value} for c in cams]

    live = {w.camera_id: w.stats() for w in runtime.workers.values()}
    primary = cam_rows[0] if cam_rows else None
    st = live.get(primary["id"]) if primary else None

    checked_out = sum(1 for r in records if r.check_out_time)
    return render(
        "recognition/live.html", request=request,
        cameras=cam_rows,
        primary_camera_id=primary["id"] if primary else 1,
        secondary_streams=[{
            "id": c["id"], "label": f'{c["name"]} ({c["role"]})',
            "recognition_enabled": True,
            "resolution": (live.get(c["id"], {}).get("stream", {}) or {}).get("resolution", "—"),
            "fps": (live.get(c["id"], {}).get("stream", {}) or {}).get("fps", 0),
        } for c in cam_rows[1:]],
        camera_profile={
            "device": f'{primary["name"]} ({primary["role"]})' if primary else "No camera",
            "resolution": (st or {}).get("stream", {}).get("resolution", "—"),
            "fps": (st or {}).get("stream", {}).get("fps", 0),
            "type": "RTSP / Hikvision",
        },
        stats={
            "recognized_today": len(records), "total_employees": total,
            "checked_in_today": len(records), "checked_out_today": checked_out,
            "unknown_attempts": n_unknown,
            "last_event": events[0].timestamp if events else None,
            "summary": f"{len(records)} of {total} employees seen today",
        },
        today_records=records, recent_events=events, unknown_attempts=[],
        attendance_summary=f"{len(records)} of {total} employees seen today",
        employee_logs_serialized=[], unknown_attempts_serialized=[],
    )


@router.get("/recognition/stream", response_class=HTMLResponse)
def recognition_stream(request: Request):
    return render("recognition/stream.html", request=request)


@router.get("/recognition/logs")
def recognition_logs():
    """Polled by the live page to refresh its side panels."""
    day = today()
    with session_scope() as s:
        records = _daily_rows(s, day)
        total = s.execute(select(func.count(Employee.id))
                          .where(Employee.is_active.is_(True))).scalar() or 0
        events = _events(s, 15)
    return {
        "employees": [{
            "employee_id": r.employee.employee_id if r.employee else "",
            "name": r.employee.full_name if r.employee else "",
            "department": r.employee.department if r.employee else "",
            "check_in": r.check_in_time.isoformat() if r.check_in_time else None,
            "check_out": r.check_out_time.isoformat() if r.check_out_time else None,
            "check_in_snapshot": r.check_in_snapshot,
            "check_out_snapshot": r.check_out_snapshot,
            "status": ("checked_out" if r.check_out_time
                       else "checked_in" if r.check_in_time else "waiting"),
            "working_hours_seconds": int(r.working_hours.total_seconds()) if r.working_hours else 0,
            "recognition_count": r.recognition_count,
        } for r in records],
        "events": [{
            "name": e.employee_name, "camera": e.camera, "action": e.action_type,
            "transition": e.transition, "score": round(e.confidence, 3),
            "time": e.timestamp.strftime("%H:%M:%S"), "snapshot": e.snapshot,
        } for e in events],
        "unknown_attempts": [],
        "stats": {"recognized_today": len(records), "total_employees": total,
                  "checked_in_today": len(records),
                  "summary": f"{len(records)} of {total} employees seen today"},
    }


@router.get("/login", response_class=HTMLResponse)
def login(request: Request):
    return render("auth/login.html", request=request)
