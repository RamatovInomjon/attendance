"""HTML routes rendering the project's original Bootstrap templates.

These serve `templates/` — the UI the project shipped with — fed from the v3
recognition core.  `app/web/django_compat.py` supplies the Django constructs the
markup needs; `app/web/viewmodels.py` maps the v3 schema into the field names it
reads.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from urllib.parse import urlsplit, urlunsplit

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import func, or_, select

from app.config import settings
from app.db.models import (
    Camera, DailyAttendance, Employee, FaceEmbedding, PresenceStatus, RecognitionEvent, UnknownSighting,
)
from app.db.session import session_scope
from app.runtime import runtime
from app.services.attendance import business_date
from app.web.django_compat import build_env
from app.web.viewmodels import DailyVM, EmployeeVM, EventVM, Page

log = logging.getLogger(__name__)
router = APIRouter(tags=["pages"])
env = build_env()


def render(name: str, *, request=None, current_view: str = "", **ctx) -> HTMLResponse:
    """Render a page with the common navigation and date context."""
    ctx.setdefault("request", request)
    ctx.setdefault("current_view", current_view)
    ctx.setdefault("today", today())
    return HTMLResponse(env.get_template(name).render(**ctx))


def today() -> date:
    return business_date(datetime.now(settings.tz))


def _recorded_presence_clause():
    """Attendance counts only after either an IN or OUT transition is recorded."""
    return or_(
        DailyAttendance.check_in_time.is_not(None),
        DailyAttendance.check_out_time.is_not(None),
    )


def _daily_rows(s, day: date, *, recorded_only: bool = False,
                limit: int | None = None, recent_first: bool = False):
    """Fetch daily rows; optional filtering/bounding is reserved for dashboard views."""
    query = (
        select(DailyAttendance, Employee)
        .join(Employee, Employee.id == DailyAttendance.employee_id)
        .where(DailyAttendance.business_date == day)
    )
    if recorded_only:
        query = query.where(_recorded_presence_clause())
    if recent_first:
        query = query.order_by(
            func.coalesce(DailyAttendance.check_out_time, DailyAttendance.check_in_time).desc(),
            DailyAttendance.id.desc(),
        )
    else:
        query = query.order_by(DailyAttendance.check_in_time)
    if limit is not None:
        query = query.limit(limit)
    rows = s.execute(query).all()
    return [DailyVM.of(d, e) for d, e in rows]


# The dashboard is a recent operational summary, not the full attendance register.
DASHBOARD_DAILY_ROWS_LIMIT = 50


def _dashboard_daily_rows(s, day: date):
    """Return at most 50 newest recorded rows; `/attendance` intentionally remains full."""
    return _daily_rows(
        s, day, recorded_only=True, limit=DASHBOARD_DAILY_ROWS_LIMIT, recent_first=True,
    )


def _dashboard_attendance_metrics(s, day: date) -> tuple[int, int, int]:
    """Count all active recorded attendance independently from the bounded table slice."""
    filters = (
        DailyAttendance.business_date == day,
        Employee.is_active.is_(True),
        _recorded_presence_clause(),
    )
    present = s.execute(
        select(func.count(DailyAttendance.id))
        .join(Employee, Employee.id == DailyAttendance.employee_id)
        .where(*filters)
    ).scalar() or 0
    check_in_times = s.execute(
        select(DailyAttendance.check_in_time)
        .join(Employee, Employee.id == DailyAttendance.employee_id)
        .where(*filters, DailyAttendance.check_in_time.is_not(None))
    ).scalars().all()
    late = sum(
        1 for check_in in check_in_times
        if (check_in.astimezone(settings.tz).hour, check_in.astimezone(settings.tz).minute) > (9, 0)
    )
    return present, len(check_in_times) - late, late


def _events(s, limit: int = 12):
    rows = s.execute(
        select(RecognitionEvent, Employee.full_name, Employee.department, Camera.name)
        .join(Employee, Employee.id == RecognitionEvent.employee_id)
        .outerjoin(Camera, Camera.id == RecognitionEvent.camera_id)
        .order_by(RecognitionEvent.ts.desc()).limit(limit)
    ).all()
    return [EventVM.of(e, n, d, c) for e, n, d, c in rows]


def _month_start(value: date) -> date:
    return value.replace(day=1)


def _previous_months(value: date, count: int) -> list[date]:
    """Return calendar-month starts ending with the month containing ``value``."""
    cursor = _month_start(value)
    months = []
    for _ in range(count):
        months.append(cursor)
        cursor = (cursor - timedelta(days=1)).replace(day=1)
    return list(reversed(months))


def _dashboard_chart_data(s, day: date) -> tuple[list[dict], list[dict]]:
    """Build complete chart series with one grouped query per chart period."""
    week_start = day - timedelta(days=6)
    daily_counts = dict(s.execute(
        select(DailyAttendance.business_date, func.count(DailyAttendance.id))
        .where(
            DailyAttendance.business_date.between(week_start, day),
            _recorded_presence_clause(),
        )
        .group_by(DailyAttendance.business_date)
    ).all())
    weekly = [
        {"label": (week_start + timedelta(days=offset)).strftime("%d.%m"),
         "present": daily_counts.get(week_start + timedelta(days=offset), 0)}
        for offset in range(7)
    ]

    month_starts = _previous_months(day, 6)
    month_key = func.strftime("%Y-%m", DailyAttendance.business_date)
    monthly_counts = dict(s.execute(
        select(month_key, func.count(DailyAttendance.id))
        .where(
            DailyAttendance.business_date.between(month_starts[0], day),
            _recorded_presence_clause(),
        )
        .group_by(month_key)
    ).all())
    monthly = [
        {"label": month.strftime("%m.%Y"), "present": monthly_counts.get(month.strftime("%Y-%m"), 0)}
        for month in month_starts
    ]
    return weekly, monthly


def _camera_health(cameras: list[Camera]) -> list[dict]:
    """Describe enabled configured cameras without changing worker lifecycle."""
    worker_stats = {camera_id: worker.stats() for camera_id, worker in runtime.workers.items()}
    health = []
    for camera in cameras:
        stats = worker_stats.get(camera.id)
        if stats is None:
            health.append({
                "id": camera.id, "name": camera.name, "role": camera.role.value,
                "state": "offline", "state_label": "Offline", "available": False,
                "detail": "Ishchi mavjud emas", "fps": None, "pipeline_errors": None,
            })
            continue

        stream = stats.get("stream") or {}
        connected = bool(stream.get("connected")) and not bool(stream.get("stale"))
        health.append({
            "id": camera.id, "name": camera.name, "role": camera.role.value,
            "state": "online" if connected else "unavailable",
            "state_label": "Onlayn" if connected else "Mavjud emas",
            "available": connected,
            "detail": "Oqim faol" if connected else "Oqim ulanmagan yoki eskirgan",
            "fps": stream.get("fps") if connected else None,
            "pipeline_errors": stats.get("pipeline_errors", 0),
        })
    return health


# ---------------------------------------------------------------- dashboard --
@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    day = today()
    with session_scope() as s:
        total = s.execute(select(func.count(Employee.id))
                          .where(Employee.is_active.is_(True))).scalar() or 0
        present, on_time, late = _dashboard_attendance_metrics(s, day)
        records = _dashboard_daily_rows(s, day)
        events = _events(s, 12)
        weekly_chart_data, monthly_chart_data = _dashboard_chart_data(s, day)
        cameras = s.execute(select(Camera).where(Camera.enabled.is_(True))
                            .order_by(Camera.name)).scalars().all()

    camera_health = _camera_health(cameras)

    return render(
        "dashboard/index.html", request=request, current_view="dashboard:home",
        total_employees=total, present_today=present, on_time_today=on_time,
        absent_today=max(0, total - present), late_arrivals=late,
        present_ratio=round(present / total * 100) if total else 0,
        today_attendance=records, recent_events=events,
        camera_health=camera_health, weekly_chart_data=weekly_chart_data,
        monthly_chart_data=monthly_chart_data,
        today=day,
    )


# ---------------------------------------------------------------- employees --
@router.get("/employees", response_class=HTMLResponse)
def employees_list(request: Request, query: str | None = None, department: str | None = None):
    with session_scope() as s:
        enrollment_counts = (
            select(
                FaceEmbedding.employee_id.label("employee_id"),
                func.count(FaceEmbedding.id).label("enrollment_count"),
            )
            .group_by(FaceEmbedding.employee_id)
            .subquery()
        )
        q = (
            select(
                Employee,
                func.coalesce(enrollment_counts.c.enrollment_count, 0).label("enrollment_count"),
            )
            .outerjoin(enrollment_counts, enrollment_counts.c.employee_id == Employee.id)
            .where(Employee.is_active.is_(True))
        )
        if query:
            q = q.where(or_(
                Employee.full_name.ilike(f"%{query}%"),
                Employee.external_id.ilike(f"%{query}%"),
            ))
        if department:
            q = q.where(Employee.department == department)
        emps = [EmployeeVM.of(employee, enrollment_count=count) for employee, count in
                s.execute(q.order_by(Employee.full_name)).all()]
        depts = [d for (d,) in s.execute(
            select(Employee.department)
            .where(Employee.is_active.is_(True))
            .distinct()
            .order_by(Employee.department)) if d]
    return render("employees/list.html", request=request, current_view="employees:list", employees=emps,
                  page_obj=Page(emps), is_paginated=False, query=query or "",
                  department=department or "", departments=depts)


# Declared before /employees/{employee_id}: a path param would otherwise
# swallow "add" and fail int conversion with a 422.
@router.get("/employees/add", response_class=HTMLResponse)
def employee_add(request: Request):
    with session_scope() as s:
        cams = [{"id": c.id, "label": c.name, "url": c.rtsp_url, "role": c.role.value}
                for c in s.execute(select(Camera)).scalars()]
    return render("employees/register.html", request=request, current_view="employees:register", employee=None, action="add",
                  cameras=cams, available_cameras=cams, use_ip_camera=bool(cams),
                  default_camera_url=cams[0]["url"] if cams else "")


@router.get("/employees/{employee_id}", response_class=HTMLResponse)
def employee_detail(request: Request, employee_id: int):
    with session_scope() as s:
        e = s.get(Employee, employee_id)
        if not e:
            raise HTTPException(404, "Employee not found")
        rows = s.execute(
            select(DailyAttendance).where(DailyAttendance.employee_id == employee_id)
            .order_by(DailyAttendance.business_date.desc()).limit(30)).scalars().all()
        hist = [DailyVM.of(d, e) for d in rows]
        n_emb = s.execute(
            select(func.count(FaceEmbedding.id)).where(FaceEmbedding.employee_id == employee_id)
        ).scalar() or 0
        vm = EmployeeVM.of(e, enrollment_count=n_emb)
    return render("employees/detail.html", request=request, current_view="employees:list", employee=vm,
                  attendance_history=hist, records=hist, embedding_count=n_emb)


# --------------------------------------------------------------- attendance --
MAX_ATTENDANCE_RANGE_DAYS = 366


def _parse_attendance_date(value: str | None, field: str) -> date | None:
    """Parse a query date without allowing malformed values to escape as 500s."""
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, f"{field} ISO sana bo'lishi kerak") from exc


def _attendance_date_range(target_date: str | None, start_date: str | None,
                           end_date: str | None) -> tuple[date, date, date, str | None]:
    """Return a bounded inclusive filter range and an optional normalization notice."""
    selected_day = _parse_attendance_date(target_date, "target_date") or today()
    start = _parse_attendance_date(start_date, "start_date")
    end = _parse_attendance_date(end_date, "end_date")
    if start is None and end is None:
        start = end = selected_day
    elif start is None:
        start = end
    elif end is None:
        end = start

    notice = None
    if start > end:
        start, end = end, start
        notice = "Sana oralig'i tartibga keltirildi."
    if (end - start).days >= MAX_ATTENDANCE_RANGE_DAYS:
        raise HTTPException(422, "Sana oralig'i 366 kundan oshmasligi kerak")
    return selected_day, start, end, notice


def attendance_records_query(start: date, end: date, *, query: str | None = None,
                             department: str | None = None, status: str | None = None):
    """Build the shared employee-attendance filter query used by HTML and CSV views."""
    statement = (
        select(DailyAttendance, Employee)
        .join(Employee, Employee.id == DailyAttendance.employee_id)
        .where(DailyAttendance.business_date.between(start, end))
    )
    if query:
        statement = statement.where(or_(
            Employee.full_name.ilike(f"%{query}%"),
            Employee.external_id.ilike(f"%{query}%"),
        ))
    if department:
        statement = statement.where(Employee.department == department)
    if status:
        statement = statement.where(DailyAttendance.status == status)
    return statement


def attendance_newest_first(statement):
    """Keep attendance reporting order deterministic across consumers."""
    return statement.order_by(
        DailyAttendance.business_date.desc(),
        func.coalesce(DailyAttendance.check_out_time, DailyAttendance.check_in_time).desc(),
        DailyAttendance.id.desc(),
    )


@router.get("/attendance", response_class=HTMLResponse)
def attendance_list(request: Request, target_date: str | None = None,
                    query: str | None = None, department: str | None = None,
                    start_date: str | None = None, end_date: str | None = None,
                    status: str | None = None):
    selected_day, start, end, range_notice = _attendance_date_range(
        target_date, start_date, end_date,
    )
    with session_scope() as s:
        records = [DailyVM.of(record, employee) for record, employee in s.execute(
            attendance_newest_first(attendance_records_query(
                start, end, query=query, department=department, status=status,
            ))
        ).all()]
        departments = [value for (value,) in s.execute(
            select(Employee.department)
            .where(Employee.is_active.is_(True))
            .distinct()
            .order_by(Employee.department)
        ) if value]

    return render(
        "attendance/list.html", request=request, current_view="attendance:list", records=records,
        daily_records=records, page_obj=Page(records), is_paginated=False,
        selected_date=selected_day, start_date=start.isoformat(), end_date=end.isoformat(),
        query=query or "", employee=query or "", department=department or "", status=status or "",
        departments=departments, range_notice=range_notice,
    )


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
    return render("attendance/unknown.html", request=request, current_view="attendance:unknown",
                  unknown_attempts=attempts, page_obj=Page(attempts), is_paginated=False)


# ------------------------------------------------------------------ cameras --
def _safe_stream_url(value: str | None) -> str | None:
    """Keep a camera stream's host/path while never rendering its credentials."""
    if not value:
        return None
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.hostname:
        return None
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{host}:{parsed.port}" if parsed.port else host
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _camera_diagnostics_context(cameras: list[Camera]) -> list[dict]:
    """Present configuration and worker observations without changing either system."""
    worker_stats = {}
    worker_failures = {}
    for camera_id, worker in runtime.workers.items():
        try:
            worker_stats[camera_id] = worker.stats()
        except Exception as exc:  # Diagnostics must not make camera failures fatal to the page.
            log.warning("camera %s stats unavailable: %s", camera_id, exc)
            worker_failures[camera_id] = "Ishchi statistikasi olinmadi"
    diagnostics = []
    for camera in cameras:
        stats = worker_stats.get(camera.id)
        stream = (stats or {}).get("stream") or {}
        online = bool(stream.get("connected")) and not bool(stream.get("stale"))
        resolution = stream.get("resolution") if online else None
        if isinstance(resolution, str) and resolution.lower().replace("×", "x") == "0x0":
            resolution = None
        observed_fps = stream.get("fps") if online else None
        if isinstance(observed_fps, bool) or not isinstance(observed_fps, (int, float)) or observed_fps <= 0:
            observed_fps = None

        pipeline = stats or {}
        diagnostics.append({
            "configured": {
                "id": camera.id, "name": camera.name, "role": camera.role.value,
                "ip": camera.ip or None, "rtsp_url": _safe_stream_url(camera.rtsp_url),
                "enabled": camera.enabled,
            },
            "observed": {
                "state": "online" if online else "offline",
                "state_label": "Onlayn" if online else "Mavjud emas",
                "resolution": resolution,
                "fps": observed_fps,
                "detail": (
                    "RTSP oqimi faol" if online
                    else worker_failures.get(camera.id, "Ishchi yoki yangilangan RTSP oqimi mavjud emas")
                ),
            },
            "pipeline": {
                "fps": pipeline.get("processed_fps"),
                "latency_ms": (
                    pipeline.get("latency_ms")
                    if pipeline.get("latency_ms") is not None
                    else (pipeline.get("timings") or {}).get("total")
                ),
                "dropped_frames": pipeline.get("dropped_frames"),
                "errors": pipeline.get("pipeline_errors"),
                "last_error": pipeline.get("last_error"),
            },
            "drift": {
                "state": "nvr-owned", "comparison": "unavailable",
                "label": "Ma'lumotlarni solishtirib bo'lmaydi",
                "detail": "Encoder qiymatlari NVR tomonidan boshqariladi",
            },
        })
    return diagnostics


@router.get("/cameras", response_class=HTMLResponse)
def cameras(request: Request):
    with session_scope() as s:
        cams = s.execute(select(Camera)).scalars().all()
    diagnostics = _camera_diagnostics_context(cams)
    return render("camera/settings.html", request=request, current_view="camera:settings",
                  camera_diagnostics=diagnostics)


@router.get("/cameras/rtsp", response_class=HTMLResponse)
@router.get("/cameras/add", response_class=HTMLResponse)
def cameras_add(request: Request):
    with session_scope() as s:
        cams = s.execute(select(Camera)).scalars().all()
    return render("camera/rtsp_register.html", request=request, current_view="camera:rtsp_register", camera=None,
                  cameras=[{"id": c.id, "name": c.name} for c in cams], action="add")


# -------------------------------------------------------------- recognition --
def _available_value(*values):
    """Return the first present runtime value while preserving legitimate zeroes."""
    return next((value for value in values if value is not None), None)


def _positive_number(value):
    """Return a positive numeric metric, excluding booleans and invalid values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return float(value)


def _available_resolution(value) -> str | None:
    """Reject the uninitialized resolution emitted by a fresh RtspSource."""
    if not isinstance(value, str) or not value.strip():
        return None
    resolution = value.strip()
    if resolution.lower().replace("×", "x") == "0x0":
        return None
    return resolution


def _last_frame_label(value) -> str | None:
    """Format an observed worker timestamp without inventing one when it is absent."""
    if value is None:
        return None
    parsed = value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        timestamp = value / 1000 if value > 10_000_000_000 else value
        parsed = datetime.fromtimestamp(timestamp, settings.tz)
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
    if isinstance(parsed, datetime):
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=settings.tz)
        return parsed.astimezone(settings.tz).strftime("%H:%M:%S")
    return str(parsed)


def _live_camera_context(cameras: list[Camera]) -> list[dict]:
    """Map configured cameras and existing worker stats into presentation-only health data."""
    role_order = {"IN": 0, "OUT": 1, "BOTH": 2}
    role_labels = {
        "IN": "Kirish kamerasi", "OUT": "Chiqish kamerasi", "BOTH": "Umumiy kamera",
    }
    contexts = []
    for camera in sorted(cameras, key=lambda item: (role_order.get(item.role.value, 3), item.name)):
        worker = runtime.workers.get(camera.id)
        stats = None
        if worker is not None:
            try:
                stats = worker.stats() or {}
            except Exception as exc:  # a broken diagnostics call must not hide the live page
                log.warning("camera %s stats unavailable: %s", camera.id, exc)

        role = camera.role.value
        if stats is None:
            contexts.append({
                "id": camera.id, "name": camera.name, "role": role,
                "role_label": role_labels.get(role, "Kamera"),
                "state": "offline", "state_label": "Offline",
                "state_detail": "Ishchi mavjud emas",
                "resolution": None, "camera_fps": None, "algorithm_fps": None,
                "latency_ms": None, "dropped_frames": None, "last_frame": None,
                "pipeline_errors": None, "pipeline_status": "unavailable",
                "pipeline_status_label": "Pipeline ma'lumoti mavjud emas",
                "last_error": None,
            })
            continue

        stream = stats.get("stream") or {}
        timings = stats.get("timings") or {}
        connected = bool(stream.get("connected")) and not bool(stream.get("stale"))
        state = "online" if connected else "unavailable"
        state_label = "Onlayn" if connected else "Mavjud emas"
        state_detail = "Oqim faol" if connected else "Oqim ulanmagan yoki eskirgan"

        algorithm_fps = _available_value(
            stats.get("processed_fps"), stats.get("algorithm_fps"), stats.get("pipeline_fps"),
        )
        total_ms = _positive_number(timings.get("total"))
        dropped_frames = _available_value(
            stats.get("dropped_frames"), stream.get("dropped_frames"), stream.get("dropped"),
        )

        pipeline_errors = stats.get("pipeline_errors") if "pipeline_errors" in stats else None
        last_error = stats.get("last_error") or None
        if last_error or (isinstance(pipeline_errors, (int, float)) and pipeline_errors > 0):
            pipeline_status, pipeline_status_label = "error", "Pipeline xatosi"
        elif pipeline_errors == 0 and connected:
            pipeline_status, pipeline_status_label = "healthy", "Pipeline sog'lom"
        else:
            pipeline_status, pipeline_status_label = "unavailable", "Pipeline ma'lumoti mavjud emas"

        contexts.append({
            "id": camera.id, "name": camera.name, "role": role,
            "role_label": role_labels.get(role, "Kamera"),
            "state": state, "state_label": state_label, "state_detail": state_detail,
            "resolution": _available_resolution(stream.get("resolution")),
            "camera_fps": _positive_number(stream.get("fps")),
            "algorithm_fps": algorithm_fps,
            "latency_ms": _available_value(
                stats.get("latency_ms"), stream.get("latency_ms"), total_ms,
            ),
            "dropped_frames": dropped_frames,
            "last_frame": _last_frame_label(_available_value(
                stream.get("last_frame_time"), stream.get("last_frame_ts"),
                stream.get("last_frame"), stats.get("last_frame_time"),
            )),
            "pipeline_errors": pipeline_errors, "pipeline_status": pipeline_status,
            "pipeline_status_label": pipeline_status_label, "last_error": last_error,
        })
    return contexts


def _event_payload(events: list[EventVM]) -> list[dict]:
    """Serialize quality-best recognition evidence for initial render and polling."""
    return [{
        "name": event.employee_name, "department": event.employee_department,
        "camera": event.camera, "action": event.action_type,
        "transition": event.transition, "score": round(event.confidence, 3),
        "time": event.timestamp.strftime("%H:%M:%S"), "snapshot": event.snapshot,
    } for event in events]


def _unknown_activity(s, day: date, limit: int = 30) -> list[dict]:
    """Return newest quality-best unknown sightings with camera presentation data."""
    rows = s.execute(
        select(UnknownSighting, Camera.name)
        .outerjoin(Camera, Camera.id == UnknownSighting.camera_id)
        .where(UnknownSighting.business_date == day)
        .order_by(UnknownSighting.last_seen.desc(), UnknownSighting.id.desc())
        .limit(limit)
    ).all()
    return [{
        "id": sighting.id, "camera_id": sighting.camera_id,
        "camera": camera_name or (f"Kamera #{sighting.camera_id}" if sighting.camera_id else "—"),
        "attempt_count": sighting.frames or 0,
        "first_seen": sighting.first_seen.astimezone(settings.tz).isoformat(),
        "last_seen": sighting.last_seen.astimezone(settings.tz).isoformat(),
        "first_seen_label": sighting.first_seen.astimezone(settings.tz).strftime("%H:%M:%S"),
        "last_seen_label": sighting.last_seen.astimezone(settings.tz).strftime("%H:%M:%S"),
        "snapshot": f"/media/{sighting.snapshot}" if sighting.snapshot else None,
    } for sighting, camera_name in rows]


@router.get("/recognition", response_class=HTMLResponse)
def recognition_live(request: Request):
    day = today()
    with session_scope() as s:
        records = _daily_rows(s, day)
        total = s.execute(select(func.count(Employee.id))
                          .where(Employee.is_active.is_(True))).scalar() or 0
        n_unknown = s.execute(select(func.count(UnknownSighting.id))
                              .where(UnknownSighting.business_date == day)).scalar() or 0
        events = _events(s, 30)
        unknown_attempts = _unknown_activity(s, day, 30)
        cams = s.execute(select(Camera).where(Camera.enabled.is_(True))).scalars().all()

    camera_rows = _live_camera_context(cams)

    checked_out = sum(1 for r in records if r.check_out_time)
    checked_in = sum(1 for r in records if r.check_in_time)
    summary = f"Bugun {total} xodimdan {len(records)} nafari qayd etildi"
    return render(
        "recognition/live.html", request=request, current_view="recognition:live",
        cameras=camera_rows,
        stats={
            "recognized_today": len(records), "total_employees": total,
            "checked_in_today": checked_in, "checked_out_today": checked_out,
            "unknown_attempts": n_unknown,
            "last_event": events[0].timestamp if events else None,
            "summary": summary,
        },
        today_records=records, recent_events=events, unknown_attempts=unknown_attempts,
        attendance_summary=summary,
        recent_events_serialized=_event_payload(events),
        unknown_attempts_serialized=unknown_attempts,
    )


@router.get("/recognition/stream", response_class=HTMLResponse)
def recognition_stream(request: Request):
    return render("recognition/stream.html", request=request, current_view="recognition:live")


@router.get("/recognition/logs")
def recognition_logs():
    """Polled by the live page to refresh its side panels."""
    day = today()
    with session_scope() as s:
        records = _daily_rows(s, day)
        total = s.execute(select(func.count(Employee.id))
                          .where(Employee.is_active.is_(True))).scalar() or 0
        n_unknown = s.execute(select(func.count(UnknownSighting.id))
                              .where(UnknownSighting.business_date == day)).scalar() or 0
        events = _events(s, 30)
        unknown_attempts = _unknown_activity(s, day, 30)
    checked_in = sum(1 for record in records if record.check_in_time)
    checked_out = sum(1 for record in records if record.check_out_time)
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
        "events": _event_payload(events),
        "unknown_attempts": unknown_attempts,
        "stats": {"recognized_today": len(records), "total_employees": total,
                  "checked_in_today": checked_in, "checked_out_today": checked_out,
                  "unknown_attempts": n_unknown,
                  "summary": f"Bugun {total} xodimdan {len(records)} nafari qayd etildi"},
    }


@router.get("/login", response_class=HTMLResponse)
def login(request: Request):
    return render("auth/login.html", request=request, current_view="auth:login")
