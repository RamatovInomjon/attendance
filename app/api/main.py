"""FastAPI surface: dashboard, live views, attendance, health."""
from __future__ import annotations

import asyncio
import csv
import io
import logging
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select

from app.config import settings
from app.db.models import (
    Camera, DailyAttendance, Employee, PresenceStatus, RecognitionEvent,
)
from app.db.session import session_scope
from app.runtime import runtime
from app.services.attendance import business_date
from app.services.auth import ensure_default_admin
from app.api import auth as auth_router
from app.api import pages as pages_router
from app.api import ws as ws_router

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Before anything is served: make sure the console is reachable. On a fresh
    # database this creates the default admin; once any account exists it does
    # nothing, so a changed password or a deleted default is never resurrected.
    try:
        if ensure_default_admin():
            log.warning("created the default admin account - change its password at /users")
    except Exception:
        log.exception("could not ensure a default admin account")
    runtime.start()
    yield
    runtime.stop()


app = FastAPI(title="EmAtSy v3", version="3.0.0", lifespan=lifespan)

# Everything hangs off the configured prefix. The edge proxy forwards the path
# unchanged, so when this is served at /faceid the app must really answer on
# /faceid/... - mounts, routes and all.
PREFIX = settings.url_prefix.rstrip("/")

settings.media_dir.mkdir(parents=True, exist_ok=True)
app.mount(f"{PREFIX}/media", StaticFiles(directory=str(settings.media_dir)), name="media")

# The original UI's stylesheet and scripts.
_static = settings.root / "static"
if _static.is_dir():
    app.mount(f"{PREFIX}/static", StaticFiles(directory=str(_static)), name="static")

# HTML pages (original Bootstrap templates) and the live-view WebSocket.
# Deny by default. This must be added BEFORE the routers so that every route,
# including any added later, is behind it unless explicitly listed public in
# app/api/auth.py. Without it the pages below answer anyone with the URL.
app.middleware("http")(auth_router.auth_middleware)

app.include_router(auth_router.router, prefix=PREFIX)
app.include_router(pages_router.router, prefix=PREFIX)
app.include_router(ws_router.router, prefix=PREFIX)


@app.get(f"{PREFIX}/health", include_in_schema=False)
def _health_probe():
    """Unauthenticated liveness probe for the edge proxy's monitoring.

    Deliberately says nothing about cameras, people or the gallery - it exists
    to answer "is the process up", and anything richer would leak operational
    detail to an endpoint that has to stay public.
    """
    return {"status": "ok"}


def _local(dt):
    return dt.astimezone(settings.tz) if dt else None


def _today() -> date:
    return business_date(datetime.now(settings.tz))


# --------------------------------------------------------------- stream ----
@app.get("/video/{camera_id}")
async def video(camera_id: int):
    w = runtime.workers.get(camera_id)
    if w is None:
        raise HTTPException(404, "camera not running")

    async def gen():
        # Each viewer renders its own copy from the shared latest frame, so
        # opening a second tab does not steal the first one's frames.
        while True:
            jpg = w.render()
            if jpg:
                yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            await asyncio.sleep(1 / 10)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")


# ------------------------------------------------------------------ api ----
@app.get("/api/health")
def health():
    ok = all(w.source.connected and not w.source.is_stale for w in runtime.workers.values())
    errs = sum(w.pipeline_errors for w in runtime.workers.values())
    return {
        "status": ("degraded" if errs else "healthy") if ok and runtime.workers else "degraded",
        "pipeline_errors": errs,
        "gallery": {"embeddings": len(runtime.gallery or []),
                    "people": runtime.gallery.n_people if runtime.gallery else 0},
        "cameras": [w.stats() for w in runtime.workers.values()],
    }


@app.get("/api/events")
def events(limit: int = 40):
    return runtime.events(limit)


def _filtered_attendance_rows(day: str | None = None, query: str | None = None,
                              department: str | None = None, start_date: str | None = None,
                              end_date: str | None = None, status: str | None = None):
    """Fetch CSV/API attendance using the same predicates as the list page."""
    selected_day, start, end, _ = pages_router._attendance_date_range(day, start_date, end_date)
    with session_scope() as s:
        rows = s.execute(
            pages_router.attendance_newest_first(pages_router.attendance_records_query(
                start, end, query=query, department=department, status=status,
            ))
        ).all()
    return selected_day, start, end, [{
            "date": r.business_date.isoformat(),
            "employee_id": e.id, "name": e.full_name, "department": e.department,
            "check_in": _local(r.check_in_time).isoformat() if r.check_in_time else None,
            "check_out": _local(r.check_out_time).isoformat() if r.check_out_time else None,
            "worked_hours": round((r.worked_seconds or 0) / 3600.0, 2),
            "presence": r.presence.value, "status": r.status, "events": r.event_count,
        } for r, e in rows]


@app.get("/api/attendance")
def attendance(day: str | None = None, query: str | None = None,
               department: str | None = None, start_date: str | None = None,
               end_date: str | None = None, status: str | None = None):
    _, _, _, rows = _filtered_attendance_rows(
        day, query, department, start_date, end_date, status,
    )
    return rows


@app.get("/api/attendance/export")
def export(day: str | None = None, query: str | None = None,
           department: str | None = None, start_date: str | None = None,
           end_date: str | None = None, status: str | None = None):
    selected_day, start, end, rows = _filtered_attendance_rows(
        day, query, department, start_date, end_date, status,
    )
    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(["Date", "Employee", "Department", "Check in", "Check out", "Worked hours", "Status"])
    for r in rows:
        wr.writerow([
            r["date"], r["name"], r["department"],
            r["check_in"][11:16] if r["check_in"] else "",
            r["check_out"][11:16] if r["check_out"] else "",
            f'{r["worked_hours"]:.2f}', r["status"],
        ])
    buf.seek(0)
    suffix = selected_day.isoformat() if start == end else f"{start}_{end}"
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=attendance_{suffix}.csv"})


@app.get("/api/debug/captures")
def debug_captures():
    """What the debug folder holds, per person."""
    merged: dict[str, int] = {}
    for w in runtime.workers.values():
        for k, v in w.debug.summary().items():
            merged[k] = merged.get(k, 0) + v
    return {"dir": str(settings.debug_dir), "enabled": settings.debug_capture,
            "captured": dict(sorted(merged.items(), key=lambda kv: -kv[1]))}


@app.post("/api/gallery/reload")
def reload_gallery():
    g = runtime.reload_gallery()
    return {"embeddings": len(g), "people": g.n_people}
