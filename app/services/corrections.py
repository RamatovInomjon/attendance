"""Admin corrections: voiding a wrong recognition, resolving an unknown face.

Two operations, one purpose. Attendance gets fixed, which is what an operator
asks for - and every correction leaves behind a LABEL, which is what the
recognition thresholds have never had.

WHY LABELS ARE THE POINT
------------------------
Every threshold in this system was set against a proxy. `augment_live_floor` is
0.35 because corridor crops gave a name to 21.7% of the faces the pipeline had
judged as nobody, against 2-3% for enrolment photographs - but "judged as
nobody" is not "not an employee". Some of that 21.7% was the feature working:
an enrolled person the studio photograph missed. Nothing in the data separates
the two, so the number is an upper bound on false accepts rather than a
measurement of them.

An admin can separate them, one face at a time:

  * a VOIDED event is a confirmed false accept - a face that reached a name it
    should not have, with the score and the crop that did it
  * a sighting marked VISITOR is a confirmed non-employee, the impostor probe
    that turns every rate in bench/far_live_rows.py into a real FAR
  * a sighting resolved to an EMPLOYEE is a confirmed miss, which is the other
    half: it says what the thresholds are costing

None of this writes attendance out of thin air. Voiding removes an event's
effect by rebuilding the day from the events that still stand; resolving an
unknown records who it was and offers its face to the gallery, and deliberately
does NOT invent a check-in. A UI that can author attendance from a dropdown is
a UI that can be wrong in a way nobody can see.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import delete, select

from app.config import settings
from pathlib import Path

from app.db.models import (
    Camera, Employee, FaceEmbedding, RecognitionEvent, UnknownSighting,
)
from app.services.attendance import AttendanceService
from app.services.augment import TAG

log = logging.getLogger(__name__)

RESOLVED_KINDS = ("employee", "visitor", "unsure")


def capture_stem(ts, score: float, camera: str) -> str:
    """The debug-capture stem for one pass, found on disk.

    An event's `snapshot` column CANNOT be used for this. It names a body crop
    (`snapshots/body_31_2_1974_1788437454.jpg` - employee, camera, track,
    epoch), while a debug capture is named for what a human reads:
    `20260903_130030_0.253_Exit_001`, in LOCAL time with the score to three
    places. The two schemes share nothing, so deriving one from the other
    silently produced a key that matched no gallery row - which meant voiding a
    recognition quietly failed to remove the crop that caused it, the one thing
    voiding is most needed for.

    So it is rebuilt from the event's own fields and confirmed against the
    directory. Returns "" when no capture survives - `debug_capture` may be off,
    or retention may have swept it.
    """
    import glob
    if not ts:
        return ""
    local = ts.astimezone(settings.tz)
    prefix = f"{local:%Y%m%d_%H%M%S}_{float(score or 0):.3f}_{camera or ''}_"
    hits = sorted(glob.glob(str(settings.debug_dir / "*" / f"{prefix}*.json")))
    return Path(hits[0]).stem if hits else ""


def evidence(stem: str) -> dict:
    """The images behind one pass, by debug stem. Containment-checked."""
    out: dict = {}
    root = settings.debug_dir.resolve()
    if not stem or not root.is_dir():
        return out
    for kind, suffix in (("face", "_face.jpg"), ("aligned", "_aligned.jpg"),
                         ("frame", "_frame.jpg")):
        for person in root.iterdir():
            if not person.is_dir():
                continue
            p = (person / f"{stem}{suffix}").resolve()
            if p.is_file() and p.is_relative_to(root):
                out[kind] = p
                break
    return out


def void_event(event_id: int, *, by: str, reason: str = "") -> dict:
    """Mark one recognition as wrong, and undo everything it caused.

    Three effects, in this order, because each depends on the last:

      1. the event is flagged - never deleted, so the mistake stays auditable
      2. that person's day is REBUILT from the events that still stand
      3. any gallery row added from the same capture is removed

    Step 3 matters more than it looks. If a wrong recognition was promoted into
    the gallery, that face is now a permanent reference for the wrong person and
    the same error gets easier every day. Voiding the event without removing the
    row would fix one attendance record and leave the cause in place.
    """
    from app.db.session import session_scope

    out: dict = {"ok": False}
    with session_scope() as s:
        e = s.get(RecognitionEvent, int(event_id))
        if e is None:
            out["error"] = "no such event"
            return out
        if e.voided_at is not None:
            out["error"] = "already voided"
            return out
        if e.employee_id is None:
            out["error"] = "this event names nobody"
            return out

        employee_id, bdate = e.employee_id, e.business_date
        name = s.execute(select(Employee.full_name)
                         .where(Employee.id == employee_id)).scalar() or str(employee_id)
        e.voided_at = datetime.now(timezone.utc)
        e.voided_by = (by or "")[:64]
        e.void_reason = (reason or "")[:160]
        s.flush()

        replayed = AttendanceService().rebuild(s, employee_id, bdate)

        cam = s.execute(select(Camera.name)
                        .where(Camera.id == e.camera_id)).scalar() or ""
        removed = 0
        key = capture_stem(e.ts, e.score, cam)
        if key:
            removed = int(s.execute(delete(FaceEmbedding).where(
                FaceEmbedding.employee_id == employee_id,
                FaceEmbedding.source_file == f"{TAG}{key}")).rowcount or 0)

        out = {"ok": True, "employee_id": employee_id, "name": name,
               "business_date": str(bdate), "replayed": replayed,
               "gallery_rows_removed": removed, "capture": key}
    if out.get("gallery_rows_removed"):
        # The floors of the remaining crops are a statement about the gallery,
        # so removing one can only lower them - free accuracy, collected here.
        from app.services import augment
        augment.recalibrate()
    log.info("correction: voided event %s (%s) by %s - %s", event_id,
             out.get("name"), by, reason or "no reason given")
    return out


def unvoid_event(event_id: int, *, by: str) -> dict:
    """Undo a void. The admin was wrong about the admin being right.

    The gallery row is NOT restored: it was deleted, and re-adding a face to
    somebody's identity is a decision that belongs on the review page with the
    crop in front of a human, not as a side effect of clicking undo.
    """
    from app.db.session import session_scope
    with session_scope() as s:
        e = s.get(RecognitionEvent, int(event_id))
        if e is None or e.voided_at is None:
            return {"ok": False, "error": "not a voided event"}
        e.voided_at = None
        e.voided_by = None
        e.void_reason = None
        s.flush()
        replayed = AttendanceService().rebuild(s, e.employee_id, e.business_date)
    log.info("correction: un-voided event %s by %s", event_id, by)
    return {"ok": True, "replayed": replayed}


def resolve_sighting(sighting_id: int, *, kind: str, employee_id: int | None,
                     by: str) -> dict:
    """Record what an unknown face actually was.

    `visitor` is the valuable one and the reason the button exists: a confirmed
    non-employee is an impostor probe, and impostor probes are what every
    threshold here has been set without. `employee` names a miss and makes the
    stored face available to the gallery - available, not added: it goes to the
    review page and through the same floor and the same refusal as any other
    crop, because a face chosen from a dropdown is no more trustworthy than one
    chosen from a grid.

    Neither writes attendance. See the module docstring.
    """
    from app.db.session import session_scope

    kind = (kind or "").strip().lower()
    if kind not in RESOLVED_KINDS:
        return {"ok": False, "error": f"kind must be one of {RESOLVED_KINDS}"}
    if kind == "employee" and not employee_id:
        return {"ok": False, "error": "naming an employee needs an employee"}

    with session_scope() as s:
        u = s.get(UnknownSighting, int(sighting_id))
        if u is None:
            return {"ok": False, "error": "no such sighting"}
        name = None
        if kind == "employee":
            name = s.execute(select(Employee.full_name)
                             .where(Employee.id == int(employee_id))).scalar()
            if name is None:
                return {"ok": False, "error": "no such employee"}
            u.resolved_employee_id = int(employee_id)
        else:
            u.resolved_employee_id = None
        u.resolved_kind = kind
        u.resolved_by = (by or "")[:64]
        u.resolved_at = datetime.now(timezone.utc)
        has_vector = u.vector is not None
    log.info("correction: sighting %s resolved as %s%s by %s", sighting_id, kind,
             f" ({name})" if name else "", by)
    return {"ok": True, "kind": kind, "name": name,
            "offerable": bool(has_vector and kind == "employee")}


def labelled_probes() -> dict:
    """The ground truth the corrections have produced so far.

    Consumed by bench/far_live_rows.py and the threshold calibration. Returned
    as counts plus vectors, so a caller can measure a real FAR instead of the
    proxy every number in this system currently rests on.
    """
    import numpy as np
    from app.db.session import session_scope

    with session_scope() as s:
        visitors = s.execute(
            select(UnknownSighting.vector).where(
                UnknownSighting.resolved_kind == "visitor",
                UnknownSighting.vector.is_not(None))).scalars().all()
        missed = s.execute(
            select(UnknownSighting.vector, UnknownSighting.resolved_employee_id)
            .where(UnknownSighting.resolved_kind == "employee",
                   UnknownSighting.vector.is_not(None))).all()
        voided = s.execute(
            select(RecognitionEvent.employee_id, RecognitionEvent.score,
                   RecognitionEvent.snapshot)
            .where(RecognitionEvent.voided_at.is_not(None))).all()

    def _stack(blobs):
        if not blobs:
            return np.zeros((0, 512), np.float32)
        V = np.stack([np.frombuffer(b, np.float32) for b in blobs])
        return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)

    return {
        "visitors": _stack(visitors),
        "missed": _stack([m[0] for m in missed]),
        "missed_employee_id": np.array([m[1] for m in missed], np.int64),
        "false_accepts": [{"employee_id": v[0], "score": v[1], "snapshot": v[2]}
                          for v in voided],
    }
