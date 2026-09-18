"""Offer the departure a missing check-out lost, from passes already stored.

THE GAP
-------
About 2.2 person-days a day end with a check-in and no check-out (35 of 462
over the 16 complete dates to 2026-09-18). On three quarters of them the face
path saw nothing later at all - but the person still walked out past a camera,
and that walk left a `reid_pass` behind with a body feature and usually a face
template. This offers the best of those as a candidate departure. It never
writes one.

THIS IS NOT THE RULE `pseudo_gallery.py` REFUSES
------------------------------------------------
That one is open-set - "is this unknown track an employee?" - where the base
rate (~45 recoverable employees among ~1100 unknown tracks a day) makes a body
rule write more wrong rows than right, at every threshold measured. This is
constrained verification: a named person checked in this morning, so which of
today's EXIT passes is them? The anchor is a pass the FACE named the SAME DAY,
which is why clothing may be used at all - it changes overnight, not at lunch.

WHAT WAS MEASURED, AND WHAT IT LICENSES
---------------------------------------
`bench/missing_checkout_eval.py`, 386 complete days held out with the answer
hidden and no ranker allowed to read `employee_id`, against a median of 266
candidates each:

    ranker          right pass  right person   at margin >= 0.05
    body                 45.3%         54.7%   57.3% on 62% of days
    face                 33.4%         54.7%   98.0% on 39% of days
    fused                51.0%         78.2%   95.4% on 45% of days

Three things follow, and all three are built in here.

**Face leads and body only backs it up.** Body alone names the right person
barely more than half the time - the same conclusion pseudo_gallery.py reached
from the other direction, arrived at independently here.

**The margin is the gate, not the score.** Ungated, the best ranker is right
about the person 78% of the time; gated on the runner-up margin it is right
95-98%, on fewer days. Coverage is the price and it is the right one to pay.

**It suggests and never writes.** A missing check-out is VISIBLE - the timesheet
flags the day incomplete and somebody looks at it. A wrong one is invisible: it
writes a plausible `worked_seconds` that nothing in the data reveals. Even at
95% a silent writer puts an invented departure in the record every twentieth
day, and converts a visible gap into an error nobody will ever find. At two a
day a person can confirm them, so the machine ranks and the human decides -
exactly the split `app/services/augment.py` uses for the gallery.

Those figures are also an OPTIMISTIC bound: they come from days where the exit
WAS captured, and the days needing repair are the ones where capture failed.
Treat them as an ordering of the options, not as a promise about accuracy here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone

import numpy as np
from sqlalchemy import select

from app.config import recognizer_key, settings
from app.db.models import (
    Camera, CameraRole, DailyAttendance, Employee, PresenceStatus,
    RecognitionEvent, ReidPass, UnknownSighting,
)

log = logging.getLogger(__name__)

# The fusion the bench measured. Face carries most of the weight because body
# alone is barely better than a coin toss at naming the person; body is kept
# because it is the only evidence on a pass whose face never cleared the gates.
FACE_WEIGHT, BODY_WEIGHT = 0.7, 0.3


def _unit(blob, dim):
    if not blob or not dim:
        return None
    a = np.frombuffer(blob, dtype=np.float32)
    if a.size != dim:
        return None
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-9 else None


@dataclass
class Candidate:
    """One EXIT pass offered as the departure that was never recorded."""
    pass_id: int
    ts: datetime
    camera: str
    face_sim: float | None
    body_sim: float | None
    score: float
    margin: float
    snapshot: str | None

    @property
    def time(self) -> str:
        return self.ts.astimezone(settings.tz).strftime("%H:%M:%S")


def _anchor(s, employee_id: int, bdate: date, check_in: datetime):
    """The employee's own face-named pass that day: the clothing reference.

    Nearest the check-in, because light and clothing drift across a day and the
    arrival is the thing the check-in actually was.
    """
    rows = s.execute(
        select(ReidPass).where(ReidPass.employee_id == employee_id,
                               ReidPass.business_date == bdate)
    ).scalars().all()
    usable = [r for r in rows if r.vector is not None or r.face_vector is not None]
    if not usable:
        return None
    return min(usable, key=lambda r: abs((r.last_seen - check_in).total_seconds())
               if r.last_seen and check_in else 1e9)


def _comparable(cand, anchor, kind: str):
    """A cosine across two model spaces succeeds and means nothing.

    Every recognizer and every ReID model here emits unit vectors, so nothing
    errors and the number looks ordinary - which is exactly why the provenance
    columns exist and why this refuses rather than scores.
    """
    if kind == "face":
        a, b = _unit(anchor.face_vector, anchor.face_dim), _unit(cand.face_vector, cand.face_dim)
        same = recognizer_key(anchor.face_model or "") == recognizer_key(cand.face_model or "")
        same = same and bool(anchor.face_model)
    else:
        a, b = _unit(anchor.vector, anchor.dim), _unit(cand.vector, cand.dim)
        same = (anchor.model_name or "") == (cand.model_name or "") and bool(anchor.model_name)
    if a is None or b is None or not same or a.shape != b.shape:
        return None
    return float(a @ b)


def _snapshots_for(s, passes, bdate: date) -> dict[int, str]:
    """A body crop per candidate, via the same (camera, track, day) join the
    unknown page uses - `unknown_sighting` and `reid_pass` share no key."""
    from app.web.django_compat import media_path
    if not passes:
        return {}
    rows = s.execute(
        select(UnknownSighting).where(
            UnknownSighting.business_date == bdate,
            UnknownSighting.camera_id.in_({p.camera_id for p in passes}),
            UnknownSighting.track_id.in_({p.track_id for p in passes}))
    ).scalars().all()
    by_key: dict[tuple, list] = {}
    for u in rows:
        by_key.setdefault((u.camera_id, u.track_id), []).append(u)
    out: dict[int, str] = {}
    for p in passes:
        cand = by_key.get((p.camera_id, p.track_id))
        if not cand or p.last_seen is None:
            continue
        u = min(cand, key=lambda x: abs((x.last_seen - p.last_seen).total_seconds()))
        # More than a few minutes apart is a different track across a tracker
        # reset. A wrong crop beside a name is worse than no crop.
        if abs((u.last_seen - p.last_seen).total_seconds()) > 180 or not u.snapshot:
            continue
        out[p.pass_id if hasattr(p, "pass_id") else p.id] = media_path(u.snapshot)
    return out


def suggest(s, employee_id: int, bdate: date, *, limit: int = 3
            ) -> list[Candidate]:
    """Ranked candidate departures for a day that never recorded one.

    Returns [] - not an error - when the day already has a check-out, when no
    anchor pass exists, or when nothing is comparable. The caller shows nothing
    in that case, which is the honest answer.
    """
    daily = s.execute(select(DailyAttendance).where(
        DailyAttendance.employee_id == employee_id,
        DailyAttendance.business_date == bdate)).scalar_one_or_none()
    if daily is None or daily.check_in_time is None or daily.check_out_time is not None:
        return []

    anchor = _anchor(s, employee_id, bdate, daily.check_in_time)
    if anchor is None:
        return []

    rows = s.execute(
        select(ReidPass).where(ReidPass.business_date == bdate,
                               ReidPass.direction == "EXIT")
    ).scalars().all()

    scored = []
    for r in rows:
        if r.id == anchor.id or r.last_seen is None:
            continue
        if r.last_seen <= daily.check_in_time:
            continue
        f = _comparable(r, anchor, "face")
        b = _comparable(r, anchor, "body")
        if f is None and b is None:
            continue
        if f is not None and b is not None:
            total = FACE_WEIGHT * f + BODY_WEIGHT * b
        else:
            total = f if f is not None else b
        scored.append((total, f, b, r))

    if not scored:
        return []
    scored.sort(key=lambda x: -x[0])
    runner_up = scored[1][0] if len(scored) > 1 else -1.0

    snaps = _snapshots_for(s, [r for _t, _f, _b, r in scored[:limit]], bdate)
    out = []
    for i, (total, f, b, r) in enumerate(scored[:limit]):
        out.append(Candidate(
            pass_id=r.id, ts=r.last_seen, camera=r.camera_name or "—",
            face_sim=f, body_sim=b, score=float(total),
            margin=float(total - runner_up) if i == 0 else 0.0,
            snapshot=snaps.get(r.id)))
    return out


def confident(cands: list[Candidate]) -> Candidate | None:
    """The top candidate, if it clears the margin gate the bench measured.

    Below the gate the page still lists what was found, unmarked: an operator
    who knows the person can use a weak candidate that the rule may not
    recommend. What the gate governs is what the system is willing to CALL a
    likely departure.
    """
    if not cands:
        return None
    top = cands[0]
    if top.margin < settings.checkout_suggest_margin:
        return None
    if top.score < settings.checkout_suggest_min_score:
        return None
    return top


def confirm(pass_id: int, *, employee_id: int, by: str) -> dict:
    """Write the confirmed departure as a real check-out, and rebuild the day.

    Modelled on `corrections.promote_sighting`, and for the same reasons:

    * `source="body"` - a fourth kind of evidence beside live, tracklet and
      manual. A human confirmed an identity the FACE never established, so it
      must be separable after the fact: threshold calibration has to exclude it
      exactly as it excludes "manual", and anyone auditing an hours figure needs
      to see which departures a person vouched for.
    * `score=0.0`, because no recognition happened. Putting the body similarity
      here would feed a ReID cosine into a table whose scores are face
      similarities, and every measurement over it would silently mix the two.
    * The day is REBUILT, not nudged. `worked_seconds` accumulates and
      `presence` is a latch, so a check-out inserted before an existing sighting
      changes what that sighting means; only a replay in timestamp order gets
      that right.
    """
    from app.db.session import session_scope
    from app.services.attendance import AttendanceService, business_date

    if not employee_id:
        return {"ok": False, "error": "confirming a departure needs an employee"}

    with session_scope() as s:
        r = s.get(ReidPass, int(pass_id))
        if r is None:
            return {"ok": False, "error": "no such pass"}
        if r.last_seen is None:
            return {"ok": False, "error": "pass has no timestamp"}
        name = s.execute(select(Employee.full_name)
                         .where(Employee.id == int(employee_id))).scalar()
        if name is None:
            return {"ok": False, "error": "no such employee"}

        bdate = business_date(r.last_seen)
        # One walk, one transition. A pass already carrying an event for this
        # person must not be confirmed twice into the same day.
        existing = s.execute(
            select(RecognitionEvent.id).where(
                RecognitionEvent.employee_id == int(employee_id),
                RecognitionEvent.business_date == bdate,
                RecognitionEvent.ts == r.last_seen,
                RecognitionEvent.voided_at.is_(None))).scalar()
        if existing is not None:
            return {"ok": False,
                    "error": "this pass is already in attendance; void that "
                             "event first"}

        role = CameraRole.BOTH
        if r.camera_id is not None:
            cam_role = s.execute(select(Camera.role)
                                 .where(Camera.id == r.camera_id)).scalar()
            if cam_role is not None:
                role = cam_role

        ev = RecognitionEvent(
            employee_id=int(employee_id), camera_id=r.camera_id, role=role,
            ts=r.last_seen, business_date=bdate,
            score=0.0, margin=0.0,
            track_id=r.track_id if r.track_id is not None else -1,
            face_px=0, votes="body", snapshot=None, accepted=True,
            transition="",                       # the rebuild computes it
            direction="EXIT",
            direction_reason=f"body match confirmed by {(by or '')[:60]}"[:96],
            source="body",
        )
        s.add(ev)
        s.flush()
        event_id = ev.id
        replayed = AttendanceService().rebuild(s, int(employee_id), bdate)
        transition = ev.transition

    log.info("checkout recovery: pass %s confirmed as EXIT for %s (%s) by %s, "
             "%d events replayed", pass_id, name, transition, by, replayed)
    return {"ok": True, "name": name, "event_id": event_id,
            "transition": transition, "business_date": bdate.isoformat(),
            "replayed": replayed}
