"""What each kind of account may do.

Three roles, and the two that matter here are new: an **operator** corrects
attendance without being handed the cameras, and a **viewer** may look but not
touch. The tests worth having are the ones that check the RULE rather than the
rendering - a hidden nav link is not a control, so every restriction is proved
against the route, and `test_no_admin_path_answers_an_operator` walks the live
route table so a page added under /cameras or /gallery later cannot quietly
answer the wrong people.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.core.security import COOKIE_NAME, read_session, sign_session
from app.services import auth as auth_svc

client = TestClient(app, follow_redirects=False)


def _cookie(role: str, *, uid: int = 1) -> dict:
    return {"Cookie": f"{COOKIE_NAME}={sign_session(user_id=uid, username=role, is_admin=role == 'admin', role=role)}"}


ADMIN, OPERATOR, VIEWER = _cookie("admin"), _cookie("operator"), _cookie("viewer")

# The surfaces an operator asked to be kept away from, by the path that serves
# them rather than by the link that points at them.
ADMIN_PATHS = ["/cameras", "/gallery/review", "/users", "/recognition",
               "/recognition/logs", "/video/1", "/api/health",
               # Enrolment is the gallery by another door; the debug folder
               # is the pipeline's own evidence.
               "/employees/add", "/api/debug/captures"]


# ---- the capability table ----------------------------------------------

@pytest.mark.parametrize("role,admin,correct", [
    ("admin", True, True),
    ("operator", False, True),
    ("viewer", False, False),
])
def test_capabilities(role, admin, correct):
    s = {"uid": 1, "u": role, "adm": role == "admin", "r": role}
    assert auth_svc.can_admin(s) is admin
    assert auth_svc.can_correct(s) is correct


def test_nobody_is_not_a_role():
    assert auth_svc.can_admin(None) is False
    assert auth_svc.can_correct(None) is False
    assert auth_svc.session_role(None) == auth_svc.ROLE_VIEWER


def test_an_unrecognised_role_loses_access_rather_than_gaining_it():
    """This value arrives from a cookie and from a form. Failing open would
    turn a typo into a promotion."""
    s = {"uid": 1, "u": "x", "adm": False, "r": "superuser"}
    assert auth_svc.session_role(s) == auth_svc.ROLE_VIEWER
    assert auth_svc.can_correct(s) is False


def test_a_cookie_minted_before_roles_existed_still_works():
    """Old cookies stay valid for up to 12 hours after a deploy. An absent
    role must mean what the account could do yesterday, not nothing."""
    payload = read_session(sign_session(user_id=1, username="a", is_admin=True))
    payload.pop("r")
    assert auth_svc.session_role(payload) == auth_svc.ROLE_ADMIN
    assert auth_svc.can_correct(payload) is True

    old_operator = {"uid": 2, "u": "b", "adm": False}
    assert auth_svc.session_role(old_operator) == auth_svc.ROLE_VIEWER


def test_the_cookie_carries_the_role():
    assert read_session(sign_session(user_id=1, username="a", is_admin=False,
                                     role="operator"))["r"] == "operator"


# ---- the routes ---------------------------------------------------------

@pytest.mark.parametrize("path", ADMIN_PATHS)
def test_operator_is_refused_every_admin_path(path):
    assert client.get(path, headers=OPERATOR).status_code == 403


@pytest.mark.parametrize("path", ADMIN_PATHS)
def test_viewer_is_refused_every_admin_path(path):
    assert client.get(path, headers=VIEWER).status_code == 403


def test_admin_reaches_them():
    for path in ("/cameras", "/gallery/review", "/users"):
        assert client.get(path, headers=ADMIN).status_code == 200


def test_no_admin_path_answers_an_operator():
    """Walk the real route table. Anything under an admin prefix must refuse
    an operator, so a route added there later is covered without being
    listed here - the same reasoning as `test_no_route_is_public_by_accident`.
    """
    from app.api.auth import _is_admin_path
    leaks = []
    for route in app.routes:
        path = getattr(route, "path", "")
        if "GET" not in (getattr(route, "methods", set()) or set()):
            continue
        if "{" in path or not _is_admin_path(path):
            continue
        r = client.get(path, headers=OPERATOR)
        if r.status_code != 403:
            leaks.append((path, r.status_code))
    assert not leaks, f"admin paths that answered an operator: {leaks}"


def test_a_refusal_is_403_and_not_a_trip_through_the_login_form():
    """They are signed in. Redirecting them to a form they would immediately
    pass would send them straight back here, which reads as a broken page."""
    r = client.get("/cameras", headers=OPERATOR)
    assert r.status_code == 403
    assert "login" not in r.headers.get("location", "")


def test_an_operator_may_still_change_their_own_password():
    """Carved out of the /users prefix deliberately: `users_password` does its
    own check and has always let an account change its own."""
    r = client.post("/users/password", headers=OPERATOR,
                    data={"username": "operator", "password": "x"})
    assert r.status_code != 403


def test_the_operator_keeps_the_pages_the_job_needs():
    for path in ("/", "/attendance", "/attendance/unknown", "/employees"):
        assert client.get(path, headers=OPERATOR).status_code == 200


# ---- corrections --------------------------------------------------------

def test_an_operator_may_correct_and_a_viewer_may_not():
    """The whole point of the role. 404 is a pass here: it means the request
    got past authorization and failed to find event 999999."""
    for headers, allowed in ((OPERATOR, True), (VIEWER, False)):
        r = client.post("/attendance/event/999999/void", headers=headers,
                        data={"reason": "test"})
        assert (r.status_code != 403) is allowed
        r = client.post("/attendance/unknown/999999/resolve", headers=headers,
                        data={"kind": "visitor"})
        assert (r.status_code != 403) is allowed


def test_an_operator_may_look_at_the_evidence_they_are_judging():
    """A correction made without seeing the crop is a guess, so the evidence
    image follows the correction permission and not the admin one."""
    r = client.get("/attendance/event/999999/evidence/face", headers=OPERATOR)
    assert r.status_code == 404
    assert client.get("/attendance/event/999999/evidence/face",
                      headers=VIEWER).status_code == 403


def test_a_viewer_is_not_shown_who_voided_an_event_or_why():
    """The void note names an operator and quotes their reason - "bu u emas".
    That is the correction trail, addressed to the people who can act on it;
    a viewer sees the event struck through and nothing about who struck it."""
    from datetime import datetime, timezone
    from sqlalchemy import delete
    from app.db.models import CameraRole, Employee, RecognitionEvent
    from app.db.session import session_scope

    ts = datetime(2026, 8, 26, 9, 0, tzinfo=timezone.utc)
    with session_scope() as s:
        emp = Employee(full_name="Void Trail", is_active=True)
        s.add(emp); s.flush()
        ev = RecognitionEvent(employee_id=emp.id, camera_id=None, ts=ts,
                              business_date=ts.date(), role=CameraRole.IN,
                              score=0.5, voided_at=ts, voided_by="operator",
                              void_reason="bu u emas")
        s.add(ev); s.flush()
        emp_id, ev_id = emp.id, ev.id
    try:
        path = f"/attendance/day/{emp_id}/2026-08-26"
        seen = client.get(path, headers=OPERATOR)
        assert seen.status_code == 200
        assert "Bekor qilingan" in seen.text and "bu u emas" in seen.text

        hidden = client.get(path, headers=VIEWER)
        assert hidden.status_code == 200
        assert "Bekor qilingan" not in hidden.text
        assert "bu u emas" not in hidden.text and "operator" not in hidden.text
    finally:
        with session_scope() as s:
            s.execute(delete(RecognitionEvent).where(RecognitionEvent.id == ev_id))
            s.execute(delete(Employee).where(Employee.id == emp_id))


# ---- the account itself -------------------------------------------------

def test_role_and_is_admin_can_never_disagree():
    """Two fields, one truth. A row claiming is_admin with role 'viewer'
    would authorise differently depending on which one the reader consults."""
    auth_svc.create_user("role-check", "password1", role="operator")
    row = next(u for u in auth_svc.list_users() if u["username"] == "role-check")
    assert row["role"] == "operator" and row["is_admin"] is False

    auth_svc.set_role("role-check", "admin")
    row = next(u for u in auth_svc.list_users() if u["username"] == "role-check")
    assert row["role"] == "admin" and row["is_admin"] is True


def test_is_admin_still_makes_an_admin_for_callers_that_predate_roles():
    auth_svc.create_user("legacy-admin", "password1", is_admin=True)
    row = next(u for u in auth_svc.list_users() if u["username"] == "legacy-admin")
    assert row["role"] == "admin"


def test_the_last_admin_cannot_be_demoted():
    """Same reasoning as `set_active`: an unreachable console is worse than a
    stale account, and demotion locks everyone out just as thoroughly."""
    for u in auth_svc.list_users():
        if u["is_admin"] and u["username"] != "inomjon":
            auth_svc.set_role(u["username"], "viewer")
    with pytest.raises(auth_svc.AuthError):
        auth_svc.set_role("inomjon", "operator")
    still = next(u for u in auth_svc.list_users() if u["username"] == "inomjon")
    assert still["role"] == "admin" and still["is_admin"] is True


def test_an_unknown_role_is_refused_rather_than_stored():
    auth_svc.create_user("role-typo", "password1", role="operator")
    with pytest.raises(auth_svc.AuthError):
        auth_svc.set_role("role-typo", "administrator")


# ---- enrolment and the ops endpoints --------------------------------------

def test_enrolment_and_the_ops_endpoints_refuse_everyone_but_an_admin():
    """/employees/add and POST /api/employees/ create an identity the cameras
    will trust from then on; /api/gallery/reload and /api/debug/captures act
    on the pipeline. All of them answered any signed-in role."""
    for headers in (OPERATOR, VIEWER):
        assert client.post("/api/employees/", headers=headers).status_code == 403
        assert client.post("/api/gallery/reload", headers=headers).status_code == 403
        assert client.get("/api/debug/captures", headers=headers).status_code == 403
    # An admin gets past the gate: the 400 is the handler objecting to an
    # empty form, which is exactly the point.
    assert client.post("/api/employees/", headers=ADMIN).status_code == 400


def test_only_an_admin_may_write_attendance_from_an_unknown():
    """Labelling a face and authoring a pass are different powers. An operator
    keeps the first - naming a miss is their job - and is refused the second,
    because a check-in typed from a dropdown is the one correction that adds a
    record rather than removing one."""
    from datetime import datetime, timezone
    from sqlalchemy import delete, select
    from app.db.models import (
        DailyAttendance, Employee, RecognitionEvent, UnknownSighting)
    from app.db.session import session_scope
    from app.services.attendance import business_date

    ts = datetime(2026, 8, 26, 9, 0, tzinfo=timezone.utc)
    with session_scope() as s:
        emp = Employee(full_name="Promote Gate", is_active=True)
        s.add(emp); s.flush()
        u = UnknownSighting(camera_id=None, track_id=3, first_seen=ts,
                            last_seen=ts, business_date=business_date(ts))
        s.add(u); s.flush()
        emp_id, sid = emp.id, u.id
    try:
        data = {"employee_id": str(emp_id), "direction": "ENTER"}
        r = client.post(f"/attendance/unknown/{sid}/resolve", headers=OPERATOR,
                        data=data)
        assert r.status_code == 303
        assert "error=" in r.headers.get("location", "")
        with session_scope() as s:
            assert s.get(UnknownSighting, sid).promoted_event_id is None

        r = client.post(f"/attendance/unknown/{sid}/resolve", headers=ADMIN,
                        data=data)
        assert r.status_code == 303
        assert "error=" not in r.headers.get("location", "")
        with session_scope() as s:
            assert s.get(UnknownSighting, sid).promoted_event_id is not None
    finally:
        with session_scope() as s:
            s.execute(delete(RecognitionEvent).where(
                RecognitionEvent.employee_id == emp_id))
            # The promotion rebuilds the day, which CREATES this row. Left
            # behind it is a person still inside the building on a stale date,
            # and the start-up sweep in test_runtime_sweep.py finds it.
            s.execute(delete(DailyAttendance).where(
                DailyAttendance.employee_id == emp_id))
            s.execute(delete(UnknownSighting).where(UnknownSighting.id == sid))
            s.execute(delete(Employee).where(Employee.id == emp_id))


def test_an_operator_may_still_label_an_unknown_without_a_direction():
    """The gate is on the direction, not on the whole form."""
    r = client.post("/attendance/unknown/999999/resolve", headers=OPERATOR,
                    data={"kind": "visitor"})
    assert r.status_code != 403


def test_the_day_page_enlarges_the_body_crop_and_not_the_debug_frame():
    """The full-frame link opened `data/debug`, which is written only when
    debug capture is on and is swept by retention - so on a deployed system it
    opened onto nothing while the event itself was still listed. The body crop
    is served from media/, which is part of the record, so it survives as long
    as the row does."""
    from datetime import datetime, timezone
    from sqlalchemy import delete
    from app.db.models import (
        CameraRole, DailyAttendance, Employee, RecognitionEvent)
    from app.db.session import session_scope

    ts = datetime(2026, 8, 26, 9, 0, tzinfo=timezone.utc)
    with session_scope() as s:
        emp = Employee(full_name="Crop Only", is_active=True)
        s.add(emp); s.flush()
        s.add(RecognitionEvent(
            employee_id=emp.id, camera_id=None, role=CameraRole.IN, ts=ts,
            business_date=ts.date(), score=0.5,
            snapshot="snapshots/body_test.jpg"))
        emp_id = emp.id
    try:
        html = client.get(f"/attendance/day/{emp_id}/2026-08-26",
                          headers=ADMIN).text
        assert "evidence/frame" not in html
        at = html.index("data-evidence-src")
        assert "body_test.jpg" in html[at:at + 160]
    finally:
        with session_scope() as s:
            s.execute(delete(RecognitionEvent).where(
                RecognitionEvent.employee_id == emp_id))
            s.execute(delete(DailyAttendance).where(
                DailyAttendance.employee_id == emp_id))
            s.execute(delete(Employee).where(Employee.id == emp_id))
