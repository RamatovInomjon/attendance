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
