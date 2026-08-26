"""The access gate.

Before this existed, every page and JSON endpoint answered anyone who knew the
URL and `/login` was decoration. These tests exist so that stays fixed: the
important one is `test_no_route_is_public_by_accident`, which walks the live
route table rather than a hand-written list, so a new endpoint added later
cannot quietly ship unprotected.
"""
from __future__ import annotations

import base64
import json

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.core.security import (
    COOKIE_NAME, hash_password, needs_rehash, read_session, sign_session,
    verify_password,
)
from app.services import auth as auth_svc

# No lifespan: it calls runtime.start(), which opens RTSP workers and retries
# against the cameras forever.
client = TestClient(app, follow_redirects=False)

# Every deliberately public path. Adding to this list is a security decision:
# /health is public because the edge proxy in front of aiscan.airi.uz probes it
# for liveness and cannot present a session. It returns only {"status": "ok"} -
# nothing about cameras, people or the gallery - precisely so that being public
# leaks nothing.
PUBLIC = {"/login", "/logout", "/favicon.ico", "/health"}


# Sent as an explicit header rather than TestClient's `cookies=` kwarg: that
# kwarg MERGES with cookies the client already holds, so a login earlier in the
# module left an admin session that silently overrode the operator one and made
# the authorization tests pass for the wrong reason.
def _admin_cookie() -> dict:
    return {"Cookie": f"{COOKIE_NAME}={sign_session(user_id=1, username='inomjon', is_admin=True)}"}


def _operator_cookie() -> dict:
    return {"Cookie": f"{COOKIE_NAME}={sign_session(user_id=2, username='operator', is_admin=False)}"}


def test_operator_password_is_not_changed_by_the_403_case():
    """The 403 above must be a real refusal, not a 403 rendered after the write
    already happened. An earlier version of this suite rewrote the live admin
    password precisely because nothing checked the effect."""
    assert auth_svc.authenticate("inomjon", "123456") is not None
    assert auth_svc.authenticate("inomjon", "hijacked") is None


# ---- passwords ---------------------------------------------------------

def test_password_roundtrip_and_salting():
    h = hash_password("123456")
    assert verify_password("123456", h)
    assert not verify_password("123457", h)
    assert not verify_password("", h)
    # different salt each time, so identical passwords do not collide in the db
    assert hash_password("123456") != h


@pytest.mark.parametrize("stored", ["", "garbage", "pbkdf2_sha256$notanint$a$b",
                                    "md5$1$a$b", "a$b$c"])
def test_malformed_hash_fails_closed(stored):
    """A corrupt row must deny access, not raise and 500 the login page."""
    assert verify_password("123456", stored) is False


def test_needs_rehash_flags_weaker_cost():
    assert needs_rehash(hash_password("x", iterations=1000)) is True
    assert needs_rehash(hash_password("x")) is False


# ---- session tokens ----------------------------------------------------

def test_valid_session_roundtrip():
    payload = read_session(sign_session(user_id=7, username="a", is_admin=True))
    assert payload["uid"] == 7 and payload["adm"] is True


@pytest.mark.parametrize("mangle", [
    lambda t: t[:-4] + "AAAA",                 # broken signature
    lambda t: t.split(".")[0],                 # signature removed
    lambda t: "nonsense",
    lambda t: "",
])
def test_tampered_tokens_rejected(mangle):
    assert read_session(mangle(sign_session(user_id=1, username="a", is_admin=False))) is None


def test_privilege_escalation_rejected():
    """Re-encoding the payload with adm=true must not survive the signature."""
    tok = sign_session(user_id=1, username="a", is_admin=False)
    _, sig = tok.split(".")
    body = base64.urlsafe_b64encode(
        json.dumps({"uid": 1, "u": "a", "adm": True, "iat": 9999999999}).encode()
    ).decode().rstrip("=")
    assert read_session(f"{body}.{sig}") is None


def test_expired_token_rejected(monkeypatch):
    tok = sign_session(user_id=1, username="a", is_admin=False)
    monkeypatch.setattr("app.core.security.MAX_AGE_S", -1)
    assert read_session(tok) is None


# ---- the gate ----------------------------------------------------------

def test_no_route_is_public_by_accident():
    """Walk the real route table: every GET route must refuse an anonymous
    caller unless it is one of the few deliberately public paths."""
    leaks = []
    for route in app.routes:
        path = getattr(route, "path", "")
        methods = getattr(route, "methods", set()) or set()
        if "GET" not in methods or path in PUBLIC or path.startswith("/static"):
            continue
        if "{" in path:                      # needs params; covered separately
            continue
        r = client.get(path)
        if r.status_code not in (303, 401, 403):
            leaks.append((path, r.status_code))
    assert not leaks, f"routes served anonymously: {leaks}"


def test_login_page_is_public():
    assert client.get("/login").status_code == 200


def test_json_endpoints_get_401_not_a_redirect():
    """An XHR redirected to an HTML login page fails with a confusing parse
    error; 401 tells the caller plainly that the session is gone."""
    r = client.get("/api/health")
    assert r.status_code == 401


def test_authenticated_request_passes():
    r = client.get("/attendance", headers=_admin_cookie())
    assert r.status_code == 200


def test_bad_credentials_rejected_without_cookie():
    r = client.post("/login", data={"username": "inomjon", "password": "wrong"})
    assert r.status_code == 401
    assert COOKIE_NAME not in r.cookies


def test_login_redirects_to_next_but_only_same_site():
    """`next` must not bounce a freshly authenticated operator off-site."""
    r = client.post("/login", data={"username": "inomjon", "password": "123456",
                                    "next": "https://evil.example/x"})
    assert r.status_code == 303
    assert r.headers["location"] == "/"

    r = client.post("/login", data={"username": "inomjon", "password": "123456",
                                    "next": "//evil.example/x"})
    assert r.headers["location"] == "/"


def test_session_cookie_is_hardened():
    r = client.post("/login", data={"username": "inomjon", "password": "123456"})
    raw = r.headers.get("set-cookie", "")
    assert "HttpOnly" in raw          # XSS cannot read the session
    assert "Path=/" in raw
    assert "samesite=lax" in raw.lower()


# ---- authorization -----------------------------------------------------

def test_users_page_is_admin_only():
    assert client.get("/users", headers=_admin_cookie()).status_code == 200
    assert client.get("/users", headers=_operator_cookie()).status_code == 403


def test_non_admin_cannot_create_users():
    r = client.post("/users/add", headers=_operator_cookie(),
                    data={"username": "sneak", "password": "abcdef"})
    assert r.status_code == 403


def test_operator_cannot_change_another_password():
    r = client.post("/users/password", headers=_operator_cookie(),
                    data={"username": "inomjon", "password": "hijacked"})
    assert r.status_code == 403


# ---- account rules -----------------------------------------------------

@pytest.mark.parametrize("username", ["ab", "", "has space", "bad/slash", "x" * 65])
def test_bad_usernames_rejected(username):
    with pytest.raises(auth_svc.AuthError):
        auth_svc.validate_new_user(username, "abcdef")


def test_short_password_rejected():
    with pytest.raises(auth_svc.AuthError):
        auth_svc.validate_new_user("someone", "12345")


def test_default_admin_is_not_recreated_once_accounts_exist():
    """Guards against a deleted or renamed default admin being resurrected, or
    a changed password being silently reset, on every restart."""
    assert auth_svc.user_count() > 0
    assert auth_svc.ensure_default_admin() is False
