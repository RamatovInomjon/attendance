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
import time

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


# --- next= confinement under a sub-path deployment -------------------------
# Found by driving the deployed site in a browser: opening /faceid/login
# directly leaves next="/" (the default), which was same-site and so allowed,
# and dropped the operator on aiscan.airi.uz's root - a different project.

import pytest as _pytest


@_pytest.mark.parametrize("raw,expected", [
    (None,             "/faceid/"),
    ("",               "/faceid/"),
    ("/",              "/faceid/"),   # the default - used to escape the app
    ("/faceid/",       "/faceid/"),
    ("/faceid",        "/faceid"),
    ("/faceid/users",  "/faceid/users"),
    ("/manim/",        "/faceid/"),   # same ORIGIN, different application
    ("/ppe/",          "/faceid/"),
    ("//evil.example", "/faceid/"),
    ("/\\evil.example", "/faceid/"),   # browsers read "/\" as "//"
    ("https://evil.example", "/faceid/"),
    ("/faceidevil",    "/faceid/"),   # prefix must match a path SEGMENT
])
def test_next_is_confined_to_the_deployment_prefix(raw, expected, monkeypatch):
    from app.config import settings
    from app.api import auth as A
    monkeypatch.setattr(settings, "url_prefix", "/faceid", raising=False)
    assert A._safe_next(raw) == expected


@_pytest.mark.parametrize("raw,expected", [
    (None,            "/"),
    ("/",             "/"),
    ("/users",        "/users"),
    ("//evil.example", "/"),
    ("/\\evil.example", "/"),
])
def test_next_still_works_without_a_prefix(raw, expected, monkeypatch):
    """Root deployments must keep behaving exactly as before."""
    from app.config import settings
    from app.api import auth as A
    monkeypatch.setattr(settings, "url_prefix", "", raising=False)
    assert A._safe_next(raw) == expected


# ---- cross-site posts ----------------------------------------------------
# SameSite=Lax already keeps the cookie off a form another site posts. This is
# the second lock: a browser attaches the page's Origin (older ones its
# Referer) to every POST, and one naming another host is refused outright.

def test_a_post_from_another_origin_is_refused():
    creds = {"username": "inomjon", "password": "123456"}
    for headers in ({"Origin": "https://evil.example"},
                    {"Referer": "https://evil.example/page"},
                    {"Origin": "null"}):          # sandboxed frame / privacy redirect
        r = client.post("/login", data=creds, headers=headers)
        assert r.status_code == 403, headers
        assert COOKIE_NAME not in r.cookies
    # Signed in makes no difference: refused before the handler runs, and
    # the write it asked for did not happen.
    r = client.post("/users/password", data={"username": "inomjon", "password": "hijacked"},
                    headers={**_admin_cookie(), "Origin": "https://evil.example"})
    assert r.status_code == 403
    assert auth_svc.authenticate("inomjon", "123456") is not None


def test_same_site_and_proxied_posts_still_pass():
    creds = {"username": "inomjon", "password": "123456"}
    assert client.post("/login", data=creds, headers={"Origin": "http://testserver"}).status_code == 303
    # Behind the edge proxy the Host header is whatever the proxy forwarded;
    # X-Forwarded-Host names the public site the browser's Origin will carry.
    r = client.post("/login", data=creds, headers={"Origin": "https://aiscan.airi.uz",
                                                   "X-Forwarded-Host": "aiscan.airi.uz"})
    assert r.status_code == 303
    # Reads are not state changes and the check must leave them alone. (The
    # jar is cleared first: signed in, /login is a redirect regardless.)
    client.cookies.clear()
    assert client.get("/login", headers={"Origin": "https://evil.example"}).status_code == 200


def test_a_browser_behind_the_proxy_is_not_refused_its_own_form():
    """The bug this rule caused, before it read Sec-Fetch-Site.

    Adding a viewer account through the public URL returned "Cross-site
    request refused". The browser posts to aiscan.airi.uz with that Origin;
    the proxy forwards it upstream with its own Host and no X-Forwarded-Host,
    so comparing the two called the operator's own form a stranger. The
    browser's own Sec-Fetch-Site says what really happened, and nothing
    between the two can rewrite it.
    """
    r = client.post("/users/add",
                    data={"username": "kuzatuvchi", "password": "viewer-pw-123",
                          "full_name": "Kuzatuvchi", "role": "viewer"},
                    headers={**_admin_cookie(),
                             "Origin": "https://aiscan.airi.uz",
                             "Sec-Fetch-Site": "same-origin"})
    assert r.status_code == 303, r.text
    assert auth_svc.authenticate("kuzatuvchi", "viewer-pw-123") is not None


def test_sec_fetch_site_decides_whenever_the_browser_sends_it():
    creds = {"username": "inomjon", "password": "123456"}
    # Refused on the browser's word, whatever the Origin claims...
    r = client.post("/login", data=creds, headers={"Origin": "http://testserver",
                                                   "Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403
    # ...and allowed on it, whatever the Host comparison would have said. A
    # same-SITE post is allowed deliberately: the sibling apps on this host
    # share the origin outright, so no header can tell their pages from ours.
    for site in ("same-origin", "same-site", "none"):
        client.cookies.clear()
        r = client.post("/login", data=creds, headers={"Origin": "https://elsewhere.example",
                                                       "Sec-Fetch-Site": site})
        assert r.status_code == 303, site


def test_an_operator_can_declare_the_public_host_for_older_browsers():
    """No Sec-Fetch-Site (Safari before 16.4) and a proxy that rewrites Host:
    `trusted_hosts` is the escape hatch, and it is the only one - an unlisted
    stranger is still refused."""
    from app.config import settings
    creds = {"username": "inomjon", "password": "123456"}
    before = settings.trusted_hosts
    settings.trusted_hosts = "aiscan.airi.uz, other.example"
    try:
        for host in ("https://aiscan.airi.uz", "https://other.example"):
            client.cookies.clear()
            assert client.post("/login", data=creds,
                               headers={"Origin": host}).status_code == 303, host
        client.cookies.clear()
        assert client.post("/login", data=creds,
                           headers={"Origin": "https://evil.example"}).status_code == 403
    finally:
        settings.trusted_hosts = before


def test_logout_accepts_get_and_post():
    """The nav signs out with a form (POST); bookmarks and old links use GET."""
    for r in (client.get("/logout"), client.post("/logout", headers=_admin_cookie())):
        assert r.status_code == 303 and r.headers["location"].endswith("/login")
        assert "Max-Age=0" in r.headers.get("set-cookie", "")


# ---- the cookie itself ---------------------------------------------------

def test_session_cookie_is_secure_only_behind_tls():
    """The LAN deployment is plain http, where a Secure cookie is silently
    never sent back and every login looks like it did not stick."""
    r = client.post("/login", data={"username": "inomjon", "password": "123456"})
    assert "secure" not in r.headers.get("set-cookie", "").lower()
    r = client.post("/login", data={"username": "inomjon", "password": "123456"},
                    headers={"X-Forwarded-Proto": "https"})
    assert "secure" in r.headers.get("set-cookie", "").lower()


def test_session_cookie_is_scoped_to_the_deployment_prefix(monkeypatch):
    """On the shared host the other projects under the same origin must never
    receive it, and a sign-out must also clear a cookie issued at "/" before
    the change - a browser keys cookies on path."""
    from fastapi.responses import Response
    from app.config import settings
    from app.api import auth as A
    monkeypatch.setattr(settings, "url_prefix", "/faceid", raising=False)
    resp = Response()
    A._set_session_cookie(resp, {"uid": 1, "username": "inomjon", "is_admin": True})
    assert "Path=/faceid;" in resp.headers["set-cookie"]
    cleared = A.logout().headers.getlist("set-cookie")
    assert any("Path=/faceid;" in h for h in cleared)
    assert any("Path=/;" in h for h in cleared)


# ---- login rate limiting -------------------------------------------------

def test_five_failures_lock_the_address_out_for_a_minute(monkeypatch):
    from app.api import auth as A
    monkeypatch.setattr(A, "_LOGIN_FAILURES", {})
    for _ in range(A.LOGIN_MAX_FAILURES):
        assert client.post("/login", data={"username": "inomjon",
                                           "password": "wrong"}).status_code == 401
    # Even the right password: it is the ADDRESS that is locked.
    r = client.post("/login", data={"username": "inomjon", "password": "123456"})
    assert r.status_code == 429
    assert "Retry-After" in r.headers
    assert COOKIE_NAME not in r.cookies
    # ...and only for a minute.
    assert 0 < A._login_locked_for("testclient") <= A.LOGIN_LOCKOUT_S
    assert A._login_locked_for("testclient", now=time.time() + A.LOGIN_LOCKOUT_S + 1) == 0


def test_failures_outside_the_window_do_not_add_up(monkeypatch):
    from app.api import auth as A
    monkeypatch.setattr(A, "_LOGIN_FAILURES", {})
    t = 1_000_000.0
    for i in range(A.LOGIN_MAX_FAILURES - 1):
        A._note_login_failure("10.0.0.9", now=t + i)
    A._note_login_failure("10.0.0.9", now=t + 2 * A.LOGIN_WINDOW_S)   # a fresh window
    assert A._login_locked_for("10.0.0.9", now=t + 2 * A.LOGIN_WINDOW_S) == 0


def test_a_successful_login_clears_the_count(monkeypatch):
    from app.api import auth as A
    monkeypatch.setattr(A, "_LOGIN_FAILURES", {})
    for _ in range(2):
        client.post("/login", data={"username": "inomjon", "password": "wrong"})
    assert "testclient" in A._LOGIN_FAILURES
    assert client.post("/login", data={"username": "inomjon",
                                       "password": "123456"}).status_code == 303
    assert "testclient" not in A._LOGIN_FAILURES


def test_the_client_address_is_the_proxys_last_forwarded_hop():
    """Behind the proxy every request comes from the proxy's address; keying
    on that would lock the whole site after five failures by anyone."""
    from fastapi import Request
    from app.api import auth as A

    def req(headers):
        return Request({"type": "http", "method": "POST", "path": "/login",
                        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
                        "query_string": b"", "server": ("testserver", 80),
                        "client": ("10.0.0.1", 1234), "scheme": "http"})
    assert A._client_ip(req({})) == "10.0.0.1"
    # The proxy appends the real client LAST; anything before it was written
    # by the client and can say whatever it likes.
    assert A._client_ip(req({"X-Forwarded-For": "1.2.3.4, 203.0.113.7"})) == "203.0.113.7"
