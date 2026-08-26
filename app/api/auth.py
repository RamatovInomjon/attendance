"""Sign-in, sign-out, user administration, and the gate in front of everything.

The gate is middleware rather than a per-route dependency deliberately. There
are ~15 page routes, several JSON endpoints, an MJPEG video feed and a
WebSocket; a dependency has to be remembered on each one, and the failure mode
of forgetting is a silently public endpoint. Middleware is deny-by-default:
a new route is protected the moment it is added, and anything public has to be
named in ONE list below, where it can be reviewed.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from app.core.security import COOKIE_NAME, MAX_AGE_S, read_session, sign_session
from app.services import auth as auth_svc

log = logging.getLogger(__name__)

router = APIRouter()

# Exact paths served without a session. Everything else requires one.
_PUBLIC_NAMES = ("/login", "/logout", "/favicon.ico", "/health")
_PUBLIC_PREFIX_NAMES = ("/static/",)


def _is_public(path: str) -> bool:
    """Public paths, resolved against the deployment prefix.

    These are compared against the RAW request path, which under a sub-path
    deployment arrives as /faceid/login - so the prefix has to be applied here
    too, or the login page itself would demand a session and loop forever.
    """
    from app.config import settings
    p = settings.url_prefix.rstrip("/")
    if path in {f"{p}{n}" for n in _PUBLIC_NAMES}:
        return True
    return path.startswith(tuple(f"{p}{n}" for n in _PUBLIC_PREFIX_NAMES))


def _p(path: str) -> str:
    """Prefix an absolute app path for the current deployment."""
    from app.config import settings
    return f"{settings.url_prefix.rstrip('/')}{path}"


def current_user(request: Request) -> dict | None:
    return getattr(request.state, "user", None)


async def auth_middleware(request: Request, call_next):
    """Attach the signed-in user, or turn the request away.

    HTML requests get a redirect to the login form with the original path in
    `next`; anything expecting JSON gets a 401, because redirecting an XHR to
    an HTML login page produces a confusing parse error rather than a clear
    'you are logged out'.
    """
    path = request.url.path
    session = read_session(request.cookies.get(COOKIE_NAME))
    request.state.user = session

    if session or _is_public(path):
        return await call_next(request)

    accept = request.headers.get("accept", "")
    wants_json = (
        path.startswith(_p("/api/"))
        or "application/json" in accept
        or request.headers.get("x-requested-with") == "XMLHttpRequest"
    )
    if wants_json:
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)

    nxt = request.url.path
    if request.url.query:
        nxt = f"{nxt}?{request.url.query}"
    return RedirectResponse(_p(f"/login?next={_quote(nxt)}"), status_code=303)


def _quote(value: str) -> str:
    from urllib.parse import quote
    return quote(value, safe="")


def _safe_next(raw: str | None) -> str:
    """Only same-site absolute paths. A bare `next` would otherwise let a
    crafted link bounce a freshly authenticated operator to another host."""
    if not raw or not raw.startswith("/") or raw.startswith("//"):
        return _p("/")
    return raw


def _set_session_cookie(response, user: dict) -> None:
    response.set_cookie(
        COOKIE_NAME,
        sign_session(user_id=user["uid"], username=user["username"],
                     is_admin=user["is_admin"]),
        max_age=MAX_AGE_S,
        httponly=True,      # not readable from JS, so XSS cannot lift the session
        samesite="lax",     # blocks cross-site form posts, keeps normal links working
        path="/",
    )


# ---- routes ------------------------------------------------------------

def _render_login(request: Request, *, error: str = "", next_url: str = "/",
                  status: int = 200) -> HTMLResponse:
    from app.api.pages import render
    resp = render("auth/login.html", request=request, current_view="auth:login",
                  error=error, next_url=next_url)
    return HTMLResponse(resp.body, status_code=status)


@router.get("/login", response_class=HTMLResponse)
def login_form(request: Request, next: str = "/"):
    if current_user(request):
        return RedirectResponse(_safe_next(next), status_code=303)
    return _render_login(request, next_url=_safe_next(next))


@router.post("/login")
async def login_submit(request: Request,
                       username: str = Form(""), password: str = Form(""),
                       next: str = Form("/")):
    target = _safe_next(next)
    # PBKDF2 verify is ~140 ms of CPU; off the event loop so one login cannot
    # stall frame delivery to every other viewer.
    from starlette.concurrency import run_in_threadpool
    user = await run_in_threadpool(auth_svc.authenticate, username, password)
    if not user:
        log.warning("failed login for %r from %s", username,
                    request.client.host if request.client else "?")
        return _render_login(request, error="Incorrect username or password.",
                             next_url=target, status=401)
    resp = RedirectResponse(target, status_code=303)
    _set_session_cookie(resp, user)
    log.info("login: %s", user["username"])
    return resp


@router.get("/logout")
@router.post("/logout")
def logout():
    resp = RedirectResponse(_p("/login"), status_code=303)
    resp.delete_cookie(COOKIE_NAME, path="/")
    return resp


@router.get("/users", response_class=HTMLResponse)
def users_page(request: Request, error: str = "", created: str = ""):
    from app.api.pages import render
    me = current_user(request)
    if not me or not me.get("adm"):
        return HTMLResponse(
            render("auth/forbidden.html", request=request,
                   current_view="auth:users").body, status_code=403)
    return render("auth/users.html", request=request, current_view="auth:users",
                  users=auth_svc.list_users(), me=me, error=error, created=created)


@router.post("/users/add")
def users_add(request: Request, username: str = Form(""), password: str = Form(""),
              full_name: str = Form(""), is_admin: str = Form("")):
    me = current_user(request)
    if not me or not me.get("adm"):
        return JSONResponse({"detail": "Admin only"}, status_code=403)
    try:
        # Checkbox posts "on" when ticked and is absent otherwise; the user
        # asked for new accounts to be admins by default, so an explicit "0"
        # from the form is the only thing that makes a non-admin.
        admin = str(is_admin).lower() not in {"0", "false", "no"}
        auth_svc.create_user(username, password, is_admin=admin, full_name=full_name)
    except auth_svc.AuthError as e:
        return RedirectResponse(_p(f"/users?error={_quote(str(e))}"), status_code=303)
    return RedirectResponse(_p(f"/users?created={_quote(username.strip().lower())}"),
                            status_code=303)


@router.post("/users/toggle")
def users_toggle(request: Request, username: str = Form(""), active: str = Form("1")):
    me = current_user(request)
    if not me or not me.get("adm"):
        return JSONResponse({"detail": "Admin only"}, status_code=403)
    try:
        auth_svc.set_active(username, str(active) not in {"0", "false", "no"})
    except auth_svc.AuthError as e:
        return RedirectResponse(_p(f"/users?error={_quote(str(e))}"), status_code=303)
    return RedirectResponse(_p("/users"), status_code=303)


@router.post("/users/password")
def users_password(request: Request, username: str = Form(""), password: str = Form("")):
    me = current_user(request)
    if not me:
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)
    target = auth_svc.normalize_username(username)
    # An admin may reset anyone; everyone else only themselves.
    if not me.get("adm") and target != me.get("u"):
        return JSONResponse({"detail": "You may only change your own password"},
                            status_code=403)
    try:
        auth_svc.set_password(target, password)
    except auth_svc.AuthError as e:
        return RedirectResponse(f"/users?error={_quote(str(e))}", status_code=303)
    return RedirectResponse(_p("/users?created=password-changed"), status_code=303)
