"""Operator accounts: authentication, creation, and the first-run admin.

The console shows attendance movements and face crops of identified people.
Before this existed every page and every JSON endpoint answered anyone who
knew the URL, so `/login` was decoration - the data was one guessed path away.
"""
from __future__ import annotations

import logging
import re

from sqlalchemy import func, select

from app.core.security import hash_password, needs_rehash, verify_password
from app.db.models import User
from app.db.session import session_scope
from app.db.models import utcnow

log = logging.getLogger(__name__)

USERNAME_RE = re.compile(r"^[a-zA-Z0-9._-]{3,64}$")
MIN_PASSWORD_LEN = 6

DEFAULT_USERNAME = "inomjon"
DEFAULT_PASSWORD = "123456"


class AuthError(ValueError):
    """Rejected input, safe to show a human."""


def normalize_username(raw: str) -> str:
    return (raw or "").strip().lower()


def validate_new_user(username: str, password: str) -> tuple[str, str]:
    u = normalize_username(username)
    if not USERNAME_RE.match(u):
        raise AuthError(
            "Username must be 3-64 characters, letters, digits, dot, dash or underscore."
        )
    if len(password or "") < MIN_PASSWORD_LEN:
        raise AuthError(f"Password must be at least {MIN_PASSWORD_LEN} characters.")
    return u, password


def authenticate(username: str, password: str) -> dict | None:
    """Return a small dict on success, None on any failure.

    One indistinguishable failure for 'no such user', 'wrong password' and
    'account disabled': telling them apart tells an attacker which usernames
    are real. The dummy verify on the no-user path keeps the response time
    flat so absence cannot be timed either.
    """
    u = normalize_username(username)
    with session_scope() as s:
        row = s.execute(select(User).where(User.username == u)).scalar_one_or_none()
        if row is None:
            verify_password(password or "x", _DUMMY_HASH)
            return None
        if not row.is_active or not verify_password(password or "", row.password_hash):
            return None

        # Opportunistic upgrade: if the stored cost is below what we now use,
        # this login is the only moment the plaintext is available to redo it.
        if needs_rehash(row.password_hash):
            row.password_hash = hash_password(password)
        row.last_login_at = utcnow()
        return {"uid": row.id, "username": row.username,
                "is_admin": bool(row.is_admin), "full_name": row.full_name or ""}


def create_user(username: str, password: str, *, is_admin: bool = False,
                full_name: str = "") -> dict:
    u, pw = validate_new_user(username, password)
    with session_scope() as s:
        exists = s.execute(
            select(func.count()).select_from(User).where(User.username == u)
        ).scalar_one()
        if exists:
            raise AuthError(f"User {u!r} already exists.")
        row = User(username=u, password_hash=hash_password(pw),
                   is_admin=bool(is_admin), full_name=full_name.strip(), is_active=True)
        s.add(row)
        s.flush()
        log.info("created user %r (admin=%s)", u, bool(is_admin))
        return {"uid": row.id, "username": row.username, "is_admin": row.is_admin}


def set_password(username: str, password: str) -> None:
    u = normalize_username(username)
    if len(password or "") < MIN_PASSWORD_LEN:
        raise AuthError(f"Password must be at least {MIN_PASSWORD_LEN} characters.")
    with session_scope() as s:
        row = s.execute(select(User).where(User.username == u)).scalar_one_or_none()
        if row is None:
            raise AuthError(f"No such user: {u}")
        row.password_hash = hash_password(password)


def set_active(username: str, active: bool) -> None:
    """Disable rather than delete, so old audit rows keep a name to point at.

    Refuses to disable the last active admin - an unreachable console is worse
    than a stale account.
    """
    u = normalize_username(username)
    with session_scope() as s:
        row = s.execute(select(User).where(User.username == u)).scalar_one_or_none()
        if row is None:
            raise AuthError(f"No such user: {u}")
        if not active and row.is_admin:
            others = s.execute(
                select(func.count()).select_from(User).where(
                    User.is_admin.is_(True), User.is_active.is_(True), User.id != row.id)
            ).scalar_one()
            if not others:
                raise AuthError("Cannot disable the last active admin.")
        row.is_active = bool(active)


def list_users() -> list[dict]:
    with session_scope() as s:
        rows = s.execute(select(User).order_by(User.username)).scalars().all()
        return [{"id": r.id, "username": r.username, "full_name": r.full_name or "",
                 "is_admin": bool(r.is_admin), "is_active": bool(r.is_active),
                 "created_at": r.created_at, "last_login_at": r.last_login_at}
                for r in rows]


def user_count() -> int:
    with session_scope() as s:
        return s.execute(select(func.count()).select_from(User)).scalar_one()


def ensure_default_admin() -> bool:
    """Create the first admin if the table is empty. Returns True if created.

    Runs at startup so a fresh deployment is reachable without a manual step,
    and does nothing once any account exists - it must never resurrect a
    deleted default or reset a changed password.
    """
    if user_count():
        return False
    create_user(DEFAULT_USERNAME, DEFAULT_PASSWORD, is_admin=True, full_name="Inomjon")
    log.warning(
        "created default admin %r with the default password - change it at "
        "/users", DEFAULT_USERNAME)
    return True


# A real hash of a random string, compared against when the username does not
# exist so that path costs the same ~140 ms as a real verify.
_DUMMY_HASH = hash_password("not-a-real-password-placeholder")
