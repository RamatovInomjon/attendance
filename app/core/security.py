"""Password hashing and signed session cookies.

Standard library only, deliberately. Adding `passlib`/`bcrypt`/`itsdangerous`
means another pip install on every deployment, and pip installs on this host
have four times dragged a CPU-only `onnxruntime` back in on top of
`onnxruntime-gpu`, silently costing ~10x inference speed. PBKDF2-HMAC-SHA256 is
in `hashlib`, and an HMAC-signed cookie is twenty lines; neither is worth a
dependency.

Two separate concerns live here:

* **Passwords at rest** — PBKDF2-HMAC-SHA256, per-user random salt, stored as a
  self-describing string so the iteration count can be raised later without
  invalidating existing hashes.
* **Who is logged in** — an HMAC-signed, expiring cookie payload. No server-side
  session store: there is one operator machine, and a signed cookie needs no
  shared state and survives a service restart.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time

from app.config import settings

# ---- passwords ---------------------------------------------------------

# OWASP's floor for PBKDF2-HMAC-SHA256 is 600k. Measured on this host at
# ~140 ms per verify: imperceptible on a login form, and expensive enough in
# bulk to make offline guessing against a stolen database slow.
PBKDF2_ITERATIONS = 600_000
_ALGO = "pbkdf2_sha256"


def hash_password(password: str, *, iterations: int = PBKDF2_ITERATIONS) -> str:
    """-> 'pbkdf2_sha256$<iterations>$<salt_b64>$<hash_b64>'."""
    if not password:
        raise ValueError("password must not be empty")
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{_ALGO}${iterations}${_b64(salt)}${_b64(dk)}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check. False for anything malformed rather than raising:
    a corrupt hash must fail closed, not 500 the login page."""
    if not password or not stored:
        return False
    try:
        algo, iters, salt_b64, hash_b64 = stored.split("$")
        if algo != _ALGO:
            return False
        expected = _unb64(hash_b64)
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), _unb64(salt_b64), int(iters),
            dklen=len(expected),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(dk, expected)


def needs_rehash(stored: str, *, iterations: int = PBKDF2_ITERATIONS) -> bool:
    """True when a hash was made with a weaker cost than we now use."""
    try:
        algo, iters, _, _ = stored.split("$")
    except ValueError:
        return True
    return algo != _ALGO or int(iters) < iterations


# ---- session cookie ----------------------------------------------------

COOKIE_NAME = "ematsy_session"
MAX_AGE_S = 12 * 3600          # one working day; re-login next morning


def _secret() -> bytes:
    """The signing key.

    A default key would mean anyone who has read the source can forge a session
    cookie for any user, so refuse to sign with the placeholder. `scripts/
    seed_admin.py` writes a real key into .env on first run.
    """
    key = (settings.secret_key or "").strip()
    if not key or key == "change-me-in-production":
        raise RuntimeError(
            "settings.secret_key is unset or still the placeholder. Session "
            "cookies would be forgeable. Run scripts/seed_admin.py, or set "
            "secret_key in .env to a random value."
        )
    return key.encode("utf-8")


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(txt: str) -> bytes:
    return base64.urlsafe_b64decode(txt + "=" * (-len(txt) % 4))


def sign_session(*, user_id: int, username: str, is_admin: bool) -> str:
    payload = {"uid": int(user_id), "u": username,
               "adm": bool(is_admin), "iat": int(time.time())}
    body = _b64(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = hmac.new(_secret(), body.encode("ascii"), hashlib.sha256).digest()
    return f"{body}.{_b64(sig)}"


def read_session(token: str | None) -> dict | None:
    """Verify signature and age. None means 'not logged in' for every failure
    mode - tampered, truncated, expired, or simply absent."""
    if not token or "." not in token:
        return None
    body, _, sig = token.partition(".")
    try:
        expected = hmac.new(_secret(), body.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(_unb64(sig), expected):
            return None
        payload = json.loads(_unb64(body))
    except Exception:
        return None
    if not isinstance(payload, dict) or "uid" not in payload:
        return None
    if time.time() - float(payload.get("iat", 0)) > MAX_AGE_S:
        return None
    return payload


def new_secret_key() -> str:
    return secrets.token_urlsafe(48)
