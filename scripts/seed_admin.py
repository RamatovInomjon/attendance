#!/usr/bin/env python3
"""First-run setup for console access: a signing key, then the default admin.

Run once per deployment:

    python scripts/seed_admin.py

Idempotent. It writes a `secret_key` into `.env` only if one is missing or is
still the shipped placeholder, and creates the default admin only when no
account exists at all - so re-running it never resets a changed password or
resurrects a deleted account.

Also usable to add accounts by hand, which is the way back in if every admin
password is lost:

    python scripts/seed_admin.py --username dilshod --password 's3cret'
    python scripts/seed_admin.py --username inomjon --reset-password 'new-one'
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PLACEHOLDER = "change-me-in-production"


def ensure_secret_key() -> bool:
    """Put a real signing key in .env. Returns True if it wrote one.

    Sessions are signed with this. Left at the placeholder, anyone who has read
    the source could forge a cookie for any user, so app/core/security.py
    refuses to sign at all until it is set.
    """
    from app.core.security import new_secret_key

    env_path = ROOT / ".env"
    lines = env_path.read_text().splitlines() if env_path.exists() else []
    for i, line in enumerate(lines):
        if line.strip().startswith("secret_key="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            if value and value != PLACEHOLDER:
                print("  secret_key: already set, left alone")
                return False
            lines[i] = f"secret_key={new_secret_key()}"
            env_path.write_text("\n".join(lines) + "\n")
            print("  secret_key: replaced the placeholder with a random key")
            return True

    lines.append(f"secret_key={new_secret_key()}")
    env_path.write_text("\n".join(lines) + "\n")
    print(f"  secret_key: written to {env_path}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--username")
    ap.add_argument("--password")
    ap.add_argument("--reset-password")
    ap.add_argument("--full-name", default="")
    ap.add_argument("--no-admin", action="store_true",
                    help="create an operator rather than an admin")
    args = ap.parse_args()

    ensure_secret_key()

    # Import only after .env exists, so settings picks the key up.
    from app.db.session import init_db
    from app.services import auth as auth_svc

    init_db()

    if args.reset_password:
        if not args.username:
            print("  --reset-password needs --username", file=sys.stderr)
            return 2
        auth_svc.set_password(args.username, args.reset_password)
        print(f"  password reset for {args.username!r}")
        return 0

    if args.username:
        if not args.password:
            print("  --username needs --password", file=sys.stderr)
            return 2
        try:
            u = auth_svc.create_user(args.username, args.password,
                                     is_admin=not args.no_admin,
                                     full_name=args.full_name)
            print(f"  created {u['username']!r} (admin={u['is_admin']})")
        except auth_svc.AuthError as e:
            print(f"  {e}", file=sys.stderr)
            return 1
        return 0

    if auth_svc.ensure_default_admin():
        print(f"  created default admin {auth_svc.DEFAULT_USERNAME!r} "
              f"with password {auth_svc.DEFAULT_PASSWORD!r}")
        print("  CHANGE THIS PASSWORD at /users after first sign-in.")
    else:
        n = auth_svc.user_count()
        print(f"  {n} account(s) already exist - nothing to do")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
