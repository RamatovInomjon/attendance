"""Keep the test suite off the live database.

`session_scope()` binds to `data/ematsy.db` — the real attendance database. A
test that creates a user or resets a password therefore edits production data,
and one did: an early version of `test_operator_cannot_change_another_password`
sent an admin cookie by accident, the request succeeded instead of returning
403, and it rewrote the real admin's password. Every later login test then
failed for a reason that had nothing to do with the code under test.

So the whole suite runs against a fresh in-memory database. `StaticPool` keeps
one connection alive for its lifetime, which is what makes `sqlite://` survive
across the many short-lived sessions `session_scope()` opens.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The signing key is read at import time by app.core.security. Set a throwaway
# one so tests never depend on a real .env and never sign with the placeholder.
import os
os.environ.setdefault("secret_key", "test-only-key-not-used-in-production-0123456789")


@pytest.fixture(scope="session", autouse=True)
def isolated_database():
    from app.db import session as db_session
    from app.db.models import Base

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, class_=Session,
                           expire_on_commit=False, future=True)

    real_engine, real_factory = db_session.engine, db_session.SessionLocal
    db_session.engine, db_session.SessionLocal = engine, factory
    try:
        yield engine
    finally:
        db_session.engine, db_session.SessionLocal = real_engine, real_factory
        engine.dispose()


@pytest.fixture(scope="session", autouse=True)
def seed_accounts(isolated_database):
    """A known admin and a known operator, so authorization tests have both
    sides to check without reaching for the real accounts."""
    from app.services import auth as auth_svc
    auth_svc.create_user("inomjon", "123456", is_admin=True, full_name="Inomjon")
    auth_svc.create_user("operator", "operator-pw", is_admin=False, full_name="Operator")
    return {"admin": "inomjon", "operator": "operator"}
