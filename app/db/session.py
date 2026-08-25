"""Synchronous SQLAlchemy session factory.

Sync, deliberately.  The previous codebase drove a single shared SQLite
connection (`poolclass=StaticPool`) from a thread pool where every task called
`asyncio.new_event_loop()`, ran one coroutine and closed the loop — while the
uvicorn loop used the same connection for page requests.  Sync sessions with a
real pool, WAL, and a busy timeout are simpler and actually correct; FastAPI
runs sync route handlers in its threadpool without complaint.
"""
from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.db.models import Base

settings.data_dir.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    settings.database_url_sync,
    echo=False,
    future=True,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False, "timeout": 30},
)


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_conn, _rec):
    """WAL lets readers and the writer coexist; busy_timeout absorbs the rest."""
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA busy_timeout=30000")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


SessionLocal = sessionmaker(bind=engine, class_=Session, expire_on_commit=False, future=True)


def init_db() -> None:
    """Create every table.  Called at startup — the old code defined this and
    never called it, which is why the app could not start."""
    Base.metadata.create_all(engine)


@contextmanager
def session_scope():
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
