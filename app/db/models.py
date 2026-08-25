"""Schema for the v3 pipeline.

Two deliberate departures from the old Django-derived schema:

* Cameras carry a **role**.  Without it there is no way to express "this
  sighting means the person arrived", which was the central missing capability.
* An append-only **recognition_event** table sits underneath the daily summary.
  The summary is derived state; the events are the record.  The old code wrote
  only the summary, and wrote `check_out_time` nowhere at all.

All datetimes are timezone-aware UTC.  Business dates are derived from local
time at the configured day boundary, never from `utcnow().date()`.
"""
from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, Date, DateTime, Enum, Float, ForeignKey, Index,
    Integer, LargeBinary, String, Text, TypeDecorator, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UtcDateTime(TypeDecorator):
    """Always store UTC, always return an aware UTC datetime.

    SQLite has no native timezone support and silently drops tzinfo, so a column
    written as aware UTC reads back naive.  That produced two distinct failures
    in testing: a 09:02 check-in rendering as 04:02, and a naive/aware
    comparison TypeError.  Normalising in the type fixes both at the source
    instead of at every call site, and keeps the code portable to PostgreSQL
    where the underlying column really is timestamptz.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class Base(DeclarativeBase):
    pass


class CameraRole(str, enum.Enum):
    IN = "IN"
    OUT = "OUT"
    BOTH = "BOTH"


class PresenceStatus(str, enum.Enum):
    INSIDE = "INSIDE"
    OUTSIDE = "OUTSIDE"


class Employee(Base):
    __tablename__ = "employee"

    id = Column(Integer, primary_key=True)
    external_id = Column(String(64), unique=True, index=True)   # face_id_users user_id
    folder = Column(String(128))
    full_name = Column(String(255), nullable=False)
    department = Column(String(255), default="")
    position = Column(String(255), default="")
    phone = Column(String(64), default="")
    user_type = Column(String(64), default="")
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(UtcDateTime(), default=utcnow)
    updated_at = Column(UtcDateTime(), default=utcnow, onupdate=utcnow)

    embeddings = relationship("FaceEmbedding", back_populates="employee",
                              cascade="all, delete-orphan")


class FaceEmbedding(Base):
    """One row per enrolment image.

    Kept per-image rather than averaged into a centroid: matching on the max
    similarity across a person's embeddings handles pose and lighting spread far
    better than one mean vector, and it makes a bad enrolment photo visible
    instead of quietly poisoning the centroid.
    """
    __tablename__ = "face_embedding"

    id = Column(Integer, primary_key=True)
    employee_id = Column(Integer, ForeignKey("employee.id", ondelete="CASCADE"),
                         nullable=False, index=True)
    source_file = Column(String(255))
    vector = Column(LargeBinary, nullable=False)      # float32 512-d, L2-normalized
    dim = Column(Integer, default=512)
    model_name = Column(String(128))
    quality = Column(Float, default=0.0)              # aligner face score
    created_at = Column(UtcDateTime(), default=utcnow)

    employee = relationship("Employee", back_populates="embeddings")


class Camera(Base):
    __tablename__ = "camera"

    id = Column(Integer, primary_key=True)
    name = Column(String(128), nullable=False)
    role = Column(Enum(CameraRole), nullable=False, default=CameraRole.BOTH)
    rtsp_url = Column(Text, nullable=False)
    enabled = Column(Boolean, default=True)
    ip = Column(String(64), default="")
    notes = Column(Text, default="")

    # Direction-of-travel geometry.  `role` alone cannot decide check-in vs
    # check-out: both cameras overlook the same corridor, so a person leaving
    # can walk through the entrance camera's view.  A tripwire across the
    # walkway plus the depth trend gives the actual direction. Coordinates are
    # normalized 0..1 so they survive any resolution change.
    line_x1 = Column(Float, nullable=True)
    line_y1 = Column(Float, nullable=True)
    line_x2 = Column(Float, nullable=True)
    line_y2 = Column(Float, nullable=True)
    inside_side = Column(Integer, default=1)          # sign of the cross product that is "inside"
    depth_grows_inward = Column(Boolean, default=True)  # face grows when walking inward
    min_travel = Column(Float, default=0.06)

    created_at = Column(UtcDateTime(), default=utcnow)
    updated_at = Column(UtcDateTime(), default=utcnow, onupdate=utcnow)


class RecognitionEvent(Base):
    """Append-only log of every accepted recognition. The source of truth."""
    __tablename__ = "recognition_event"

    id = Column(Integer, primary_key=True)
    employee_id = Column(Integer, ForeignKey("employee.id", ondelete="SET NULL"),
                         nullable=True, index=True)
    camera_id = Column(Integer, ForeignKey("camera.id", ondelete="SET NULL"), nullable=True)
    role = Column(Enum(CameraRole), nullable=False)
    ts = Column(UtcDateTime(), nullable=False, index=True)
    business_date = Column(Date, nullable=False, index=True)
    score = Column(Float, default=0.0)
    margin = Column(Float, default=0.0)
    track_id = Column(Integer, default=-1)
    face_px = Column(Integer, default=0)
    votes = Column(String(32), default="")
    snapshot = Column(String(255), nullable=True)
    accepted = Column(Boolean, default=True)
    transition = Column(String(32), default="")   # "" | CHECK_IN | CHECK_OUT | RE_SIGHTING
    direction = Column(String(16), default="UNKNOWN")     # ENTER | EXIT | UNKNOWN
    direction_reason = Column(String(96), default="")     # why, for auditing

    __table_args__ = (
        Index("ix_event_emp_date", "employee_id", "business_date"),
        Index("ix_event_date_ts", "business_date", "ts"),
    )


class DailyAttendance(Base):
    """Derived summary. Rebuildable from recognition_event at any time."""
    __tablename__ = "daily_attendance"

    id = Column(Integer, primary_key=True)
    employee_id = Column(Integer, ForeignKey("employee.id", ondelete="CASCADE"),
                         nullable=False, index=True)
    business_date = Column(Date, nullable=False, index=True)

    check_in_time = Column(UtcDateTime(), nullable=True)
    check_out_time = Column(UtcDateTime(), nullable=True)
    worked_seconds = Column(Integer, default=0)      # accumulated across in/out pairs
    presence = Column(Enum(PresenceStatus), default=PresenceStatus.OUTSIDE)
    entered_at = Column(UtcDateTime(), nullable=True)   # open interval start

    event_count = Column(Integer, default=0)
    status = Column(String(32), default="PRESENT")   # PRESENT | NO_CHECKOUT | ABSENT
    check_in_snapshot = Column(String(255), nullable=True)
    check_out_snapshot = Column(String(255), nullable=True)
    created_at = Column(UtcDateTime(), default=utcnow)
    updated_at = Column(UtcDateTime(), default=utcnow, onupdate=utcnow)

    employee = relationship("Employee")

    __table_args__ = (
        UniqueConstraint("employee_id", "business_date", name="uq_daily_emp_date"),
    )


class UnknownSighting(Base):
    """One row per unknown *track*, not per detection.

    The old code inserted a row and wrote a JPEG on every unrecognized frame,
    so one visitor standing in view produced a stream of both.
    """
    __tablename__ = "unknown_sighting"

    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey("camera.id", ondelete="SET NULL"), nullable=True)
    track_id = Column(Integer, default=-1)
    first_seen = Column(UtcDateTime(), default=utcnow, index=True)
    last_seen = Column(UtcDateTime(), default=utcnow)
    business_date = Column(Date, nullable=False, index=True)
    frames = Column(Integer, default=1)
    best_score = Column(Float, default=0.0)      # best match that still missed
    nearest_employee_id = Column(Integer, nullable=True)
    vector = Column(LargeBinary, nullable=True)
    snapshot = Column(String(255), nullable=True)
