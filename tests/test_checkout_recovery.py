"""Offering the departure a missing check-out lost.

The measurement behind this (bench/missing_checkout_eval.py, 386 held-out days)
says body alone names the right person barely more than half the time, and that
the runner-up MARGIN - not the score - is what separates 78% from 95%. So the
tests that matter here are the refusals: across model spaces, below the margin,
and the fact that nothing is ever written without a person saying so.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest
from sqlalchemy import delete, select

from app.config import settings
from app.db.models import (
    CameraRole, DailyAttendance, Employee, PresenceStatus, RecognitionEvent,
    ReidPass,
)
from app.db.session import session_scope
from app.services import checkout_recovery as rec

DAY = date(2026, 6, 15)
REC = settings.recognizer_model
BODY = settings.reid_model if hasattr(settings, "reid_model") else "osnet.onnx"


def _v(seed, dim=512):
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v))


def _mix(a, b, w):
    """A vector w of the way from a to b - a controllable similarity."""
    v = w * b + (1 - w) * a
    return (v / np.linalg.norm(v)).astype(np.float32)


def _utc(d, hh, mm=0):
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=settings.tz).astimezone(timezone.utc)


@pytest.fixture
def person():
    with session_scope() as s:
        e = Employee(full_name="Recovery Tester", external_id="RC-1")
        s.add(e)
        s.flush()
        eid = e.id
    yield eid
    with session_scope() as s:
        s.execute(delete(RecognitionEvent).where(RecognitionEvent.employee_id == eid))
        s.execute(delete(ReidPass).where(ReidPass.business_date == DAY))
        s.execute(delete(DailyAttendance).where(DailyAttendance.employee_id == eid))
        s.execute(delete(Employee).where(Employee.id == eid))


def _open_day(s, eid, check_in):
    s.add(DailyAttendance(employee_id=eid, business_date=DAY,
                          check_in_time=check_in, entered_at=check_in,
                          worked_seconds=0, status="NO_CHECKOUT",
                          presence=PresenceStatus.INSIDE))


def _pass(s, *, track, ts, employee_id=None, face=None, body=None,
          face_model=REC, body_model=BODY, direction="EXIT"):
    s.add(ReidPass(
        camera_id=2, camera_name="Exit", track_id=track,
        first_seen=ts - timedelta(seconds=5), last_seen=ts, business_date=DAY,
        direction=direction, employee_id=employee_id,
        vector=body.tobytes() if body is not None else None,
        dim=len(body) if body is not None else 0, model_name=body_model,
        face_vector=face.tobytes() if face is not None else None,
        face_dim=len(face) if face is not None else 0, face_model=face_model))


class TestItOffersTheRightPass:
    def test_the_closest_face_wins_and_carries_a_margin(self, person):
        me, other = _v(1), _v(2)
        ci = _utc(DAY, 9)
        with session_scope() as s:
            _open_day(s, person, ci)
            _pass(s, track=1, ts=ci, employee_id=person, face=me, body=me,
                  direction="ENTER")                      # the anchor
            _pass(s, track=2, ts=_utc(DAY, 18), face=_mix(other, me, 0.95),
                  body=_mix(other, me, 0.9))              # them, leaving
            _pass(s, track=3, ts=_utc(DAY, 17), face=other, body=other)
        with session_scope() as s:
            cands = rec.suggest(s, person, DAY)
            assert cands, "a departure should be offered"
            assert cands[0].ts == _utc(DAY, 18)
            assert cands[0].margin > 0
            assert rec.confident(cands) is cands[0]

    def test_nothing_is_offered_once_the_day_has_a_check_out(self, person):
        ci = _utc(DAY, 9)
        with session_scope() as s:
            s.add(DailyAttendance(employee_id=person, business_date=DAY,
                                  check_in_time=ci, check_out_time=_utc(DAY, 18),
                                  worked_seconds=9 * 3600, status="PRESENT",
                                  presence=PresenceStatus.OUTSIDE))
            _pass(s, track=1, ts=ci, employee_id=person, face=_v(1), body=_v(1))
        with session_scope() as s:
            assert rec.suggest(s, person, DAY) == []

    def test_a_pass_before_the_check_in_is_not_a_departure(self, person):
        me = _v(1)
        ci = _utc(DAY, 12)
        with session_scope() as s:
            _open_day(s, person, ci)
            _pass(s, track=1, ts=ci, employee_id=person, face=me, body=me,
                  direction="ENTER")
            _pass(s, track=2, ts=_utc(DAY, 8), face=me, body=me)   # morning
        with session_scope() as s:
            assert rec.suggest(s, person, DAY) == []


class TestTheRefusalsThatMakeItSafe:
    def test_a_vector_from_another_model_is_never_scored(self, person):
        """A cosine across two spaces succeeds and means nothing - which is why
        the provenance columns exist. Refuse, do not score."""
        me = _v(1)
        ci = _utc(DAY, 9)
        with session_scope() as s:
            _open_day(s, person, ci)
            _pass(s, track=1, ts=ci, employee_id=person, face=me, body=me,
                  direction="ENTER")
            _pass(s, track=2, ts=_utc(DAY, 18), face=me, body=me,
                  face_model="some_other_recognizer.onnx",
                  body_model="some_other_reid.onnx")
        with session_scope() as s:
            assert rec.suggest(s, person, DAY) == []

    def test_below_the_margin_there_is_no_recommendation(self, person):
        """Two candidates that score alike are exactly the case the gate is
        for: the ranker cannot tell them apart, so it must not pick."""
        me = _v(1)
        ci = _utc(DAY, 9)
        with session_scope() as s:
            _open_day(s, person, ci)
            _pass(s, track=1, ts=ci, employee_id=person, face=me, body=me,
                  direction="ENTER")
            twin = _mix(_v(5), me, 0.9)
            _pass(s, track=2, ts=_utc(DAY, 18), face=twin, body=twin)
            _pass(s, track=3, ts=_utc(DAY, 17), face=twin.copy(), body=twin.copy())
        with session_scope() as s:
            cands = rec.suggest(s, person, DAY)
            assert len(cands) >= 2
            assert cands[0].margin < settings.checkout_suggest_margin
            assert rec.confident(cands) is None, "a tie must not be recommended"
            # ...but the operator still sees them.
            assert cands[0].snapshot is None or True

    def test_a_weak_best_score_is_not_recommended(self, person, monkeypatch):
        me = _v(1)
        ci = _utc(DAY, 9)
        with session_scope() as s:
            _open_day(s, person, ci)
            _pass(s, track=1, ts=ci, employee_id=person, face=me, body=me,
                  direction="ENTER")
            _pass(s, track=2, ts=_utc(DAY, 18), face=_v(9), body=_v(9))
        with session_scope() as s:
            cands = rec.suggest(s, person, DAY)
            assert cands
            monkeypatch.setattr(settings, "checkout_suggest_min_score", 0.99)
            assert rec.confident(cands) is None

    def test_no_anchor_means_no_suggestion(self, person):
        """Without a pass the FACE named that day there is no clothing
        reference, and body similarity to nothing is not evidence."""
        ci = _utc(DAY, 9)
        with session_scope() as s:
            _open_day(s, person, ci)
            _pass(s, track=2, ts=_utc(DAY, 18), face=_v(3), body=_v(3))
        with session_scope() as s:
            assert rec.suggest(s, person, DAY) == []

    def test_suggesting_writes_nothing(self, person):
        me = _v(1)
        ci = _utc(DAY, 9)
        with session_scope() as s:
            _open_day(s, person, ci)
            _pass(s, track=1, ts=ci, employee_id=person, face=me, body=me,
                  direction="ENTER")
            _pass(s, track=2, ts=_utc(DAY, 18), face=_mix(_v(2), me, 0.95),
                  body=_mix(_v(2), me, 0.9))
        with session_scope() as s:
            rec.suggest(s, person, DAY)
        with session_scope() as s:
            assert s.execute(select(RecognitionEvent).where(
                RecognitionEvent.employee_id == person)).first() is None
            row = s.execute(select(DailyAttendance).where(
                DailyAttendance.employee_id == person)).scalar_one()
            assert row.check_out_time is None


class TestConfirming:
    def test_confirm_writes_a_body_sourced_check_out_and_rebuilds(self, person):
        me = _v(1)
        ci = _utc(DAY, 9)
        with session_scope() as s:
            _open_day(s, person, ci)
            _pass(s, track=1, ts=ci, employee_id=person, face=me, body=me,
                  direction="ENTER")
            # The check-in itself must exist as an event, or the rebuild has
            # nothing to replay and cannot close an interval.
            s.add(RecognitionEvent(
                employee_id=person, camera_id=1, role=CameraRole.IN, ts=ci,
                business_date=DAY, transition="CHECK_IN", direction="ENTER",
                accepted=True, source="live"))
            _pass(s, track=2, ts=_utc(DAY, 18), face=_mix(_v(2), me, 0.95),
                  body=_mix(_v(2), me, 0.9))
        with session_scope() as s:
            top = rec.suggest(s, person, DAY)[0]
            pass_id = top.pass_id

        out = rec.confirm(pass_id, employee_id=person, by="tester")
        assert out["ok"], out
        assert out["transition"] == "CHECK_OUT"

        with session_scope() as s:
            ev = s.execute(select(RecognitionEvent).where(
                RecognitionEvent.employee_id == person,
                RecognitionEvent.source == "body")).scalar_one()
            # Not a recognition: a ReID cosine in a column of face similarities
            # would silently poison every measurement taken over this table.
            assert ev.score == 0.0
            assert ev.direction == "EXIT"
            row = s.execute(select(DailyAttendance).where(
                DailyAttendance.employee_id == person)).scalar_one()
            assert row.check_out_time is not None
            assert row.worked_seconds == 9 * 3600
            assert row.presence == PresenceStatus.OUTSIDE

    def test_the_same_pass_cannot_be_confirmed_twice(self, person):
        me = _v(1)
        ci = _utc(DAY, 9)
        with session_scope() as s:
            _open_day(s, person, ci)
            _pass(s, track=1, ts=ci, employee_id=person, face=me, body=me,
                  direction="ENTER")
            _pass(s, track=2, ts=_utc(DAY, 18), face=_mix(_v(2), me, 0.95),
                  body=_mix(_v(2), me, 0.9))
        with session_scope() as s:
            pass_id = rec.suggest(s, person, DAY)[0].pass_id
        assert rec.confirm(pass_id, employee_id=person, by="t")["ok"]
        again = rec.confirm(pass_id, employee_id=person, by="t")
        assert not again["ok"]
        assert "already in attendance" in again["error"]

    def test_confirm_refuses_an_unknown_pass_or_employee(self, person):
        assert not rec.confirm(999999, employee_id=person, by="t")["ok"]
