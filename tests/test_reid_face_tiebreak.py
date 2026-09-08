"""Letting the face break a cross-camera tie the body could not.

The runner-up margin in `ReidWorker._match` is expensive: measured on this
corridor's 281-pass corpus it drops correct matches from 55 to 29 while the
wrong count stays at zero either way. Those are not impostors being caught -
they are passes where two people's CLOTHING scored alike and the body axis had
nothing left to say.

The face is asked only in that case. What these tests pin is that it stays a
TIE-BREAK and never becomes a second way in:

* every candidate must still clear the body threshold, so the calibrated accept
  decision is untouched;
* a clear body winner is decided by the body, face or no face;
* "no face" is not a low score, it is no evidence - a tie between a pass with a
  face and a pass without one is not a tie the face may break;
* two candidates whose faces are equally close is still a tie, and abstaining
  is the answer.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pytest

from app.config import settings
from app.db.models import ReidPass
from app.db.session import session_scope
from app.services.reid_worker import ReidWorker

DAY = date(2026, 9, 8)
NOW = datetime(2026, 9, 8, 9, 0, tzinfo=timezone.utc)
D = 512


def _unit(seed, d=D):
    r = np.random.default_rng(seed)
    v = r.standard_normal(d).astype(np.float32)
    return v / np.linalg.norm(v)


def _near(v, seed, amount):
    """A vector at a controlled cosine from `v`.

    The scale matters in 512 dimensions: a unit vector's components are about
    1/sqrt(512), so noise of 0.02 per component lands near cosine 0.9 and noise
    of 0.5 destroys the direction entirely.
    """
    r = np.random.default_rng(seed)
    x = v + r.standard_normal(len(v)).astype(np.float32) * amount
    return x / np.linalg.norm(x)


BODY = _unit(1)
FACE = _unit(2)


def _worker():
    return ReidWorker.__new__(ReidWorker)


def _run(s, *, a_face, b_face, query_face, a_amt=0.020, b_amt=0.0205):
    """Two Entrance candidates and one Exit query; returns 'a', 'b' or None."""
    a = ReidPass(camera_id=1, track_id=1, first_seen=NOW, last_seen=NOW,
                 business_date=DAY, dim=D, model_name="m",
                 vector=_near(BODY, 3, a_amt).tobytes(),
                 face_vector=None if a_face is None else a_face.tobytes(),
                 face_dim=0 if a_face is None else D)
    b = ReidPass(camera_id=1, track_id=2, first_seen=NOW, last_seen=NOW,
                 business_date=DAY, dim=D, model_name="m",
                 vector=_near(BODY, 4, b_amt).tobytes(),
                 face_vector=None if b_face is None else b_face.tobytes(),
                 face_dim=0 if b_face is None else D)
    q = ReidPass(camera_id=2, track_id=3, first_seen=NOW, last_seen=NOW,
                 business_date=DAY, dim=D, model_name="m",
                 vector=BODY.tobytes())
    s.add_all([a, b, q])
    s.flush()
    got = _worker()._match(s, q, BODY, query_face)
    if got is None:
        return None
    return "a" if got[0] == a.id else "b"


def test_a_clear_body_winner_is_decided_by_the_body():
    with session_scope() as s:
        # b is far enough behind that the margin is satisfied outright.
        assert _run(s, a_face=None, b_face=None, query_face=None,
                    a_amt=0.020, b_amt=0.060) == "a"
        s.rollback()


def test_a_body_tie_with_no_faces_abstains():
    with session_scope() as s:
        assert _run(s, a_face=None, b_face=None, query_face=None) is None
        s.rollback()


def test_the_face_breaks_the_tie_toward_whichever_candidate_it_agrees_with():
    with session_scope() as s:
        # The face agrees with b, which is SECOND on body.
        assert _run(s, a_face=_unit(9), b_face=_near(FACE, 5, 0.02),
                    query_face=_near(FACE, 6, 0.02)) == "b"
        s.rollback()
    with session_scope() as s:
        # ...and with a, which is first. The rule has no preference of its own.
        assert _run(s, a_face=_near(FACE, 5, 0.02), b_face=_unit(9),
                    query_face=_near(FACE, 6, 0.02)) == "a"
        s.rollback()


def test_one_candidate_with_a_face_is_not_a_tie_the_face_can_break():
    """'No face' is not a low score. Ranking it against a real one would invent
    a comparison that was never made."""
    with session_scope() as s:
        assert _run(s, a_face=_near(FACE, 5, 0.02), b_face=None,
                    query_face=_near(FACE, 6, 0.02)) is None
        s.rollback()


def test_two_wrong_faces_abstain_rather_than_pick_the_least_wrong():
    with session_scope() as s:
        assert _run(s, a_face=_unit(9), b_face=_unit(10),
                    query_face=_near(FACE, 6, 0.02)) is None
        s.rollback()


def test_two_equally_close_faces_are_still_a_tie():
    with session_scope() as s:
        assert _run(s, a_face=_near(FACE, 5, 0.02), b_face=_near(FACE, 7, 0.02),
                    query_face=_near(FACE, 6, 0.02)) is None
        s.rollback()


def test_the_tie_break_can_be_turned_off():
    old = settings.reid_match_face_tiebreak
    try:
        settings.reid_match_face_tiebreak = False
        with session_scope() as s:
            assert _run(s, a_face=_unit(9), b_face=_near(FACE, 5, 0.02),
                        query_face=_near(FACE, 6, 0.02)) is None
            s.rollback()
    finally:
        settings.reid_match_face_tiebreak = old


def test_the_body_threshold_still_governs_every_accept():
    """The property that keeps the measured false-accept rate measured: face
    may choose BETWEEN plausible candidates, never make an implausible one
    plausible."""
    with session_scope() as s:
        # Both bodies are unrelated to the query, so nothing clears 0.6769.
        far_a, far_b = _unit(20), _unit(21)
        a = ReidPass(camera_id=1, track_id=1, first_seen=NOW, last_seen=NOW,
                     business_date=DAY, dim=D, model_name="m",
                     vector=far_a.tobytes(),
                     face_vector=_near(FACE, 5, 0.02).tobytes(), face_dim=D)
        b = ReidPass(camera_id=1, track_id=2, first_seen=NOW, last_seen=NOW,
                     business_date=DAY, dim=D, model_name="m",
                     vector=far_b.tobytes(),
                     face_vector=_unit(9).tobytes(), face_dim=D)
        q = ReidPass(camera_id=2, track_id=3, first_seen=NOW, last_seen=NOW,
                     business_date=DAY, dim=D, model_name="m",
                     vector=BODY.tobytes())
        s.add_all([a, b, q])
        s.flush()
        assert float(far_a @ BODY) < settings.reid_match_threshold
        assert _worker()._match(s, q, BODY, _near(FACE, 6, 0.02)) is None
        s.rollback()


def test_both_halves_of_a_match_point_at_each_other():
    with session_scope() as s:
        a = ReidPass(camera_id=1, track_id=1, first_seen=NOW, last_seen=NOW,
                     business_date=DAY, dim=D, model_name="m",
                     vector=_near(BODY, 3, 0.02).tobytes())
        q = ReidPass(camera_id=2, track_id=3, first_seen=NOW, last_seen=NOW,
                     business_date=DAY, dim=D, model_name="m",
                     vector=BODY.tobytes())
        s.add_all([a, q])
        s.flush()
        got = _worker()._match(s, q, BODY, None)
        assert got is not None and got[0] == a.id
        assert q.matched_pass_id == a.id and a.matched_pass_id == q.id
        assert q.match_score == pytest.approx(a.match_score)
        s.rollback()
