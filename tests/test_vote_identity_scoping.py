"""A track's evidence must belong to the identity the track committed.

Reported from the dashboard: a row named "Qo'shmatov Axmat" showing an aligned
face that was plainly "Nuriddinov Azizjon". The recogniser was not at fault and
the gallery was clean - the two identities are near-orthogonal (cos ~= -0.02),
each with a tight within-person cluster.

The fault was in TrackVote. `best_score` and `best_snapshot` were a single
running maximum over every frame, *regardless of which person each frame
matched*, while the identity came separately from the K-of-N majority. When one
track saw two people - colleagues walking together, or a ByteTrack id switch -
the vote committed person A while the maximum belonged to a frame of person B.
The emitted event then carried A's name with B's score and B's face.

Live instance (cam1, 2026-08-27 09:21:18, track 601): committed one employee,
reported 0.277 and an aligned crop that both belonged to another employee who
had passed the same camera 14 seconds earlier at 0.227.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.gallery import Match, TrackVote


A_ID, B_ID = 50, 24                      # Qo'shmatov, Nuriddinov
FACE_A = np.full((3, 8, 8), 0.25, np.float32)
FACE_B = np.full((3, 8, 8), 0.75, np.float32)


def _vote() -> TrackVote:
    return TrackVote(window=5, required=3)


def test_committed_identity_reports_its_own_score_not_the_tracks_maximum():
    v = _vote()
    for s in (0.201, 0.215, 0.222):                 # A wins the majority
        v.add(Match(A_ID, s, 0.1, None), snapshot=FACE_A, quality=1.0)
    v.add(Match(B_ID, 0.277, 0.1, None), snapshot=FACE_B, quality=1.0)   # B scores higher

    assert v.committed == A_ID
    assert v.best_score == pytest.approx(0.222), (
        "reported score must be the committed identity's best, not the "
        "track-wide maximum belonging to somebody else"
    )


def test_committed_identity_shows_its_own_face():
    v = _vote()
    for s in (0.201, 0.215, 0.222):
        v.add(Match(A_ID, s, 0.1, None), snapshot=FACE_A, quality=1.0)
    v.add(Match(B_ID, 0.277, 0.1, None), snapshot=FACE_B, quality=1.0)

    assert np.array_equal(v.best_snapshot, FACE_A), (
        "the dashboard crop must be the committed person's face; showing the "
        "higher-scoring stranger is the reported bug"
    )


def test_the_other_identity_is_still_available_for_diagnosis():
    """Scoping the report must not throw away what else the track saw."""
    v = _vote()
    for s in (0.201, 0.215, 0.222):
        v.add(Match(A_ID, s, 0.1, None), snapshot=FACE_A, quality=1.0)
    v.add(Match(B_ID, 0.277, 0.1, None), snapshot=FACE_B, quality=1.0)

    assert v.best_for(B_ID)[0] == pytest.approx(0.277)
    assert np.array_equal(v.best_for(B_ID)[1], FACE_B)


def test_order_does_not_matter():
    """The stranger's frame may arrive before the majority forms."""
    v = _vote()
    v.add(Match(B_ID, 0.277, 0.1, None), snapshot=FACE_B, quality=1.0)
    for s in (0.201, 0.215, 0.222):
        v.add(Match(A_ID, s, 0.1, None), snapshot=FACE_A, quality=1.0)

    assert v.committed == A_ID
    assert v.best_score == pytest.approx(0.222)
    assert np.array_equal(v.best_snapshot, FACE_A)


def test_before_a_commit_the_running_best_is_still_reported():
    """With no identity decided there is nothing to scope to, so the best
    observation available is the right thing to show."""
    v = _vote()
    v.add(Match(A_ID, 0.201, 0.1, None), snapshot=FACE_A, quality=1.0)
    v.add(Match(B_ID, 0.277, 0.1, None), snapshot=FACE_B, quality=1.0)

    assert v.committed is None
    assert v.best_score == pytest.approx(0.277)
    assert np.array_equal(v.best_snapshot, FACE_B)


def test_a_single_identity_track_is_unaffected():
    """The common case must behave exactly as before."""
    v = _vote()
    for s in (0.21, 0.33, 0.28):
        v.add(Match(A_ID, s, 0.1, None), snapshot=FACE_A, quality=1.0)

    assert v.committed == A_ID
    assert v.best_score == pytest.approx(0.33)
    assert np.array_equal(v.best_snapshot, FACE_A)


def test_misses_never_become_the_reported_evidence():
    """A miss carries a score but no identity; it must not be reportable."""
    v = _vote()
    for s in (0.201, 0.215, 0.222):
        v.add(Match(A_ID, s, 0.1, None), snapshot=FACE_A, quality=1.0)
    v.add(Match(None, 0.99, 0.0, B_ID), snapshot=FACE_B, quality=1.0)

    assert v.committed == A_ID
    assert v.best_score == pytest.approx(0.222)
    assert np.array_equal(v.best_snapshot, FACE_A)
