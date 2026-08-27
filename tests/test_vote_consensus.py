"""Identity is decided by consensus over the whole pass.

The rule (docs/RECOGNITION_PLAN.md, T2): a track commits a person when that
person accounts for at least `vote_consensus` (0.65) of the frames that produced
ANY identity, and at least `vote_min_recognitions` (5) frames agree on them.

Two properties of the rule matter and are easy to get wrong:

* **Misses are excluded from the denominator.** A frame that matched nobody
  says nothing about which of two candidates is right, so it does not dilute
  the winner. 8 for A, 2 for B and 20 misses is 80% for A.
* **A fraction alone cannot judge a short pass.** One agreeing frame out of one
  is 100% consensus and would clear any percentage rule, which is why the
  absolute floor exists. Below it the pass names nobody.

The old rule was "first to 3 of the last 5", which could settle an identity
from the opening frames and never revisit it - and live scores for one person
spanned 0.196-0.473 in a single day, so the opening frames are not reliably a
pass's best evidence.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.gallery import Match, TrackVote

A, B, C = 50, 24, 31
FACE_A = np.full((3, 8, 8), 0.25, np.float32)
FACE_B = np.full((3, 8, 8), 0.75, np.float32)


def _vote(consensus=0.65, min_recognitions=5) -> TrackVote:
    return TrackVote(window=5, required=3,
                     consensus=consensus, min_recognitions=min_recognitions)


def _add(v, emp, n, score=0.30, snapshot=None):
    for _ in range(n):
        v.add(Match(emp, score, 0.10, None), snapshot=snapshot, quality=1.0)


def _miss(v, n):
    for _ in range(n):
        v.add(Match(None, 0.10, 0.0, C), snapshot=None, quality=0.0)


# --- the headline rule -----------------------------------------------------

def test_clear_consensus_commits():
    """21 of 30 identified frames agree -> 70%, above the bar."""
    v = _vote()
    _add(v, A, 21); _add(v, B, 9)
    assert v.finalize() == A
    assert v.agreement == pytest.approx(21 / 30)


def test_a_split_pass_commits_nobody():
    """12 for A, 10 for B: the leader has 55% and the pass disagreed with itself."""
    v = _vote()
    _add(v, A, 12); _add(v, B, 10)
    assert v.finalize() is None


def test_misses_do_not_dilute_the_winner():
    """8 for A, 2 for B, 20 misses -> 80% for A, not 27%.

    This is the owner's decision, and it is the difference between committing
    and rejecting: counting the 20 misses would put A at 8/30 = 27%.
    """
    v = _vote()
    _add(v, A, 8); _add(v, B, 2); _miss(v, 20)
    assert v.identified == 10
    assert v.missed == 20
    assert v.agreement == pytest.approx(0.8)
    assert v.finalize() == A


# --- the floor -------------------------------------------------------------

def test_a_single_frame_cannot_decide_a_pass():
    """1 of 1 is 100% consensus and clears any percentage rule. The floor is
    the only thing standing between that and a named attendance row."""
    v = _vote()
    _add(v, A, 1)
    assert v.agreement == pytest.approx(1.0)
    assert v.finalize() is None


def test_four_unanimous_frames_are_still_too_few():
    v = _vote()
    _add(v, A, 4)
    assert v.finalize() is None


def test_five_unanimous_frames_are_enough():
    v = _vote()
    _add(v, A, 5)
    assert v.finalize() == A


def test_the_floor_counts_AGREEING_frames_not_total_frames():
    """4 for A out of 40 identified is neither enough frames nor enough share."""
    v = _vote()
    _add(v, A, 4); _add(v, B, 36)
    assert v.finalize() == B          # B has 36/40 = 90%


# --- order independence ----------------------------------------------------

def test_arrival_order_cannot_change_the_answer():
    """The old rule was first-past-the-post, so order decided it. It must not."""
    early = _vote(); _add(early, B, 4); _add(early, A, 21); _add(early, B, 5)
    late = _vote();  _add(late, A, 21); _add(late, B, 9)
    assert early.finalize() == late.finalize() == A


def test_a_late_surge_can_take_the_lead():
    """B trails badly at first and still wins - impossible under first-to-3."""
    v = _vote()
    _add(v, A, 4); _add(v, B, 30)
    assert v.finalize() == B


# --- provisional vs final --------------------------------------------------

def test_nothing_is_decided_mid_pass():
    """`decided` stays False for the track's whole life, which is what keeps
    recognition running instead of stopping at the first agreement (T1)."""
    v = _vote()
    _add(v, A, 30)
    assert v.decided is False
    assert v.committed is None
    v.finalize()
    assert v.decided is True


def test_provisional_leads_the_live_overlay():
    v = _vote()
    _add(v, A, 2)
    assert v.provisional_id is None          # under `required`, no name yet
    _add(v, A, 1)
    assert v.provisional_id == A             # 3 frames: good enough to display
    assert v.finalize() is None              # but not to commit - under the floor


def test_provisional_follows_the_lead_change():
    v = _vote()
    _add(v, A, 5)
    assert v.provisional_id == A
    _add(v, B, 9)
    assert v.provisional_id == B


def test_finalize_is_idempotent():
    """A second call must not be able to rewrite an identity already written
    to attendance."""
    v = _vote()
    _add(v, A, 10)
    first = v.finalize()
    _add(v, B, 100)                     # evidence arriving after the decision
    assert v.finalize() == first == A


# --- evidence stays scoped to the identity we name -------------------------

def test_best_frame_belongs_to_the_committed_identity():
    """The 99cf420 property must survive the rewrite: the face shown is the
    best frame OF THE PERSON NAMED, never a higher-scoring stranger's."""
    v = _vote()
    for s in (0.20, 0.21, 0.22, 0.23, 0.24):
        v.add(Match(A, s, 0.10, None), snapshot=FACE_A, quality=1.0)
    v.add(Match(B, 0.90, 0.50, None), snapshot=FACE_B, quality=1.0)
    assert v.finalize() == A
    assert v.best_score == pytest.approx(0.24)
    assert np.array_equal(v.best_snapshot, FACE_A)


def test_evidence_is_scoped_to_the_provisional_leader_mid_pass():
    """Before the pass ends there is no final identity, but the live overlay
    still must not pair one person's name with another's face."""
    v = _vote()
    for s in (0.20, 0.21, 0.22):
        v.add(Match(A, s, 0.10, None), snapshot=FACE_A, quality=1.0)
    v.add(Match(B, 0.90, 0.50, None), snapshot=FACE_B, quality=1.0)
    assert v.provisional_id == A
    assert np.array_equal(v.best_snapshot, FACE_A)


def test_a_rejected_pass_reports_no_identity():
    """Nobody named means nothing for attendance to write - the caller records
    an unknown sighting instead."""
    v = _vote()
    _add(v, A, 2); _add(v, B, 2)
    assert v.finalize() is None
    assert v.decided is False


# --- degenerate input ------------------------------------------------------

def test_an_empty_track_names_nobody():
    v = _vote()
    assert v.finalize() is None
    assert v.agreement == 0.0
    assert v.provisional_id is None


def test_a_track_of_pure_misses_names_nobody():
    v = _vote()
    _miss(v, 50)
    assert v.identified == 0
    assert v.agreement == 0.0
    assert v.finalize() is None


@pytest.mark.parametrize("share,expected", [
    (0.64, None),      # just under the bar
    (0.65, A),         # exactly on it
    (0.66, A),
])
def test_the_consensus_boundary(share, expected):
    """100 identified frames, so the share is exact."""
    v = _vote()
    n = round(share * 100)
    _add(v, A, n); _add(v, B, 100 - n)
    assert v.finalize() == expected
