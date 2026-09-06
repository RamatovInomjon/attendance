"""One walk past this corridor is one attendance decision.

Both cameras overlook the same space, and their direction geometry is mirrored,
so a single person walking through is seen twice and labelled ENTER by one
camera and EXIT by the other. The debounce was scoped to `camera_id`, so both
halves passed it and both moved attendance state.

The damage, from the live database on 2026-08-26:

    emp 9   cam1 07:31:04.898  IN   ENTER  CHECK_IN
            cam2 07:31:09.470  OUT  EXIT   CHECK_OUT   -> worked_seconds = 4
    emp 6   cam1 13:08:54      IN   ENTER  CHECK_IN
            cam2 13:10:06      OUT  EXIT   CHECK_OUT   -> worked_seconds = 72

Seven such pairs within 10 s, 25 within the 90 s cooldown, and six rows left
with `check_out_time` BEFORE `check_in_time`. Reconfirmed live on 2026-08-31:
all eight walk-throughs triggered both cameras, 2-4 s apart.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.services.arbiter import PassArbiter, PendingPass


@dataclass
class FakeTrack:
    embedded_frames: int
    best_score: float


def _pass(emp, cam, mono, frames, score):
    return PendingPass(employee_id=emp, camera_id=cam, role=None, ts=None,
                       monotonic=mono, track=FakeTrack(frames, score),
                       snapshot=None, camera_name=f"cam{cam}")


def test_both_halves_of_one_walk_yield_one_decision():
    a = PassArbiter(window_s=15.0)
    a.submit(_pass(9, 1, 100.0, 18, 0.41))
    a.submit(_pass(9, 2, 104.6, 6, 0.27))       # the 4.6 s gap from emp 9
    assert a.due(110.0) == []                    # window still open
    groups = a.due(116.0)
    assert len(groups) == 1
    assert groups[0].winner.camera_id == 1
    assert [o.camera_id for o in groups[0].others] == [2]


def test_the_strongest_view_wins_not_the_first():
    """A plain person-scoped debounce keeps whichever camera finished its track
    first, which is an accident of when the person left each field of view. The
    weaker view wins it half the time."""
    a = PassArbiter(window_s=15.0)
    a.submit(_pass(7, 2, 100.0, 4, 0.55))       # arrives first, but 4 frames
    a.submit(_pass(7, 1, 102.0, 22, 0.31))      # arrives later, far more evidence
    g = a.due(120.0)[0]
    assert g.winner.camera_id == 1
    assert g.winner.track.embedded_frames == 22


def test_score_breaks_a_tie_on_frames():
    a = PassArbiter(window_s=15.0)
    a.submit(_pass(3, 1, 100.0, 10, 0.30))
    a.submit(_pass(3, 2, 101.0, 10, 0.44))
    assert a.due(120.0)[0].winner.camera_id == 2


def test_a_genuine_leave_and_return_stays_two_decisions():
    """The window must not swallow a real second trip. 15 s against the 2-4 s
    the two halves of one walk actually arrive apart."""
    a = PassArbiter(window_s=15.0)
    a.submit(_pass(5, 1, 100.0, 12, 0.40))
    first = a.due(120.0)
    a.submit(_pass(5, 2, 400.0, 12, 0.40))      # five minutes later
    second = a.due(420.0)
    assert len(first) == 1 and len(second) == 1
    assert first[0].others == [] and second[0].others == []


def test_different_people_are_never_fused():
    a = PassArbiter(window_s=15.0)
    a.submit(_pass(1, 1, 100.0, 10, 0.4))
    a.submit(_pass(2, 2, 101.0, 10, 0.4))
    groups = a.due(120.0)
    assert len(groups) == 2
    assert all(g.others == [] for g in groups)


def test_a_group_is_handed_out_exactly_once():
    """Both camera threads drain the arbiter. If a group could be returned
    twice, one walk would be written twice - the very thing this prevents."""
    a = PassArbiter(window_s=15.0)
    a.submit(_pass(4, 1, 100.0, 10, 0.4))
    assert len(a.due(120.0)) == 1
    assert a.due(120.0) == []


def test_nothing_is_released_before_the_window_closes():
    a = PassArbiter(window_s=15.0)
    a.submit(_pass(4, 1, 100.0, 10, 0.4))
    assert a.due(114.9) == []
    assert len(a.due(115.1)) == 1


def test_drain_releases_everything_for_shutdown():
    """Held passes belong to people who really walked past. Losing them at
    shutdown is the same bug as never finalizing an in-flight track."""
    a = PassArbiter(window_s=15.0)
    a.submit(_pass(1, 1, 100.0, 10, 0.4))
    a.submit(_pass(2, 2, 100.0, 10, 0.4))
    assert len(a.drain()) == 2
    assert a.drain() == []


def test_the_window_comes_from_settings_by_default():
    from app.config import settings
    assert PassArbiter().window == settings.cross_camera_window_s
    assert settings.cross_camera_window_s >= 5.0, (
        "the two halves of one walk arrived 2-4 s apart in live measurement")


def test_fused_count_is_reported():
    a = PassArbiter(window_s=15.0)
    a.submit(_pass(9, 1, 100.0, 18, 0.41))
    a.submit(_pass(9, 2, 104.6, 6, 0.27))
    a.due(120.0)
    assert a.stats()["fused"] == 1


# --- direction is resolved on its own evidence -----------------------------

@dataclass
class DirTrack:
    embedded_frames: int
    best_score: float
    direction: str
    direction_reason: str


def _dpass(emp, cam, mono, frames, score, direction, reason):
    return PendingPass(employee_id=emp, camera_id=cam, role=None, ts=None,
                       monotonic=mono, track=DirTrack(frames, score, direction, reason),
                       snapshot=None, camera_name=f"cam{cam}")


def test_the_view_that_ended_last_decides_a_contradiction():
    """The export's 14 contradicted groups: a person walks to the door area
    and comes back. The Exit camera saw the outbound half (EXIT), the
    Entrance camera the return (ENTER). The stronger face used to decide the
    direction; when EXIT won, somebody who never left was checked out."""
    a = PassArbiter(window_s=15.0)
    a.submit(_dpass(30, 2, 100.0, 34, 0.65, "EXIT", "line+depth agree (EXIT)"))     # stronger face
    a.submit(_dpass(30, 1, 109.5, 11, 0.37, "ENTER", "line+depth agree (ENTER)"))   # later, weaker
    g = a.due(130.0)[0]
    assert g.winner.camera_id == 2                     # identity evidence: still the stronger view
    assert g.direction == "ENTER"                      # direction: the later movement
    assert g.direction_reason.startswith("line+depth agree (ENTER) [over cam2:EXIT")
    assert g.contradicted
    assert a.stats()["contradictions"] == 1


def test_a_whole_u_turn_beats_a_half_view_of_it():
    """One camera saw the person cross out and come back: a tripwire finding
    that they are inside. It outranks the other camera's EXIT half."""
    a = PassArbiter(window_s=15.0)
    a.submit(_dpass(5, 2, 100.0, 40, 0.6, "EXIT", "line+depth agree (EXIT)"))
    a.submit(_dpass(5, 1, 104.0, 20, 0.5, "ENTER", "crossed and returned (inside)"))
    g = a.due(120.0)[0]
    assert g.direction == "ENTER"
    assert g.direction_reason.startswith("crossed and returned (inside) [over cam2:EXIT")


def test_a_tripwire_verdict_outranks_depth_only_whatever_the_order():
    a = PassArbiter(window_s=15.0)
    a.submit(_dpass(8, 1, 100.0, 50, 0.7, "ENTER", "line+depth agree (ENTER)"))
    a.submit(_dpass(8, 2, 106.0, 9, 0.3, "EXIT", "depth only (EXIT, travel 0.21)"))
    g = a.due(120.0)[0]
    assert g.direction == "ENTER"
    assert not g.contradicted


def test_a_view_without_a_verdict_never_overrides_one():
    a = PassArbiter(window_s=15.0)
    a.submit(_dpass(8, 1, 100.0, 12, 0.4, "ENTER", "line only (ENTER)"))
    a.submit(_dpass(8, 2, 108.0, 60, 0.8, "UNKNOWN", "stationary (0.010 < 0.06)"))
    g = a.due(120.0)[0]
    assert g.winner.camera_id == 2          # best face
    assert g.direction == "ENTER"           # but the other view's direction


def test_agreeing_views_keep_the_winners_own_reason():
    a = PassArbiter(window_s=15.0)
    a.submit(_dpass(8, 1, 100.0, 12, 0.4, "ENTER", "line+depth agree (ENTER)"))
    a.submit(_dpass(8, 2, 103.0, 6, 0.3, "ENTER", "depth only (ENTER, travel 0.30)"))
    g = a.due(120.0)[0]
    assert g.direction == "ENTER"
    assert "[over" not in g.direction_reason


# --- the window stretches while the person is still in view ---------------

def test_a_group_waits_for_a_live_track_of_the_same_person():
    """The return leg of a U-turn is a track that is still running when the
    outbound pass's window would close. Deciding then is deciding on half
    the walk - and 55 such pairs completed 15-60 s apart in the export."""
    a = PassArbiter(window_s=15.0)
    a.submit(_dpass(30, 2, 100.0, 34, 0.65, "EXIT", "line+depth agree (EXIT)"))
    a.note_live(1, {30}, 114.0)                 # Entrance camera has them in view
    assert a.due(116.0) == []                    # window passed, but still live
    a.note_live(1, {30}, 124.0)
    assert a.due(125.0) == []
    a.submit(_dpass(30, 1, 126.0, 11, 0.37, "ENTER", "line+depth agree (ENTER)"))
    a.note_live(1, set(), 127.0)                 # ...and now they have left
    g = a.due(128.0)
    assert len(g) == 1
    assert g[0].direction == "ENTER" and g[0].contradicted


def test_a_live_note_goes_stale_and_the_group_closes():
    a = PassArbiter(window_s=15.0)
    a.submit(_dpass(30, 2, 100.0, 34, 0.65, "EXIT", "line+depth agree (EXIT)"))
    a.note_live(1, {30}, 114.0)
    assert a.due(116.0) == []
    assert len(a.due(118.5)) == 1               # no fresh note for >3 s


def test_the_hold_is_capped_even_while_live():
    a = PassArbiter(window_s=15.0, max_hold_s=60.0)
    a.submit(_dpass(30, 2, 100.0, 34, 0.65, "EXIT", "line+depth agree (EXIT)"))
    for t in range(101, 160, 2):
        a.note_live(1, {30}, float(t))
        assert a.due(float(t)) == []
    a.note_live(1, {30}, 161.0)
    assert len(a.due(161.0)) == 1


def test_somebody_else_being_live_does_not_hold_a_group():
    a = PassArbiter(window_s=15.0)
    a.submit(_dpass(30, 2, 100.0, 34, 0.65, "EXIT", "line+depth agree (EXIT)"))
    a.note_live(1, {31}, 115.5)
    assert len(a.due(116.0)) == 1
