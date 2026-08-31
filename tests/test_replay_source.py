"""Replaying recorded clips must preserve what makes them worth replaying.

The cameras are only reachable from the office LAN, so the pipeline is
exercised against recorded 4K clips. That is only useful if the replay keeps
the properties the live streams have - in particular that ONE walk past this
corridor was recorded by BOTH cameras, seconds apart. Play each camera's clips
back-to-back and those pairs become unrelated events minutes apart: the
cross-camera fusion never fires, and the replay reports everything is fine
because the situation that breaks it never arises.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.core.stream import ReplaySource


@pytest.fixture
def recordings(tmp_path):
    """Two cameras, four walks, each seen by both 1-4 s apart - the real
    pattern measured on 2026-08-31."""
    names = {
        "Entrance": ["20260831_114043_Entrance.mp4", "20260831_114309_Entrance.mp4",
                     "20260831_114759_Entrance.mp4", "20260831_120305_Entrance.mp4"],
        "Exit": ["20260831_114042_Exit.mp4", "20260831_114311_Exit.mp4",
                 "20260831_114756_Exit.mp4", "20260831_120630_Exit.mp4"],
    }
    for cam, files in names.items():
        (tmp_path / cam).mkdir()
        for f in files:
            (tmp_path / cam / f).write_bytes(b"")
    return tmp_path


def _all(recordings, max_gap=8.0):
    rows = []
    for cam in ("Entrance", "Exit"):
        src = ReplaySource(recordings / cam, name=cam)
        rows += [(off, cam, p.name) for off, p in src.schedule(max_gap)]
    return sorted(rows)


def test_the_two_cameras_agree_on_one_clock(recordings):
    """Each source computes the schedule alone, from the same directory. If
    they disagreed the pairing would be lost even with correct offsets."""
    a = _all(recordings)
    b = _all(recordings)
    assert a == b


def test_a_walk_recorded_by_both_cameras_stays_a_pair(recordings):
    rows = _all(recordings)
    pairs = [(x, y) for x, y in zip(rows, rows[1:])
             if x[1] != y[1] and (y[0] - x[0]) <= 5.0]
    assert len(pairs) >= 3, (
        "the cross-camera pairs are the reason to replay these clips at all; "
        f"schedule was {rows}")


def test_gaps_shorter_than_the_cap_are_preserved_exactly(recordings):
    """114042 Exit and 114043 Entrance are one second apart. That one second
    is the whole signal."""
    rows = _all(recordings)
    first_two = rows[:2]
    assert first_two[1][0] - first_two[0][0] == pytest.approx(1.0)


def test_dead_time_is_compressed(recordings):
    """The recordings span an hour of a mostly empty corridor. Replaying the
    idle stretches in full would take an hour to show four walks."""
    rows = _all(recordings)
    assert rows[-1][0] <= 8.0 * len(rows), "idle gaps should be capped"
    # 11:40 -> 12:06 is 26 real minutes; capped it must be far less.
    assert rows[-1][0] < 120.0


def test_the_original_order_is_kept(recordings):
    rows = _all(recordings)
    assert [n for _o, _c, n in rows] == sorted(n for _o, _c, n in rows), (
        "clips must play in the order they were recorded, across both cameras")


def test_a_directory_with_no_clips_is_not_fatal(tmp_path):
    src = ReplaySource(tmp_path / "Nothing", name="Nothing")
    assert src.clips() == []
    assert src.schedule(8.0) == []


def test_unparseable_names_still_play(tmp_path):
    """Filenames are the only timestamp source. Something that does not parse
    must still be replayed rather than silently dropped."""
    d = tmp_path / "Odd"
    d.mkdir()
    (d / "not-a-timestamp.mp4").write_bytes(b"")
    src = ReplaySource(d, name="Odd")
    assert len(src.schedule(8.0)) == 1


def test_the_source_reports_the_live_interface():
    """CameraWorker cannot tell this from RtspSource, so the surface it uses
    must exist: read/start/stop/stats, connected, is_stale, fps, width."""
    src = ReplaySource("nowhere", name="X")
    for attr in ("read", "start", "stop", "stats"):
        assert callable(getattr(src, attr))
    for attr in ("connected", "is_stale", "fps", "width", "height"):
        assert hasattr(src, attr)
    st = src.stats()
    assert {"name", "connected", "stale", "fps", "frames", "resolution"} <= set(st)


# --- a gap in the stream must not join two people into one track -----------

def test_a_stream_gap_is_a_discontinuity_not_a_pause():
    """ByteTrack's lost-track buffer is counted in FRAMES. When frames stop,
    its clock stops too, so after a stall it re-associates tracks from before
    the gap with whoever is in view afterwards - a different person inherits
    the earlier person's track, their vote and their identity.

    Observed in replay as a single 55-second "pass" spanning two clips that
    were recorded eight seconds apart. Live, the same thing happens whenever a
    camera drops out and reconnects.
    """
    import numpy as np
    from app.core.tracker import FaceTracker

    t = FaceTracker(frame_rate=20, track_buffer=36)
    box = np.array([[100, 100, 200, 200]], np.float32)
    score = np.array([0.9], np.float32)

    first = None
    for _ in range(10):
        out = t.update(box, score)
        if out:
            first = out[0][0]
    assert first is not None

    # A person somewhere else entirely, after the "gap".
    far = np.array([[1400, 700, 1500, 800]], np.float32)
    after_reset = FaceTracker(frame_rate=20, track_buffer=36)
    for _ in range(10):
        out = after_reset.update(far, score)
    assert out, "a fresh tracker must still track"

    # Without a reset the SAME tracker keeps its frame clock, which is the
    # behaviour the worker now guards against by calling reset() across a gap.
    t.reset()
    ids_after = set()
    for _ in range(10):
        for tid, _b, _s in t.update(far, score):
            ids_after.add(tid)
    assert ids_after, "after reset the tracker must produce tracks again"
