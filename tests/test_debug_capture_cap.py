"""The per-person capture cap must bound the folder without losing the newest.

`debug_max_per_person` is 40, and data/debug/Inomjon_Ramatov held 242 sidecars.
The count lived in memory, per DebugCapture instance - and there is one
instance per camera worker, per process start - so the real cap was
40 x cameras x restarts. Seeding it from what is on disk makes it one number.

Seeding alone then broke the thing the folder is for. Two people were already
past the cap, so their next recognition wrote nothing, and the events page -
which shows the capture beside each recognition, and is where a wrong
check-out is judged - had no face to show and said the frame was not saved.
So the window rolls: the oldest capture goes, the newest is always kept.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from app.config import settings
from app.services import debug_capture
from app.services.debug_capture import DebugCapture

PERSON = "Cap Person"


@pytest.fixture
def plenty_of_disk(monkeypatch):
    """The free-disk guard must not decide these tests on whatever this
    machine happens to have free."""
    monkeypatch.setattr(debug_capture.shutil, "disk_usage",
                        lambda p: SimpleNamespace(free=10 ** 13, total=10 ** 13, used=0))


def _capture(dc, n=1, name=PERSON):
    return dc.capture(
        name=name, frame_bgr=np.zeros((64, 64, 3), np.uint8), box=(8, 8, 40, 40),
        aligned_chw=None, camera="Entrance", role="in", score=0.5, margin=0.1,
        track_id=n, ts=datetime(2026, 9, 1, 8, 0, n, tzinfo=timezone.utc))


def test_the_cap_counts_what_is_already_on_disk(tmp_path, plenty_of_disk):
    folder = tmp_path / "Cap_Person"
    folder.mkdir()
    for i in range(3):
        (folder / f"20260831_080000_0.500_Entrance_{i + 1:03d}.json").write_text("{}")

    first = DebugCapture(enabled=True, root=tmp_path, max_per_person=5)
    assert all(_capture(first, n) for n in range(1, 5)), "every capture is written"
    assert first.summary()["Cap_Person"] == 5
    assert len(list(folder.glob("*.json"))) == 5, "three on disk plus two is the cap"

    # A second instance - the other camera, or the process after a restart -
    # starts from the five on disk, not from zero.
    second = DebugCapture(enabled=True, root=tmp_path, max_per_person=5)
    assert _capture(second, 9)
    assert len(list(folder.glob("*.json"))) == 5


def test_the_window_rolls_so_the_newest_recognition_always_has_evidence(
        tmp_path, plenty_of_disk):
    """The bug on the events page: a person past the cap wrote nothing, so
    every later recognition had no face to show."""
    folder = tmp_path / "Cap_Person"
    folder.mkdir()
    for i in range(5):
        for suffix in (".json", "_face.jpg", "_aligned.jpg", "_frame.jpg"):
            (folder / f"20260831_08000{i}_0.500_Entrance_{i + 1:03d}{suffix}").write_text("x")

    dc = DebugCapture(enabled=True, root=tmp_path, max_per_person=5)
    stem = _capture(dc, 1)
    assert stem, "a person at the cap still gets their capture"
    assert (folder / f"{stem}.json").exists()
    assert len(list(folder.glob("*.json"))) == 5, "and the folder stays bounded"

    # The OLDEST went, with all four of its files, and nothing newer did.
    assert not list(folder.glob("20260831_080000_*")), "oldest capture retired whole"
    assert (folder / "20260831_080004_0.500_Entrance_005.json").exists()


def test_names_keep_increasing_even_as_old_captures_are_evicted(tmp_path, plenty_of_disk):
    """The sequence is seeded from the highest on disk, not from the count, so
    an evicted name is never handed out twice - two captures of one person in
    the same second would otherwise write to the same files."""
    folder = tmp_path / "Cap_Person"
    folder.mkdir()
    (folder / "20260831_080000_0.500_Entrance_007.json").write_text("{}")

    dc = DebugCapture(enabled=True, root=tmp_path, max_per_person=2)
    assert _capture(dc, 1).endswith("_008")
    assert _capture(dc, 2).endswith("_009")


def test_the_sidecar_records_the_threshold_actually_in_force(tmp_path, plenty_of_disk):
    dc = DebugCapture(enabled=True, root=tmp_path, max_per_person=5)
    stem = _capture(dc)
    meta = json.loads((tmp_path / "Cap_Person" / f"{stem}.json").read_text())
    assert meta["threshold"] == settings.threshold_for(settings.recognizer_model)


def test_a_full_disk_skips_the_capture_without_spending_the_cap(tmp_path, monkeypatch):
    """The same guard `frame()` always had. A full disk must fail the audit
    trail, not the service - and a capture that wrote nothing must not use up
    a slot, or the folder would stay empty once the disk was freed."""
    monkeypatch.setattr(debug_capture.shutil, "disk_usage",
                        lambda p: SimpleNamespace(free=0, total=10 ** 13, used=10 ** 13))
    dc = DebugCapture(enabled=True, root=tmp_path, max_per_person=5)
    assert _capture(dc) is None
    assert not (tmp_path / "Cap_Person").exists(), "nothing written"
    assert dc.summary().get("Cap_Person", 0) == 0, "a skipped capture costs no slot"
