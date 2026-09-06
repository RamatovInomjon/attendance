"""The per-person capture cap must mean what it says.

`debug_max_per_person` is 40, and data/debug/Inomjon_Ramatov held 242 sidecars.
The count lived in memory, per DebugCapture instance - and there is one
instance per camera worker, per process start - so the real cap was
40 x cameras x restarts. Seeding it from what is on disk is what makes it one
number.
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
    stems = [_capture(first, n) for n in range(1, 5)]
    assert [bool(s) for s in stems] == [True, True, False, False], \
        "three on disk plus two new ones is the cap"
    assert first.summary()["Cap_Person"] == 5
    assert len(list(folder.glob("*.json"))) == 5

    # A second instance - the other camera, or the process after a restart -
    # starts from the five on disk, not from zero.
    second = DebugCapture(enabled=True, root=tmp_path, max_per_person=5)
    assert _capture(second, 9) is None
    assert len(list(folder.glob("*.json"))) == 5


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
