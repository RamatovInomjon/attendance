"""One reid_pass row per pass, however many times the pass is flushed.

Production logged 2,102 `UNIQUE constraint failed: reid_pass.camera_id,
reid_pass.track_id, reid_pass.first_seen` between 3 and 22 September - about 5%
of all passes. The sequence: a pass's crops pause past the stale limit, so
`_close_stale` writes it as UNKNOWN and drops it from `_open`; the SAME track's
next crop opens a new pass under the same key; the second write then collides.
The half that failed was the completed track - the one carrying the name, the
direction and the face template - so 5.1% of named recognitions were left with
an unnamed, directionless row. These tests reproduce that sequence.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import delete, select

from app.config import settings
from app.db.models import ReidPass
from app.db.session import session_scope
from app.services.reid_worker import ReidWorker

CAM, TRACK = 7, 4242


class _FakeReid:
    model_name = "fake_reid.onnx"
    dim = 8

    def embed(self, crops):
        rng = np.random.default_rng(len(crops))
        v = rng.standard_normal((len(crops), 8)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class _CountingPseudo:
    """Stands in for PseudoGallery.place and counts how often a pass is placed."""
    def __init__(self):
        self.calls = 0

    def place(self, s, row, **kw):
        self.calls += 1
        row.pseudo_person_id = 999
        return SimpleNamespace(code="P-TEST")

    def stats(self):
        return {}


@pytest.fixture
def worker(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "persons_dir", tmp_path)
    w = ReidWorker(model_path="unused")
    w.reid = _FakeReid()
    w.pseudo = _CountingPseudo()
    yield w
    with session_scope() as s:
        s.execute(delete(ReidPass).where(ReidPass.camera_id == CAM))


def _crop(first_seen, ts):
    return SimpleNamespace(track_id=TRACK, first_seen=first_seen, ts=ts, score=0.9,
                           image=np.full((32, 16, 3), 120, np.uint8))


def _completed(first_seen, *, employee_id=31, name="Xamdamov Rustam",
               direction="EXIT", face=True):
    f = np.ones(512, np.float32) / np.sqrt(512) if face else None
    return SimpleNamespace(track_id=TRACK, first_seen=first_seen, employee_id=employee_id,
                           name=name, direction=direction, best_score=0.55,
                           face_template=f, best_ipd=30.0, face_frames=12)


def _rows():
    with session_scope() as s:
        return [dict(employee_id=r.employee_id, name=r.name, direction=r.direction,
                     crops=r.crops, face_frames=r.face_frames,
                     has_face=r.face_vector is not None, folder=r.folder)
                for r in s.execute(select(ReidPass).where(ReidPass.camera_id == CAM)).scalars()]


def test_a_paused_pass_resumed_and_completed_is_one_named_row(worker):
    t0 = time.time() - 600
    # Crops arrive, then pause long enough for the stale sweep to write the pass.
    for k in range(3):
        worker._on_crop(CAM, "Exit", _crop(t0, t0 + k))
    worker._close_stale(t0 + 3 + 3600)
    assert len(_rows()) == 1 and _rows()[0]["employee_id"] is None
    # The same track keeps going, then completes WITH a name.
    for k in range(3):
        worker._on_crop(CAM, "Exit", _crop(t0, t0 + 100 + k))
    worker._on_pass(CAM, "Exit", _completed(t0), None)

    rows = _rows()
    assert len(rows) == 1, "the second flush must merge, not insert"
    r = rows[0]
    assert r["employee_id"] == 31 and r["name"] == "Xamdamov Rustam"
    assert r["direction"] == "EXIT", "the completion's whole-track direction wins over UNKNOWN"
    assert r["crops"] == 6, "both halves' crops are counted"
    assert r["has_face"] and r["face_frames"] == 12
    assert r["folder"].startswith("known/")
    assert worker.passes_merged == 1 and worker.errors == 0


def test_a_completion_that_finds_nothing_open_still_names_the_row(worker):
    """Crops paused, the stale sweep wrote the pass, and NO new crop came: the
    completion used to find nothing open and be dropped without a word."""
    t0 = time.time() - 600
    for k in range(3):
        worker._on_crop(CAM, "Exit", _crop(t0, t0 + k))
    worker._close_stale(t0 + 3 + 3600)
    worker._on_pass(CAM, "Exit", _completed(t0), None)

    (r,) = _rows()
    assert r["employee_id"] == 31 and r["direction"] == "EXIT" and r["has_face"]
    assert worker.names_attached == 1


def test_an_unnamed_later_half_never_erases_a_name(worker):
    t0 = time.time() - 600
    for k in range(3):
        worker._on_crop(CAM, "Exit", _crop(t0, t0 + k))
    worker._on_pass(CAM, "Exit", _completed(t0), None)          # named first
    for k in range(3):
        worker._on_crop(CAM, "Exit", _crop(t0, t0 + 200 + k))    # then more body crops
    worker._close_stale(t0 + 200 + 3600)

    (r,) = _rows()
    assert r["employee_id"] == 31 and r["direction"] == "EXIT"
    assert r["crops"] == 6


def test_a_merged_pass_is_placed_in_a_pseudo_person_only_once(worker):
    """`place()` counts every call as a pass; placing both halves would turn one
    walk into two visits of the same person."""
    t0 = time.time() - 600
    for k in range(3):
        worker._on_crop(CAM, "Exit", _crop(t0, t0 + k))
    worker._close_stale(t0 + 3 + 3600)
    assert worker.pseudo.calls == 1
    for k in range(3):
        worker._on_crop(CAM, "Exit", _crop(t0, t0 + 100 + k))
    worker._close_stale(t0 + 103 + 3600)
    assert worker.pseudo.calls == 1, "an already-grouped row is not grouped again"
    assert len(_rows()) == 1


def test_distinct_passes_are_still_distinct_rows(worker):
    t0 = time.time() - 600
    for first in (t0, t0 + 50):          # same track id, different first_seen
        for k in range(3):
            worker._on_crop(CAM, "Exit", _crop(first, first + k))
    worker._close_stale(t0 + 50 + 3600)
    assert len(_rows()) == 2
