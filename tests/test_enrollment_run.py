"""Re-enrolment: keep the corridor crops, and keep the write lock short.

`scripts/enroll.py` rebuilds the gallery from the photographs. Two things about
HOW it does that matter more than the rebuild itself:

* `live:` rows are admin-reviewed corridor crops carrying their own floors. The
  wipe used to take them with everything else, so every re-enrolment silently
  undid the augmentation work - and nothing in the log said so.
* the detect/align/embed of every photograph ran INSIDE the write transaction,
  so the capture threads sat on the SQLite write lock for ten seconds and more.
"""
from __future__ import annotations

import json
from contextlib import contextmanager

import numpy as np
import pytest
from sqlalchemy import delete, select

from app.config import settings
from app.db.models import Employee, FaceEmbedding
from app.db.session import session_scope
from app.services import enrollment
from app.services.augment import TAG

EXT = "EXT-RERUN"
CROP = f"{TAG}20260901_090000_0.412_Exit_001"


@pytest.fixture(autouse=True)
def restore_gallery_rows():
    """The wipe deletes every enrolment row in the shared in-memory database,
    so what the other test files seeded is saved here and put back after."""
    with session_scope() as s:
        saved = [dict(employee_id=r.employee_id, source_file=r.source_file,
                      vector=r.vector, dim=r.dim, model_name=r.model_name,
                      quality=r.quality, threshold=r.threshold)
                 for r in s.execute(select(FaceEmbedding)).scalars()]
        emp_before = set(s.execute(select(Employee.id)).scalars())
    yield
    with session_scope() as s:
        s.execute(delete(FaceEmbedding))
        for row in saved:
            s.add(FaceEmbedding(**row))
        made = [i for i in s.execute(select(Employee.id)).scalars()
                if i not in emp_before]
        if made:
            s.execute(delete(FaceEmbedding).where(FaceEmbedding.employee_id.in_(made)))
            s.execute(delete(Employee).where(Employee.id.in_(made)))


@pytest.fixture
def sessions(monkeypatch):
    """Counts open sessions, so the test can tell whether an image was
    embedded while a transaction was open."""
    state = {"open": 0, "embedded_inside": 0}
    real = enrollment.session_scope

    @contextmanager
    def counting():
        state["open"] += 1
        try:
            with real() as s:
                yield s
        finally:
            state["open"] -= 1

    monkeypatch.setattr(enrollment, "session_scope", counting)
    return state


def _unit(seed):
    v = np.random.default_rng(seed).standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def _gallery(tmp_path, n=3):
    folder = tmp_path / f"001_{EXT}"
    folder.mkdir()
    (folder / "metadata.json").write_text(json.dumps(
        {"user_id": EXT, "full_name": "Rerun Person", "department": "QA"}))
    for i in range(n):
        (folder / f"image_{i + 1:02d}.png").write_bytes(b"not decoded here")
    return tmp_path


def _enroller(monkeypatch, sessions):
    """No model is loaded; `embed_file` is the only thing run() needs."""
    e = enrollment.Enroller(detector=object(), aligner=object(), recognizer=object())

    def fake_embed(path):
        sessions["embedded_inside"] += sessions["open"]
        return _unit(int(path.stem.split("_")[-1])), 0.98, 400.0, 180.0

    monkeypatch.setattr(e, "embed_file", fake_embed)
    return e


def test_re_enrolment_keeps_the_corridor_crops_and_replaces_the_photographs(
        tmp_path, monkeypatch, sessions):
    with session_scope() as s:
        emp = Employee(external_id=EXT, full_name="Rerun Person", is_active=True)
        s.add(emp); s.flush()
        s.add_all([
            FaceEmbedding(employee_id=emp.id, source_file="old_photo.png",
                          vector=_unit(1).tobytes(), dim=512,
                          model_name=settings.recognizer_model),
            FaceEmbedding(employee_id=emp.id, source_file=CROP,
                          vector=_unit(2).tobytes(), dim=512,
                          model_name=settings.recognizer_model, threshold=0.35),
            # A corridor crop from ANOTHER recognizer is meaningless to this
            # one, and keeping it would make load_gallery() refuse the very
            # gallery the run rebuilds.
            FaceEmbedding(employee_id=emp.id, source_file=f"{TAG}stale_model_crop",
                          vector=_unit(3).tobytes(), dim=512,
                          model_name="some_other_recognizer.onnx", threshold=0.35),
        ])
        emp_id = emp.id

    rep = _enroller(monkeypatch, sessions).run(_gallery(tmp_path), wipe=True)
    assert (rep.people, rep.images_seen, rep.embedded) == (1, 3, 3)
    assert sessions["embedded_inside"] == 0, \
        "every image is embedded before the transaction opens, never inside it"

    with session_scope() as s:
        rows = s.execute(select(FaceEmbedding.source_file, FaceEmbedding.threshold)
                         .where(FaceEmbedding.employee_id == emp_id)).all()
        dept = s.get(Employee, emp_id).department
    assert sorted(f for f, _t in rows) == \
        ["image_01.png", "image_02.png", "image_03.png", CROP]
    assert dict(rows)[CROP] == pytest.approx(0.35), "the crop keeps its floor"
    assert dept == "QA", "the employee record is still upserted from metadata.json"
