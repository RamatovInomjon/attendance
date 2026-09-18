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


def test_the_rebuild_names_everyone_it_left_without_a_face(monkeypatch, tmp_path, sessions):
    """The wipe removes every non-corridor row and the rebuild restores only
    what has a photo folder. Someone enrolled without one is unrecognisable
    from then on - and was, silently, for three people. Now the report says."""
    with session_scope() as s:
        s.add(Employee(external_id="EXT-NOFOLDER", full_name="No Folder Person",
                       is_active=True))
    rep = _enroller(monkeypatch, sessions).run(gallery_dir=_gallery(tmp_path))
    names = [n for _i, n in rep.orphans]
    assert "No Folder Person" in names
    assert not any(EXT in n for n in names), "the person with a folder was rebuilt"
    assert "LEFT WITHOUT A FACE" in rep.summary()



def _gallery_without_metadata(tmp_path, ext_id, n=2):
    """A folder of photographs and nothing else.

    This is how the enrolment export reached gpu6: the images were copied, the
    `metadata.json` beside them was not.
    """
    folder = tmp_path / f"013_{ext_id}"
    folder.mkdir()
    for i in range(n):
        (folder / f"image_{i + 1:02d}.png").write_bytes(b"not decoded here")
    return tmp_path


def test_a_folder_without_metadata_does_not_overwrite_a_known_person(
        monkeypatch, tmp_path, sessions):
    """A rebuild must not degrade a record it knows nothing about.

    `metadata.json` is the master record and still wins where it has a value.
    A folder WITHOUT one says nothing about the person - but these fields were
    assigned unconditionally, so on gpu6, where only the photographs had been
    copied, a gallery rebuild rewrote all 54 full names to the folder suffix
    ("Xamdamov Rustam" -> "Rustam") and blanked every department, position and
    phone. The employee ids survived, so attendance still joined and nothing
    looked broken; the loss showed up only as first names in the UI.
    """
    ext = "Rustam"
    with session_scope() as s:
        s.add(Employee(external_id=ext, full_name="Xamdamov Rustam",
                       department="Buxgalteriya", position="Bosh buxgalter",
                       phone="+998901112233", is_active=True))

    _enroller(monkeypatch, sessions).run(
        gallery_dir=_gallery_without_metadata(tmp_path, ext))

    with session_scope() as s:
        emp = s.execute(
            select(Employee).where(Employee.external_id == ext)).scalar_one()
        assert emp.full_name == "Xamdamov Rustam"
        assert emp.department == "Buxgalteriya"
        assert emp.position == "Bosh buxgalter"
        assert emp.phone == "+998901112233"
        assert emp.folder == f"013_{ext}"      # the rebuild still took effect


def test_metadata_still_wins_where_it_has_a_value(monkeypatch, tmp_path, sessions):
    """Preserving the old value must not stop a real update from landing."""
    with session_scope() as s:
        s.add(Employee(external_id=EXT, full_name="Stale Name",
                       department="Stale Dept", is_active=True))

    _enroller(monkeypatch, sessions).run(gallery_dir=_gallery(tmp_path))

    with session_scope() as s:
        emp = s.execute(
            select(Employee).where(Employee.external_id == EXT)).scalar_one()
        assert emp.full_name == "Rerun Person"
        assert emp.department == "QA"


def test_a_person_seen_for_the_first_time_still_falls_back_to_the_folder(
        monkeypatch, tmp_path, sessions):
    """With no metadata AND no existing row there is nothing better to use."""
    ext = "Brand New"
    _enroller(monkeypatch, sessions).run(
        gallery_dir=_gallery_without_metadata(tmp_path, ext))

    with session_scope() as s:
        emp = s.execute(
            select(Employee).where(Employee.external_id == ext)).scalar_one()
        assert emp.full_name == ext
        assert emp.department == ""


def test_the_rebuild_logs_one_line_per_folder_and_survives_an_empty_gallery(
        monkeypatch, tmp_path, sessions, caplog):
    """The per-folder line belongs to the folder loop, not the orphan loop.

    It had drifted one loop down, where `folder` and `rows` are whatever the
    last iteration left behind. Two failures: every real "enrolled X" line is
    lost and the last folder's is repeated once per orphan (observed on gpu6 as
    `enrolled 054_Inomjon` three times), and with an EMPTY gallery directory
    plus an active employee both names are unbound, so the rebuild dies with
    UnboundLocalError inside the write transaction.
    """
    import logging

    with session_scope() as s:
        s.add(Employee(external_id="EXT-ORPHAN", full_name="Orphan One",
                       is_active=True))

    # Separate roots: a gallery dir is SCANNED for folders, so an empty dir
    # nested inside the other one would be enrolled as a person of its own.
    empty = tmp_path / "empty_root"
    empty.mkdir()
    real = tmp_path / "real_root"
    real.mkdir()
    with caplog.at_level(logging.INFO):
        rep = _enroller(monkeypatch, sessions).run(gallery_dir=empty)
    assert any("Orphan One" in r.getMessage() for r in caplog.records), \
        "the orphan must still be reported"
    assert rep.people == 0

    # And with real folders, one line each, naming the folder it enrolled.
    caplog.clear()
    with caplog.at_level(logging.INFO):
        _enroller(monkeypatch, sessions).run(gallery_dir=_gallery(real))
    enrolled = [r for r in caplog.records if "enrolled" in str(r.msg)]
    assert len(enrolled) == 1, f"one line per folder, got {len(enrolled)}"
    assert enrolled[0].args[0] == f"001_{EXT}"
