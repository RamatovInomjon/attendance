"""Bootstrapping a first gallery row for somebody who has none.

The check this path cannot make is the one augmentation relies on: with zero
embeddings there is no "own" similarity, so `own - other` is meaningless and
`augment.scan()` refuses every candidate. What replaces it is the floor - and
the test that matters is that the floor is still enforced, so a human asserting
the wrong identity cannot produce a row that fires for somebody else.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from sqlalchemy import delete, select

from app.config import settings
from app.db.models import Employee, FaceEmbedding
from app.db.session import session_scope
from app.services import augment
from app.services.augment import Candidate

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "bootstrap_face", Path(__file__).resolve().parent.parent / "scripts" / "bootstrap_face.py")
bootstrap_face = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bootstrap_face)


def _unit(seed: int, dim: int = 512) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def faceless_person():
    with session_scope() as s:
        e = Employee(full_name="Faceless Person", external_id="WEB-NOFACE",
                     is_active=True)
        s.add(e)
        s.flush()
        eid = e.id
    yield eid
    with session_scope() as s:
        s.execute(delete(FaceEmbedding).where(FaceEmbedding.employee_id == eid))
        s.execute(delete(Employee).where(Employee.id == eid))


def test_faceless_employees_finds_exactly_the_people_with_no_embedding(
        faceless_person):
    ids = [i for i, _n in bootstrap_face.faceless_employees()]
    assert faceless_person in ids

    with session_scope() as s:
        s.add(FaceEmbedding(employee_id=faceless_person, vector=_unit(1).tobytes(),
                            dim=512, model_name=settings.recognizer_model))
    assert faceless_person not in [i for i, _n in bootstrap_face.faceless_employees()]


def test_the_floor_still_refuses_a_crop_another_face_already_reaches(
        faceless_person, monkeypatch):
    """The human asserted the identity; the measurement must still be able to
    say no. A crop that a real corridor face reaches at or above the floor is a
    lookalike, and a bootstrapped row is no more exempt than an augmented one.
    """
    v = _unit(7)
    c = Candidate(key="k", employee_id=faceless_person, name="Faceless Person",
                  camera="Exit", when="2026-09-06T11:25:28", score=0.49)
    c.vec = v

    # A corridor probe that IS this crop: similarity 1.0, far above any floor.
    monkeypatch.setattr(augment, "corridor_probes",
                        lambda days=14, dim=None: (np.stack([v]), np.array([-1])))
    augment.calibrate([c], np.zeros((0, 512), np.float32),
                      np.zeros(0, np.int64), {})

    assert c.rejected, "a face reaching the crop above its floor must refuse it"
    assert "lookalike" in c.rejected


def test_a_clean_crop_is_accepted_and_carries_the_corridor_floor(
        faceless_person, monkeypatch):
    v = _unit(11)
    c = Candidate(key="k2", employee_id=faceless_person, name="Faceless Person",
                  camera="Exit", when="2026-09-06T11:25:28", score=0.49)
    c.vec = v

    # Nothing resembles it: an unrelated probe, and an unrelated gallery.
    monkeypatch.setattr(augment, "corridor_probes",
                        lambda days=14, dim=None: (np.stack([_unit(99)]), np.array([-1])))
    other = _unit(123)[None]
    augment.calibrate([c], other, np.array([12345], np.int64), {12345: "Someone"})

    assert not c.rejected, c.rejected
    expected = max(settings.threshold_for(settings.recognizer_model),
                   settings.augment_live_floor)
    assert c.threshold == pytest.approx(expected)
    # The floor is what protects the row, so it must never sit at the ordinary
    # recognition threshold for a corridor crop.
    assert c.threshold >= settings.augment_live_floor


def test_captures_are_filtered_by_score_frames_aligner_and_gate(tmp_path, monkeypatch):
    """--min-score is the operator's assertion; the rest are quality gates."""
    monkeypatch.setattr(settings, "debug_dir", tmp_path)
    person = tmp_path / "someone"
    person.mkdir()

    def write(stem, **over):
        meta = {"employee_id": 55, "name": "X", "camera": "Exit",
                "timestamp_local": "2026-09-06T11:25:28", "score": 0.50,
                "embedded_frames": 40, "quality": {"aligner_score": 1.0, "gate": "PASS"}}
        meta.update(over)
        (person / f"{stem}.json").write_text(json.dumps(meta))
        (person / f"{stem}_aligned.jpg").write_bytes(b"x")

    write("good")
    write("low_score", score=0.20)
    write("few_frames", embedded_frames=2)
    write("bad_gate", quality={"aligner_score": 1.0, "gate": "BLUR"})
    write("no_align", quality={"aligner_score": 0.5, "gate": "PASS"})
    (person / "orphan.json").write_text(json.dumps({"employee_id": 55, "score": 0.9}))

    rows = bootstrap_face.captures_for({55})[55]
    keys = {r["key"] for r in rows}
    assert "orphan" not in keys, "a capture with no aligned crop is unusable"

    ok = [r for r in rows if r["score"] >= 0.45 and r["frames"] >= 8
          and r["aligner"] >= 0.95 and r["gate"] in (None, "PASS")]
    assert [r["key"] for r in ok] == ["good"]
    assert rows[0]["score"] >= rows[-1]["score"], "best score first"
