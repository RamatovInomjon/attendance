"""Face vectors from one recognizer must never be scored by another.

The gallery has refused a wrong-model rebuild for weeks (`test_gallery_model_guard`),
but three other tables store face vectors too - `unknown_sighting.vector`,
`reid_pass.face_vector`, `pseudo_person.face_templates` - and none of them
recorded which recognizer wrote them. A recognizer swap leaves every one of
those rows behind at the same width, and comparing them against the new model
SUCCEEDS: unit-length, plausible cosines, nothing to say it is noise. Those
vectors feed the corridor-crop floors, the labelled impostor set and the
cross-camera tie-break, so the failure would have set thresholds from noise.

Each table now stamps `recognizer_key()` and each reader compares it. These
tests pin every reader, the key itself, the migration that backfills old rows,
and the two other things the swap exposed: an override tuned for one model
must not apply to another, and a track whose head flips to a second person
must not carry the first person's evidence.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pytest
from sqlalchemy import delete, select

from app.config import recognizer_key, settings
from app.db.models import FaceEmbedding, PseudoPerson, ReidPass, UnknownSighting
from app.db.session import session_scope

NOW = datetime(2026, 9, 16, 9, 0, tzinfo=timezone.utc)
DAY = date(2026, 9, 16)
OTHER = "adaface_ir101_finetune_fp16.onnx"        # the previous recognizer


def _unit(seed, d=512):
    r = np.random.default_rng(seed)
    v = r.standard_normal(d).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture(autouse=True)
def _other_model_is_really_other():
    assert recognizer_key(OTHER) != settings.recognizer_key


# ---- the key --------------------------------------------------------------

@pytest.mark.parametrize("a,b", [
    ("ir101S3v2s_sr10final_fp16.onnx", "ir101S3v2s_sr10final.onnx"),
    ("ir101S3v2s_sr10final_fp16.onnx", "ir101S3v2s_sr10final_fp16.onnx.enc"),
    ("/x/models/adaface_ir101_finetune.onnx", "adaface_ir101_finetune_fp32.onnx"),
])
def test_precision_and_encryption_are_the_same_recognizer(a, b):
    assert recognizer_key(a) == recognizer_key(b)


def test_different_weights_are_different_recognizers():
    assert recognizer_key("ir101S3v2s_sr10final_fp16.onnx") != recognizer_key(OTHER)
    assert recognizer_key("ir50S3v2s_sr10final.onnx") != recognizer_key("ir101S3v2s_sr10final.onnx")


def test_nothing_is_not_a_recognizer():
    assert recognizer_key(None) == "" and recognizer_key("") == ""
    assert recognizer_key(None) != settings.recognizer_key


# ---- corridor probes and labelled probes ----------------------------------

@pytest.fixture
def sightings():
    mine, theirs, unstamped = _unit(1), _unit(2), _unit(3)
    with session_scope() as s:
        rows = [
            UnknownSighting(camera_id=None, track_id=1, business_date=DAY, first_seen=NOW,
                            last_seen=NOW, vector=mine.tobytes(),
                            model_name=settings.recognizer_model,
                            resolved_kind="visitor"),
            UnknownSighting(camera_id=None, track_id=2, business_date=DAY, first_seen=NOW,
                            last_seen=NOW, vector=theirs.tobytes(), model_name=OTHER,
                            resolved_kind="visitor"),
            UnknownSighting(camera_id=None, track_id=3, business_date=DAY, first_seen=NOW,
                            last_seen=NOW, vector=unstamped.tobytes(), model_name=None,
                            resolved_kind="visitor"),
        ]
        s.add_all(rows)
        s.flush()
        ids = [r.id for r in rows]
    yield mine, theirs, unstamped
    with session_scope() as s:
        s.execute(delete(UnknownSighting).where(UnknownSighting.id.in_(ids)))


def _contains(P, v):
    return len(P) and float((P @ v).max()) > 0.999


def test_corridor_probes_come_only_from_the_recognizer_in_use(sightings):
    from app.services import augment
    mine, theirs, unstamped = sightings
    P, _who = augment.corridor_probes(days=3650)
    assert _contains(P, mine)
    assert not _contains(P, theirs), "the previous recognizer's vector is noise here"
    assert not _contains(P, unstamped), "unknown provenance reads as 'not this model'"


def test_labelled_probes_keep_the_label_but_not_the_other_models_vector(sightings):
    from app.services import corrections
    mine, theirs, _unstamped = sightings
    V = corrections.labelled_probes()["visitors"]      # already stacked, unit rows
    assert _contains(V, mine)
    assert not _contains(V, theirs)


# ---- the pseudo-person gallery --------------------------------------------

def test_face_templates_from_another_recognizer_are_not_loaded():
    from app.services.pseudo_gallery import PseudoGallery, _pack
    face = _unit(7, 8)
    with session_scope() as s:
        for p in s.query(PseudoPerson).all():
            s.delete(p)
        for r in s.query(ReidPass).all():
            s.delete(r)
        s.flush()
        stale = PseudoPerson(code="P-STALE", first_seen=NOW, last_seen=NOW, n_passes=1,
                             face_templates=_pack(face.reshape(1, -1)), face_dim=8,
                             face_model=OTHER, body_dim=0, body_model="")
        s.add(stale)
        s.flush()
        row = ReidPass(camera_id=1, camera_name="Entrance", track_id=1, first_seen=NOW,
                       last_seen=NOW, business_date=DAY, direction="ENTER", name="",
                       folder="", crops=5, dim=8, model_name="m")
        s.add(row)
        s.flush()
        g = PseudoGallery()
        placed = g.place(s, row, face=face, body=None, face_ipd=30.0, body_model="m")
        assert placed is not None and placed.id != stale.id, \
            "an identical face under the OLD recognizer must not match"
        assert placed.face_model == settings.recognizer_key
        for p in s.query(PseudoPerson).all():
            s.delete(p)
        for r in s.query(ReidPass).all():
            s.delete(r)
        s.flush()


# ---- the cross-camera face tie-break --------------------------------------

def test_the_tie_break_ignores_faces_written_by_another_recognizer():
    from app.services.reid_worker import ReidWorker
    body, face = _unit(11), _unit(12)

    def near(v, seed, amt):
        r = np.random.default_rng(seed)
        x = v + r.standard_normal(len(v)).astype(np.float32) * amt
        return x / np.linalg.norm(x)

    def run(model):
        with session_scope() as s:
            a = ReidPass(camera_id=1, track_id=1, first_seen=NOW, last_seen=NOW,
                         business_date=DAY, dim=512, model_name="m",
                         vector=near(body, 3, 0.020).tobytes(),
                         face_vector=_unit(9).tobytes(), face_dim=512, face_model=model)
            b = ReidPass(camera_id=1, track_id=2, first_seen=NOW, last_seen=NOW,
                         business_date=DAY, dim=512, model_name="m",
                         vector=near(body, 4, 0.0205).tobytes(),
                         face_vector=near(face, 5, 0.02).tobytes(), face_dim=512,
                         face_model=model)
            q = ReidPass(camera_id=2, track_id=3, first_seen=NOW, last_seen=NOW,
                         business_date=DAY, dim=512, model_name="m", vector=body.tobytes())
            s.add_all([a, b, q])
            s.flush()
            got = ReidWorker.__new__(ReidWorker)._match(s, q, body, near(face, 6, 0.02))
            out = None if got is None else ("b" if got[0] == b.id else "a")
            s.rollback()
            return out

    assert run(settings.recognizer_key) == "b", "same recognizer: the face decides"
    assert run(OTHER) is None, "another recognizer's faces are no evidence: abstain"


# ---- the migration ----------------------------------------------------------

def test_backfill_stamps_old_rows_with_the_gallerys_recognizer():
    """Rows written before provenance was recorded were scored against the
    gallery in the database at the time, so its model is theirs - but only
    while that gallery is still the one they were scored against."""
    import importlib.util
    from pathlib import Path as _P
    from app.db.models import Employee
    spec = importlib.util.spec_from_file_location(
        "migrate_under_test", _P(__file__).resolve().parents[1] / "scripts" / "migrate.py")
    migrate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migrate)

    with session_scope() as s:
        emp = Employee(external_id="EXT-BACKFILL", full_name="Backfill", is_active=True)
        s.add(emp)
        s.flush()
        s.add(FaceEmbedding(employee_id=emp.id, source_file="image_01.png",
                            vector=_unit(20).tobytes(), dim=512,
                            model_name=settings.recognizer_model, quality=0.9))
        u = UnknownSighting(camera_id=None, track_id=99, business_date=DAY, first_seen=NOW,
                            last_seen=NOW, vector=_unit(21).tobytes(), model_name=None)
        s.add(u)
        s.flush()
        uid, eid = u.id, emp.id
        gallery_models = {recognizer_key(m) for m in
                          s.execute(select(FaceEmbedding.model_name)).scalars()}
    try:
        done = migrate.backfill_face_model()
        with session_scope() as s:
            stamped = s.get(UnknownSighting, uid).model_name
        if len(gallery_models) == 1:
            assert stamped == settings.recognizer_key
            assert done.get("unknown_sighting", 0) >= 1
        else:
            assert stamped is None, "two gallery models in one table: nothing may be guessed"
    finally:
        with session_scope() as s:
            s.execute(delete(UnknownSighting).where(UnknownSighting.id == uid))
            s.execute(delete(FaceEmbedding).where(FaceEmbedding.employee_id == eid))
            s.execute(delete(Employee).where(Employee.id == eid))


# ---- the override is a number on one model's scale ------------------------

def test_an_override_applies_only_to_the_recognizer_it_was_tuned_for(monkeypatch):
    calibrated = settings.threshold_for()
    monkeypatch.setattr(settings, "recognition_threshold_override", 0.22)
    monkeypatch.setattr(settings, "recognition_threshold_override_model", "")
    assert settings.threshold_for() == calibrated, "unbound: ignored"
    monkeypatch.setattr(settings, "recognition_threshold_override_model", OTHER)
    assert settings.threshold_for() == calibrated, "bound to another model: ignored"
    monkeypatch.setattr(settings, "recognition_threshold_override_model",
                        settings.recognizer_model + ".enc")
    assert settings.threshold_for() == 0.22, "bound to this one: in force"


# ---- a head that jumps to another person -----------------------------------

def test_iou_of_the_same_head_is_high_and_of_a_swapped_head_is_low():
    from app.core.pipeline import _iou
    a = np.array([100, 100, 220, 220], np.float32)
    assert _iou(a, a + 8) > 0.7, "one person, one frame later (measured 0.7-0.97)"
    assert _iou(a, np.array([260, 100, 380, 220], np.float32)) == 0.0, "the neighbour"
    assert _iou(a, np.array([0, 0, 0, 0], np.float32)) == 0.0


def test_reset_identity_forgets_the_person_and_keeps_the_track():
    from app.core.direction import Direction, Trajectory
    from app.core.gallery import Match
    from app.core.pipeline import CameraPipeline as Pipeline, TrackState

    pipe = Pipeline.__new__(Pipeline)
    pipe.head_jumps = 0
    st = TrackState(track_id=42, first_seen=1.0, last_seen=5.0,
                    box=np.array([0, 0, 50, 100], np.float32), vote=pipe._new_vote())
    st.vote.add(Match(employee_id=7, score=0.5, margin=0.2, runner_up=None))
    st.employee_id, st.name, st.score, st.emitted = 7, "Seven", 0.5, True
    st.face_sum, st.face_frames, st.best_vector = _unit(1), 3, _unit(1)
    st.trajectory.add(1.0, (0, 0, 10, 10), 100, 100)
    st.direction = Direction.ENTER
    st.attempts, st.body_saves = 9, 2

    pipe._reset_identity(st)

    assert st.employee_id is None and st.name == "…" and not st.emitted
    assert st.vote.provisional_id is None and st.vote.identified == 0
    assert st.face_sum is None and st.face_frames == 0 and st.best_vector is None
    assert len(st.trajectory.points) == 0 and st.direction is Direction.UNKNOWN
    assert st.track_id == 42 and st.first_seen == 1.0, "the track is the same object"
    assert st.attempts == 9 and st.body_saves == 2, "track bookkeeping survives"
