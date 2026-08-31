"""A gallery built by one recognizer is meaningless to another.

Face recognizers generally emit 512-d L2-normalised vectors, so comparing one
model's queries against another's embeddings SUCCEEDS: the matmul runs, cosines come back
in the usual range, and nothing anywhere reports a problem. The matching is
simply wrong - in the gallery and in every live comparison at once.

That is what switching `recognizer_model` without re-enrolling does, which is
precisely the step a deployment is most likely to forget.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.services.enrollment import _model_key, load_gallery


@pytest.fixture(autouse=True)
def restore_gallery_rows():
    """These tests wipe `face_embedding` to control what load_gallery sees.
    The suite shares ONE in-memory database, so anything left behind changes
    what later tests observe - the dashboard test reads enrolment counts."""
    from app.db.models import Employee, FaceEmbedding
    from app.db.session import session_scope
    from sqlalchemy import delete, select
    with session_scope() as s:
        saved = [dict(employee_id=r.employee_id, source_file=r.source_file,
                      vector=r.vector, dim=r.dim, model_name=r.model_name,
                      quality=r.quality)
                 for r in s.execute(select(FaceEmbedding)).scalars()]
        emp_before = set(s.execute(select(Employee.id)).scalars())
    yield
    with session_scope() as s:
        s.execute(delete(FaceEmbedding))
        for row in saved:
            s.add(FaceEmbedding(**row))
        # Explicit: the test engine has no `PRAGMA foreign_keys`, so cascades
        # do not fire and SQLite recycles the freed ids.
        made = [i for i in s.execute(select(Employee.id)).scalars()
                if i not in emp_before]
        if made:
            s.execute(delete(FaceEmbedding).where(
                FaceEmbedding.employee_id.in_(made)))
            s.execute(delete(Employee).where(Employee.id.in_(made)))


class TestModelKey:
    """What counts as "the same recognizer" for gallery purposes."""

    def test_precision_variants_are_the_same_model(self):
        """fp16 and fp32 of one export differ by 0.0012 d-prime on this
        gallery - inside the noise. The live gallery is stamped with the fp32
        name while the deployed model is fp16; treating those as incompatible
        would refuse to start on a gallery that is perfectly good."""
        assert _model_key("adaface_ir101_finetune.onnx") == \
               _model_key("adaface_ir101_finetune_fp16.onnx")

    def test_the_vault_suffix_is_the_same_model(self):
        assert _model_key("adaface_ir101_finetune_fp16.onnx.enc") == \
               _model_key("adaface_ir101_finetune.onnx")

    def test_genuinely_different_models_are_different(self):
        assert _model_key("some_other_arch_fp16.onnx") != \
               _model_key("adaface_ir101_finetune_fp16.onnx")

    def test_a_missing_model_name_does_not_crash(self):
        assert _model_key(None) == ""


class TestLoadGallery:
    def _seed(self, model_name: str, n: int = 3):
        from app.db.models import Employee, FaceEmbedding
        from app.db.session import session_scope
        from sqlalchemy import delete
        with session_scope() as s:
            s.execute(delete(FaceEmbedding))
            e = Employee(full_name=f"Guard {model_name}", is_active=True)
            s.add(e); s.flush()
            for i in range(n):
                v = np.random.default_rng(i).standard_normal(512).astype(np.float32)
                s.add(FaceEmbedding(employee_id=e.id, vector=v.tobytes(),
                                    dim=512, model_name=model_name, quality=0.9))

    def test_a_matching_gallery_loads(self):
        from app.config import settings
        self._seed(settings.recognizer_model)
        g = load_gallery()
        assert len(g) == 3

    def test_a_gallery_from_another_recognizer_is_refused(self):
        """Refused, not warned. A warning in a log nobody reads still leaves
        every attendance record for that day wrong, with nothing in the data
        to show it."""
        self._seed("some_other_model.onnx")
        with pytest.raises(RuntimeError, match="gallery/recognizer mismatch"):
            load_gallery()

    def test_the_error_names_both_models_and_the_way_out(self):
        # Deliberately not a real model name: whichever recognizer is
        # configured, this must be a different one.
        self._seed("some_other_recognizer.onnx")
        with pytest.raises(RuntimeError) as exc:
            load_gallery()
        msg = str(exc.value)
        assert "some_other_recognizer" in msg
        assert "scripts/enroll.py" in msg

    def test_a_half_migrated_gallery_is_refused(self):
        """The dangerous middle state: a re-enrolment that crashed partway,
        leaving rows from both models. Every score is then drawn from a mixture
        of two incomparable spaces."""
        from app.config import settings
        from app.db.models import Employee, FaceEmbedding
        from app.db.session import session_scope
        from sqlalchemy import delete, select
        self._seed(settings.recognizer_model)
        with session_scope() as s:
            emp = s.execute(select(Employee.id)).scalars().first()
            v = np.random.default_rng(9).standard_normal(512).astype(np.float32)
            s.add(FaceEmbedding(employee_id=emp, vector=v.tobytes(), dim=512,
                                model_name="a_different_recognizer.onnx", quality=0.9))
        with pytest.raises(RuntimeError, match="mismatch"):
            load_gallery()

    def test_strict_can_be_turned_off_for_tooling(self):
        self._seed("some_other_model.onnx")
        assert len(load_gallery(strict=False)) == 3

    def test_an_empty_gallery_is_not_a_mismatch(self):
        from app.db.models import FaceEmbedding
        from app.db.session import session_scope
        from sqlalchemy import delete
        with session_scope() as s:
            s.execute(delete(FaceEmbedding))
        assert len(load_gallery()) == 0
