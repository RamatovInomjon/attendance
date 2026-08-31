"""A face must be embedded as itself, and a gallery must match its recognizer.

Both failures guarded here are silent by construction: they produce no error,
no warning, and plausible-looking numbers.

This file previously also covered a keypoint-conditioned recognizer. It was
evaluated on native-4K corridor footage against the deployed IR-101, agreed with
it on 12 of 12 passes, and cost 1.5x the compute - so it was removed. What
survives is the property that mattered underneath it: results are paired with
inputs BY POSITION all the way through
detect -> align -> gate -> embed, and any stage that quietly returns a shorter
list re-pairs every face after the gap onto the wrong track.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.config import settings
from app.core.aligner import AlignedFace, FaceAligner


class TestAlignerPairing:
    """`zip(pending, faces)` in the pipeline pairs by position."""

    @pytest.fixture(scope="class")
    def aligner(self):
        return FaceAligner(settings.model_path(settings.aligner_model),
                           crop_size=settings.align_crop_size,
                           margin=settings.align_margin, mode=settings.align_mode)

    def test_one_face_per_box_even_when_a_box_is_off_frame(self, aligner):
        """A crop entirely outside the frame used to be dropped silently, so
        every later face was attributed to another track - a wrong identity
        that looks entirely plausible."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = [
            np.array([100, 100, 180, 180], np.float32),      # normal
            np.array([-400, -400, -320, -320], np.float32),  # entirely off-frame
            np.array([300, 200, 380, 280], np.float32),      # normal
        ]
        faces = aligner.align(frame, boxes, is_bgr=True)
        assert len(faces) == len(boxes)

    def test_each_face_reports_its_own_crop_box(self, aligner):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = [np.array([50, 50, 130, 130], np.float32),
                 np.array([400, 300, 480, 380], np.float32)]
        faces = aligner.align(frame, boxes, is_bgr=True)
        for box, face in zip(boxes, faces):
            cx = (box[0] + box[2]) / 2
            crop_cx = (face.crop_box[0] + face.crop_box[2]) / 2
            assert abs(cx - crop_cx) < 2.0, (
                "face is paired with another box\'s crop")

    def test_the_shape_the_recognizer_expects(self, aligner):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        f = aligner.align(frame, [np.array([100, 100, 200, 200], np.float32)],
                          is_bgr=True)[0]
        assert f.aligned.shape == (3, 112, 112)
        assert f.aligned.dtype == np.float32
        assert f.landmarks.shape == (5, 2)


class TestRecognizerContract:
    def test_a_model_needing_more_than_an_image_is_refused(self, tmp_path):
        """The pipeline supplies one thing: the aligned face. A model wanting a
        second input would be handed nothing for it and would return vectors
        that are quietly wrong - in the gallery and in every live match at
        once, with nothing in the data to reveal it. So it is refused at load
        rather than silently misused.

        Built as a real two-input graph, not asserted against the source: the
        core ships as a compiled extension, where there is no source to read.
        """
        import onnx
        from onnx import helper, TensorProto
        from app.core.recognizer import FaceRecognizer

        img = helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                            ["b", 3, 112, 112])
        extra = helper.make_tensor_value_info("keypoints", TensorProto.FLOAT,
                                              ["b", 5, 2])
        out = helper.make_tensor_value_info("embedding", TensorProto.FLOAT,
                                            ["b", 512])
        node = helper.make_node("ReduceMean", ["input"], ["embedding"],
                                axes=[2, 3], keepdims=0)
        graph = helper.make_graph([node], "two_input", [img, extra], [out])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        # onnx writes the newest IR version it knows; this runtime caps lower.
        model.ir_version = 10
        path = tmp_path / "two_input.onnx"
        onnx.save(model, str(path))

        with pytest.raises(ValueError, match="inputs"):
            FaceRecognizer(path)

    def test_embeddings_come_back_normalised(self):
        from app.core.recognizer import FaceRecognizer
        rec = FaceRecognizer(settings.model_path(settings.recognizer_model),
                             batch_size=settings.embed_batch)
        out = rec.embed(np.random.uniform(-1, 1, (3, 3, 112, 112)).astype(np.float32))
        assert out.shape == (3, 512)
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-4)

    def test_a_single_face_may_be_passed_unbatched(self):
        from app.core.recognizer import FaceRecognizer
        rec = FaceRecognizer(settings.model_path(settings.recognizer_model),
                             batch_size=settings.embed_batch)
        assert rec.embed(
            np.random.uniform(-1, 1, (3, 112, 112)).astype(np.float32)).shape == (1, 512)

    def test_batching_does_not_change_an_embedding(self):
        """The pipeline embeds whatever a frame holds; a batch of four must
        give each face the same vector it would get alone."""
        from app.core.recognizer import FaceRecognizer
        rec = FaceRecognizer(settings.model_path(settings.recognizer_model),
                             batch_size=settings.embed_batch)
        faces = np.random.uniform(-1, 1, (4, 3, 112, 112)).astype(np.float32)
        batched = rec.embed(faces)
        for i in range(len(faces)):
            alone = rec.embed(faces[i : i + 1])[0]
            assert float(batched[i] @ alone) > 0.9999
