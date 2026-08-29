"""AdaFace IR-101 (WebFace12M) embedding extractor on ONNX Runtime.

Benchmarked on the deployment GPU (RTX 3070 Laptop, CUDA EP):

    ir101 fp32   6.20 ms @ bs1   3.17 ms/face @ bs8
    ir101 fp16   4.57 ms @ bs1   1.90 ms/face @ bs8      <- default
    ir18  fp32   1.43 ms @ bs1   0.81 ms/face @ bs8      <- fallback

FP16 is the default: it is ~1.4x faster than FP32 and the accuracy difference on
the enrolment gallery is below the noise floor (see docs/BENCHMARKS.md).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

from app.core.onnx_env import best_providers
from app.core.model_vault import load_model

log = logging.getLogger(__name__)


class FaceRecognizer:
    """(N, 3, 112, 112) in [-1, 1] -> (N, 512) L2-normalized embeddings."""

    def __init__(self, model_path: str | Path, providers=None, batch_size: int = 8):
        if providers is None:
            providers = best_providers()

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        # load_model returns a path for a plain .onnx (ONNX Runtime mmaps it)
        # or decrypted bytes for a licensed .onnx.enc, so protected weights
        # are never written to disk in the clear.
        self.session = ort.InferenceSession(load_model(model_path), sess_options=opts,
                                            providers=providers)
        self.batch_size = batch_size
        self.model_name = model_path

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.dtype = np.float16 if "float16" in inp.type else np.float32

        # Keypoint-conditioned models (AdaFace ViT + KPRPE) take a second input
        # of 5 landmarks. Detected from the graph rather than configured, so
        # swapping `recognizer_model` is the only switch needed and the two can
        # never disagree.
        extra = [i for i in self.session.get_inputs()[1:]]
        self.keypoint_name = next(
            (i.name for i in extra if "key" in i.name.lower() or "point" in i.name.lower()),
            extra[0].name if extra else None)
        self.needs_keypoints = self.keypoint_name is not None
        self.keypoint_dtype = np.float32
        if self.needs_keypoints:
            kp = next(i for i in extra if i.name == self.keypoint_name)
            self.keypoint_dtype = np.float16 if "float16" in kp.type else np.float32

        # A fixed batch dimension is a research export; run it one row at a time
        # rather than failing, and say so, because it costs ~2.4x.
        b = inp.shape[0]
        self.fixed_batch = b if isinstance(b, int) and b > 0 else None
        if self.fixed_batch:
            log.warning("recognizer %s has a FIXED batch of %d - throughput will "
                        "suffer; convert it with scripts/convert_recognizer.py",
                        Path(str(model_path)).name, self.fixed_batch)
            self.batch_size = self.fixed_batch

    @property
    def provider(self) -> str:
        return self.session.get_providers()[0]

    def embed(self, aligned: np.ndarray, normalize: bool = True,
              keypoints: np.ndarray | None = None) -> np.ndarray:
        """(N,3,112,112) -> (N,512). `keypoints` (N,5,2) in 0-1 for KPRPE models.

        A keypoint model given no keypoints is refused rather than fed zeros:
        it would return plausible-looking vectors that are quietly wrong, and
        that error would land in the gallery and in every live match at once,
        with nothing in the data to reveal it.
        """
        aligned = np.asarray(aligned)
        if aligned.ndim == 3:
            aligned = aligned[None]
        if aligned.dtype != self.dtype:
            aligned = aligned.astype(self.dtype)

        if self.needs_keypoints:
            if keypoints is None:
                raise ValueError(
                    f"{Path(str(self.model_name)).name} is keypoint-conditioned "
                    f"and requires `keypoints` (N,5,2) normalised to 0-1; got None. "
                    f"AlignedFace.keypoints carries them.")
            keypoints = np.asarray(keypoints)
            if keypoints.ndim == 2:
                keypoints = keypoints[None]
            if len(keypoints) != len(aligned):
                raise ValueError(f"keypoints/faces length mismatch: "
                                 f"{len(keypoints)} vs {len(aligned)}")
            keypoints = keypoints.astype(self.keypoint_dtype)

        outs = []
        for i in range(0, len(aligned), self.batch_size):
            chunk = aligned[i : i + self.batch_size]
            feeds = {self.input_name: chunk}
            if self.needs_keypoints:
                feeds[self.keypoint_name] = keypoints[i : i + self.batch_size]
            outs.append(self.session.run(None, feeds)[0])
        emb = np.concatenate(outs, axis=0).astype(np.float32)

        if normalize:
            emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        return emb
