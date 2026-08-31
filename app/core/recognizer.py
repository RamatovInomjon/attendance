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

        # One input, one output. A model wanting anything else is not a drop-in
        # replacement and must not be treated as one - it would be fed nothing
        # for its second input and return plausible-looking vectors that are
        # quietly wrong, in the gallery and in every live match at once.
        extra = self.session.get_inputs()[1:]
        if extra:
            raise ValueError(
                f"{Path(str(model_path)).name} takes {len(extra) + 1} inputs "
                f"({', '.join(i.name for i in extra)} besides the image). This "
                f"pipeline supplies only the aligned face.")

        # A fixed batch dimension is a research export; run it one row at a time
        # rather than failing, and say so, because it costs ~2.4x.
        b = inp.shape[0]
        self.fixed_batch = b if isinstance(b, int) and b > 0 else None
        if self.fixed_batch:
            log.warning("recognizer %s has a FIXED batch of %d - a research "
                        "export. Throughput will suffer: the pipeline embeds a "
                        "whole frame's faces at once and this forces them one "
                        "at a time. Re-export with a dynamic batch dimension.",
                        Path(str(model_path)).name, self.fixed_batch)
            self.batch_size = self.fixed_batch

    @property
    def provider(self) -> str:
        return self.session.get_providers()[0]

    def embed(self, aligned: np.ndarray, normalize: bool = True) -> np.ndarray:
        """(N,3,112,112) in [-1,1] -> (N,512) L2-normalized embeddings."""
        aligned = np.asarray(aligned)
        if aligned.ndim == 3:
            aligned = aligned[None]
        if aligned.dtype != self.dtype:
            aligned = aligned.astype(self.dtype)

        outs = []
        for i in range(0, len(aligned), self.batch_size):
            outs.append(self.session.run(
                None, {self.input_name: aligned[i : i + self.batch_size]})[0])
        emb = np.concatenate(outs, axis=0).astype(np.float32)

        if normalize:
            emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        return emb
