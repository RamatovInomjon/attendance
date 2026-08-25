"""AdaFace IR-101 (WebFace12M) embedding extractor on ONNX Runtime.

Benchmarked on the deployment GPU (RTX 3070 Laptop, CUDA EP):

    ir101 fp32   6.20 ms @ bs1   3.17 ms/face @ bs8
    ir101 fp16   4.57 ms @ bs1   1.90 ms/face @ bs8      <- default
    ir18  fp32   1.43 ms @ bs1   0.81 ms/face @ bs8      <- fallback

FP16 is the default: it is ~1.4x faster than FP32 and the accuracy difference on
the enrolment gallery is below the noise floor (see docs/BENCHMARKS.md).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from app.core.onnx_env import best_providers


class FaceRecognizer:
    """(N, 3, 112, 112) in [-1, 1] -> (N, 512) L2-normalized embeddings."""

    def __init__(self, model_path: str | Path, providers=None, batch_size: int = 8):
        if providers is None:
            providers = best_providers()

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
        self.batch_size = batch_size

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.dtype = np.float16 if "float16" in inp.type else np.float32

    @property
    def provider(self) -> str:
        return self.session.get_providers()[0]

    def embed(self, aligned: np.ndarray, normalize: bool = True) -> np.ndarray:
        aligned = np.asarray(aligned)
        if aligned.ndim == 3:
            aligned = aligned[None]
        if aligned.dtype != self.dtype:
            aligned = aligned.astype(self.dtype)

        outs = []
        for i in range(0, len(aligned), self.batch_size):
            chunk = aligned[i : i + self.batch_size]
            outs.append(self.session.run(None, {self.input_name: chunk})[0])
        emb = np.concatenate(outs, axis=0).astype(np.float32)

        if normalize:
            emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        return emb
