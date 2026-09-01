"""Person re-identification: a body crop in, one embedding out.

The face recogniser answers "who is this" from 112x112 of face. This answers
"is this the same body" from a whole person, which is what survives when the
face is turned away, too small, or never enrolled at all.

Model: AdaFace-style BNNeck ReID net trained in the research project at
phd/dissertatsiya2 and exported by `scripts/export_reid.py`. On its cross-camera
test set - gallery = Entrance, query = Exit, the direction this corridor
actually runs - the deployed checkpoint scores mAP 92.24 / Rank-1 93.49, with
FNIR 31.10% at a 10% false-positive identification rate.

THAT FNIR IS THE POINT TO REMEMBER: roughly a third of genuine cross-camera
queries are missed. This is a data-collection aid and a hint, never an identity
for attendance.

THE PREPROCESSING IS NOT OPTIONAL AND NOT GUESSABLE
---------------------------------------------------
Nothing in an ONNX graph records that its input must be RGB, squashed to
256x128 without preserving aspect ratio, scaled to [0,1] and normalised by the
ImageNet statistics. Get any of it wrong and the model returns embeddings of
exactly the right shape and unit norm that are quietly wrong - every stored
feature and every match wrong with them, and nothing in the numbers to say so.

So `scripts/export_reid.py` writes the convention into the model's metadata and
this class reads it back rather than restating it. `bench/eval_reid.py` then
scores the result against the published figure; that comparison is what proves
the two halves agree.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from app.core.model_vault import load_model
from app.core.onnx_env import best_providers

log = logging.getLogger(__name__)

# Fallbacks for a model exported before the metadata was written. They match
# tools/reid/reid_data.py:120-136; a model that disagrees must carry its own.
_DEFAULT_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_STD = (0.229, 0.224, 0.225)


class PersonReID:
    """(N body crops, BGR uint8) -> (N, D) L2-normalized embeddings."""

    def __init__(self, model_path: str | Path, providers=None, batch_size: int = 8):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            load_model(model_path), sess_options=opts,
            providers=providers or best_providers())
        self.model_name = Path(str(model_path)).name.replace(".enc", "")
        self.batch_size = batch_size

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.dtype = np.float16 if "float16" in inp.type else np.float32
        extra = self.session.get_inputs()[1:]
        if extra:
            raise ValueError(
                f"{self.model_name} takes {len(extra) + 1} inputs; this pipeline "
                f"supplies only the body crop.")

        meta = dict(self.session.get_modelmeta().custom_metadata_map)

        # Shape from the graph where it is static, from metadata otherwise.
        _, _, gh, gw = inp.shape
        self.height = int(meta.get("reid_height") or (gh if isinstance(gh, int) else 256))
        self.width = int(meta.get("reid_width") or (gw if isinstance(gw, int) else 128))
        self.colour = meta.get("reid_colour", "RGB").upper()
        self.resize_mode = meta.get("reid_resize", "squash")
        self.mean = np.asarray(json.loads(meta["reid_mean"]) if "reid_mean" in meta
                               else _DEFAULT_MEAN, np.float32).reshape(3, 1, 1)
        self.std = np.asarray(json.loads(meta["reid_std"]) if "reid_std" in meta
                              else _DEFAULT_STD, np.float32).reshape(3, 1, 1)
        out = self.session.get_outputs()[0]
        self.dim = int(out.shape[-1]) if isinstance(out.shape[-1], int) else 0

        if "reid_mean" not in meta:
            log.warning(
                "%s carries no preprocessing metadata; assuming RGB/squash/"
                "ImageNet norm. Re-export with scripts/export_reid.py - a wrong "
                "assumption here degrades matching silently.", self.model_name)
        log.info("reid %s: %dx%d embed=%d %s", self.model_name,
                 self.height, self.width, self.dim, self.provider)

    @property
    def provider(self) -> str:
        return self.session.get_providers()[0]

    def preprocess(self, crops_bgr) -> np.ndarray:
        """BGR uint8 crops -> (N,3,H,W). Squash, not letterbox: the model was
        trained on `T.Resize((h, w))`, which does not preserve aspect ratio.
        Padding to preserve it would put the model outside its training
        distribution."""
        out = np.empty((len(crops_bgr), 3, self.height, self.width), np.float32)
        for i, bgr in enumerate(crops_bgr):
            img = (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                   if self.colour == "RGB" else bgr)
            # cv2.resize takes (width, height); torchvision takes (h, w).
            img = cv2.resize(img, (self.width, self.height),
                             interpolation=cv2.INTER_LINEAR)
            chw = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            out[i] = (chw - self.mean) / self.std
        return out

    def embed(self, crops_bgr, normalize: bool = True) -> np.ndarray:
        if not len(crops_bgr):
            return np.zeros((0, self.dim or 512), np.float32)
        batch = self.preprocess(crops_bgr).astype(self.dtype)
        outs = []
        for i in range(0, len(batch), self.batch_size):
            outs.append(self.session.run(
                None, {self.input_name: batch[i:i + self.batch_size]})[0])
        emb = np.concatenate(outs, axis=0).astype(np.float32)
        if normalize:                      # the graph already does; cheap insurance
            emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        return emb


def sharpness_of(crop_bgr: np.ndarray) -> float:
    """Laplacian variance, the quality term of the tracklet weighting."""
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def aggregate(embs: np.ndarray, scores=None, sharpness=None) -> np.ndarray:
    """Several crops of one pass -> one tracklet feature.

    Quality-weighted mean of the L2-normalised embeddings, renormalised:

        w = score^1.0 * sharpness_norm^0.5

    from the research project's `sifat_treklet.py`. Aggregation is worth about
    +4 mAP across every model tested, for no inference cost - a pass already
    produces several crops, so this is free accuracy.

    The weighting is not decoration. A PLAIN mean makes FNIR WORSE: blurred and
    half-occluded crops drag the template toward a generic body, and the
    template then matches everybody slightly. Weighting by detector confidence
    and sharpness is what removes that regression.
    """
    embs = np.atleast_2d(np.asarray(embs, np.float32))
    if len(embs) == 1:
        return embs[0] / (np.linalg.norm(embs[0]) + 1e-12)

    w = np.ones(len(embs), np.float32)
    if scores is not None:
        w = w * np.clip(np.asarray(scores, np.float32), 1e-3, None)
    if sharpness is not None:
        s = np.asarray(sharpness, np.float32)
        # Normalised within the pass: absolute Laplacian variance depends on
        # crop size and lighting, so only the relative ordering is meaningful.
        s = s / (s.max() + 1e-6) if s.max() > 0 else np.ones_like(s)
        w = w * np.sqrt(np.clip(s, 1e-3, None))

    f = (embs * w[:, None]).sum(axis=0) / (w.sum() + 1e-12)
    return f / (np.linalg.norm(f) + 1e-12)
