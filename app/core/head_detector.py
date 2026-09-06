"""CrowdHuman YOLOv8n person/head detector, used for *tracking*, not recognition.

Why a second detector
---------------------
Face detection is intermittent by nature: it drops the moment somebody turns,
looks down, or is motion-blurred. Measured on this site, that left tracks with a
median of 1-4 usable frames, which is fatal twice over — K-of-N voting needs 3
frames to commit an identity, and direction needs 5 trajectory points plus real
travel. Both starve, so people go unrecognized and their direction is unknown.

A head stays visible through all of that. Tracking heads gives one continuous
track per person-pass; face detection and recognition then run *inside* that
track, on whichever frames actually contain a usable face. The track carries the
identity and the trajectory; the face only has to be good once.

Output layout: [1, 6, N] -> 4 box (cx,cy,w,h) + 2 class scores
(0 = head, 1 = person), in letterboxed input coordinates. N depends on the
export geometry: 18900 at 960x960, 10710 at 960x544.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from app.core.onnx_env import best_providers
from app.core.model_vault import load_model

# Verified against real frames: class 0 boxes have a median size of 40 px
# and class 1 of 276 px on the same images, so 0 is the head.
CLS_HEAD, CLS_PERSON = 0, 1


@dataclass
class HeadDetection:
    box: np.ndarray      # xyxy in original-frame pixels
    score: float
    cls: int             # 0 head, 1 person (CLS_HEAD / CLS_PERSON)


class HeadDetector:
    def __init__(self, model_path: str | Path, size: int | None = None,
                 conf: float = 0.35, iou: float = 0.5, providers=None):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        # load_model returns a path for a plain .onnx (ONNX Runtime mmaps it)
        # or decrypted bytes for a licensed .onnx.enc, so protected weights
        # are never written to disk in the clear.
        self.session = ort.InferenceSession(load_model(model_path), sess_options=opts,
                                            providers=providers or best_providers())
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name

        # Take the geometry from the model, not from config. The 960x544 export
        # is rectangular: a 16:9 frame letterboxed into a square wastes 44% of
        # every buffer on grey padding that the CPU converts and the GPU
        # convolves. `size` is honoured only as a fallback for dynamic-axis
        # models, and kept for call-site compatibility.
        _, _, h, w = inp.shape
        self.in_h = int(h) if isinstance(h, int) else int(size or 640)
        self.in_w = int(w) if isinstance(w, int) else int(size or 640)
        self.size = max(self.in_h, self.in_w)

        self.conf = conf
        self.iou = iou

        # Preallocated once, reused for every frame. blobFromImage allocates a
        # fresh NCHW float32 buffer per call - 11 MB at 960x960 - and on this
        # host (13 GB RAM, ~1 GB free, 4 GB swapped) that allocation costs
        # 6 ms of page faults before a single pixel is converted. Measured
        # end-to-end: 41.8 ms per frame via blobFromImage at 960x960 against
        # 4.4 ms here at 960x544, a 9.5x reduction with identical output.
        self._blob = np.empty((1, 3, self.in_h, self.in_w), np.float32)
        self._canvas = np.full((self.in_h, self.in_w, 3), 114, np.uint8)
        self._rgb = np.empty((self.in_h, self.in_w, 3), np.uint8)

    @property
    def provider(self) -> str:
        return self.session.get_providers()[0]

    @property
    def input_shape(self) -> tuple[int, int]:
        return (self.in_h, self.in_w)

    def _letterbox(self, img):
        """Scale into the model's rectangle, preserving aspect. Reuses the canvas."""
        h, w = img.shape[:2]
        r = min(self.in_h / h, self.in_w / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        top, left = (self.in_h - nh) // 2, (self.in_w - nw) // 2
        if nh != self.in_h or nw != self.in_w:
            self._canvas[:] = 114          # only repaint when padding shows
        cv2.resize(img, (nw, nh), dst=self._canvas[top:top + nh, left:left + nw],
                   interpolation=cv2.INTER_LINEAR)
        return self._canvas, r, left, top

    def _to_blob(self, canvas):
        """BGR HWC uint8 -> RGB NCHW float32 in [0,1], into the preallocated buffer."""
        cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB, dst=self._rgb)
        # transpose gives a view; np.divide writes straight into the buffer with
        # the cast folded in, so nothing is allocated on this path.
        np.divide(self._rgb.transpose(2, 0, 1), 255.0, out=self._blob[0],
                  casting="unsafe")
        return self._blob

    def detect(self, frame_bgr: np.ndarray, want: int | None = CLS_HEAD) -> list[HeadDetection]:
        canvas, r, dx, dy = self._letterbox(frame_bgr)
        blob = self._to_blob(canvas)
        out = self.session.run(None, {self.input_name: blob})[0]

        pred = out[0].T                      # (anchors, 6)
        boxes_c, scores_c = pred[:, :4], pred[:, 4:]
        cls = scores_c.argmax(1)
        conf = scores_c.max(1)

        keep = conf >= self.conf
        if want is not None:
            keep &= (cls == want)
        if not keep.any():
            return []
        b, c, k = boxes_c[keep], conf[keep], cls[keep]

        xy = np.empty_like(b)
        xy[:, 0] = b[:, 0] - b[:, 2] / 2
        xy[:, 1] = b[:, 1] - b[:, 3] / 2
        xy[:, 2] = b[:, 0] + b[:, 2] / 2
        xy[:, 3] = b[:, 1] + b[:, 3] / 2
        xy[:, [0, 2]] -= dx
        xy[:, [1, 3]] -= dy
        xy /= r

        # Suppression is PER CLASS. The pipeline asks for both classes in one
        # pass, and a person box that is only head and shoulders - somebody
        # at the bottom of the frame, the closest and best face there is -
        # overlaps its own head well above the IoU limit, so class-agnostic
        # NMS dropped whichever of the two scored lower: no head, no
        # recognition on that frame; no person, a hole in the track.
        rects = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in xy]
        idx = _nms_per_class(rects, c, k, self.conf, self.iou)
        if len(idx) == 0:
            return []
        return [HeadDetection(xy[i], float(c[i]), int(k[i])) for i in idx]


def _nms_per_class(rects, conf: np.ndarray, cls: np.ndarray, thr: float, iou: float) -> list[int]:
    if hasattr(cv2.dnn, "NMSBoxesBatched"):
        idx = cv2.dnn.NMSBoxesBatched(rects, conf.tolist(), cls.tolist(), thr, iou)
        return [int(i) for i in np.array(idx).ravel()] if len(idx) else []
    keep: list[int] = []
    for c in np.unique(cls):
        members = np.flatnonzero(cls == c)
        idx = cv2.dnn.NMSBoxes([rects[i] for i in members], conf[members].tolist(), thr, iou)
        if len(idx):
            keep.extend(int(members[i]) for i in np.array(idx).ravel())
    return keep
