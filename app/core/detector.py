"""Face detection.

YOLOv8n-face is the default.  Measured on the deployment GPU against YuNet, the
other candidate, over the resolutions this corridor geometry actually needs:

    resolution     YOLOv8n-face (CUDA)    YuNet (OpenCV DNN, CPU)
    640x360               6.2 ms                  8.8 ms
    1280x736              7.5 ms                 46.8 ms
    1920x1088             ~8 ms                 139.0 ms

Both found 270/270 faces in the enrolment gallery with near-identical box sizes,
so accuracy did not separate them and speed did.  YuNet stays selectable: it is
the better choice on a CPU-only host, where YOLO has no GPU to run on.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Detection:
    box: np.ndarray   # (4,) xyxy in original-frame pixels
    score: float


class YoloFaceDetector:
    def __init__(self, model_path: str | Path, imgsz: int = 1280, conf: float = 0.35, device: int = 0):
        import logging
        from ultralytics import YOLO

        logging.getLogger("ultralytics").setLevel(logging.ERROR)
        self.model = YOLO(str(model_path))
        self.model.to(f"cuda:{device}" if device >= 0 else "cpu")
        self.imgsz = imgsz
        self.conf = conf
        self.device = device

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        r = self.model.predict(
            frame_bgr, verbose=False, device=self.device, imgsz=self.imgsz, conf=self.conf
        )[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        return [Detection(b, float(s)) for b, s in zip(boxes, scores)]


class YuNetDetector:
    """CPU fallback.  Recreated on input-size change, as the API requires."""

    def __init__(self, model_path: str | Path, conf: float = 0.6, nms: float = 0.3):
        self.model_path = str(model_path)
        self.conf = conf
        self.nms = nms
        self._det = None
        self._size = None

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        if self._det is None or self._size != (w, h):
            self._det = cv2.FaceDetectorYN.create(self.model_path, "", (w, h), self.conf, self.nms, 5000)
            self._size = (w, h)
        _, faces = self._det.detect(frame_bgr)
        if faces is None:
            return []
        out = []
        for f in faces:
            x, y, bw, bh = f[:4]
            out.append(Detection(np.array([x, y, x + bw, y + bh], dtype=np.float32), float(f[-1])))
        return out


def build_detector(kind: str, model_path, **kw):
    if kind == "yolo":
        return YoloFaceDetector(model_path, **kw)
    if kind == "yunet":
        return YuNetDetector(model_path, **kw)
    raise ValueError(f"unknown detector: {kind}")
