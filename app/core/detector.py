"""Face detection.

YOLOv8n-face is the default.  Measured on the deployment GPU against YuNet, the
other candidate, over the resolutions this corridor geometry actually needs:

    resolution     YOLOv8n-face (CUDA)    YuNet (OpenCV DNN, CPU)
    640x360               6.2 ms                  8.8 ms
    1280x736              7.5 ms                 46.8 ms
    1920x1088             ~8 ms                 139.0 ms

Both found 270/270 faces in the enrolment gallery with near-identical box sizes,
so accuracy did not separate them and speed did.

The YuNet implementation was removed on 2026-08-25 along with its weights: it
was never selected (`detector_kind` has always been "yolo"), it only made sense
on a CPU-only host, and this pipeline already refuses to start without CUDA.
Restoring it means re-adding a class here and downloading
`face_detection_yunet_2023mar.onnx` from OpenCV Zoo.

This detector runs only during ENROLMENT, from photographs. The live pipeline
tracks and aligns from head boxes and never calls it.
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



def build_detector(kind: str, model_path, **kw):
    if kind == "yolo":
        return YoloFaceDetector(model_path, **kw)
    raise ValueError(
        f"unknown detector: {kind!r}. Only 'yolo' is available; the YuNet "
        "path and its weights were removed as unused."
    )
