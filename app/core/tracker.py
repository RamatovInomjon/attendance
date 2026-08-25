"""Per-camera ByteTrack.

The Ultralytics BYTETracker implementation, wrapped so that **each camera
constructs its own instance**.  The previous codebase called
`model.track(persist=True)` on a single module-level YOLO object from every
camera thread; Ultralytics keeps tracker state on that object, so one tracker
received interleaved frames from two different scenes and track IDs migrated
across cameras.  Nothing here is shared between cameras.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker


class _Dets:
    """Minimal stand-in for an Ultralytics Boxes object.

    BYTETracker.update() reads .conf, .xywh and .cls; supplying just those keeps
    the detector decoupled from Ultralytics' result types.
    """

    def __init__(self, boxes_xyxy: np.ndarray, scores: np.ndarray):
        if len(boxes_xyxy) == 0:
            self.xywh = np.zeros((0, 4), dtype=np.float32)
            self.conf = np.zeros((0,), dtype=np.float32)
            self.cls = np.zeros((0,), dtype=np.float32)
            return
        b = np.asarray(boxes_xyxy, dtype=np.float32)
        cx = (b[:, 0] + b[:, 2]) / 2.0
        cy = (b[:, 1] + b[:, 3]) / 2.0
        w = b[:, 2] - b[:, 0]
        h = b[:, 3] - b[:, 1]
        self.xywh = np.stack([cx, cy, w, h], axis=1)
        self.conf = np.asarray(scores, dtype=np.float32)
        self.cls = np.zeros(len(b), dtype=np.float32)

    def __len__(self):
        return len(self.conf)


class FaceTracker:
    """Wraps BYTETracker; returns (track_id, xyxy, score) per update."""

    def __init__(
        self,
        frame_rate: int = 12,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.2,
        new_track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
    ):
        args = SimpleNamespace(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            fuse_score=False,
        )
        self.tracker = BYTETracker(args, frame_rate=frame_rate)

    def update(self, boxes_xyxy, scores) -> list[tuple[int, np.ndarray, float]]:
        tracks = self.tracker.update(_Dets(boxes_xyxy, scores))
        out = []
        for t in tracks:
            # Ultralytics returns [x1, y1, x2, y2, track_id, score, cls, idx]
            x1, y1, x2, y2 = t[:4]
            tid = int(t[4])
            score = float(t[5])
            out.append((tid, np.array([x1, y1, x2, y2], dtype=np.float32), score))
        return out

    def reset(self):
        self.tracker.reset()
