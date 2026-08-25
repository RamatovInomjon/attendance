"""Event-triggered clip recorder.

Continuous 4K recording is not affordable here: two streams at 16 Mbps is
~14.4 GB/hour, so a twelve-hour day would need ~173 GB against ~104 GB free.
Almost all of that would be an empty corridor.

So this keeps a rolling pre-roll buffer and only commits a clip while a track is
alive, plus padding either side. What lands on disk is the person-passes — the
only footage worth re-running an algorithm against — at a small fraction of the
size.

Every clip is written alongside a JSON sidecar naming what the live system
decided, so a recording can be replayed and the new answer compared against the
old one rather than judged by eye.
"""
from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from app.config import settings

log = logging.getLogger(__name__)


class ClipRecorder:
    def __init__(self, camera_name: str, out_dir: Path | None = None,
                 fps: float = 10.0, pre_roll_s: float = 2.0, post_roll_s: float = 2.0,
                 width: int | None = None, min_free_gb: float = 10.0,
                 max_clip_s: float = 60.0):
        self.camera = camera_name
        self.dir = Path(out_dir or settings.data_dir / "recordings") / camera_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.pre = deque(maxlen=max(1, int(pre_roll_s * fps)))
        self.post_roll_s = post_roll_s
        self.width = width                 # None keeps native resolution
        self.min_free_gb = min_free_gb
        self.max_clip_s = max_clip_s

        self._writer: cv2.VideoWriter | None = None
        self._path: Path | None = None
        self._started: float = 0.0
        self._last_active: float = 0.0
        self._frames = 0
        self._meta: dict = {}
        self._lock = threading.Lock()

        self.clips_written = 0
        self.frames_written = 0
        self.skipped_low_disk = 0

    # -- disk ------------------------------------------------------------
    def _free_gb(self) -> float:
        try:
            return shutil.disk_usage(self.dir).free / 1073741824
        except OSError:
            return 0.0

    def _scale(self, frame: np.ndarray) -> np.ndarray:
        if self.width is None or frame.shape[1] <= self.width:
            return frame
        h = int(round(frame.shape[0] * self.width / frame.shape[1]))
        return cv2.resize(frame, (self.width, h), interpolation=cv2.INTER_AREA)

    # -- main entry ------------------------------------------------------
    def offer(self, frame: np.ndarray, ts: float, active: bool, meta: dict | None = None):
        """Feed every processed frame. ``active`` = a track is currently alive."""
        with self._lock:
            small = self._scale(frame)

            if self._writer is None:
                self.pre.append(small)
                if not active:
                    return
                if self._free_gb() < self.min_free_gb:
                    self.skipped_low_disk += 1
                    if self.skipped_low_disk % 50 == 1:
                        log.warning("[%s] recording skipped: only %.1f GB free (need %.1f)",
                                    self.camera, self._free_gb(), self.min_free_gb)
                    return
                self._open(small, ts, meta or {})

            if active:
                self._last_active = ts
            if meta:
                self._meta.update(meta)

            self._writer.write(small)
            self._frames += 1
            self.frames_written += 1

            over = (ts - self._started) > self.max_clip_s
            idle = (ts - self._last_active) > self.post_roll_s
            if idle or over:
                self._close(reason="max_clip" if over else "idle")

    def _open(self, frame: np.ndarray, ts: float, meta: dict):
        stamp = datetime.fromtimestamp(ts, tz=settings.tz).strftime("%Y%m%d_%H%M%S")
        self._path = self.dir / f"{stamp}_{self.camera}.mp4"
        h, w = frame.shape[:2]
        self._writer = cv2.VideoWriter(str(self._path),
                                       cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        if not self._writer.isOpened():
            log.error("[%s] could not open %s for writing", self.camera, self._path)
            self._writer = None
            return
        self._started = ts
        self._last_active = ts
        self._frames = 0
        self._meta = dict(meta)
        for f in self.pre:                 # commit the pre-roll
            self._writer.write(f)
            self._frames += 1
        self.pre.clear()

    def _close(self, reason: str = ""):
        if self._writer is None:
            return
        self._writer.release()
        dur = self._frames / self.fps
        if self._path is not None:
            side = self._path.with_suffix(".json")
            side.write_text(json.dumps({
                "camera": self.camera, "clip": self._path.name,
                "started_local": datetime.fromtimestamp(self._started, tz=settings.tz).isoformat(),
                "frames": self._frames, "fps": self.fps,
                "duration_s": round(dur, 2), "closed_because": reason,
                **self._meta,
            }, indent=2, ensure_ascii=False, default=str))
            self.clips_written += 1
            log.info("[%s] clip %s  %.1fs  %d frames  (%s)",
                     self.camera, self._path.name, dur, self._frames, reason)
        self._writer = None
        self._path = None
        self._meta = {}

    def stop(self):
        with self._lock:
            self._close(reason="shutdown")

    def stats(self) -> dict:
        return {"clips": self.clips_written, "frames": self.frames_written,
                "recording": self._writer is not None,
                "free_gb": round(self._free_gb(), 1),
                "skipped_low_disk": self.skipped_low_disk}
