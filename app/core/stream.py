"""RTSP frame source.

Three behaviours the previous reader got wrong, all of which produce wrong
attendance data rather than a visible crash:

* **A dead camera must not replay its last frame.**  The old loop substituted
  `last_good_frame` on read failure and pushed it into the processing queue, so
  a camera that had been offline for hours kept generating recognitions from the
  face frozen in that frame.  Here a stale source stops producing entirely and
  reports `is_stale`.
* **Reconnect must use the same construction path.**  The old code opened with
  `CAP_FFMPEG` plus options and reconnected with a bare `VideoCapture(source)`.
* **The queue is drop-oldest.**  Recognition wants the newest frame; a backlog
  is latency, not work.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Frame:
    image: np.ndarray      # BGR, full resolution
    ts: float              # wall-clock capture time (time.time())
    index: int


class RtspSource:
    """Background reader thread with bounded drop-oldest queue and backoff."""

    BACKOFF = (1, 2, 4, 8, 15, 30)

    def __init__(
        self,
        url: str,
        name: str = "cam",
        transport: str = "tcp",
        queue_size: int = 2,
        stale_after_s: float = 5.0,
        open_timeout_ms: int = 8000,
    ):
        self.url = url
        self.name = name
        self.transport = transport
        self.stale_after_s = stale_after_s
        self.open_timeout_ms = open_timeout_ms

        self.q: queue.Queue[Frame] = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self.connected = False
        self.last_frame_ts = 0.0
        self.frames_read = 0
        self.reconnects = 0
        self.read_failures = 0
        self.fps = 0.0
        self.width = 0
        self.height = 0

    # -- lifecycle --------------------------------------------------------
    def start(self) -> "RtspSource":
        self._thread = threading.Thread(target=self._run, name=f"rtsp-{self.name}", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    @property
    def is_stale(self) -> bool:
        return (time.time() - self.last_frame_ts) > self.stale_after_s

    # -- internals --------------------------------------------------------
    def _open(self) -> cv2.VideoCapture | None:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{self.transport}|fflags;nobuffer|flags;low_delay"
            f"|stimeout;{self.open_timeout_ms * 1000}"
        )
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cap

    def _put(self, frame: Frame):
        """Drop-oldest: latency matters more than completeness."""
        try:
            self.q.put_nowait(frame)
        except queue.Full:
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(frame)
            except queue.Full:
                pass

    def _run(self):
        attempt = 0
        cap = None
        t_fps = time.time()
        n_fps = 0

        while not self._stop.is_set():
            if cap is None:
                cap = self._open()
                if cap is None:
                    delay = self.BACKOFF[min(attempt, len(self.BACKOFF) - 1)]
                    attempt += 1
                    self.connected = False
                    log.warning("[%s] open failed, retry in %ds (attempt %d)", self.name, delay, attempt)
                    self._stop.wait(delay)
                    continue
                attempt = 0
                self.connected = True
                self.reconnects += 1
                log.info("[%s] connected %dx%d", self.name, self.width, self.height)

            ok, img = cap.read()
            if not ok or img is None:
                self.read_failures += 1
                # A few dropped frames are normal; a run of them is a dead link.
                if self.read_failures >= 30:
                    log.warning("[%s] stream lost, reconnecting", self.name)
                    cap.release()
                    cap = None
                    self.connected = False
                    self.read_failures = 0
                    self._stop.wait(1.0)
                continue

            self.read_failures = 0
            self.frames_read += 1
            self.last_frame_ts = time.time()
            self._put(Frame(img, self.last_frame_ts, self.frames_read))

            n_fps += 1
            if self.last_frame_ts - t_fps >= 2.0:
                self.fps = n_fps / (self.last_frame_ts - t_fps)
                n_fps = 0
                t_fps = self.last_frame_ts

        if cap is not None:
            cap.release()
        self.connected = False
        log.info("[%s] reader stopped", self.name)

    def read(self, timeout: float = 1.0) -> Frame | None:
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stats(self) -> dict:
        return {
            "name": self.name, "connected": self.connected, "stale": self.is_stale,
            "fps": round(self.fps, 1), "frames": self.frames_read,
            "reconnects": self.reconnects, "resolution": f"{self.width}x{self.height}",
        }
