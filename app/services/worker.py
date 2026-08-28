"""Camera worker: joins the CV pipeline to the durable attendance state.

Runs one camera end to end — stream, recognition, event persistence — and keeps
a small snapshot of live state for the UI.  Designed to be run either in a
thread (one per camera) or as its own process; nothing here is shared between
cameras except the read-only gallery matrix.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from app.core.gallery import Gallery
from app.core.geometry import aligned_to_uint8
from app.core.direction import DirectionConfig
from app.core.pipeline import CameraPipeline, FrameResult
from app.core.stream import RtspSource
from app.db.models import CameraRole, UnknownSighting
from app.db.session import session_scope
from app.services.attendance import AttendanceService
from app.services.debug_capture import DebugCapture
from app.services.recorder import ClipRecorder

log = logging.getLogger(__name__)


class CameraWorker:
    def __init__(self, camera_id: int, name: str, role: CameraRole, rtsp_url: str,
                 gallery: Gallery, direction_cfg: DirectionConfig | None = None):
        self.camera_id = camera_id
        self.name = name
        self.role = role
        self.rtsp_url = rtsp_url
        self.direction_cfg = direction_cfg or DirectionConfig()

        self.source = RtspSource(
            rtsp_url, name=name, transport=settings.rtsp_transport,
            queue_size=settings.frame_queue_size, stale_after_s=settings.stale_after_s,
        )
        self.pipeline = CameraPipeline(name, gallery, direction_cfg=self.direction_cfg)
        self.attendance = AttendanceService()
        self.debug = DebugCapture()
        self.recorder = (ClipRecorder(
            name, fps=max(1.0, 20.0 / settings.process_every_nth),
            pre_roll_s=settings.record_pre_roll_s,
            post_roll_s=settings.record_post_roll_s,
            width=settings.record_width,
            min_free_gb=settings.record_min_free_gb,
            max_clip_s=settings.record_max_clip_s,
        ) if settings.record_clips else None)

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # live state for the UI
        self.latest: FrameResult | None = None
        self.latest_jpeg: bytes | None = None
        self._lock = threading.Lock()
        self.recent_events: list[dict] = []
        self.frames_seen = 0
        # Person-pass accounting. A completed track is roughly one pass, and is
        # the denominator for "how many of the people who walked by did we
        # recognize" - a question recognition events alone cannot answer.
        self.passes_total = 0
        self.passes_recognized = 0
        self.passes_unknown = 0
        self.passes_by_direction = {"ENTER": 0, "EXIT": 0, "UNKNOWN": 0}
        # A crash inside the per-frame path is silent from the outside: frames
        # keep flowing, timings look fine, and recognition simply stops. One
        # such bug ran 229 times before anyone noticed, so the count is exposed
        # on /api/health where it can be seen without reading the log.
        self.pipeline_errors = 0
        self.last_error = ""

        self.snapshot_dir = settings.media_dir / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    # -- lifecycle --------------------------------------------------------
    def start(self):
        self.source.start()
        self._thread = threading.Thread(target=self._run, name=f"worker-{self.name}", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self.recorder is not None:
            self.recorder.stop()
        self.source.stop()
        if self._thread:
            self._thread.join(timeout=5.0)

    # -- helpers ----------------------------------------------------------
    def _scale_box(self, box, ctx_width: int):
        full_w = self.source.width or ctx_width
        r = ctx_width / float(full_w) if full_w else 1.0
        return [v * r for v in box]

    def _save_snapshot(self, crop: np.ndarray | None, employee_id: int, kind: str) -> str | None:
        if crop is None:
            return None
        try:
            rel = f"snapshots/{kind}_{employee_id}_{int(time.time())}.jpg"
            rgb = aligned_to_uint8(crop)
            cv2.imwrite(str(settings.media_dir / rel), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            return rel
        except Exception:
            log.exception("[%s] snapshot write failed", self.name)
            return None

    def _save_body(self, crop, employee_id: int) -> str | None:
        """Write a raw BGR body crop. Separate from _save_snapshot because that
        one expects an aligned CHW face in [-1, 1]; this is already an image."""
        if crop is None:
            return None
        try:
            rel = f"snapshots/body_{employee_id}_{int(time.time())}.jpg"
            cv2.imwrite(str(settings.media_dir / rel), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            return rel
        except Exception:
            log.exception("[%s] body snapshot write failed", self.name)
            return None

    def _persist_completed(self, res: FrameResult):
        """One attendance decision per completed track — per person-pass.

        Identity and direction mature at different points in a walk: the vote
        commits within a few good frames, while direction needs real travel.
        Deciding at completion is the only moment both are final.
        """
        if not res.completed:
            return
        from app.services.attendance import business_date
        ts = datetime.fromtimestamp(res.frame.ts, tz=timezone.utc)
        bdate = business_date(ts)
        with session_scope() as s:
            for ct in res.completed:
                self.passes_total += 1
                self.passes_by_direction[ct.direction] = \
                    self.passes_by_direction.get(ct.direction, 0) + 1

                if ct.employee_id is None:
                    self.passes_unknown += 1
                    snap = self._save_snapshot(ct.crop, 0, "unknown")
                    s.add(UnknownSighting(
                        camera_id=self.camera_id, track_id=ct.track_id,
                        # first_seen was written equal to last_seen, so every
                        # duration in the table read 0.00s and the column could
                        # never answer "how long was this person in view" - the
                        # signal that separates a real transit from somebody
                        # standing still. duration_s is on the track; use it.
                        first_seen=ts - timedelta(seconds=ct.duration_s),
                        last_seen=ts, business_date=bdate,
                        frames=ct.embedded_frames, best_score=ct.best_score,
                        nearest_employee_id=ct.nearest_employee_id,
                        vector=ct.vector.astype("float32").tobytes() if ct.vector is not None else None,
                        snapshot=snap,
                    ))
                    log.info("[%s] MISS best=%.3f nearest=%-20s embedded=%d traj=%d "
                             "travel=%.3f dur=%.1fs dir=%s (%s)",
                             self.name, ct.best_score,
                             self.pipeline.gallery.name(ct.nearest_employee_id)[:20],
                             ct.embedded_frames, ct.traj_points, ct.travel,
                             ct.duration_s, ct.direction, ct.direction_reason)
                    continue

                self.passes_recognized += 1
                # The dashboard shows the BODY: a person is recognisable to a
                # human by build, clothing and posture, where a 112x112 aligned
                # face often is not - especially on the crops that turn out to
                # be wrong. The aligned face is still written next to it under
                # "evt_", so a questionable match can still be examined.
                face_snap = self._save_snapshot(ct.crop, ct.employee_id, "evt")
                snap = self._save_body(ct.person_crop, ct.employee_id) or face_snap
                # The frame that actually drove the match, kept beside the clear
                # one: on a wrong answer this is the frame that explains it.
                if ct.score_crop is not None:
                    self._save_snapshot(ct.score_crop, ct.employee_id, "why")
                # Capture the track's BEST frame, not the frame it happened to
                # be pruned on - by then the person has already walked out.
                if ct.context is not None and ct.box is not None:
                    try:
                        self.debug.capture(
                            name=ct.name, frame_bgr=ct.context, box=self._scale_box(
                                ct.box, ct.context.shape[1]),
                            aligned_chw=ct.crop, camera=self.name, role=self.role.value,
                            score=ct.best_score, margin=ct.best_margin, track_id=ct.track_id,
                            ts=ts, quality=ct.quality, native=ct.native,
                            extra={"embedded_frames": ct.embedded_frames,
                                   "traj_points": ct.traj_points,
                                   "travel": round(ct.travel, 3),
                                   "duration_s": round(ct.duration_s, 1),
                                   "direction": ct.direction,
                                   "direction_reason": ct.direction_reason,
                                   "employee_id": ct.employee_id},
                        )
                    except Exception:
                        log.exception("[%s] debug capture failed", self.name)
                d = self.attendance.record(
                    s, employee_id=ct.employee_id, camera_id=self.camera_id,
                    role=self.role, ts=ts, score=ct.best_score, margin=ct.best_margin,
                    track_id=ct.track_id, face_px=ct.face_px,
                    votes=f"emb{ct.embedded_frames}/traj{ct.traj_points}",
                    snapshot=snap, direction=ct.direction,
                    direction_reason=ct.direction_reason,
                    require_direction=self.direction_cfg.configured,
                )
                entry = {
                    "ts": ts.astimezone(settings.tz).strftime("%H:%M:%S"),
                    "name": ct.name, "employee_id": ct.employee_id,
                    "camera": self.name, "role": self.role.value,
                    "score": round(ct.best_score, 3), "transition": d.transition,
                    "snapshot": snap, "direction": ct.direction,
                }
                self.recent_events.insert(0, entry)
                del self.recent_events[40:]
                log.info("[%s] %-12s %-20s score=%.3f margin=%.3f embedded=%d traj=%d travel=%.3f "
                         "dur=%.1fs dir=%s (%s)",
                         self.name, d.transition, ct.name[:20], ct.best_score,
                         ct.best_margin, ct.embedded_frames, ct.traj_points, ct.travel,
                         ct.duration_s, ct.direction, ct.direction_reason)

    # -- main loop --------------------------------------------------------
    def _run(self):
        log.info("[%s] worker started (role=%s)", self.name, self.role.value)
        nth = 0
        while not self._stop.is_set():
            frame = self.source.read(timeout=1.0)
            if frame is None:
                continue
            self.frames_seen += 1
            nth += 1
            if nth % settings.process_every_nth:
                continue
            try:
                res = self.pipeline.process(frame)
            except Exception as e:
                self.pipeline_errors += 1
                self.last_error = f"{type(e).__name__}: {e}"[:160]
                log.exception("[%s] pipeline error", self.name)
                continue
            with self._lock:
                self.latest = res

            if self.recorder is not None:
                try:
                    live = [t for t in res.tracks if frame.ts - t.last_seen < 1.0]
                    self.recorder.offer(
                        frame.image, frame.ts, active=bool(live),
                        meta={"names": sorted({t.name for t in live if t.employee_id}),
                              "tracks": [t.track_id for t in live]})
                except Exception:
                    log.exception("[%s] recorder error", self.name)

            if res.candidates:
                ts_c = datetime.fromtimestamp(frame.ts, tz=timezone.utc)
                for c in res.candidates:
                    self.debug.frame(name=c.name, aligned_chw=c.aligned, native=c.native,
                                     score=c.score, camera=self.name, track_id=c.track_id,
                                     ts=ts_c, idx=c.index, quality=c.quality,
                                     accepted=c.accepted)
            try:
                self._persist_completed(res)
            except Exception:
                log.exception("[%s] completed-track persist error", self.name)
        log.info("[%s] worker stopped", self.name)

    # -- UI ---------------------------------------------------------------
    def render(self, max_width: int = 1280) -> bytes | None:
        """Annotated JPEG of the most recent processed frame."""
        with self._lock:
            res = self.latest
        if res is None:
            return None

        img = res.frame.image
        h, w = img.shape[:2]
        scale = min(1.0, max_width / w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            img = img.copy()

        now = time.time()
        for t in res.tracks:
            if now - t.last_seen > 2.0:
                continue
            x1, y1, x2, y2 = (t.box * scale).astype(int)
            known = t.employee_id is not None
            color = (0, 200, 0) if known else (40, 120, 240)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = t.name if known else (t.last_reason or "…")
            if known and t.score:
                label += f" {t.score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(img, label, (x1 + 3, max(10, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        banner = f"{self.name} [{self.role.value}]  {self.source.fps:.0f}fps  {res.timings['total']:.0f}ms"
        cv2.putText(img, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None

    def stats(self) -> dict:
        with self._lock:
            res = self.latest
        return {
            "camera_id": self.camera_id, "name": self.name, "role": self.role.value,
            "stream": self.source.stats(),
            "frames_processed": self.pipeline.frames_processed,
            "faces_embedded": self.pipeline.faces_embedded,
            "active_tracks": len(self.pipeline.tracks),
            "pipeline_errors": self.pipeline_errors,
            "recorder": self.recorder.stats() if self.recorder else None,
            "last_error": self.last_error,
            "passes": {
                "total": self.passes_total,
                "recognized": self.passes_recognized,
                "unknown": self.passes_unknown,
                "by_direction": dict(self.passes_by_direction),
            },
            "timings": res.timings if res else {},
        }
