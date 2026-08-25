"""Visual audit trail for recognitions.

For every committed identity (and optionally every gate-passing non-match) this
writes four things into `data/debug/<person>/`:

    <ts>_<score>_frame.jpg    the whole annotated frame, for context
    <ts>_<score>_face.jpg     the native-resolution crop the aligner was given
    <ts>_<score>_aligned.jpg  the 112x112 the recognizer actually saw
    <ts>_<score>.json         scores, margin, pose, gate, camera, track

The aligned crop is the one that matters: if a name is wrong, that image shows
whether the pipeline mis-framed the face or the gallery genuinely contains a
lookalike.  Guessing from the annotated frame alone is not enough.
"""
from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np

from app.config import settings
from app.core.geometry import aligned_to_uint8

log = logging.getLogger(__name__)


class DebugCapture:
    def __init__(self, enabled: bool | None = None, root=None, max_per_person: int | None = None):
        self.enabled = settings.debug_capture if enabled is None else enabled
        self.root = root or settings.debug_dir
        self.max_per_person = max_per_person or settings.debug_max_per_person
        self._counts: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _slug(self, name: str) -> str:
        keep = "".join(c if (c.isalnum() or c in " _-") else "_" for c in name).strip()
        return keep.replace(" ", "_") or "unknown"

    def capture(
        self, *, name: str, frame_bgr: np.ndarray, box, aligned_chw: np.ndarray | None,
        camera: str, role: str, score: float, margin: float, track_id: int,
        ts: datetime, quality=None, extra: dict | None = None,
        native: np.ndarray | None = None,
    ) -> str | None:
        if not self.enabled:
            return None
        slug = self._slug(name)
        with self._lock:
            # max_per_person == 0 means uncapped, for runs where every pass matters.
            if self.max_per_person and self._counts[slug] >= self.max_per_person:
                return None
            self._counts[slug] += 1
            n = self._counts[slug]

        try:
            d = self.root / slug
            d.mkdir(parents=True, exist_ok=True)
            stem = f"{ts.astimezone(settings.tz):%Y%m%d_%H%M%S}_{score:.3f}_{camera}_{n:03d}"

            x1, y1, x2, y2 = [int(v) for v in box]

            # 1. annotated full frame, downscaled so the folder stays usable
            ann = frame_bgr.copy()
            cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 220, 0), 3)
            label = f"{name} {score:.3f}"
            cv2.putText(ann, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(ann, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 220, 0), 2, cv2.LINE_AA)
            h, w = ann.shape[:2]
            if w > 1600:
                ann = cv2.resize(ann, (1600, int(h * 1600 / w)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(d / f"{stem}_frame.jpg"), ann, [cv2.IMWRITE_JPEG_QUALITY, 85])

            # 2. native face crop. Prefer the crop the pipeline actually used —
            # recomputing it here from a box means guessing, and any mismatch
            # between box and frame silently produces a picture of the floor.
            if native is not None and native.size:
                face = native
            else:
                side = max(x2 - x1, y2 - y1) * settings.align_margin / 2.0
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                fx1, fy1 = max(0, int(cx - side)), max(0, int(cy - side))
                fx2, fy2 = min(w, int(cx + side)), min(h, int(cy + side))
                face = frame_bgr[fy1:fy2, fx1:fx2]
            if face is not None and face.size:
                cv2.imwrite(str(d / f"{stem}_face.jpg"), face, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # 3. what the recognizer actually saw
            if aligned_chw is not None:
                rgb = aligned_to_uint8(aligned_chw)
                big = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (224, 224),
                                 interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(d / f"{stem}_aligned.jpg"), big, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # 4. the numbers behind the decision
            meta = {
                "name": name, "camera": camera, "role": role,
                "timestamp_local": ts.astimezone(settings.tz).isoformat(),
                "score": round(float(score), 4), "margin": round(float(margin), 4),
                "track_id": int(track_id),
                # `box` is in CONTEXT-FRAME pixels - the caller scales it down to
                # match the saved _frame.jpg so the overlay lines up. Deriving
                # face_px from it therefore reports a number in a different
                # coordinate space from `min_face_px`, which is full-res: a real
                # 159 px face logged as 58 px against a gate of 66, making a
                # correctly-passed frame look like it should have been rejected.
                # The gate value comes from Quality, which measured the full-res
                # box the gate actually saw.
                "box_context": [x1, y1, x2, y2],
                "box_context_px": int(max(x2 - x1, y2 - y1)),
                "face_px": (int(quality.face_px) if quality is not None
                            else None),
                "min_face_px": settings.min_face_px,
                "threshold": settings.recognition_threshold,
                "align_margin": settings.align_margin, "align_mode": settings.align_mode,
            }
            if quality is not None:
                meta["quality"] = {
                    "sharpness": round(float(quality.sharpness), 1),
                    "yaw": round(float(quality.yaw), 1),
                    "pitch": round(float(quality.pitch), 1),
                    "aligner_score": round(float(quality.aligner_score), 3),
                    "gate": quality.reason or "PASS",
                }
            if extra:
                meta.update(extra)
            (d / f"{stem}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
            return stem
        except Exception:
            log.exception("debug capture failed for %s", name)
            return None

    def frame(self, *, name: str, aligned_chw, native, score: float,
              camera: str, track_id: int, ts: datetime, idx: int,
              quality=None, accepted: bool = False) -> None:
        """Every gate-passing frame of a track, not just its best.

        The best-shot capture answers "what did it decide"; this answers "what
        did it have to choose from", which is what you need to tell a threshold
        problem from a quality problem.
        """
        if not self.enabled or not settings.save_all_frames:
            return
        try:
            import shutil as _sh
            if _sh.disk_usage(self.root).free / 1073741824 < settings.save_all_min_free_gb:
                return
            d = self.root / self._slug(name) / "frames"
            d.mkdir(parents=True, exist_ok=True)
            stem = (f"{ts.astimezone(settings.tz):%H%M%S}_t{track_id}_{idx:03d}"
                    f"_{score:.3f}_{'ok' if accepted else 'gated'}")
            if aligned_chw is not None:
                rgb = aligned_to_uint8(aligned_chw)
                cv2.imwrite(str(d / f"{stem}_aligned.jpg"),
                            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            if native is not None and native.size:
                cv2.imwrite(str(d / f"{stem}_native.jpg"), native,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
        except Exception:
            log.exception("per-frame debug write failed for %s", name)

    def summary(self) -> dict:
        with self._lock:
            return dict(self._counts)
