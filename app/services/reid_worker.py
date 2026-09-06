"""Body-crop collection and cross-camera re-identification, off the hot path.

WHY THIS IS A SEPARATE THREAD
-----------------------------
Everything here is expensive: JPEG encoding, a 6.5 GFLOP model, and database
writes. The capture thread has a 50 ms frame budget and already spends ~10 ms of
it recognising faces. Putting ReID inline would make a slow embed delay the next
frame, and a failed one count as a `pipeline_error` - so a data-collection
feature could degrade attendance, which is the one thing it must never do.

So the pipeline only *produces* crops (`FrameResult.body_crops`) and this thread
consumes them through a bounded, drop-oldest queue - the same contract
`RtspSource` uses for frames. If ReID cannot keep up, crops are dropped and
recognition is untouched. Dropping is the correct failure: these are training
samples, and there will be more.

WHAT IT COLLECTS, AND WHY BOTH KINDS
------------------------------------
* **Recognised passes** -> `persons/known/<date>/<Person_Name>/`. The face path
  has already labelled these, so they are free ground truth: labelled positives
  are what ReID training needs and what a threshold is calibrated against.
* **Unrecognised passes** -> `persons/unknown/<date>/<Camera>_t<id>_<time>/`.
  These are the people the face path misses - 51% of passes - and the reason to
  want body ReID at all.

CROSS-CAMERA MATCHING
---------------------
An unknown pass on the Exit camera is scored against unmatched unknown passes
from Entrance on the same business date. A hit gives a labelled cross-camera
identity: the same person, two cameras, no name. That pair is exactly what the
next round of ReID training and testing consumes.

It writes NO attendance, ever. On its own test set this model misses ~31% of
genuine cross-camera queries at a 10% false-positive rate; that is fine for
collecting data and nowhere near good enough to name somebody.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from app.config import settings

log = logging.getLogger(__name__)


def slug(name: str) -> str:
    """A folder name that survives every filesystem, still readable."""
    import re
    s = re.sub(r"[^\w\s-]", "", str(name), flags=re.UNICODE).strip()
    return re.sub(r"\s+", "_", s) or "Unnamed"


def _spread(times, keep: int) -> list[int]:
    """Choose `keep` of these crops, as far apart in time as possible.

    Taking the first N would cluster every kept crop at the start of the pass,
    when the person is furthest away and at one angle. Spreading them across
    the whole pass is what makes a tracklet worth aggregating: the point of
    several crops is that they disagree usefully - different distances, poses
    and lighting - and the quality weighting then picks the good ones.

    The first and last are always kept, so the selection spans the full pass
    however many are dropped from the middle.
    """
    n = len(times)
    if keep <= 0 or n <= keep:
        return list(range(n))
    if keep == 1:
        return [n // 2]
    # Evenly spaced positions across the pass, endpoints included.
    step = (n - 1) / (keep - 1)
    idx = sorted({int(round(i * step)) for i in range(keep)})
    # Rounding can collide; fill from what is left, preferring the widest gaps.
    if len(idx) < keep:
        remaining = [i for i in range(n) if i not in idx]
        remaining.sort(key=lambda i: -min(abs(i - j) for j in idx))
        idx = sorted(idx + remaining[: keep - len(idx)])
    return idx


class _Pass:
    """Crops accumulating for one person-pass, before it completes."""

    __slots__ = ("key", "camera_id", "camera_name", "track_id", "first_seen",
                 "last_seen", "crops", "scores", "times")

    def __init__(self, key, camera_id, camera_name, track_id, first_seen):
        self.key = key
        self.camera_id, self.camera_name = camera_id, camera_name
        self.track_id, self.first_seen = track_id, first_seen
        self.last_seen = first_seen
        self.crops: list = []
        self.scores: list = []
        self.times: list = []


class ReidWorker:
    """One per process, shared by every camera. Started by `Runtime`."""

    def __init__(self, model_path=None, queue_size: int = 256):
        self.enabled = bool(settings.reid_model)
        self.model_path = model_path or (
            settings.model_path(settings.reid_model) if self.enabled else None)
        self.q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._open: dict[tuple, _Pass] = {}
        self.reid = None

        self.crops_seen = 0
        self.crops_dropped = 0
        self.passes_written = 0
        self.matches_found = 0
        self.errors = 0
        self.last_error = ""

    # -- lifecycle --------------------------------------------------------
    def start(self):
        if not self.enabled:
            log.info("reid: disabled (no reid_model configured)")
            return self
        from app.core.model_vault import model_available
        if not model_available(self.model_path):
            log.error("reid: %s not found - collection and matching disabled. "
                      "Export it with scripts/export_reid.py", self.model_path)
            self.enabled = False
            return self
        self._thread = threading.Thread(target=self._run, name="reid", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=20.0)

    # -- producer side, called from the capture thread --------------------
    def submit_crops(self, camera_id, camera_name, crops):
        """Never blocks. A full queue drops the OLDEST crop, because the newest
        is the closest to the camera and therefore the most useful."""
        if not self.enabled or not crops:
            return
        for c in crops:
            self.crops_seen += 1
            try:
                self.q.put_nowait(("crop", camera_id, camera_name, c))
            except queue.Full:
                self.crops_dropped += 1
                try:
                    self.q.get_nowait()
                    self.q.put_nowait(("crop", camera_id, camera_name, c))
                except (queue.Empty, queue.Full):
                    pass

    def submit_pass(self, camera_id, camera_name, ct, ts):
        """A track completed: close its pass, embed, store, and try to match."""
        if not self.enabled:
            return
        try:
            self.q.put_nowait(("pass", camera_id, camera_name, (ct, ts)))
        except queue.Full:
            self.crops_dropped += 1

    # -- consumer ---------------------------------------------------------
    def _run(self):
        from app.core.onnx_env import preload_cuda_libs
        from app.core.reid import PersonReID
        preload_cuda_libs()
        try:
            self.reid = PersonReID(self.model_path,
                                   batch_size=settings.embed_batch)
        except Exception:
            log.exception("reid: model would not load; collection disabled")
            self.enabled = False
            return
        log.info("reid worker started (%s, %d-d)", self.reid.model_name, self.reid.dim)

        while not self._stop.is_set():
            try:
                kind, cam_id, cam_name, payload = self.q.get(timeout=1.0)
            except queue.Empty:
                self._close_stale(time.time())
                continue
            self._close_stale(time.time())
            try:
                if kind == "crop":
                    self._on_crop(cam_id, cam_name, payload)
                else:
                    self._on_pass(cam_id, cam_name, *payload)
            except Exception as e:
                self.errors += 1
                self.last_error = f"{type(e).__name__}: {e}"[:160]
                log.exception("reid: %s failed", kind)
        # Anything still open belongs to somebody who really walked past.
        for p in list(self._open.values()):
            try:
                self._flush(p, None, None, "UNKNOWN")
            except Exception:
                log.exception("reid: shutdown flush failed")
        log.info("reid worker stopped (%d crops, %d passes, %d matches, %d dropped)",
                 self.crops_seen, self.passes_written, self.matches_found,
                 self.crops_dropped)

    def _key(self, cam_id, track_id, first_seen):
        # Track ids restart on tracker.reset() and collide across cameras.
        return (cam_id, track_id, round(float(first_seen), 3))

    def _on_crop(self, cam_id, cam_name, c):
        k = self._key(cam_id, c.track_id, c.first_seen)
        p = self._open.get(k)
        if p is None:
            p = self._open[k] = _Pass(k, cam_id, cam_name, c.track_id, c.first_seen)
        p.crops.append(c.image)
        p.scores.append(c.score)
        p.times.append(c.ts)
        p.last_seen = max(p.last_seen, c.ts)

    def _on_pass(self, cam_id, cam_name, ct, ts):
        # `ct.first_seen` is the track's own, carried on CompletedTrack. It used
        # to be reconstructed as `completion_time - duration_s`, which is wrong
        # by track_max_age_s: a track is pruned three seconds AFTER it was last
        # seen. The key never matched, so every identified pass stayed open and
        # was dumped at shutdown as an UNKNOWN - recognised people appearing in
        # the unknown folders with no name and no direction.
        p = self._open.pop(self._key(cam_id, ct.track_id, ct.first_seen), None)
        if p is None:
            cand = [key for key in self._open
                    if key[0] == cam_id and key[1] == ct.track_id]
            if cand:
                p = self._open.pop(max(cand, key=lambda x: x[2]))
        if p is None or not p.crops:
            return
        self._flush(p, ct, ts, ct.direction)

    def _close_stale(self, now: float):
        """Write out passes that no longer receive crops.

        Not every track becomes a CompletedTrack: `_prune` drops tracks that
        never produced an embedded frame, so somebody who walks through with
        their face turned away leaves body crops and no completion. Without
        this they accumulate until shutdown and are written all at once with no
        direction - which is exactly what made them look like a bug.

        They ARE genuinely unknown; they just deserve to be written when they
        happen, and to be distinguishable from a pass the face path considered
        and rejected.
        """
        limit = settings.track_max_age_s + settings.body_crop_interval_s + 2.0
        # A pass at the crop cap stopped RECEIVING crops, not walking: the
        # pipeline collects nothing past body_crop_max_per_pass, so silence
        # from it says nothing about the track. Flushed at `limit` it went out
        # as UNKNOWN while the person was still in view, and the CompletedTrack
        # that arrived later - with the name - found nothing to attach to. So a
        # capped pass waits out the longest it could plausibly still be alive.
        capped = (settings.track_max_age_s
                  + settings.body_crop_max_per_pass * settings.body_crop_interval_s
                  + 60.0)
        cap = settings.body_crop_max_per_pass
        for key in [k for k, p in self._open.items()
                    if now - p.last_seen > (capped if len(p.crops) >= cap else limit)]:
            p = self._open.pop(key)
            if p.crops:
                try:
                    self._flush(p, None, None, "UNKNOWN")
                except Exception:
                    self.errors += 1
                    log.exception("reid: closing a stale pass failed")

    # -- the work ---------------------------------------------------------
    def _flush(self, p: _Pass, ct, ts, direction: str):
        from app.db.session import session_scope
        from app.db.models import ReidPass
        from app.services.attendance import business_date
        from app.core.reid import aggregate, sharpness_of

        when = ts or datetime.now(timezone.utc)
        named = ct is not None and ct.employee_id is not None
        day = when.astimezone(settings.tz).strftime("%Y%m%d")
        stamp = datetime.fromtimestamp(p.first_seen, tz=settings.tz)

        if named:
            folder = Path("known") / day / slug(ct.name)
        else:
            folder = (Path("unknown") / day /
                      f"{p.camera_name}_t{p.track_id}_{stamp:%H%M%S}")
        out = settings.persons_dir / folder
        out.mkdir(parents=True, exist_ok=True)

        keep = _spread(p.times, settings.body_crop_keep_per_pass)
        crops = [p.crops[i] for i in keep]
        scores = [p.scores[i] for i in keep]
        times = [p.times[i] for i in keep]

        kept = 0
        for img, sc, t in zip(crops, scores, times):
            fn = (f"{datetime.fromtimestamp(t, tz=settings.tz):%Y%m%d_%H%M%S}"
                  f"_{p.camera_name}_t{p.track_id}_s{sc:.3f}.jpg")
            if cv2.imwrite(str(out / fn), img, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                kept += 1

        embs = self.reid.embed(crops)
        sharp = [sharpness_of(c) for c in crops]
        feat = aggregate(embs, scores, sharp)

        with session_scope() as s:
            row = ReidPass(
                camera_id=p.camera_id, camera_name=p.camera_name,
                track_id=p.track_id,
                first_seen=datetime.fromtimestamp(p.first_seen, tz=timezone.utc),
                last_seen=datetime.fromtimestamp(p.last_seen, tz=timezone.utc),
                business_date=business_date(when), direction=direction or "UNKNOWN",
                employee_id=ct.employee_id if named else None,
                name=ct.name if named else "",
                folder=str(folder), crops=kept,
                vector=feat.astype("float32").tobytes(), dim=int(feat.shape[0]),
                model_name=self.reid.model_name,
            )
            s.add(row)
            s.flush()
            hit = None
            if not named and kept >= settings.reid_min_crops:
                hit = self._match(s, row, feat)
            pass_id, folder_str = row.id, row.folder

        (out / "_pass.json").write_text(json.dumps({
            "camera": p.camera_name, "track_id": p.track_id,
            "first_seen": stamp.isoformat(), "direction": direction,
            "employee_id": ct.employee_id if named else None,
            "name": ct.name if named else None,
            "face_score": round(float(ct.best_score), 4) if ct else None,
            "crops": kept, "reid_model": self.reid.model_name,
            "reid_vector": [round(float(x), 6) for x in feat],
            "pass_id": pass_id,
            "matched_pass_id": hit[0] if hit else None,
            "match_score": round(hit[1], 4) if hit else None,
        }, indent=2, ensure_ascii=False))

        self.passes_written += 1
        if hit:
            self.matches_found += 1
            log.info("[reid] %s t%d matched pass %d  score=%.3f margin=%.3f",
                     p.camera_name, p.track_id, hit[0], hit[1], hit[2])

    def _match(self, s, row, feat):
        """Score this pass against unmatched passes from the OTHER camera.

        Two rules, both required, mirroring `Gallery.match`: the top score must
        clear the threshold AND beat the runner-up by a margin. The genuine and
        impostor distributions overlap heavily here - the research project
        measures wrong_sim_p90 at 0.664 against right_sim_p10 at 0.521 - so a
        high score on its own is weak evidence.
        """
        from sqlalchemy import select
        from app.db.models import ReidPass

        rows = s.execute(
            select(ReidPass).where(
                ReidPass.business_date == row.business_date,
                ReidPass.camera_id != row.camera_id,
                ReidPass.matched_pass_id.is_(None),
                ReidPass.employee_id.is_(None),
                ReidPass.model_name == row.model_name,   # never mix models
                ReidPass.id != row.id,
                ReidPass.dim == row.dim,
            )
        ).scalars().all()
        if not rows:
            return None

        M = np.stack([np.frombuffer(r.vector, dtype=np.float32) for r in rows])
        M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
        sims = M @ feat
        order = np.argsort(-sims)
        top = float(sims[order[0]])
        second = float(sims[order[1]]) if len(order) > 1 else -1.0
        margin = top - second
        if top < settings.reid_match_threshold or margin < settings.reid_match_margin:
            return None

        other = rows[int(order[0])]
        row.matched_pass_id, row.match_score, row.match_margin = other.id, top, margin
        other.matched_pass_id, other.match_score, other.match_margin = row.id, top, margin
        return (other.id, top, margin)

    def stats(self) -> dict:
        return {"enabled": self.enabled, "queued": self.q.qsize(),
                "crops_seen": self.crops_seen, "crops_dropped": self.crops_dropped,
                "passes": self.passes_written, "matches": self.matches_found,
                "errors": self.errors, "last_error": self.last_error,
                "model": Path(str(self.model_path)).name if self.model_path else None}
