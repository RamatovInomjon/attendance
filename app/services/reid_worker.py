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

GROUPING THE PEOPLE WHO ARE NOT ENROLLED
----------------------------------------
An unnamed pass is also handed to `app/services/pseudo_gallery.py`, which
attaches it to a stable pseudo-identity built from its FACE and BODY features -
so the same visitor's passes end up in one place instead of being N unrelated
unknowns. Face leads and works across days; body is the fallback and only
within one, because clothing changes. See that module for the measurements.

It writes NO attendance, ever - not from the cross-camera link and not from the
pseudo-person grouping. Naming an unknown by body mislabels 10.9% of confirmed
visitors at the matching threshold, and at this corridor's base rate (~45
recoverable employees among ~1100 unknown tracks a day) that produces more
wrong attendance rows than right ones at every threshold measured.
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
        # Built on this thread, used only from it, so it needs no lock.
        from app.services.pseudo_gallery import PseudoGallery
        self.pseudo = PseudoGallery() if settings.pseudo_person else None

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

        # The FACE side of the same pass, carried on the completed track: one
        # quality-weighted template over every frame that cleared the gates.
        # It is what makes grouping an unregistered person possible at all -
        # body alone regroups 3.5-7.6% of pairs - and it is what breaks a
        # cross-camera tie the body cannot.
        face = getattr(ct, "face_template", None) if ct is not None else None
        face_ipd = float(getattr(ct, "best_ipd", 0.0) or 0.0) if ct else 0.0
        face_n = int(getattr(ct, "face_frames", 0) or 0) if ct else 0

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
                face_vector=(face.astype("float32").tobytes()
                             if face is not None else None),
                face_dim=int(face.shape[0]) if face is not None else 0,
                face_ipd=face_ipd, face_frames=face_n,
            )
            s.add(row)
            s.flush()
            hit = None
            enough = kept >= settings.reid_min_crops
            if not named and enough:
                hit = self._match(s, row, feat, face)
            # A tracklet feature built from one blurred crop is not worth a
            # cross-camera claim, and it is not worth a body LINK either - the
            # same argument, against a far larger candidate set. Such a pass is
            # still grouped, but only by its face.
            group = self._group(s, row, feat if enough else None,
                                face, face_ipd, ct, named)
            pass_id, folder_str = row.id, row.folder
            pseudo_code = group.code if group is not None else None

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
            "face_ipd": round(face_ipd, 1),
            "face_frames": face_n,
            "pseudo_person": pseudo_code,
        }, indent=2, ensure_ascii=False))

        self.passes_written += 1
        if hit:
            self.matches_found += 1
            log.info("[reid] %s t%d matched pass %d  score=%.3f margin=%.3f",
                     p.camera_name, p.track_id, hit[0], hit[1], hit[2])

    def _group(self, s, row, body, face, face_ipd, ct, named):
        """Attach this pass to a pseudo-identity, so one visitor is one person.

        Named passes come here too, with `create=False`: an employee is not a
        new visitor and must not mint a pseudo-identity, but their face may
        match a pseudo-person assembled from their OWN earlier passes - the
        ones the face path missed - and that is the only way such a group ever
        gets a name. A pass whose face was good enough to match the enrolment
        gallery is named upstream and never reaches here as an unknown.
        """
        if self.pseudo is None:
            return None
        try:
            return self.pseudo.place(
                s, row, face=face, body=body, face_ipd=face_ipd,
                body_model=self.reid.model_name,
                registry_match=((ct.employee_id, float(ct.best_score), ct.name)
                                if named else None),
                create=not named)
        except Exception:
            # Grouping is bookkeeping. It must never cost a pass its row, its
            # crops or its cross-camera match, which are already written.
            self.errors += 1
            log.exception("reid: pseudo-person grouping failed")
            return None

    def _match(self, s, row, feat, face=None):
        """Score this pass against unmatched passes from the OTHER camera.

        Two rules, both required, mirroring `Gallery.match`: the top score must
        clear the threshold AND beat the runner-up by a margin. The genuine and
        impostor distributions overlap heavily here - the research project
        measures wrong_sim_p90 at 0.664 against right_sim_p10 at 0.521 - so a
        high score on its own is weak evidence.

        THE MARGIN RULE IS EXPENSIVE, AND FACE IS WHAT CAN PAY FOR IT. Measured
        on this corridor's 281-pass corpus (bench/reid_match_eval.py), the
        margin costs more than half the correct matches - 55 accepts fall to 29
        - while the wrong count is zero either way. Those are not impostors
        being caught; they are passes where two candidates' CLOTHING scored
        alike and the body had nothing left to say.

        So when the body is undecided, the FACE is asked, and only then:

            top body < threshold                 -> no match, as before
            top beats runner-up by the margin    -> match, exactly as before
            otherwise, and only if both have a comparable face:
                the face must clear `pseudo_face_threshold` AND beat the
                runner-up's face by `reid_match_margin` -> match

        Why this shape rather than blending the two scores into one number: the
        research fusion (global-z blend, w = 0.6, FNIR@10% 11.59 -> 5.93) was
        measured under the ENROLMENT protocol, where the decision is a
        threshold on the blended score. This rule is a pairwise link with a
        runner-up margin, and a z-normalised blend is not on the scale
        `reid_match_threshold` was calibrated against. Blending here changed
        the RANKING without a calibrated bar to judge it by, which in practice
        only converted accepts into rejections - a stricter matcher wearing
        fusion's name. A tie-break adds recall exactly where the body axis has
        run out of information, and leaves the calibrated accept decision -
        every candidate must still clear the body threshold - untouched.
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
        if top < settings.reid_match_threshold:
            return None

        won, how = int(order[0]), "body"
        if margin < settings.reid_match_margin:
            won = self._face_tiebreak(rows, order, sims, face)
            if won < 0:
                return None
            how = "face"

        other = rows[won]
        score = float(sims[won])
        row.matched_pass_id, row.match_score, row.match_margin = other.id, score, margin
        other.matched_pass_id, other.match_score, other.match_margin = row.id, score, margin
        return (other.id, score, margin)

    def _face_tiebreak(self, rows, order, sims, face) -> int:
        """Index of the winner when the BODY could not separate the field.

        Only candidates that clear the body threshold on their own take part,
        so this can never admit a pass the calibrated accept rule rejected - it
        only chooses between passes that rule already considers plausible.

        `pseudo_face_threshold` is the bar because it is the one that was
        measured for exactly this question: is this pass's face the same PERSON
        as that pass's face (90-98% pair precision over three days and ~4200
        passes). The enrolment threshold answers a different question - is this
        face the person in that studio photograph - and does not transfer.
        """
        if face is None or not settings.reid_match_face_tiebreak:
            return -1
        eligible = [i for i in order
                    if float(sims[i]) >= settings.reid_match_threshold
                    and rows[i].face_vector and rows[i].face_dim == len(face)]
        if len(eligible) < 2:
            # One candidate with a face and one without is not a tie the face
            # can break: "no face" is not a low score, it is no evidence, and
            # ranking it against a real one would invent a comparison.
            return -1
        F = np.stack([np.frombuffer(rows[i].face_vector, dtype=np.float32)
                      for i in eligible])
        F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
        fs = F @ np.asarray(face, np.float32)
        o = np.argsort(-fs)
        best, runner = float(fs[o[0]]), float(fs[o[1]])
        if best < settings.pseudo_face_threshold:
            return -1
        if (best - runner) < settings.reid_match_margin:
            return -1
        return int(eligible[int(o[0])])

    def stats(self) -> dict:
        return {"enabled": self.enabled, "queued": self.q.qsize(),
                "crops_seen": self.crops_seen, "crops_dropped": self.crops_dropped,
                "passes": self.passes_written, "matches": self.matches_found,
                "errors": self.errors, "last_error": self.last_error,
                "pseudo": self.pseudo.stats() if self.pseudo else None,
                "model": Path(str(self.model_path)).name if self.model_path else None}
