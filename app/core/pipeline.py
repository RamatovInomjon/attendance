"""Per-camera recognition worker.

One instance owns one camera and everything downstream of it: its own detector,
its own tracker, its own track state.  Nothing is shared with another camera.

Per processed frame:

    decode (full-res)
      -> downscale to DETECT_WIDTH -> detect -> scale boxes back up
      -> ByteTrack (this camera's own instance)
      -> per track: quality gate, then align + embed the good frames
      -> match against the gallery, feed the track's K-of-N vote
      -> on commitment, emit a RecognitionEvent

Faces are cropped from the **full-resolution** frame, never the downscaled one:
the DFA aligner resizes its input to 160x160 internally and warps the canonical
crop out of that, so feeding it a downscaled face throws away the detail the
recognizer depends on.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.config import settings
from app.core.aligner import FaceAligner
from app.core.detector import build_detector
from app.core.direction import Direction, DirectionConfig, Trajectory
from app.core.head_detector import CLS_HEAD, HeadDetector
from app.core.gallery import Gallery, Match, TrackVote
from app.core.quality import Quality, assess
from app.core.recognizer import FaceRecognizer
from pathlib import Path

from app.core.stream import Frame, RtspSource
from app.core.tracker import FaceTracker

log = logging.getLogger(__name__)


@dataclass
class TrackState:
    track_id: int
    first_seen: float
    last_seen: float
    box: np.ndarray
    vote: TrackVote
    employee_id: int | None = None
    name: str = "…"
    score: float = 0.0
    best_quality: float = 0.0
    best_crop: np.ndarray | None = None      # aligned 112x112, for the snapshot
    attempts: int = 0
    gated: int = 0
    emitted: bool = False
    last_reason: str = ""
    face_box: np.ndarray | None = None
    trajectory: Trajectory = field(default_factory=Trajectory)
    direction: Direction = Direction.UNKNOWN
    direction_reason: str = ""
    reported_direction: Direction | None = None   # what we already wrote an event for
    embedded: int = 0
    best_face_px: int = 0
    nearest_id: int | None = None
    best_vector: np.ndarray | None = None
    best_seen: float = 0.0
    # Two frames are tracked, and they are usually different frames.
    # score-best: why the match happened - kept for diagnosis.
    best_native: np.ndarray | None = None
    best_context: np.ndarray | None = None
    best_box: np.ndarray | None = None
    # quality-best: what a person is shown. Everything displayed - aligned crop,
    # native crop, annotated frame, pose metadata - must come from THIS one
    # frame, or the picture and the numbers describe different moments and a
    # wrong result looks unexplainable.
    disp_quality: float = 0.0
    disp_quality_obj: Quality | None = None
    disp_native: np.ndarray | None = None
    disp_context: np.ndarray | None = None
    disp_box: np.ndarray | None = None
    best_quality_obj: Quality | None = None   # the Quality record, not its score


@dataclass
class RecognitionOutcome:
    """What the worker hands to the attendance layer."""
    track_id: int
    employee_id: int
    name: str
    score: float
    margin: float
    face_px: int
    votes: str
    ts: float
    crop: np.ndarray | None = None
    box: np.ndarray | None = None
    quality: Quality | None = None
    direction: str = "UNKNOWN"
    direction_reason: str = ""


@dataclass
class CompletedTrack:
    """A track that has left the scene. One of these is roughly one person-pass,
    which is the denominator for any 'how many were recognized' question."""
    track_id: int
    employee_id: int | None
    name: str
    best_score: float
    embedded_frames: int
    gated_frames: int
    direction: str
    direction_reason: str
    face_px: int
    duration_s: float
    traj_points: int = 0
    travel: float = 0.0
    native: np.ndarray | None = None
    score_crop: np.ndarray | None = None   # top-scoring frame, for diagnosis
    context: np.ndarray | None = None
    quality: object = None
    box: np.ndarray | None = None
    crop: np.ndarray | None = None
    vector: np.ndarray | None = None
    nearest_employee_id: int | None = None


@dataclass
class FrameCandidate:
    """One gate-passing face this frame, before any vote is taken."""
    track_id: int
    name: str
    score: float
    aligned: np.ndarray
    native: np.ndarray | None
    quality: object
    accepted: bool
    index: int


@dataclass
class FrameResult:
    frame: Frame
    tracks: list[TrackState] = field(default_factory=list)
    outcomes: list[RecognitionOutcome] = field(default_factory=list)
    completed: list[CompletedTrack] = field(default_factory=list)
    candidates: list[FrameCandidate] = field(default_factory=list)
    timings: dict = field(default_factory=dict)


class CameraPipeline:
    def __init__(self, name: str, gallery: Gallery, detector=None, aligner=None,
                 recognizer=None, direction_cfg: DirectionConfig | None = None,
                 head_detector=None):
        self.name = name
        self.gallery = gallery
        self.direction_cfg = direction_cfg or DirectionConfig()

        # Tracking runs on HEADS, not faces. Faces vanish the moment somebody
        # turns or looks down, which left tracks with a median of 1-4 usable
        # frames - too few for a 3-of-5 identity vote or a 5-point trajectory,
        # so people went unrecognized and their direction was undetermined.
        # A head stays visible throughout, so one track spans one person-pass;
        # face recognition then runs inside that track on whichever frames
        # actually hold a usable face.
        self.head_detector = head_detector
        if self.head_detector is None and settings.head_model:
            hp = settings.model_path(settings.head_model)
            if Path(hp).exists():
                self.head_detector = HeadDetector(hp, size=settings.head_input,
                                                  conf=settings.head_conf)

        self.detector = detector or build_detector(
            settings.detector_kind,
            settings.model_path(settings.detector_model),
            imgsz=settings.detect_width,
            conf=settings.detect_conf,
        )
        self.aligner = aligner or FaceAligner(
            settings.model_path(settings.aligner_model),
            crop_size=settings.align_crop_size,
            margin=settings.align_margin,
            mode=settings.align_mode,
        )
        self.recognizer = recognizer or FaceRecognizer(
            settings.model_path(settings.recognizer_model), batch_size=settings.embed_batch
        )
        self.tracker = FaceTracker(
            # ByteTrack measures its lost-track buffer in FRAMES
            # (max_time_lost = frame_rate/30 * track_buffer), so this must track
            # the real processing rate or the wall-clock tolerance silently
            # changes with it. It was hardcoded to 12 while the pipeline ran
            # every 2nd frame of a 20 fps stream; going to every frame doubled
            # the rate and so halved the tolerance to 0.63 s, which fragmented
            # tracks into 1-frame stubs.
            frame_rate=settings.track_frame_rate,
            track_high_thresh=settings.track_high_thresh,
            track_low_thresh=settings.track_low_thresh,
            match_thresh=settings.track_match_thresh,
            track_buffer=settings.track_buffer,
        )

        self.tracks: dict[int, TrackState] = {}
        self.frames_processed = 0
        self.faces_embedded = 0

    # -- helpers ----------------------------------------------------------
    def _prune(self, now: float) -> list["CompletedTrack"]:
        """Retire stale tracks, returning a record for each so the caller can
        count person-passes. Tracks that never produced a single embedded frame
        are dropped silently: they were a detection flicker, not a person."""
        dead = [tid for tid, t in self.tracks.items()
                if now - t.last_seen > settings.track_max_age_s]
        out = []
        for tid in dead:
            t = self.tracks.pop(tid)
            if t.embedded == 0:
                continue
            out.append(CompletedTrack(
                track_id=tid, employee_id=t.employee_id, name=t.name,
                best_score=float(max(t.best_seen, t.vote.best_score or 0.0)),
                embedded_frames=t.embedded, gated_frames=t.gated,
                direction=t.direction.value, direction_reason=t.direction_reason,
                face_px=t.best_face_px, duration_s=max(0.0, t.last_seen - t.first_seen),
                traj_points=len(t.trajectory.points), travel=t.trajectory.travel(),
                # Display artifacts all from the quality-best frame.
                native=t.disp_native, context=t.disp_context, quality=t.disp_quality_obj,
                # The best-shot box, not the box at pruning: by then the person
                # has walked on, so a completion-time box lands on empty floor.
                box=(t.disp_box.copy() if t.disp_box is not None
                     else (t.box.copy() if t.box is not None else None)),
                # Clearest frame first: this is what the dashboard shows and what
                # a person judges the result by. The top-scoring frame is kept
                # separately for diagnosis.
                crop=(t.vote.quality_snapshot if t.vote.quality_snapshot is not None
                      else (t.best_crop if t.best_crop is not None else t.vote.best_snapshot)),
                score_crop=t.vote.best_snapshot,
                vector=t.best_vector, nearest_employee_id=t.nearest_id,
            ))
        return out

    @staticmethod
    def _assign_faces(head_boxes, face_boxes):
        """Map each head to the face inside it (largest, by centre containment)."""
        out = {}
        for hi, hb in enumerate(head_boxes):
            best, best_area = None, -1.0
            for fb in face_boxes:
                cx, cy = (fb[0] + fb[2]) / 2.0, (fb[1] + fb[3]) / 2.0
                if hb[0] <= cx <= hb[2] and hb[1] <= cy <= hb[3]:
                    a = (fb[2] - fb[0]) * (fb[3] - fb[1])
                    if a > best_area:
                        best, best_area = fb, a
            if best is not None:
                out[hi] = best
        return out

    def _detect(self, frame_bgr: np.ndarray):
        """Detect on a downscale; return boxes in FULL-RES coordinates."""
        h, w = frame_bgr.shape[:2]
        if w > settings.detect_width:
            scale = settings.detect_width / w
            # INTER_LINEAR, not INTER_AREA: this downscale only feeds the
            # detector, and faces are cropped from the full-resolution frame, so
            # resample quality here costs nothing measurable (impostor max
            # 0.1831 vs 0.1827 over the gallery) while running ~3x faster —
            # more under the CPU contention of two camera workers.
            small = cv2.resize(frame_bgr, (settings.detect_width, int(round(h * scale))),
                               interpolation=cv2.INTER_LINEAR)
        else:
            scale = 1.0
            small = frame_bgr

        dets = self.detector.detect(small)
        if not dets:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)

        boxes = np.stack([d.box for d in dets]).astype(np.float32) / scale
        scores = np.array([d.score for d in dets], dtype=np.float32)
        return boxes, scores

    # -- main -------------------------------------------------------------
    def process(self, frame: Frame) -> FrameResult:
        t_all = time.perf_counter()
        res = FrameResult(frame=frame)
        now = frame.ts

        t0 = time.perf_counter()
        h, w = frame.image.shape[:2]
        scale = settings.detect_width / w if w > settings.detect_width else 1.0
        small = (cv2.resize(frame.image, (settings.detect_width, int(round(h * scale))),
                            interpolation=cv2.INTER_LINEAR) if scale != 1.0 else frame.image)

        if self.head_detector is not None:
            # One detector, not two. Heads are what we track, and the DFA aligner
            # does its own face localisation inside whatever crop it is given -
            # so a head box serves for recognition too. Measured on real frames,
            # aligning from head boxes scores 0.187 median against 0.192 from a
            # dedicated face detector, while costing 10 ms less per frame and
            # still producing a box on frames where face detection finds nothing.
            hd = self.head_detector.detect(small, want=CLS_HEAD)
            track_boxes = (np.stack([d.box for d in hd]).astype(np.float32) / scale
                           if hd else np.zeros((0, 4), np.float32))
            track_scores = (np.array([d.score for d in hd], np.float32)
                            if hd else np.zeros((0,), np.float32))
        else:
            fd_ = self.detector.detect(small)
            track_boxes = (np.stack([d.box for d in fd_]).astype(np.float32) / scale
                           if fd_ else np.zeros((0, 4), np.float32))
            track_scores = (np.array([d.score for d in fd_], np.float32)
                            if fd_ else np.zeros((0,), np.float32))
        t_det = (time.perf_counter() - t0) * 1000

        tracked = self.tracker.update(track_boxes, track_scores)
        res.completed = self._prune(now)

        # --- update track bookkeeping ---
        pending = []          # tracks that still need an identity this frame
        for ti, (tid, box, score) in enumerate(tracked):
            st = self.tracks.get(tid)
            if st is None:
                st = TrackState(
                    track_id=tid, first_seen=now, last_seen=now, box=box,
                    vote=TrackVote(window=settings.vote_window, required=settings.vote_required),
                )
                self.tracks[tid] = st
            st.last_seen = now
            st.box = box
            # Trajectory follows the head: it is present on every frame, so the
            # path is continuous even while the face is turned away.
            st.trajectory.add(now, box, frame.image.shape[1], frame.image.shape[0])
            st.face_box = box       # the head box is the alignment source
            d, why = st.trajectory.direction(self.direction_cfg)
            st.direction_reason = why          # refresh: a stale reason misleads
            if d is not Direction.UNKNOWN:
                st.direction = d
            # Only tracks showing a face this frame can be recognized.
            if not st.vote.decided:
                pending.append(st)

        # Pre-gate on head-box size before paying for alignment. face_px is
        # max(box w, box h) - it needs no landmarks, so `small()` can be decided
        # from the box alone. The retrained detector finds noticeably more
        # distant heads (9-14 px at detect scale), and every one of them used to
        # be warped and scored only to be thrown away by the same gate a few
        # microseconds later. Aligning one face costs ~15 ms; this check costs
        # nothing.
        #
        # Filtering here also fixes a track/face misalignment: the aligner drops
        # boxes whose crop falls entirely outside the frame, which made the
        # `zip(pending, faces)` below pair tracks with other tracks' faces.
        # Requiring a positive on-frame intersection means align() can never
        # return a short list.
        fh, fw = frame.image.shape[:2]
        sized = []
        for st in pending:
            x1, y1, x2, y2 = st.face_box
            face_px = float(max(x2 - x1, y2 - y1))
            on_frame = min(x2, fw) > max(x1, 0) and min(y2, fh) > max(y1, 0)
            if face_px < settings.min_face_px or not on_frame:
                st.attempts += 1
                st.gated += 1
                st.last_reason = (f"small({face_px:.0f}px)" if on_frame
                                  else "offframe")
                continue
            sized.append(st)
        pending = sized

        t_align = t_embed = 0.0
        if pending:
            t0 = time.perf_counter()
            # Hand the BGR frame straight over: the aligner converts only the
            # crops it actually reads. Converting the whole 4K frame here cost
            # 5.89 ms and a 24 MB allocation per frame with a face in view.
            faces = self.aligner.align(frame.image,
                                       [st.face_box for st in pending], is_bgr=True)
            t_align = (time.perf_counter() - t0) * 1000
            assert len(faces) == len(pending), (len(faces), len(pending))

            keep, quals = [], []
            for st, f in zip(pending, faces):
                q: Quality = assess(
                    st.face_box, f.aligned, f.landmarks, f.score,
                    min_face_px=settings.min_face_px,
                    min_laplacian_var=settings.min_laplacian_var,
                    min_aligner_score=settings.min_aligner_score,
                    max_yaw_deg=settings.max_yaw_deg,
                    max_pitch_deg=settings.max_pitch_deg,
                )
                st.attempts += 1
                if not q.ok:
                    st.gated += 1
                    st.last_reason = q.reason
                    continue
                keep.append((st, f))
                quals.append(q)

            if keep:
                t0 = time.perf_counter()
                aligned = np.stack([f.aligned for _st, f in keep])
                embs = self.recognizer.embed(aligned)
                t_embed = (time.perf_counter() - t0) * 1000
                self.faces_embedded += len(embs)

                for (st, f), q, emb in zip(keep, quals, embs):
                    m: Match = self.gallery.match(
                        emb, settings.recognition_threshold, settings.second_best_margin
                    )
                    st.embedded += 1
                    st.best_face_px = max(st.best_face_px, int(q.face_px))
                    if settings.save_all_frames:
                        x1, y1, x2, y2 = [int(v) for v in st.face_box]
                        res.candidates.append(FrameCandidate(
                            track_id=st.track_id,
                            name=self.gallery.name(m.employee_id) if m.employee_id
                                 else f"_near_{self.gallery.name(m.runner_up)}",
                            score=float(m.score), aligned=f.aligned,
                            native=frame.image[max(0, y1):y2, max(0, x1):x2],
                            quality=q, accepted=m.employee_id is not None,
                            index=st.embedded))
                    # Match.score is the top similarity even when it misses, and
                    # runner_up carries who it nearly was - both worth keeping so
                    # an unrecognized pass can be explained rather than guessed at.
                    if m.score > st.best_seen or st.best_vector is None:
                        st.best_seen = float(m.score)
                        st.best_vector = emb
                        st.nearest_id = m.employee_id if m.employee_id is not None else m.runner_up
                        st.best_quality_obj = q
                        st.best_box = np.asarray(st.face_box, dtype=np.float32).copy()
                        x1, y1, x2, y2 = [int(v) for v in st.face_box]
                        st.best_native = frame.image[max(0, y1):y2, max(0, x1):x2].copy()

                    if q.score > st.disp_quality or st.disp_native is None:
                        st.disp_quality = float(q.score)
                        st.disp_quality_obj = q
                        st.disp_box = np.asarray(st.face_box, dtype=np.float32).copy()
                        dx1, dy1, dx2, dy2 = [int(v) for v in st.face_box]
                        st.disp_native = frame.image[max(0, dy1):dy2, max(0, dx1):dx2].copy()
                        fh, fw = frame.image.shape[:2]
                        cw = 1400
                        st.disp_context = (cv2.resize(frame.image, (cw, int(fh * cw / fw)),
                                                      interpolation=cv2.INTER_AREA)
                                           if fw > cw else frame.image.copy())
                    if q.score > st.best_quality:
                        st.best_quality = q.score
                        st.best_crop = f.aligned

                    committed = st.vote.add(m, f.aligned, quality=q.score)
                    if m.employee_id is not None:
                        st.score = max(st.score, m.score)

                    if committed is not None and not st.emitted:
                        st.employee_id = committed
                        st.name = self.gallery.name(committed)
                        st.emitted = True
                        votes = "/".join("?" if v is None else str(v) for v in st.vote.votes)
                        res.outcomes.append(RecognitionOutcome(
                            track_id=st.track_id, employee_id=committed, name=st.name,
                            score=st.vote.best_score, margin=m.margin,
                            face_px=int(q.face_px), votes=votes, ts=now,
                            crop=st.vote.best_snapshot if st.vote.best_snapshot is not None else st.best_crop,
                            box=st.box.copy(), quality=q,
                            direction=st.direction.value, direction_reason=st.direction_reason,
                        ))
                        st.reported_direction = st.direction
                    elif not st.emitted:
                        st.name = "…" if m.employee_id is None else self.gallery.name(m.employee_id)

        # No attendance outcome is emitted here. Identity commits partway
        # through a walk, but direction only resolves once the person has
        # actually travelled, so an event written at commit time is always
        # NO_DIRECTION - and the later corrected event was silently dropped by
        # the per-camera debounce. The decision belongs at track completion,
        # where identity and direction are both final: one pass, one decision.
        self.frames_processed += 1
        res.tracks = list(self.tracks.values())
        res.timings = {
            "detect": round(t_det, 2), "align": round(t_align, 2),
            "embed": round(t_embed, 2), "total": round((time.perf_counter() - t_all) * 1000, 2),
            "faces": len(tracked),
        }
        return res
