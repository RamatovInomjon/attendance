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
from app.core.head_detector import CLS_HEAD, CLS_PERSON, HeadDetector
from app.core.gallery import Gallery, Match, TrackVote
from app.core.model_vault import model_available
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
    person_box: np.ndarray | None = None
    # Where the head sat inside the person box, as a fraction of that box, the
    # last time both were seen together. Used to synthesise a head point on
    # frames where the head is missed - the person box is still tracked, so the
    # trajectory stays continuous and direction keeps resolving even when the
    # head is too small or turned away to detect.
    head_offset: tuple[float, float] | None = None
    frames_without_head: int = 0
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
    best_margin: float
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


def _head_in(person: np.ndarray, heads: np.ndarray) -> np.ndarray | None:
    """The head belonging to this person box, or None.

    Association is by CONTAINMENT of the head centre, not IoU: a head is a tiny
    box inside a large one, so their IoU is near zero even when the head plainly
    belongs to that person, and an IoU threshold would reject every correct
    pair. Where several heads fall inside one person box - someone standing
    behind another - the highest one wins, since a person's own head is the
    topmost thing in their box.
    """
    if heads is None or len(heads) == 0:
        return None
    px1, py1, px2, py2 = person
    cx = (heads[:, 0] + heads[:, 2]) * 0.5
    cy = (heads[:, 1] + heads[:, 3]) * 0.5
    inside = (cx >= px1) & (cx <= px2) & (cy >= py1) & (cy <= py2)
    if not inside.any():
        return None
    idx = np.flatnonzero(inside)
    return heads[idx[np.argmin(cy[idx])]]


def _offset_of(head: np.ndarray, person: np.ndarray) -> tuple[float, float]:
    """Head centre as a fraction of the person box, for later synthesis."""
    pw = max(float(person[2] - person[0]), 1.0)
    ph = max(float(person[3] - person[1]), 1.0)
    hcx = (float(head[0]) + float(head[2])) * 0.5
    hcy = (float(head[1]) + float(head[3])) * 0.5
    return ((hcx - float(person[0])) / pw, (hcy - float(person[1])) / ph)


# Head centre when none has ever been seen for this track: horizontally centred,
# and a tenth of the way down - where a standing person's head is.
_DEFAULT_HEAD_OFFSET = (0.5, 0.10)


def _synth_head(person: np.ndarray, offset: tuple[float, float] | None) -> np.ndarray:
    """A head-sized box where this person's head should be.

    Returned as a box rather than a point because Trajectory records boxes and
    uses their AREA for the depth signal. Its size is tied to the person box, so
    the area still grows and shrinks with distance the way a real head box does.
    """
    ox, oy = offset or _DEFAULT_HEAD_OFFSET
    px1, py1 = float(person[0]), float(person[1])
    pw = max(float(person[2]) - px1, 1.0)
    ph = max(float(person[3]) - py1, 1.0)
    cx, cy = px1 + ox * pw, py1 + oy * ph
    half = max(pw * 0.18, 4.0)          # ~a head's width relative to shoulders
    return np.array([cx - half, cy - half, cx + half, cy + half], np.float32)


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
            # model_available, not Path.exists: an encrypted deployment has only
            # <name>.enc on disk, and a plain existence check silently disabled
            # head tracking and fell back to face detection.
            if model_available(hp):
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
        # Recognised passes traced so far. Tracing embeds every frame of every
        # track, which is far more GPU work than the pipeline normally does, so
        # it stops itself rather than running until someone remembers.
        self.traced_passes = 0
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
            # THE decision point. Everything before this is provisional: the
            # overlay name, the running best frame, the leader of the tally.
            # The pass is over, so the whole of its evidence is in, and the
            # consensus rule gets the last word. Attendance reads only this.
            final_id = t.vote.finalize()
            t.employee_id = final_id
            t.name = self.gallery.name(final_id) if final_id is not None else ""
            if t.employee_id is not None and settings.debug_trace_tracks:
                self.traced_passes += 1
                if self.traced_passes == settings.debug_trace_limit:
                    log.info("track tracing complete: %d recognised passes captured "
                             "in %s/<person>/frames/", self.traced_passes,
                             settings.debug_dir)
            out.append(CompletedTrack(
                track_id=tid, employee_id=t.employee_id, name=t.name,
                # A DECIDED track reports the committed identity's own best
                # score.  best_seen is the top similarity across every frame
                # including misses and other people, which is what you want to
                # explain a pass that matched nobody - and exactly what you must
                # not print next to a name, because it can belong to somebody
                # else who happened to share the track.
                best_score=float(t.vote.best_score if t.vote.decided
                                 else max(t.best_seen, t.vote.best_score or 0.0)),
                best_margin=float(t.vote.best_margin),
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
                # Highest-scoring aligned face FOR THE COMMITTED IDENTITY:
                # this is what the dashboard shows. It is the frame that drove
                # the identity, so it is the one to judge the decision by - if
                # the match is wrong, this is the image that made it wrong.
                #
                # "For the committed identity" is load-bearing. This was once
                # the best frame over ALL identities, so a track holding two
                # people showed the higher-scoring stranger under the voted
                # person's name - a correct-looking row with the wrong face.
                #
                # This used to be the *clearest* frame instead, because
                # score-best kept surfacing foreheads and backs of heads. That
                # was a symptom of min_aligner_score sitting at 0.40; at 0.90
                # nothing without a visible face can be embedded at all, so the
                # top-scoring frame is now both safe to show and more
                # informative. The clearest frame is kept alongside for
                # comparison.
                crop=(t.vote.best_snapshot if t.vote.best_snapshot is not None
                      else (t.vote.quality_snapshot if t.vote.quality_snapshot is not None
                            else t.best_crop)),
                score_crop=(t.vote.quality_snapshot if t.vote.quality_snapshot is not None
                            else t.best_crop),
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

    def _tracing(self) -> bool:
        return (settings.debug_trace_tracks
                and self.traced_passes < settings.debug_trace_limit)

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

        head_boxes = np.zeros((0, 4), np.float32)
        if self.head_detector is not None:
            # One detector pass, both classes. The aligner does its own face
            # localisation inside whatever crop it is given, so a head box
            # serves for recognition; aligning from head boxes scores 0.187
            # median against 0.192 from a dedicated face detector, 10 ms
            # cheaper, and still produces a box where face detection finds none.
            #
            # TRACKING runs on the person box, not the head. Measured over 6
            # clips: person boxes gave 7 coherent tracks with a median length of
            # 188 frames, where head boxes gave 13 tracks with a median of 9 and
            # twice the fragmentation - from the same number of detections. A
            # body is simply a larger, more persistent thing to associate.
            dets = self.head_detector.detect(small, want=None)
            hd = [d for d in dets if d.cls == CLS_HEAD]
            pd = [d for d in dets if d.cls == CLS_PERSON]
            head_boxes = (np.stack([d.box for d in hd]).astype(np.float32) / scale
                          if hd else np.zeros((0, 4), np.float32))
            src = pd if (settings.track_on == "person" and pd) else hd
            track_boxes = (np.stack([d.box for d in src]).astype(np.float32) / scale
                           if src else np.zeros((0, 4), np.float32))
            track_scores = (np.array([d.score for d in src], np.float32)
                            if src else np.zeros((0,), np.float32))
            tracking_persons = src is pd
        else:
            fd_ = self.detector.detect(small)
            track_boxes = (np.stack([d.box for d in fd_]).astype(np.float32) / scale
                           if fd_ else np.zeros((0, 4), np.float32))
            track_scores = (np.array([d.score for d in fd_], np.float32)
                            if fd_ else np.zeros((0,), np.float32))
            tracking_persons = False
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
                    vote=TrackVote(window=settings.vote_window, required=settings.vote_required,
                                   consensus=settings.vote_consensus,
                                   min_recognitions=settings.vote_min_recognitions),
                )
                self.tracks[tid] = st
            st.last_seen = now
            st.box = box

            if tracking_persons:
                st.person_box = box
                head = _head_in(box, head_boxes)
                if head is not None:
                    st.face_box = head
                    st.head_offset = _offset_of(head, box)
                    st.frames_without_head = 0
                    traj_box = head
                else:
                    # Head not detected this frame - too small, turned away, or
                    # occluded. The person is still tracked, so keep the
                    # trajectory going: direction is what this track is for, and
                    # it resolves fine without ever seeing a face. Recognition
                    # simply does not run on these frames.
                    st.face_box = None
                    st.frames_without_head += 1
                    traj_box = _synth_head(box, st.head_offset)
            else:
                st.person_box = None
                st.face_box = box
                traj_box = box

            # The trajectory ALWAYS follows a head-shaped point, never the body
            # centroid, even when tracking persons. The tripwire and the depth
            # trend were calibrated against head positions; a body centre sits
            # lower in frame and would cross the line at a different moment,
            # silently invalidating that calibration. When the head is missed we
            # synthesise its position from the person box using the offset
            # observed while both were visible, so the path stays continuous and
            # in the same geometry throughout.
            st.trajectory.add(now, traj_box, frame.image.shape[1], frame.image.shape[0])
            d, why = st.trajectory.direction(self.direction_cfg)
            st.direction_reason = why          # refresh: a stale reason misleads
            if d is not Direction.UNKNOWN:
                st.direction = d
            # Only tracks showing a head this frame can be recognized. With
            # person tracking a track can live for many frames with no head at
            # all, and those frames have nothing to align.
            #
            # Normally recognition stops once the vote commits. While tracing,
            # keep scoring for the whole life of the track so the score curve
            # across the entire pass is recorded, not just its run-up.
            if st.face_box is not None and (not st.vote.decided or self._tracing()):
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
        # defined before the branch: used after it, and `pending` can be empty
        gated_trace: list = []
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
                    # While tracing, a gated face is the interesting one: it is
                    # what the gates threw away, and you cannot judge whether
                    # they were right without seeing it scored. Embed it, tag it
                    # with the gate that rejected it, and save it - but keep it
                    # OUT of `keep`, so it never reaches the vote and cannot
                    # affect the identity.
                    if self._tracing():
                        gated_trace.append((st, f, q))
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
                    if settings.save_all_frames or self._tracing():
                        x1, y1, x2, y2 = [int(v) for v in st.face_box]
                        res.candidates.append(FrameCandidate(
                            track_id=st.track_id,
                            name=self.gallery.name(m.employee_id) if m.employee_id
                                 else f"_near_{self.gallery.name(m.runner_up)}",
                            score=float(m.score), aligned=f.aligned,
                            native=frame.image[max(0, y1):y2, max(0, x1):x2],
                            quality=q, accepted=m.employee_id is not None,
                            # st.attempts, not st.embedded: attempts counts EVERY
                            # assessed frame, so accepted and gated frames share
                            # one sequence and the pass reads in order.
                            index=st.attempts))
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

                    provisional = st.vote.add(m, f.aligned, quality=q.score)
                    if m.employee_id is not None:
                        st.score = max(st.score, m.score)

                    # Display only. The identity is not settled until the track
                    # ends, so this name may change as the pass accumulates
                    # evidence - which is the point. `emitted` records that the
                    # track has had a name at some point, for the live feed.
                    if provisional is not None:
                        st.employee_id = provisional
                        st.name = self.gallery.name(provisional)
                        st.emitted = True
                        votes = "/".join("?" if v is None else str(v) for v in st.vote.votes)
                        res.outcomes.append(RecognitionOutcome(
                            track_id=st.track_id, employee_id=provisional, name=st.name,
                            score=st.vote.best_score, margin=m.margin,
                            face_px=int(q.face_px), votes=votes, ts=now,
                            crop=st.vote.best_snapshot if st.vote.best_snapshot is not None else st.best_crop,
                            box=st.box.copy(), quality=q,
                            direction=st.direction.value, direction_reason=st.direction_reason,
                        ))
                        st.reported_direction = st.direction
                    elif not st.emitted:
                        st.name = "…" if m.employee_id is None else self.gallery.name(m.employee_id)

        # Trace: score the faces the gates rejected, so the saved frames show the
        # whole pass rather than only its winners. Deliberately after the real
        # path, and deliberately not touching any track state beyond the debug
        # candidate list.
        if gated_trace:
            try:
                g_aligned = np.stack([f.aligned for _st, f, _q in gated_trace])
                g_embs = self.recognizer.embed(g_aligned)
                for (st, f, q), emb in zip(gated_trace, g_embs):
                    gm = self.gallery.match(emb, settings.recognition_threshold,
                                            settings.second_best_margin)
                    who = (self.gallery.name(gm.employee_id) if gm.employee_id
                           else f"_near_{self.gallery.name(gm.runner_up)}")
                    x1, y1, x2, y2 = [int(v) for v in st.face_box]
                    res.candidates.append(FrameCandidate(
                        track_id=st.track_id,
                        name=who,
                        score=float(gm.score),
                        aligned=f.aligned,
                        native=frame.image[max(0, y1):y2, max(0, x1):x2],
                        quality=q,
                        accepted=False,          # filenames get "gated"
                        index=st.attempts))
            except Exception:
                log.exception("trace embedding failed")

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
