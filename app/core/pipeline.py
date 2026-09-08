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
    # Periodic body-crop bookkeeping (ReID data collection). Kept on the track
    # rather than in the caller because the cadence is per person-pass.
    last_body_save: float = 0.0
    body_saves: int = 0
    # Body crop from the quality-best frame, kept for EVERY track. The voted
    # one (`vote.best_person`) is scoped to a committed identity and is
    # therefore None for an unknown - which left unknown passes showing a
    # 112x112 aligned face on the dashboard, the one crop a human cannot judge.
    disp_person: np.ndarray | None = None
    # Where the head sat inside the person box, as a fraction of that box, the
    # last time both were seen together. Used to synthesise a head point on
    # frames where the head is missed - the person box is still tracked, so the
    # trajectory stays continuous and direction keeps resolving even when the
    # head is too small or turned away to detect.
    head_offset: tuple[float, float] | None = None
    # ...and how big the head was relative to that box, so a synthesised head
    # has the same area a detected one would. The depth signal compares areas
    # along the track, and a synthesised box of a different size than the real
    # head reads as the person moving when only the detector flickered.
    head_rel: tuple[float, float] | None = None
    frames_without_head: int = 0
    trajectory: Trajectory = field(default_factory=Trajectory)
    # Live verdict for the overlay. Attendance never reads it: the pass is
    # decided in _prune() from the whole trajectory, once, when it ends.
    direction: Direction = Direction.UNKNOWN
    direction_reason: str = ""
    embedded: int = 0
    best_face_px: int = 0
    nearest_id: int | None = None
    best_vector: np.ndarray | None = None
    best_seen: float = 0.0
    # RUNNING QUALITY-WEIGHTED SUM OF EVERY EMBEDDING THIS TRACK PRODUCED.
    #
    # The vote judges each frame separately and then needs
    # `vote_min_recognitions` of them to agree, so a pass that yields three
    # good frames names nobody however clearly each one scores. The evidence
    # was there; the rule could not reach it. One template over the whole pass
    # can, and it is measured to recover 23-30% more employee passes per day
    # at 97.4% precision (integration/docs/HISOBOT_YUZ_TANA.md, section 4).
    #
    # A SUM, not a list: at 512 floats per frame a loiterer would otherwise
    # accumulate megabytes on the capture thread, and nothing downstream wants
    # the individual frames back.
    #
    # The weight is `aligner_score * sqrt(ipd)`. The research normalises IPD
    # within the pass first; that divides every weight by one per-pass
    # constant, which cancels when the template is renormalised to unit length
    # at the end - so this is the same formula, expressed in a form that can be
    # accumulated one frame at a time.
    face_sum: np.ndarray | None = None
    face_wsum: float = 0.0
    face_frames: int = 0
    best_ipd: float = 0.0
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
    # The track's own first sighting, carried rather than reconstructed. A
    # consumer deriving it as `completion_time - duration_s` is wrong by
    # `track_max_age_s`, because a track is pruned three seconds AFTER it was
    # last seen - which silently orphaned every pass in the ReID collector.
    first_seen: float = 0.0
    traj_points: int = 0
    travel: float = 0.0
    native: np.ndarray | None = None
    score_crop: np.ndarray | None = None   # top-scoring frame, for diagnosis
    person_crop: np.ndarray | None = None  # BODY crop, what the dashboard shows
    context: np.ndarray | None = None
    quality: object = None
    box: np.ndarray | None = None
    crop: np.ndarray | None = None
    vector: np.ndarray | None = None
    nearest_employee_id: int | None = None
    # The whole pass as ONE face query, L2-normalised. `vector` above is the
    # single best-SCORING frame and is kept for diagnosis; this is built from
    # every frame that cleared the gates, weighted by how much face each one
    # actually showed. It is what the second-chance matcher in
    # app/services/worker.py asks the gallery about, and what the pseudo-person
    # gallery stores for an unregistered person.
    face_template: np.ndarray | None = None
    face_frames: int = 0
    # Best inter-pupil distance seen in the pass, in source pixels. The measure
    # of whether this template is worth trusting at all: below ~20 px a face
    # embedding links the wrong people rather than linking weakly.
    best_ipd: float = 0.0


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
class BodyCrop:
    """One periodic body crop, for ReID data collection.

    Taken on a timer rather than on recognition, so it also covers people the
    face path never identifies - who are exactly the ones a ReID model is
    wanted for. The pipeline only produces these; writing and embedding happen
    off the capture thread in app/services/reid_worker.py.
    """
    track_id: int
    ts: float                      # frame.ts, the capture instant
    image: np.ndarray              # BGR uint8, already downscaled
    score: float                   # detector confidence, the tracklet weight
    first_seen: float              # identifies the pass; track ids are reused


@dataclass
class FrameResult:
    frame: Frame
    tracks: list[TrackState] = field(default_factory=list)
    outcomes: list[RecognitionOutcome] = field(default_factory=list)
    completed: list[CompletedTrack] = field(default_factory=list)
    candidates: list[FrameCandidate] = field(default_factory=list)
    body_crops: list[BodyCrop] = field(default_factory=list)
    timings: dict = field(default_factory=dict)


def _unit(v: np.ndarray | None) -> np.ndarray | None:
    """L2-normalise a weighted sum back into a query vector, or pass on None.

    The norm is what makes the sum comparable with the gallery at all, and it
    is where the per-pass weight constant cancels - see `TrackState.face_sum`.
    A degenerate sum (every weight zero, or two embeddings that cancelled) has
    no direction to normalise and is dropped rather than scaled up into noise.
    """
    if v is None:
        return None
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 1e-6 else None


def _whole_body(person_box: np.ndarray, frame_w: int, frame_h: int,
                edge_px: int = 4, min_aspect: float = 1.5) -> bool:
    """Is this a usable body crop, or a fragment?

    A track begins the instant somebody appears at the edge of view, so its
    first box is reliably a head and one shoulder. The crop comes out wider than
    it is tall, and it is the worst possible ReID sample: the model squashes
    whatever it is given to 256x128, so a fragment is stretched into something
    that resembles no real body.

    ONLY THE LEFT AND RIGHT EDGES COUNT AS TRUNCATION. Measured over one
    corridor clip (39 person boxes): 46% touch the TOP of the frame and 10% the
    bottom, because the cameras look down a corridor and somebody walking toward
    one has their head near the top and their feet out of shot. Rejecting those
    threw away 56% of all crops, including the closest and largest - the best
    samples there are. Rejecting on left/right and shape instead keeps 85%.

    Horizontal truncation is different: a box against the left or right edge is
    a person half out of view sideways, and that really is a fragment.
    """
    x1, y1, x2, y2 = (float(v) for v in person_box[:4])
    if x1 <= edge_px or x2 >= frame_w - edge_px:
        return False
    w, h = x2 - x1, y2 - y1
    return w > 0 and h / w >= min_aspect


def _body_crop(image: np.ndarray, person_box: np.ndarray,
               max_w: int = 320) -> np.ndarray | None:
    """A display-sized copy of the person box.

    Person boxes are tall (roughly 1:2.5) and at 4K a full-resolution copy is
    several megabytes, so it is downscaled here rather than at render time -
    these are written to disk for every recognised pass and kept indefinitely.
    Aspect is preserved; the dashboard letterboxes rather than distorting.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in person_box[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = image[y1:y2, x1:x2]
    if crop.shape[1] > max_w:
        scale = max_w / crop.shape[1]
        crop = cv2.resize(crop, (max_w, max(1, int(crop.shape[0] * scale))),
                          interpolation=cv2.INTER_AREA)
    return crop.copy()


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


def _rel_size(head: np.ndarray, person: np.ndarray) -> tuple[float, float]:
    """Head width and height as fractions of the person box."""
    pw = max(float(person[2] - person[0]), 1.0)
    ph = max(float(person[3] - person[1]), 1.0)
    return (max(float(head[2] - head[0]), 1.0) / pw,
            max(float(head[3] - head[1]), 1.0) / ph)


# Head centre when none has ever been seen for this track: horizontally centred,
# and a tenth of the way down - where a standing person's head is.
_DEFAULT_HEAD_OFFSET = (0.5, 0.10)
# ...and its size, when no head was ever measured: about a third of the
# shoulder width, and a seventh of a (whole) body's height.
_DEFAULT_HEAD_REL = (0.36, 0.14)


def _synth_head(person: np.ndarray, offset: tuple[float, float] | None,
                rel: tuple[float, float] | None = None) -> np.ndarray:
    """A head-sized box where this person's head should be.

    Returned as a box rather than a point because Trajectory records boxes and
    uses their AREA for the depth signal. Both its position and its size come
    from the last frame on which the head was actually seen inside this
    person box, so the synthesised head is the detected head's stand-in and
    not a differently sized box that reads as a change of distance.
    """
    ox, oy = offset or _DEFAULT_HEAD_OFFSET
    rw, rh = rel or _DEFAULT_HEAD_REL
    px1, py1 = float(person[0]), float(person[1])
    pw = max(float(person[2]) - px1, 1.0)
    ph = max(float(person[3]) - py1, 1.0)
    cx, cy = px1 + ox * pw, py1 + oy * ph
    hw = max(pw * rw * 0.5, 4.0)
    hh = max(ph * rh * 0.5, 4.0)
    return np.array([cx - hw, cy - hh, cx + hw, cy + hh], np.float32)


class CameraPipeline:
    def __init__(self, name: str, gallery: Gallery, detector=None, aligner=None,
                 recognizer=None, direction_cfg: DirectionConfig | None = None,
                 head_detector=None):
        self.name = name
        self.gallery = gallery
        self.direction_cfg = direction_cfg or DirectionConfig()
        # Resolved from the recognizer in use, not read from a single global:
        # thresholds are calibrated per model and do not transfer. See
        # settings.recognizer_thresholds.
        self.threshold = settings.threshold_for(settings.recognizer_model)

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
                # Detect down to ByteTrack's LOW threshold, not head_conf.
                # ByteTrack's second association keeps an occluded or blurred
                # person on their track while their confidence dips into
                # [track_low_thresh, track_high_thresh); pre-filtering at
                # head_conf never let those boxes reach it, so a pass split
                # into two tracks that each needed its own five agreeing
                # frames. head_conf still gates the HEAD boxes below, which
                # feed recognition and the trajectory; new tracks still need
                # new_track_thresh, so a low box cannot start one.
                self.head_detector = HeadDetector(
                    hp, size=settings.head_input,
                    conf=min(settings.head_conf, settings.track_low_thresh))

        # The face detector only runs when there is no head detector. Building
        # it anyway cost every worker a torch CUDA context and the YOLO-face
        # weights on a card that is deliberately shared.
        if detector is not None:
            self.detector = detector
        elif self.head_detector is None:
            self.detector = build_detector(
                settings.detector_kind,
                settings.model_path(settings.detector_model),
                imgsz=settings.detect_width,
                conf=settings.detect_conf,
            )
        else:
            self.detector = None
        self.aligner = aligner or FaceAligner(
            settings.model_path(settings.aligner_model),
            crop_size=settings.align_crop_size,
            margin=settings.align_margin,
            mode=settings.align_mode,
        )
        self._log_recognizer = True
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

        log.info("recognizer %s: threshold=%.3f",
                 Path(str(settings.recognizer_model)).name, self.threshold)
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
            # THE decision point, and the only one. Everything before this is
            # provisional: the overlay name, the running best frame, the leader
            # of the tally. The pass is over, so the whole of its evidence is
            # in, and the consensus rule gets the last word.
            #
            # Exactly one attendance decision per pass follows from this:
            # worker._persist_completed consumes CompletedTrack and nothing
            # else writes attendance. res.outcomes, emitted during the pass, is
            # a live feed for scripts/live_test.py and must never be persisted -
            # it carries provisional identities that the consensus can overturn.
            # Direction is decided HERE, from the whole trajectory, exactly
            # once. The per-frame verdict on the track is for the overlay and
            # may lag by a few frames; this one has every point of the pass.
            # There is no latch and no staleness: the track keeps its full
            # history, so "where did this person end up relative to where
            # they started" is answered from evidence, however long ago they
            # last moved. See app/core/direction.py.
            direction, direction_reason = t.trajectory.direction(self.direction_cfg)

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
                # The committed identity's own best body when there is one;
                # otherwise the quality-best body of the pass. An unknown pass
                # has no committed identity, and showing its aligned face
                # instead defeats the point of showing a body at all - a person
                # is recognisable by build, clothing and posture, and least of
                # all by a tight crop of a face the system could not place.
                person_crop=(t.vote.best_person if t.vote.best_person is not None
                             else t.disp_person),
                embedded_frames=t.embedded, gated_frames=t.gated,
                direction=direction.value, direction_reason=direction_reason,
                face_px=t.best_face_px, duration_s=max(0.0, t.last_seen - t.first_seen),
                first_seen=t.first_seen,
                traj_points=len(t.trajectory), travel=t.trajectory.travel(),
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
                face_template=_unit(t.face_sum), face_frames=t.face_frames,
                best_ipd=t.best_ipd,
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
        """Detect on a downscale; return boxes in FULL-RES coordinates.

        Face-detector path only: with a head detector configured there is no
        face detector to run (see __init__), and this must not be reached.
        """
        if self.detector is None:
            raise RuntimeError("no face detector: the head detector is in use")
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

    def flush(self) -> list["CompletedTrack"]:
        """Finalize every live track, whatever its age.

        `_prune` only runs inside `process()` and is driven by `frame.ts`, so
        when frames stop - shutdown, or a camera that dies - every track still
        in view was simply dropped. The pass produced NEITHER a recognition
        event NOR an unknown sighting: somebody walked past and the system has
        no record of it at all. Anyone still mid-corridor at a restart was lost
        this way, every restart.
        """
        if not self.tracks:
            return []
        newest = max(t.last_seen for t in self.tracks.values())
        # Push the clock past every track's age so _prune retires all of them,
        # and reuse it rather than duplicating the finalization rules.
        return self._prune(newest + settings.track_max_age_s + 1.0)

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
            hd = [d for d in dets if d.cls == CLS_HEAD and d.score >= settings.head_conf]
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

        fh0, fw0 = frame.image.shape[:2]
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

                # PERIODIC BODY CROP, taken here and nowhere else.
                #
                # It requires a DETECTED HEAD inside the person box, which is
                # the only reliable way to know the crop contains one. Geometry
                # cannot tell the difference: a box running to the top of frame
                # is sometimes a person standing close with their head fully in
                # view, and sometimes a person whose head is above the frame
                # entirely. Rejecting all of them threw away 46% of crops
                # including the closest and best; accepting all of them produced
                # headless torsos, which are near-useless for ReID - the head
                # and shoulders are where much of the signal is.
                #
                # This is deliberately BEFORE the face size, quality and
                # identity gates, so it still covers people the face path never
                # names - precisely the ones a ReID model is wanted for. Only
                # the head DETECTION is required, not a recognisable face.
                #
                # Cost is bounded twice: once per `body_crop_interval_s` per
                # track, and never more than `body_crop_max_per_pass`. A
                # rejected frame does NOT advance the timer, so the next usable
                # frame is taken immediately rather than waiting another
                # interval.
                if (settings.body_crop_interval_s > 0
                        and head is not None
                        and st.body_saves < settings.body_crop_max_per_pass
                        and now - st.last_body_save >= settings.body_crop_interval_s
                        and _whole_body(box, fw0, fh0)):
                    crop = _body_crop(frame.image, box, max_w=settings.body_crop_width)
                    if crop is not None:
                        res.body_crops.append(BodyCrop(
                            track_id=tid, ts=now, image=crop,
                            score=float(score), first_seen=st.first_seen))
                        st.last_body_save = now
                        st.body_saves += 1

                synth = head is None
                if head is not None:
                    st.face_box = head
                    st.head_offset = _offset_of(head, box)
                    st.head_rel = _rel_size(head, box)
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
                    traj_box = _synth_head(box, st.head_offset, st.head_rel)
            else:
                synth = False
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
            st.trajectory.add(now, traj_box, frame.image.shape[1], frame.image.shape[0],
                              synth=synth)
            # Live verdict for the overlay, over the whole track so far. It
            # is a few vectorised operations, but a long track asks them every
            # frame, so beyond the first 100 points it is refreshed every
            # fifth frame - the final decision in _prune() uses every point
            # regardless.
            n_pts = len(st.trajectory)
            if n_pts <= 100 or n_pts % 5 == 0:
                st.direction, st.direction_reason = \
                    st.trajectory.direction(self.direction_cfg)
            # Only tracks showing a head this frame can be recognized. With
            # person tracking a track can live for many frames with no head at
            # all, and those frames have nothing to align.
            #
            # EVERY gate-passing frame is recognized, for as long as the person
            # is in view. Recognition used to stop the moment the vote
            # committed, which capped a pass at a handful of frames; the
            # identity is now a consensus taken when the track ends, so more
            # frames is strictly more evidence for that decision. There is no
            # cap: accuracy comes first, and if the frame budget is ever
            # genuinely exhausted the answer is more GPU, not less evidence.
            if st.face_box is not None:
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
            # `zip(pending, faces)` below pairs tracks to faces BY POSITION, so
            # a short list would silently attribute one person's face to another
            # track. The aligner now guarantees one face per box, but this was
            # an `assert` - which vanishes under `python -O`, exactly the build
            # a customer runs. Checked for real, and the frame's recognition is
            # abandoned rather than risk a wrong name.
            if len(faces) != len(pending):
                log.error("aligner returned %d faces for %d boxes; skipping "
                          "recognition this frame rather than mispairing them",
                          len(faces), len(pending))
                faces, pending = [], []

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

                # One GEMM for the whole frame, and ONE gallery reference for
                # every read below. reload_gallery() swaps the attribute from
                # another thread, so re-reading it per face could score against
                # one snapshot and resolve the name against the next.
                gallery = self.gallery
                matches = gallery.match_batch(
                    embs, self.threshold, settings.second_best_margin)
                for (st, f), q, emb, m in zip(keep, quals, embs, matches):
                    st.embedded += 1
                    st.best_face_px = max(st.best_face_px, int(q.face_px))
                    # Into the tracklet template, regardless of what this frame
                    # matched. The template is a better QUERY, not a second
                    # vote: excluding the frames that matched nobody would
                    # rebuild the same evidence the live rule already has.
                    w = float(q.aligner_score) * float(np.sqrt(max(q.ipd, 1e-6)))
                    if w > 0.0:
                        st.face_sum = (emb * w if st.face_sum is None
                                       else st.face_sum + emb * w)
                        st.face_wsum += w
                        st.face_frames += 1
                    st.best_ipd = max(st.best_ipd, float(q.ipd))
                    if settings.save_all_frames or self._tracing():
                        x1, y1, x2, y2 = [int(v) for v in st.face_box]
                        res.candidates.append(FrameCandidate(
                            track_id=st.track_id,
                            name=gallery.name(m.employee_id) if m.employee_id
                                 else f"_near_{gallery.name(m.runner_up)}",
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
                        # ...and the body from that same frame, so an
                        # unrecognised pass still has a person to show.
                        if st.person_box is not None:
                            st.disp_person = _body_crop(
                                frame.image, st.person_box,
                                max_w=settings.body_crop_width)
                    if q.score > st.best_quality:
                        st.best_quality = q.score
                        st.best_crop = f.aligned

                    # Crop the body only when this frame will become the new
                    # best for its identity. A 4K person region is a large copy
                    # and every other frame's would be discarded immediately.
                    body = None
                    if (m.employee_id is not None and st.person_box is not None
                            and m.score > st.vote.best_for(m.employee_id)[0]):
                        body = _body_crop(frame.image, st.person_box)
                    provisional = st.vote.add(m, f.aligned, quality=q.score,
                                              person=body)
                    if m.employee_id is not None:
                        st.score = max(st.score, m.score)

                    # DISPLAY ONLY. Attendance never reads any of this: the one
                    # decision per pass is taken in _prune() from vote.finalize()
                    # and written by worker._persist_completed. res.outcomes is a
                    # live feed consumed by scripts/live_test.py.
                    #
                    # The name follows the leader and may change as the pass
                    # accumulates evidence - that is the point of deciding at the
                    # end. An outcome is appended only when the leader CHANGES,
                    # not every frame: the leader is recomputed per frame now, so
                    # an unguarded append would emit one record per frame for the
                    # whole time a person is in view.
                    if provisional is not None:
                        changed = st.employee_id != provisional
                        st.employee_id = provisional
                        st.name = gallery.name(provisional)
                        st.emitted = True
                        if changed:
                            votes = "/".join("?" if v is None else str(v)
                                             for v in st.vote.votes)
                            res.outcomes.append(RecognitionOutcome(
                                track_id=st.track_id, employee_id=provisional, name=st.name,
                                score=st.vote.best_score, margin=m.margin,
                                face_px=int(q.face_px), votes=votes, ts=now,
                                crop=(st.vote.best_snapshot
                                      if st.vote.best_snapshot is not None else st.best_crop),
                                box=st.box.copy(), quality=q,
                                direction=st.direction.value,
                                direction_reason=st.direction_reason,
                            ))
                    elif not st.emitted:
                        st.name = "…" if m.employee_id is None else gallery.name(m.employee_id)

        # Trace: score the faces the gates rejected, so the saved frames show the
        # whole pass rather than only its winners. Deliberately after the real
        # path, and deliberately not touching any track state beyond the debug
        # candidate list.
        if gated_trace:
            try:
                g_aligned = np.stack([f.aligned for _st, f, _q in gated_trace])
                g_embs = self.recognizer.embed(g_aligned)
                for (st, f, q), emb in zip(gated_trace, g_embs):
                    gm = self.gallery.match(emb, self.threshold,
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
