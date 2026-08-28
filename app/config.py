"""Single source of truth for every tunable.

The previous codebase had the recognition threshold in three places with three
different values (0.85 in env.example, 0.62 in constants.py, and the one that
actually ran: 0.5 hardcoded).  Everything here is read from the environment or
a .env file exactly once, at import.
"""
from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # ---- paths ----------------------------------------------------------
    root: Path = ROOT
    models_dir: Path = ROOT / "models"
    data_dir: Path = ROOT / "data"
    media_dir: Path = ROOT / "media"
    gallery_dir: Path = ROOT / "face_id_users"

    database_url: str = f"sqlite+aiosqlite:///{ROOT / 'data' / 'ematsy.db'}"
    database_url_sync: str = f"sqlite:///{ROOT / 'data' / 'ematsy.db'}"

    # ---- models ---------------------------------------------------------
    detector_kind: str = "yolo"                       # only option; YuNet removed
    # CrowdHuman YOLOv8n (person + head). Used only for tracking - see the note
    # in app/core/head_detector.py. Set to "" to fall back to face-box tracking.
    # Exported from the same yolov8n_640.pt weights at a 960 input. Training
    # size and inference size are independent for a fully-convolutional YOLO, so
    # no retraining was needed - only a re-export.
    #
    # Why 960: after letterboxing a 4K frame, a head reaches the model at 16 px
    # median at 640 input, which is right at the floor for small-object
    # detection. At 960 it arrives at 24 px. Measured on 25 real frames: 30
    # heads at 640 vs 32 at 960 (5.9 -> 12.0 ms). 1280 found no more than 960
    # while costing 21.7 ms, so the gain stops there.
    # Retrained 100 epochs on full CrowdHuman (mAP50 .812, mAP50-95 .509) and
    # exported at the frame's native 16:9 rather than a square. A 960x540 frame
    # letterboxed to 960x960 spends 44% of every buffer on grey padding that the
    # CPU converts and the GPU convolves. Measured against the old 960x960
    # export on 168 real corridor frames: finds all 111 heads it found plus 15
    # more (9-14 px, distant), mean confidence .730 vs .713, median IoU .915 -
    # strictly better, and detect() drops 60.0 -> 22.7 ms.
    # fp16. Validated on 168 real corridor frames against the fp32 export:
    # identical 126 detections, zero found by only one of them, mean confidence
    # 0.7295 vs 0.7296, median IoU 0.9848 - while GPU inference drops 4.33 ->
    # 3.46 ms and the file halves to 6.2 MB.
    head_model: str = "yolov8n_head_960x544_fp16.onnx"
    head_input: int = 960          # fallback only; geometry comes from the model
    head_conf: float = 0.35
    # Which class ByteTrack associates on. The detector finds both in one pass.
    # Measured over 6 real clips, same detections either way:
    #   head   -> 13 tracks, median  9 frames, 31% fragments
    #   person ->  7 tracks, median 188 frames, 14% fragments
    # A body is larger and more persistent, so association survives the moments
    # a head turns away or is briefly missed. Recognition still runs on the head
    # box found inside each tracked person, and the trajectory still follows a
    # head-shaped point so the calibrated direction geometry is unaffected.
    track_on: str = "person"           # person | head
    detector_model: str = "yolov8n-face.pt"
    aligner_model: str = "dfa_mobilenet_aligner.onnx"
    # Fine-tuned AdaFace IR-101 (NIST FRVT submission build). Exported with
    # a dynamic batch dim and a named "embedding" output, so no graph surgery
    # is needed. Genuinely different weights from the base: mean cosine 0.53.
    # fp16. Full-gallery revalidation is numerically indistinguishable from
    # fp32 - d-prime 10.03 both, rank-1 100%, genuine mean 0.8484 both, weakest
    # genuine pair 0.4551 -> 0.4557 - so the calibrated threshold still applies.
    # 6.19 -> 4.57 ms per batch, and 261 MB -> 130 MB.
    recognizer_model: str = "adaface_ir101_finetune_fp16.onnx"

    # 960, not 1280: detector recall is unchanged (40/40 on the gallery, and
    # YOLOv8n-face holds 100% recall down to a 16 px face — see
    # bench/detect_resolution.py), while both the downscale and the forward
    # pass get cheaper.
    detect_width: int = 960        # downscale for detection; crops come from full-res
    detect_conf: float = 0.35
    # The DFA graph resizes its input to 160 regardless, so a larger crop_size is
    # a resample for nothing: measured d-prime is 12.37-12.53 across 144..320.
    align_crop_size: int = 160
    # 1.30 agrees with three independent lines of evidence: the measured sweep
    # (flat 1.0-1.35, falling from 1.43), the break-even arithmetic (160/1.43 =
    # 112, the output size), and the CVLFace authors' own reference input, whose
    # face fills 57% of the frame. See app/core/aligner.py for the full note.
    align_margin: float = 1.30
    # "sharp" warps the 112 crop from the native-resolution crop instead of the
    # aligner's internal 160 tensor - one resample instead of two.
    # Gallery: d-prime 12.53 -> 12.64, weakest genuine pair 0.547 -> 0.571.
    # "reference" restores bit-exact CVLFace PyTorch numerics.
    align_mode: str = "sharp"
    embed_batch: int = 8

    # ---- quality gates --------------------------------------------------
    # Calibrated against bench/degrade_test.py, which degraded known gallery
    # faces to corridor conditions and re-matched.  The recognizer held up far
    # better than the original guesses assumed:
    #     60 px face          -> 0.99
    #     sharpness 78 (k=5)  -> 0.96
    #     sharpness 43 (k=9)  -> 0.71-0.84
    #     35 deg pitch        -> 0.91-0.95
    # The first walkthrough gated 104 of 116 observations, mostly on blur, while
    # scores that low are still perfectly recognizable.  Gates now reject only
    # genuinely unusable frames; the threshold and the 3-of-5 vote do the real
    # filtering.
    # Measured against HEAD boxes, which run ~1.33x the face box, so this is
    # about a 42 px face.
    #
    # Was 66. Measured 2026-08-28 by replaying 20 recordings: the face-size gate
    # was rejecting 5 575 of 7 578 frames that had a head box - by far the
    # largest loss in the pipeline, dwarfing aligner (418), yaw (54) and blur
    # (40) put together. A 26-second pass embedded 12 of its ~520 frames.
    #
    # Sweeping it showed 66 was costing recognitions for nothing: at 56 one more
    # person is recognised, embeddings per pass rise 43%, and the mean committed
    # score RISES 0.324 -> 0.358, because more frames per pass means a better
    # best frame. Held flat down to 40; at 32 unnamed tracks start climbing.
    #
    # 56 is the conservative end of that plateau - it captures the whole measured
    # gain while staying furthest from the size where AdaFace degrades. 40-48
    # yields still more evidence and may be better, but that cannot be justified
    # until false accepts can be measured against labelled impostors.
    min_face_px: int = 56
    min_laplacian_var: float = 25.0
    # The DFA aligner's own confidence that it found a face - and the only gate
    # that can tell a face from the BACK OF A HEAD. It is not a face detector:
    # given a hairline it happily regresses symmetric landmarks, which then
    # produce a confident-looking "yaw 6, pitch 3" from meaningless points.
    # Sharpness does not help either - hair texture scored 473 where real faces
    # score 50-270.
    #
    # 0.40 was far below anything that discriminates. Measured:
    #   enrolment faces (n=220)   min 0.9929, none below 0.99
    #   live crops <0.70          backs and tops of heads, no usable face
    #   live crops 0.70-0.90      steep top-down, face barely visible
    #   live crops 0.90-0.99      face visible, looking down
    #   live crops >=0.99         clear near-frontal faces
    #   the reported false accept 0.604 -> matched a real person at 0.219
    # Over 174 live crops, 12 above-threshold matches came from sub-0.90 crops,
    # the worst being aligner 0.545 matching at 0.274 - higher than several
    # genuine recognitions that day.
    #
    # 0.90 keeps every crop with a visible face and drops all 12. It costs
    # recognition opportunities on people walking head-down, which is the right
    # trade: a miss costs nothing because they are seen again, while a false
    # accept records one person as another and nothing in the data reveals it.
    min_aligner_score: float = 0.900
    max_yaw_deg: float = 45.0
    max_pitch_deg: float = 40.0

    # ---- recognition ----------------------------------------------------
    # CALIBRATED 2026-08-19 from two real walkthroughs (bench/calibrate_threshold.py):
    #   not-enrolled walker : 43 gate-passing obs, best score max 0.184
    #   enrolled walkers    : 87 gate-passing obs, best score max 0.514
    # 0.26 sits 41% above the worst impostor ever observed live, and the
    # 3-of-5 same-identity vote requires three such frames to agree.
    # Zero false commits in simulation at every threshold from 0.22 up.
    # The gallery-photo measurement suggested 0.40; live crops are a lower-
    # scoring domain, which is exactly why this had to be measured.
    # PROVISIONAL for adaface_ir101_finetune. Thresholds do not transfer between
    # models: this one compresses impostor scores hard (gallery max 0.208 vs the
    # base model's 0.392), so the whole scale sits lower. 0.14 is the base
    # deployment threshold rescaled by the same fraction of each model's
    # gallery FAR=0 point (0.26/0.395 = 0.66; 0.210 x 0.66 = 0.138).
    # Confirm against a live walkthrough before trusting it.
    # 0.18, raised from 0.14 after a confirmed false positive at 0.157
    # (a forehead crop matched to the wrong person). The lowest accepts are
    # where the errors live: today's accepts ran 0.157-0.522, and the single
    # known error sat at the very bottom. 0.18 drops 4 of 43 current accepts.
    #
    # For attendance a false accept is far worse than a miss - one person is
    # recorded as another, and nothing in the data reveals it - while a miss
    # costs nothing, because the person is seen again on their next pass.
    # Still provisional: it needs a labelled set, not one confirmed error.
    recognition_threshold: float = 0.18
    # Live scores are compressed (0.26-0.53) vs gallery scores (~0.90), so a
    # fixed margin is proportionally weaker here.  0.08 would have excluded the
    # one weak event seen in the first live run (margin 0.065).
    second_best_margin: float = 0.045
    vote_window: int = 5
    vote_required: int = 3

    # Identity is decided by CONSENSUS over the whole pass, not by the first few
    # frames to agree.  A track commits a person when that person accounts for
    # at least `vote_consensus` of the frames that produced ANY identity, and at
    # least `vote_min_recognitions` frames agree on them.
    #
    # Misses are deliberately NOT in the denominator: a frame that matched
    # nobody says nothing about which of two candidates is right, and counting
    # them would punish a person who was turned away for most of a pass.
    #
    # The floor exists because a fraction alone cannot judge a short pass - one
    # agreeing frame out of one is 100% consensus and would clear any percentage
    # rule.  Below the floor the pass commits nobody and is recorded as an
    # unknown sighting, which is the FAR-favouring choice: better to miss a
    # quick walk-through than to name the wrong person from two frames.
    vote_consensus: float = 0.65
    vote_min_recognitions: int = 5

    # How long a direction verdict may be relied on after the evidence for it
    # left the trajectory window. The window is 90 points - 4.5 s at 20 fps - so
    # a verdict older than that was computed from data the trajectory no longer
    # holds, and nothing has confirmed it since.
    #
    # Without this, `st.direction` latched the last non-UNKNOWN verdict forever
    # while `direction_reason` kept refreshing, so a person standing still was
    # committed with a direction up to TEN MINUTES old. Seven live instances on
    # 2026-08-27, including check-outs on the entrance camera for people who had
    # not moved. The latch itself is right - somebody pausing at a door should
    # keep their direction - it just may not outlive the window that produced it.
    direction_max_age_s: float = 5.0

    # ---- tracking -------------------------------------------------------
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.2
    track_match_thresh: float = 0.8
    # Frames, not seconds: ByteTrack keeps a lost track for
    # frame_rate/30 * track_buffer frames. At track_frame_rate=20 and
    # track_buffer=36 that is 24 frames = 1.2 s of occlusion tolerance, which
    # is what the pipeline had at 10 fps with the old 12/30 defaults.
    track_frame_rate: int = 20
    track_buffer: int = 36
    track_max_age_s: float = 3.0

    # ---- stream ---------------------------------------------------------
    rtsp_transport: str = "tcp"
    # Every frame. The old value of 2 dated from 12 fps cameras sharing the GPU
    # with a training job; both premises are gone and the streams now run 20 fps.
    # Detection is 22.7 ms against a 50 ms budget per camera at 20 fps, and full
    # rate doubles trajectory density - which is what the 55% UNKNOWN direction
    # rate was starving on (min_points=5 rarely reached at 10 fps).
    process_every_nth: int = 1
    stale_after_s: float = 5.0
    # 2 slots dropped ~8% of frames to burst spill once the pipeline ran every
    # frame. The queue is drop-oldest, so extra slots only fill when the
    # pipeline is momentarily behind - at a 15 ms median against a 50 ms budget
    # it sits near-empty, and the worst case costs 2 frames (~100 ms) of
    # staleness on a track that lasts seconds.
    frame_queue_size: int = 4

    # ---- attendance -----------------------------------------------------
    event_cooldown_s: int = 90
    day_boundary_hour: int = 4
    timezone: str = "Asia/Tashkent"
    open_interval_policy: str = "flag"       # flag | close_at_eod

    # ---- debug ----------------------------------------------------------
    # When on, every recognition writes the full annotated frame, the native
    # face crop and the aligned 112x112 the recognizer actually saw, plus a JSON
    # sidecar - so a wrong name can be inspected instead of guessed at.
    debug_capture: bool = True
    debug_capture_unknown: bool = True     # also keep gate-passing non-matches
    # Capped, not uncapped. 0 means unbounded, which is right for a debugging
    # session and wrong for an install that runs for months - the directory
    # grows without limit and it is full of face crops. Set 0 explicitly for a
    # capture run.
    debug_max_per_person: int = 40
    debug_dir: Path = ROOT / "data" / "debug"

    # ---- recording (DEBUG ONLY - off by default) -------------------------
    # Clip recording exists to build a replay corpus while tuning: it is how
    # the detector A/B, the direction comparison and the fp16 validation were
    # all measured with the cameras switched off.
    #
    # It is NOT for production. One day of two cameras produced 1,156 clips and
    # 3.2 GB, and those clips are video of identified people - a customer site
    # accumulating that silently is a privacy problem as much as a disk one.
    #
    # Turn it on for a debugging session:  record_clips=1 python scripts/run.py
    record_clips: bool = False
    record_width: int | None = 1920      # None = native 4K
    record_pre_roll_s: float = 2.0
    record_post_roll_s: float = 2.0
    record_max_clip_s: float = 60.0
    record_min_free_gb: float = 10.0

    # Save every gate-passing frame of every track, not just the best one.
    # Debug only, same reasoning as record_clips: it wrote 15,724 images in a
    # day. The best-shot capture below is what a production install keeps.
    save_all_frames: bool = False

    # ---- track tracing (DEBUG ONLY) --------------------------------------
    # Normally recognition stops the moment the 3-of-5 vote commits: once the
    # identity is known, embedding more frames costs GPU time and changes
    # nothing. That also means the saved frames only cover the run-up to the
    # decision, so you cannot see how the score behaved across the WHOLE pass.
    #
    # With this on, every frame of a track is embedded and scored for as long
    # as the track lives, and each aligned face is written with its score. A
    # 5-second pass at 20 fps yields ~100 images named by score, so the
    # approach/retreat curve can be read straight off the filenames.
    #
    # Expensive - it embeds frames the pipeline would otherwise skip - so it
    # stops itself after debug_trace_limit recognised passes.
    debug_trace_tracks: bool = False
    debug_trace_limit: int = 10       # recognised passes, then tracing stops
    save_all_min_free_gb: float = 8.0

    # ---- retention ------------------------------------------------------
    snapshot_retention_days: int = 90

    # ---- api ------------------------------------------------------------
    # Served under a sub-path by a reverse proxy, e.g. /faceid. The edge proxy
    # in front of aiscan.airi.uz does NOT strip the prefix - the backend
    # receives /faceid/... verbatim - so FastAPI's root_path is not enough and
    # every route, static mount, link and WebSocket URL must carry it.
    # Empty means served at the root, which is how it runs locally.
    url_prefix: str = ""
    host: str = "0.0.0.0"
    port: int = 8000
    secret_key: str = Field(default="change-me-in-production")

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def model_path(self, name: str) -> Path:
        return self.models_dir / name


settings = Settings()
