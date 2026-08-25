# EmAtSy v3 — Implementation Plan

Face-recognition attendance for two Hikvision cameras at AIRI, rebuilt around
YOLOv8n-face → ByteTrack → CVLFace DFA → AdaFace IR-101.

**Status:** plan approved, implementation in progress.
**Target host:** this laptop — Ryzen 7 5800H (16T), RTX 3070 Laptop 8GB, 13GB RAM.

---

## 1. Why a rebuild and not a patch

The existing `fast_api/` app cannot start (no schema creation, `requirements.txt`
is the Django-era lockfile), and three of its defects are architectural rather
than local: both cameras share one YOLO model and therefore one tracker; the
camera's identity is passed into the attendance decision and then ignored, so
IN/OUT does not exist; and `check_out_time` is never written by anything. See
`docs/AUDIT.md` for the full finding list.

What is worth keeping: the SQLAlchemy models and Django-compatible Jinja
templates (the UI is fine), and — most valuable — the verified ONNX exports in
`../cvlface_test/`, which match the reference PyTorch pipeline to 4.6e-5.

The new code lives in `app/`. `fast_api/` stays in place until `app/` reaches
parity, then is deleted.

---

## 2. Measurements this plan is built on

All numbers measured on the deployment host, not quoted from papers.
Reproduce with `bench/bench_models.py` and `bench/validate_gallery.py`.

### 2.1 Detector — YOLOv8n-face wins on speed, ties on accuracy

| resolution | YOLOv8n-face (CUDA) | YuNet (OpenCV DNN, CPU) |
|---|---|---|
| 640×360 | **6.2 ms** | 8.8 ms |
| 960×544 | **6.7 ms** | 23.5 ms |
| 1280×736 | **7.5 ms** | 46.8 ms |
| 1920×1088 | ~8 ms | 139.0 ms |

Over the 270-image enrolment gallery both found **270/270** faces with
near-identical box widths (median 227 vs 224 px), so accuracy did not separate
them. YuNet's design point is CPU inference at small resolutions; this host has
a GPU and a wide corridor that needs ≥1280px detection. **Decision: YOLOv8n-face
at imgsz=1280.** YuNet stays selectable in config as the CPU-only fallback.

### 2.2 Recognizer — IR-101 FP16

| model | bs1 | bs8 | d′ | rank-1 | FRR @ FAR=0 |
|---|---|---|---|---|---|
| ir101 fp32 | 6.20 ms | 3.17 ms/f | — | — | — |
| **ir101 fp16** | **4.57 ms** | **1.90 ms/f** | **11.33** | **100.00%** | **0.0000** |
| ir18 fp32 | 1.43 ms | 0.81 ms/f | 7.60 | 99.63% | 0.0167 |

IR-101 is 3× the cost of IR-18 and clearly better separated. At 4.6 ms/face it
is affordable here. **Decision: IR-101 FP16.** IR-18 stays configurable for
future multi-camera scale-out.

### 2.3 Gallery separation (54 people, 268 images, IR-101 FP16)

```
                   n     mean     std      min       p1      p50      max
genuine          540   0.8979   0.0847   0.4128   0.5804   0.9239   0.9947
impostor       35775   0.0417   0.0651  -0.2048  -0.0958   0.0372   0.3969

d-prime = 11.33      1:N leave-one-out rank-1 = 270/270 (100.00%)
FAR<=1e-3 -> threshold 0.300 (FRR 0.0000)
FAR<=1e-4 -> threshold 0.370 (FRR 0.0000)
FAR = 0   -> threshold 0.400 (FRR 0.0000)
```

**This measurement is enrolment-photo vs enrolment-photo.** Corridor crops are a
different domain — smaller, higher pitch angle, motion blur. The real operating
threshold will be lower and the separation narrower. `0.40` is provisional until
the calibration walkthrough (§7) is recorded.

Data quality notes from the same run:
- `029_Lyudmila` `image_04`/`image_05` scored 0.413 against their siblings —
  a different subject. Moved to `face_id_users/_quarantine/`, metadata updated.
- `007_Azizbek` vs `036_Anvar` is the most confusable pair at 0.397. Below any
  sensible threshold, but the pair to watch when tuning.

### 2.4 Decode

4K H.265 read at 41.8 ms/frame — i.e. stream rate, not decode-bound. After the
switch to H.264 @12fps the budget is ~83 ms/frame per camera, comfortable.

---

## 3. Camera configuration — applied

Both cameras are **Hikvision DS-2CD2083G2-I** (8MP, 120dB WDR), mounted high on
a wide-angle lens looking down a long corridor. Applied via ISAPI on 2026-08-19;
originals backed up to the scratchpad before the change.

| setting | was | now | why |
|---|---|---|---|
| codec | H.265 | **H.264** | cheaper decode, better ffmpeg/NVDEC support |
| resolution | 3840×2160 | **3840×2160** | unchanged — resolution *is* face pixels here |
| framerate | 20 | **12** | 12 fps is plenty through a doorway; halves decode |
| GOP | 50 | **12** | ~1s keyframe interval → fast reconnect recovery |
| bitrate | 6144 | **8192** kbps VBR | H.264 needs more bits than H.265 for equal quality |
| WDR | **off** | **open, level 50** | windows were fully blown out; entrants were silhouettes |

Roles: `192.168.1.2` = **IN**, `192.168.1.64` = **OUT**.
Stream URL: `rtsp://admin:***@<ip>:554/Streaming/Channels/101`

Still open, pending the walkthrough: capping max shutter at 1/250 to stop
walking faces motion-blurring. Deferred because it trades noise for sharpness
and should be judged on real captures.

---

## 4. Architecture

```
                  ┌─────────────── camera worker (1 process per camera) ───────────────┐
  RTSP 4K H.264   │                                                                     │
  ──────────────► │  decode ──► downscale 1280 ──► YOLOv8n-face ──► ByteTrack           │
    12 fps        │    │                                              │                 │
                  │    │                                              ▼                 │
                  │    │                                       per-track state          │
                  │    │                                     quality score each frame   │
                  │    │                                              │                 │
                  │    └──── crop face from FULL-RES 4K ◄─────────────┘                 │
                  │                     │                                               │
                  │                     ▼                                               │
                  │            DFA aligner (112×112)                                    │
                  │                     │                                               │
                  │                     ▼                                               │
                  │          AdaFace IR-101 FP16 → 512-d                                │
                  │                     │                                               │
                  │                     ▼                                               │
                  │        gallery matmul + K-of-N vote                                 │
                  └─────────────────────┬───────────────────────────────────────────────┘
                                        │  RecognitionEvent{employee, camera, role, ts, score}
                                        ▼
                            attendance state machine  (OUTSIDE ⇄ INSIDE)
                                        │
                                        ▼
                          SQLite/Postgres — events + daily rows
```

**The one non-obvious constraint.** The DFA aligner resizes its input to 160×160
internally and warps the 112×112 crop out of *that* tensor. Feed it a 4K frame
and a 200px face becomes 8px before the warp. So the detector localises on a
cheap downscale, and the crop handed to the aligner is cut from the
**full-resolution** frame. This is why the pipeline keeps the 4K frame around
instead of discarding it after detection.

### Why one process per camera

The old code shared a single global `_yolo_model` across camera threads;
Ultralytics keeps tracker state on the model instance, so one tracker received
interleaved frames from two different scenes. Process isolation makes that
structurally impossible rather than a convention someone can break later, and
gives each camera an independent restart boundary when a decoder wedges.

---

## 5. Module layout

```
app/
  config.py              pydantic-settings; every threshold in one place
  core/
    geometry.py          umeyama + batched warp   [done, bit-exact vs reference]
    detector.py          YOLOv8n-face / YuNet     [done]
    aligner.py           DFA mobilenet, batched   [done]
    recognizer.py        AdaFace IR-101 FP16      [done]
    quality.py           size / blur / pose / det-score gates
    tracker.py           ByteTrack, one instance per camera
    gallery.py           embedding matrix, matmul match, K-of-N vote
    stream.py            RTSP reader, backoff, staleness detection
    pipeline.py          per-camera worker loop
  services/
    enrollment.py        build gallery from face_id_users/
    attendance.py        IN/OUT state machine
  db/
    models.py            SQLAlchemy
    schema.py            create/migrate
  api/                   FastAPI routes + live view
scripts/
  enroll.py              one-shot gallery build
  record_calibration.py  raw capture for threshold measurement
  calibrate.py           FAR/FRR sweep on real captures
  run.py                 supervisor: launches both camera workers + API
```

---

## 6. Attendance logic

Camera role drives direction; a per-employee state machine decides whether an
event counts. This replaces the old elapsed-time heuristic, under which standing
near the entrance camera for six minutes checked you out.

```python
def on_event(emp_id, role, ts, camera_id, score):
    st = state[emp_id]                                   # INSIDE | OUTSIDE

    if ts - st.last_seen_on[camera_id] < COOLDOWN:       # 90 s
        return                                           # debounce

    if role == "IN" and st.status == "OUTSIDE":
        st.status = "INSIDE"
        daily.check_in_time = min(daily.check_in_time or ts, ts)
        st.entered_at = ts

    elif role == "OUT" and st.status == "INSIDE":
        st.status = "OUTSIDE"
        daily.check_out_time = ts                        # last OUT wins
        daily.working_hours += ts - st.entered_at        # accumulate

    log(AttendanceEvent(emp_id, camera_id, role, ts, score))   # always
```

Properties the old code lacked: hours **accumulate** across in/out pairs so a
lunch break subtracts; `check_out_time` is the *last* OUT so leaving and
returning does not truncate the day; loitering logs events but never moves the
summary.

Day boundary is **04:00 local** (`Asia/Tashkent`), not midnight, so a late shift
ending at 01:00 files against the day it started. Timestamps are stored
timezone-aware UTC and converted only at the presentation edge.

Open intervals at end of day are **flagged** (`status = NO_CHECKOUT`), never
auto-closed with a plausible-looking time.

---

## 7. Accuracy plan

Ordered by expected return.

1. **Best-shot per track.** Score every detection on face area × sharpness ×
   frontality; recognize the best frame, not the first. The old code committed
   an identity from whatever frame a track first appeared in.
2. **K-of-N vote.** Require 3 of the last 5 quality-passing frames to agree
   before committing an identity.
3. **Quality gates** before embedding: min 60 px face, Laplacian variance,
   detector score ≥0.35, yaw/pitch from the aligner's 5 landmarks.
4. **Margin rule**: accept only when best score clears threshold *and* beats
   second-best by ≥0.05.
5. **Multi-embedding gallery**: keep all 3–5 enrolment embeddings per person and
   match on max similarity, not a single averaged centroid.
6. **Recalibrate on real captures** (§2.3 caveat) once the walkthrough is in.

Deliberately **not** in v3: liveness/anti-spoofing. It matters for attendance
fraud, but these cameras are 3 m up behind a wide-angle lens — the realistic
spoof (holding a phone to the lens) is not practical at that geometry. Revisit
if a door-level camera is ever added.

---

## 8. Phases

| # | phase | status |
|---|---|---|
| 1 | Core CV modules | **done** — geometry bit-exact vs reference, all validated |
| 2 | Camera config | **done** — H.264/12fps/GOP12/WDR + shutter 1/250 |
| 3 | Enrolment | **done** — 54 people, 268/268 images, 0 failures |
| 4 | Tracking + quality + best-shot | **done** — per-camera ByteTrack, gates calibrated |
| 5 | Stream + pipeline worker | **done** — backoff, staleness, 20-43 fps sustained |
| 6 | Attendance state machine + DB | **done** — IN/OUT verified, NO_CHECKIN/NO_CHECKOUT flags |
| 7 | API + live view | **done** — dashboard, MJPEG, JSON, CSV |
| 8 | Calibration | **done** — threshold 0.26 from two real walkthroughs |
| 9 | Ops | **done** — maintenance job, systemd units, docs |
| 10 | Domain adaptation | **tool ready** — `enroll_from_live.py`, awaiting review pass |
| 11 | Authentication | **not started** — see Known limits |

---

## 9. Configuration baseline

Provisional values; §8 phase 8 replaces the threshold with a measured one.

```ini
# stream
RTSP_URL_IN            = rtsp://admin:***@192.168.1.2:554/Streaming/Channels/101
RTSP_URL_OUT           = rtsp://admin:***@192.168.1.64:554/Streaming/Channels/101
RTSP_TRANSPORT         = tcp
STALE_AFTER_S          = 5
RECONNECT_BACKOFF_S    = 1,2,4,8,15,30

# detection
DETECT_WIDTH           = 1280        # downscale for YOLO; crops come from 4K
DETECT_CONF            = 0.35
PROCESS_EVERY_NTH      = 2           # ~6 recognition passes/s per camera

# quality gates
MIN_FACE_PX            = 60
MIN_LAPLACIAN_VAR      = 80
MIN_ALIGNER_SCORE      = 0.50
MAX_YAW_DEG            = 35

# recognition
RECOGNITION_THRESHOLD  = 0.40        # PROVISIONAL — from gallery, not corridor
SECOND_BEST_MARGIN     = 0.05
VOTE_WINDOW            = 5
VOTE_REQUIRED          = 3

# attendance
EVENT_COOLDOWN_S       = 90
DAY_BOUNDARY           = 04:00
TIMEZONE               = Asia/Tashkent
OPEN_INTERVAL_POLICY   = flag
```

---

## 10. Known risks

| risk | mitigation |
|---|---|
| Corridor faces smaller than the gallery's 225 px median | 4K retained; measure in walkthrough; raise `DETECT_WIDTH` if needed |
| Threshold from enrolment photos does not transfer | §7.6 recalibration — the walkthrough is the gate on going live |
| High mounting angle → large pitch, the known weak axis for ArcFace | quality gates reject extreme pose; best-shot picks the most frontal frame |
| 13 GB RAM, 2 GB free with Chrome open | 4K frames are 24 MB; bounded queues, drop-oldest; close browsers when running |
| Disk 97% full (28 GB) | snapshot retention job from day one; weights symlinked, not copied |
