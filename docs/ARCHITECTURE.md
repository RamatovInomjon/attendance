# Architecture

## The pipeline

```
RTSP 4K H.264 @12fps
   │
   ├─ RtspSource            background thread, drop-oldest queue(2), backoff reconnect
   │
   ▼  Frame (full-res BGR + capture timestamp)
CameraPipeline.process()          one instance per camera, nothing shared
   │
   ├─ downscale 3840→1280 (INTER_AREA, 2.4 ms)
   ├─ YOLOv8n-face @1280           → boxes, rescaled back to FULL-RES coords
   ├─ ByteTrack (this camera's own instance) → stable track ids
   │
   ├─ for tracks without an identity yet:
   │     ├─ crop from the FULL-RES frame, margin ×1.4, resized to 224²
   │     ├─ DFA aligner (batched)  → 5 landmarks + canonical 112×112 crop
   │     ├─ quality gate           → size / sharpness / yaw / pitch / aligner score
   │     ├─ AdaFace IR-101 FP16    → 512-d, L2-normalized
   │     ├─ gallery matmul         → best person, runner-up, margin
   │     └─ TrackVote              → 3-of-5 agreement before committing
   │
   ▼  RecognitionOutcome
CameraWorker._persist()
   │
   ├─ AttendanceService.record()   debounce → role + presence → transition
   └─ RecognitionEvent + DailyAttendance
```

## The two constraints that shaped it

**1. The aligner must be fed a crop, never a frame.**
`dfa_mobilenet_aligner.onnx` resizes its input to 160×160 *inside the graph*.
Hand it a 3840×2160 frame and a 200 px face becomes ~8 px before anything else
happens. So detection localises on a cheap downscale, and the crop handed to the
aligner comes out of the full-resolution frame.

**Why the margin is 1.30.** The margin decides how many pixels of face survive
into the 112×112 output. Inside the aligner the crop is 160 px wide, so the face
spans `160/margin` px, and the warp resamples that to 112:

| margin | face px inside aligner | vs 112 output |
|---|---|---|
| 1.00 | 160 | downsample, full detail |
| **1.30** | **123** | **slight downsample — deployed** |
| 1.43 | 112 | 1:1, break-even |
| 1.80 | 89 | upsampling — inventing detail |
| 2.30 | 70 | heavy upsampling |

Measured on the gallery (`bench/sweep_crop_pipeline.py`), d′ is flat to 1.35 and
falls away from exactly 1.43, as that arithmetic predicts:

```
1.00  12.44   1.20  12.46   1.30  12.47   1.43  12.29   1.60  12.13   2.00  11.69
1.10  12.45   1.25  12.49   1.35  12.50   1.50  12.33   1.80  11.96
```

Three independent lines of evidence agree on ~1.3: this sweep, the break-even
arithmetic above, and the CVLFace authors' own reference `input.png`, whose face
fills 57% of the frame — an implied margin of **1.30**. General literature puts
the useful range at 1.1–1.5, which brackets it.

**Why `crop_size` is 160, not 224.** It barely matters — the graph resizes to
160 whatever you send. Measured d′ across 144→320 is 12.37–12.53, i.e. flat. So
160 is chosen because it is the size the network actually consumes: no resample
for nothing, and the smallest tensor.

| crop_size | 144 | 160 | 192 | 224 | 256 | 320 |
|---|---|---|---|---|---|---|
| d′ | 12.37 | **12.53** | 12.45 | 12.47 | 12.53 | 12.52 |

**The two-stage sharp warp.** The reference implementation warps the 112 crop
out of the aligner's internal 160 tensor. That resamples twice: a 292 px crop
goes *down* to 160 then to 112, and a small 126 px crop goes *up* to 160 and
back down. Instead the aligner is used for landmarks only (batched at 160, all
the 1050-anchor aggregator can consume) and the canonical crop is warped from
the **native-resolution** crop — one downsample either way.

| | d′ | weakest genuine pair | crop sharpness |
|---|---|---|---|
| reference warp | 12.53 | 0.547 | 581 |
| **sharp warp** | **12.64** | **0.571** | **946** |

Full-gallery re-validation after the switch: **d′ 11.33 → 12.63**, weakest
genuine pair **0.413 → 0.571**, rank-1 still 100%. This departs from the
reference PyTorch numerics the ONNX export was verified against, which is only
sound because enrolment and inference share the code path — so the gallery and
the query are always produced identically. `align_mode = "reference"` restores
bit-exact behaviour.

**2. Nothing may be shared between cameras.**
Each `CameraPipeline` builds its own detector, aligner, recognizer and
`FaceTracker`. The previous implementation called `model.track(persist=True)` on
one module-level YOLO object from every camera thread; Ultralytics keeps tracker
state on that object, so a single tracker received interleaved frames from two
different scenes and track ids migrated between cameras.

## Identity is a track-level decision

A single frame never names anyone. Each track accumulates votes and commits only
on **3-of-5 agreement** (`TrackVote`), and a match must clear both the
**threshold** and a **margin** over the runner-up person.

`Gallery.match()` is one matrix product against all 268 embeddings, then a
max-reduce *per person* — so a person's five enrolment images compete as one
identity, and the best-matching pose wins. Embeddings are stored per image
rather than averaged into a centroid: max-similarity handles pose spread better
than a mean vector, and a bad enrolment photo stays visible instead of quietly
dragging the centroid.

## Attendance

Direction comes from the camera's **role**; whether an event counts comes from
the employee's **presence state**. Both are required.

```
                 IN event                    OUT event
   OUTSIDE ─────────────────────► INSIDE ─────────────────────► OUTSIDE
      │  (already OUTSIDE:            │  (already INSIDE:           │
      │   re-sighting, logged,        │   re-sighting, logged,      │
      └───  summary unchanged)        └───  summary unchanged) ─────┘
```

- **Debounce** — same person, same camera, within 90 s: dropped entirely.
- **Accumulate** — `worked_seconds` sums each INSIDE interval, so a lunch break
  subtracts instead of vanishing.
- **Last OUT wins** — leaving and returning does not truncate the day.
- **Open intervals are flagged**, never closed with an invented time.

Verified end to end (`bench/test_attendance.py`): a 09:02→18:10 day with a lunch
break and a short step-out yields 9.13 h gross − 1.08 h outside = **8.05 h**.

## Time

Every timestamp column uses a `UtcDateTime` decorator that stores UTC and always
returns an **aware** UTC datetime. SQLite has no timezone support and silently
drops tzinfo; during testing that produced a 09:02 check-in rendering as 04:02,
and a naive/aware comparison `TypeError`. Normalising in the type fixes both at
the source and keeps the code portable to PostgreSQL, where the column really is
`timestamptz`.

Business dates roll at **04:00 local** (`Asia/Tashkent`), not midnight, so a
late shift ending at 01:00 files against the day it started.

## Failure behaviour

| condition | behaviour |
|---|---|
| RTSP read fails once | ignored; transient loss is normal |
| 30 consecutive failures | release, reconnect with the same construction path |
| repeated open failure | backoff 1→2→4→8→15→30 s, then hold |
| no frame for 5 s | `is_stale`, surfaced in `/api/health` and the dashboard |
| camera offline | **stops producing frames** — never replays the last one |

That last row is the important one: the previous reader substituted its last
good frame on failure and pushed it into the processing queue, so a camera that
had been down for hours kept generating recognitions from a frozen face.
