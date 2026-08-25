# Benchmarks

All measured on the deployment host: Ryzen 7 5800H (16T), RTX 3070 Laptop 8 GB,
CUDA 12.4, onnxruntime-gpu 1.23.2. Reproduce with the scripts named per section.

## Detector — `bench/bench_models.py`, `bench/bench_detect_acc.py`

| resolution | YOLOv8n-face (CUDA) | YuNet (OpenCV DNN, CPU) |
|---|---|---|
| 640×360 | **6.2 ms** | 8.8 ms |
| 960×544 | **6.7 ms** | 23.5 ms |
| 1280×736 | **7.5 ms** | 46.8 ms |
| 1920×1088 | ~8 ms | 139.0 ms |

Accuracy over the 270-image gallery:

| detector | found | face width p5 / median / p95 |
|---|---|---|
| YuNet | 270/270 | 181 / 224 / 304 px |
| YOLOv8n-face | 270/270 | 178 / 227 / 315 px |

Both perfect, so speed decided. YuNet's design point is CPU inference at low
resolution; this host has a GPU and a wide corridor needing ≥1280 px detection.
**YOLOv8n-face**, with YuNet kept selectable for CPU-only hosts.

## Recognizer — `bench/bench_models.py`, `bench/validate_gallery.py`

| model | bs1 | bs4 | bs8 | bs16 |
|---|---|---|---|---|
| ir101 fp32 | 6.20 ms | 3.54 ms/f | 3.17 ms/f | 2.65 ms/f |
| **ir101 fp16** | **4.57 ms** | **2.29 ms/f** | **1.90 ms/f** | **1.54 ms/f** |
| ir18 fp32 | 1.43 ms | 0.92 ms/f | 0.81 ms/f | 0.73 ms/f |

Separation on 54 people:

| model | d′ | rank-1 | thr @ FAR=0 | FRR there |
|---|---|---|---|---|
| **ir101 fp16** | **11.33** | **100.00%** | 0.400 | **0.0000** |
| ir18 | 7.60 | 99.63% | 0.405 | 0.0167 |

IR-101 costs 3× IR-18 and is clearly better separated; at 4.6 ms/face it is
affordable. **IR-101 FP16.**

## Gallery separation (IR-101 FP16, 54 people / 268 images)

```
                   n     mean     std      min       p1      p50      max
genuine          540   0.8979   0.0847   0.4128   0.5804   0.9239   0.9947
impostor       35775   0.0417   0.0651  -0.2048  -0.0958   0.0372   0.3969

d-prime = 11.33        1:N leave-one-out rank-1 = 270/270 (100.00%)

 thresh        FAR       FRR
   0.30    0.00095    0.0000
   0.35    0.00020    0.0000
   0.40    0.00000    0.0000     ← deployed (provisional)
   0.50    0.00000    0.0019
```

**This is enrolment-photo vs enrolment-photo.** Corridor crops are a different
domain. See `docs/OPERATIONS.md` → Calibration.

Data-quality findings from the same run:
- `029_Lyudmila/image_04,05` scored 0.413 against their siblings — a different
  subject. Moved to `face_id_users/_quarantine/`; metadata updated. Re-run after
  removal: 268/268 images embedded, **0 outliers**.
- Closest impostor pair: `007_Azizbek` / `036_Anvar` at 0.397 — the pair to
  watch as the gallery grows.

## Aligner crop margin — `bench/sweep_margin.py`

| margin | pad/side | d′ | rank-1 | min genuine |
|---|---|---|---|---|
| 1.00 | 0% | 12.44 | 100.00% | 0.546 |
| 1.15 | 7% | 12.44 | 100.00% | 0.536 |
| **1.30** | 15% | **12.47** | 100.00% | 0.544 |
| 1.50 | 25% | 12.33 | 100.00% | 0.542 |
| 1.80 | 40% | 11.96 | 100.00% | 0.549 |
| 2.30 | 65% | 8.37 | 99.63% | 0.065 |

Flat 1.0–1.5, collapsing at 2.3 — too much context shrinks the face inside the
aligner's internal 160² resize. Deployed at **1.4**.

## Alignment stage — `bench/bench_models.py`

| batch | total | per face |
|---|---|---|
| 1 | 1.42 ms | 1.42 ms |
| 8 | 3.03 ms | 0.38 ms |
| 16 | 4.96 ms | 0.31 ms |

Negligible, hence batching every face in a frame into one session run.

## Numerical fidelity

The batched warp in `app/core/geometry.py` was checked against the verified
per-image implementation in `cvlface_test/inference.py`:

```
theta  max|diff| = 0.0
warp   max|diff| = 0.0     allclose = True
```

Bit-exact, so the upstream verification (embeddings within 4.6e-5 of the
reference PyTorch models) carries over unchanged.

## Downscale cost (4K → 1280)

| interpolation | time |
|---|---|
| INTER_AREA | 2.42 ms |
| INTER_LINEAR | 0.57 ms |
| INTER_NEAREST | 0.21 ms |

INTER_AREA deployed — best quality for downscaling, and 2.4 ms is immaterial
against a 167 ms budget.

## End-to-end, live — `scripts/live_test.py`

**Superseded (12 fps cameras, every 2nd frame).** Kept for history:

| camera | detect p50 | total p50 | sustained | needed |
|---|---|---|---|---|
| Entrance | 19.4 ms | 19.6 ms | 20.9 fps | 6 fps |
| Exit | 19.6 ms | 19.7 ms | 43.4 fps | 6 fps |

## Full-rate pipeline (2026-08-25)

Cameras now stream 20 fps and `process_every_nth` is 1, so the budget is 50 ms
per frame per camera rather than 100 ms.

### Where the time went

The old `detect` figure hid a 4K→960 resize *and* a preprocessing path that
dominated everything else. Measured on the deployment host:

| stage | old | new |
|---|---|---|
| 4K→960 resize | 0.9 ms | 0.9 ms |
| letterbox | 3.6 ms | in-place, ~0.2 ms |
| **blob conversion** | **26–49 ms** | **6.3 ms** |
| GPU inference | 12.5 ms | 7–14 ms |
| postprocess/NMS | 0.3 ms | 0.3 ms |

`cv2.dnn.blobFromImage` allocates a fresh NCHW float32 buffer per call — 11 MB
at 960×960. This host runs 13 GB RAM with ~1 GB free and 4 GB swapped, where a
fresh 11 MB allocation costs 6 ms of page faults before converting a pixel
(`memcpy` of the same 2.7 MB takes 0.07 ms — bandwidth is fine, allocation is
not). `HeadDetector` now converts into preallocated buffers.

The GPU was never the bottleneck: 12.5 ms of a ~50 ms frame.

### Clean-machine, end-to-end (`CameraPipeline.process`, real clip)

| configuration | median | p90 | duty @20 fps |
|---|---|---|---|
| old code + 960×960 model | 23.03 ms | 30.45 ms | 46% |
| new code + 960×544 model | 10.13 ms | 18.49 ms | 20% |

### Live, both cameras

| | processed / streamed | effective |
|---|---|---|
| before (`every_nth=2`) | 50% by design | 10 fps |
| after (`every_nth=1`) | 94.9% / 95.5% | **19.0 / 19.1 fps** |

The residual ~5% is drop-oldest queue spill during bursts; `frame_queue_size`
went 2→4 to absorb them.

### What full rate buys

Same 20 clips, direction and recognition outcomes:

| | tracks | UNKNOWN direction | recognized | median traj points |
|---|---|---|---|---|
| `every_nth=2` | 10 | 1 (10.0%) | 3 | 29 |
| `every_nth=1` | 14 | **0 (0.0%)** | 4 | **80** |

Direction needs `min_points=5` plus real travel; at 10 fps a brisk walker often
did not supply it, which is what the 55% live UNKNOWN rate was starving on.

## Head detector — retrained model (2026-08-25)

`yolov8n_full_640/weights/best.pt`, 100 epochs on full CrowdHuman
(mAP50 0.812, mAP50-95 0.509), exported at 960×544 rather than 960×960: a 16:9
frame letterboxed into a square spends 44% of every buffer on grey padding that
the CPU converts and the GPU convolves.

Against the previous 960×960 export over 168 real corridor frames:

| | heads found | mean conf | notes |
|---|---|---|---|
| old | 111 | 0.713 | — |
| **new** | **126** | **0.730** | superset: found all 111, plus 15 |

Median IoU on agreed boxes 0.915. The 15 extras are distant heads 9–14 px tall
at detect scale (agreed median 16 px) — verified by eye as real people, not
false positives. They are below `min_face_px` for recognition but start tracks
earlier, which is exactly what trajectory density needs.

## Attendance logic — `bench/test_attendance.py`

Simulated day: arrive 09:02, lunch 13:05→13:50, step out 15:00→15:20, leave 18:10.

```
final: in=09:02  out=18:10  worked=8.05h  presence=OUTSIDE
gross span 9.13h  minus 1.08h outside = 8.05h worked
events logged: 8 (2 debounced calls wrote nothing)
```

---

# Live calibration — 2026-08-19

Two real walkthroughs on the deployed corridor. `bench/calibrate_threshold.py`
replays both and simulates the 3-of-5 track vote.

## Camera fix found by the first walkthrough

The **shutter was set to 1/25 s**. A walking person moves several centimetres
during that exposure, which is why live sharpness was 36 against the gallery's
291 and why blur caused 103 of 116 gate rejections. Set to **1/250**:

| | before (1/25) | after (1/250) |
|---|---|---|
| gate-pass rate | 12/116 (10%) | 85/200 (42.5%) |
| best score seen | 0.226 | **0.514** |
| scene brightness | 114.6 | 112.3 (negligible loss) |

## Threshold calibration

| run | subject | gate-passing obs | best-score max |
|---|---|---|---|
| walk 1 | **not enrolled** | 43 | **0.184** |
| walk 2 | enrolled (Inomjon Ramatov, Nishonov Kamoliddin) | 87 | **0.514** |

Track-vote simulation (3-of-5 agreement on the same identity):

| threshold | impostor obs ≥ thr | genuine obs ≥ thr | false commits | true commits |
|---|---|---|---|---|
| 0.22 | 0/43 | 39/87 | 0/4 | 3/8 |
| **0.26** | **0/43** | ~32/87 | **0/4** | **3/8** |
| 0.30 | 0/43 | 24/87 | 0/4 | 2/8 |
| 0.35 | 0/43 | 21/87 | 0/4 | 1/8 |
| 0.45 | 0/43 | 4/87 | 0/4 | 0/8 |

**Deployed: 0.26** — 41% above the highest impostor score ever observed live,
while still committing the genuine tracks. Above 0.30 true positives start
dropping without buying any impostor rejection, because the impostor
distribution has already ended by 0.184.

The impostor run is also qualitatively reassuring: its best matches were
scattered randomly across identities (18 × Shukrullo, 15 × Nozimjon, 13 ×
Shoxrux …) with no winner — noise, not a near-miss. A false commit would require
three frames above threshold *agreeing on one person*.

## Quality-gate recalibration — `bench/degrade_test.py`

Known gallery faces degraded to corridor conditions and re-matched:

| condition | face px | sharpness | score |
|---|---|---|---|
| enrolment original | 205 | 291 | 1.000 |
| resized to 60 px | 60 | 170 | **0.995** |
| motion blur k=5 | 92 | 78 | **0.956** |
| motion blur k=9 | 92 | 47 | **0.708** |
| 35° pitch | 92 | 306 | **0.927** |

The recognizer is far more robust than the original gates assumed, which were
discarding 90% of observations. Recalibrated:

| gate | was | now |
|---|---|---|
| `min_face_px` | 60 | 50 |
| `min_laplacian_var` | 80 | **25** |
| `min_aligner_score` | 0.50 | 0.40 |
| `max_yaw_deg` | 35 | 45 |
| `max_pitch_deg` | 30 | 40 |

Replayed against walk 1: **12 → 43 usable observations (3.6×)**.

## First live events

```
    time camera    role  score  margin   px transition   name
13:56:40 Entrance  IN    0.469   0.298   85 CHECK_IN     Bo'ronov Nazim
13:56:41 Entrance  IN    0.457   0.278   92 CHECK_IN     Cho'tboyev Muhammad
13:56:45 Entrance  IN    0.471   0.354  210 CHECK_IN     Naziraxon Ibroximova
14:05:44 Exit      OUT   0.340   0.141  108 RE_SIGHTING  Inomjon Ramatov
14:06:29 Entrance  IN    0.435   0.065  149 CHECK_IN     Anvarxodja Kadirov
```

The 0.065 margin on the last row prompted raising `second_best_margin` to 0.08.

## Remaining gap

Live scores peak at ~0.51 where gallery-to-gallery scores sit at ~0.90. That gap
is domain, not model: clean 1280×720 enrolment photos vs small, angled,
still-somewhat-blurred crops from 3 m up. `scripts/enroll_from_live.py` closes
it by adding confirmed corridor crops to the gallery — the highest-value
remaining accuracy work.
