# Audit of the previous implementation (`fast_api/`)

Reviewed 2026-08-19, 3,360 LOC. Full write-up with diagrams:
<https://claude.ai/code/artifact/82d968c0-4985-4dc3-8c6f-7ce560ca7fcd>

The package itself was deleted on 2026-08-25. This audit is kept because
several findings explain *why* `app/` is built the way it is - the design
choices here are reactions to the problems listed below.

## Blocking — could not start

| # | finding |
|---|---|
| B1 | `init_db()` defined, never called; lifespan goes straight to `SELECT … FROM camera_cameraconfig`. No `db.sqlite3` present — only an orphaned `-wal`/`-shm` pair. |
| B2 | `requirements.txt` contains zero of `fastapi`, `uvicorn`, `sqlalchemy`, `aiosqlite`. It is the Django-era lockfile. |
| B3 | InsightFace provider list falls through to CPU silently while logging `⚡ loaded on CUDA` unconditionally. |

## Critical — wrong or exposed data

| # | finding | how `app/` avoids it |
|---|---|---|
| C1 | One global `_yolo_model`, so one Ultralytics tracker received interleaved frames from both cameras; track ids migrated across scenes | one detector + tracker instance per `CameraPipeline` |
| C2 | `camera_id` passed into `_process_attendance_async` and never read — IN/OUT decided by elapsed time alone | `Camera.role` carried into the state machine |
| C3 | `check_out_time` and `working_hours` never written by anything, yet read by dashboard, CSV and logs | state machine writes both; `bench/test_attendance.py` asserts it |
| C4 | `StaticPool` = one SQLite connection shared across a thread pool that opened a fresh event loop per write | sync sessions, real pool, WAL + `busy_timeout` |
| C5 | `datetime.utcnow()` writes vs `date.today()` reads — every time 5 h early in Tashkent, pre-05:00 events filed a day late | `UtcDateTime` type + 04:00 local business date |
| C6 | No auth anywhere; `/media` a public static mount of face snapshots; `GET /api/cameras/` returned RTSP credentials; CORS `*` with credentials | still open — see OPERATIONS "Known limits" |
| C7 | Identity committed from a single frame, then frozen for an hour | quality gate → best-shot → 3-of-5 vote |

## High

- `align_and_recognize` cropped a YOLO box then ran InsightFace's **full stack,
  including SCRFD detection, again** on that crop — two detectors doing one
  detector's job, and the padded crop was not the 112² warp ArcFace expects.
- `CUDA_LAUNCH_BLOCKING=1` set at import, plus a per-frame `torch.cuda.synchronize()`.
- Dead camera replayed `last_good_frame` forever, generating attendance events
  from an offline camera.
- MJPEG and WebSocket consumers both `display_queue.get()` — two viewers stole
  each other's frames.
- GStreamer path hardcoded H.265 depay/decode; H.264 cameras silently fell
  through to the FFMPEG backend.
- `register_employee` was `async def` containing `time.sleep()` — blocked the
  whole event loop for up to 30 s.
- Matching was a Python loop that recomputed `emb / norm(emb)` per candidate.
- Every unknown detection wrote a DB row **and** a JPEG.
- `active_tracks` never pruned.
- `f"{(record.working_hours or 0) / 3600:.1f}h"` — latent `TypeError` the moment
  the column held a real `timedelta`.

## Medium

`constants.py`, `camera_manager.py` and `network_diagnostics.py` (377 lines)
were never imported by anything. `crud.py` helpers duplicated inline in
`services.py`. Overlay label was `f"Employee #{id}"`. `/api/v1/register`
returned a hardcoded `{"id": 0}`. Registration averaged ≤24 embeddings behind a
single blur check. Login page rendered with no POST handler. No migrations, no
tests, no Dockerfile, no README.

## Status

`app/` superseded it, and `fast_api/` was **deleted on 2026-08-25**. The last
live dependency - the employee-enrolment router - was replaced by a native v3
enrolment path; all 15 remaining modules were unreachable from both entrypoints.
`tests/test_ui_pages.py` keeps a guard asserting the legacy router is never
remounted, so the boundary cannot quietly come back.
