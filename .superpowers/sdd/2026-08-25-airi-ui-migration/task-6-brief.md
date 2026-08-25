# Task 6: Two-Camera Live Recognition Console

Work in `/home/inomjon/projectAI/face_rec/face_recognition_airi`. Preserve accepted Tasks 1-5 and all recognition transport/core behavior.

## Files

Modify `app/api/pages.py`, `templates/recognition/live.html`, `static/js/camera_stream.js`, `tests/test_ui_pages.py`, and necessary shared CSS. Modify `/recognition/logs` response only backward-compatibly if required.

## Data/Interface Requirements

Normalize every enabled camera into presentation context with: DB id/name/role, `state`, resolution, camera/stream FPS, algorithm/processed FPS, latency, dropped frames, last-frame time, and pipeline error status from existing worker `stats()`. Missing worker/stat fields produce explicit `offline`/unavailable values, not fabricated zero-as-healthy. Do not start/stop/reconfigure workers.

Stable markup selectors: `data-camera-id`, `data-camera-role`, `data-stream-canvas`, `data-stream-state`. Keep current stream endpoints, decoder, polling cadence, and `/recognition/logs` top-level keys `employees`, `events`, `unknown_attempts`, `stats`.

## UI Requirements

Render Uzbek AIRI-style live console with entrance `Kirish kamerasi` and exit `Chiqish kamerasi` cards side by side on wide screens, stacked on narrow screens. Each shows connection state, resolution, `Kamera FPS`, `Algoritm FPS`, latency, drops, and last frame. Include attendance summary, recent recognized events with quality-best snapshot, confidence, camera, action/time, unknown activity, and visible pipeline errors.

Dashboard/live must use event/attendance quality-best snapshots returned by existing APIs; no `why_*` diagnostics. Evidence uses shared modal safely. Cap dynamically rendered recent feed at newest 30 nodes. Use delegated handlers and textContent/DOM construction for external WebSocket/poll fields—resolve the deferred Task 2 innerHTML injection risk in touched live-feed code.

Change only selectors/render targets needed by new markup in `camera_stream.js`; preserve transport/decoder/fullscreen behavior. No unbounded timers/listeners.

Tests must assert both camera role labels, stable data selectors, metrics labels, offline rendering with missing workers where practical, quality evidence/no `why_*`, top-level polling JSON contract, and safe bounded feed implementation. Preserve full suite.

Run `/home/inomjon/anaconda3/envs/yolo/bin/python -m pytest tests/test_ui_pages.py -v`, `node --check static/js/camera_stream.js`, and report exact output.

## Constraints

- No model, threshold, tracking, direction, recorder, attendance transition, camera config, or stream transport changes.
- Do not dispatch subagents.
- Self-review and write `.superpowers/sdd/2026-08-25-airi-ui-migration/task-6-report.md`.

