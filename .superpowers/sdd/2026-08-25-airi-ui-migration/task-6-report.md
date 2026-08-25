# Task 6 Report: Two-Camera Live Recognition Console

## Status

Completed.

## Changed files

- `app/api/pages.py`
  - Added presentation-only normalization for every enabled camera: DB identity, localized role, stream state, resolution, camera FPS, explicit processed FPS, latency, dropped frames, last frame, and pipeline status/error details.
  - Missing workers render as `offline`; missing worker fields remain unavailable rather than becoming healthy zeroes.
  - Kept worker lifecycle untouched. No worker is started, stopped, or reconfigured by the page.
  - Added newest quality-best unknown sightings and expanded quality-best recent events to 30 entries.
  - Preserved `/recognition/logs` top-level keys: `employees`, `events`, `unknown_attempts`, and `stats`; added only backward-compatible values/fields beneath those keys.
- `templates/recognition/live.html`
  - Replaced the legacy primary/secondary layout with Uzbek entrance/exit camera cards that are side by side on wide screens and stacked on narrow screens.
  - Added stable `data-camera-id`, `data-camera-role`, `data-stream-canvas`, and `data-stream-state` selectors.
  - Added attendance summary, quality-best recent recognitions, confidence/camera/action/time, unknown activity, and visible pipeline errors.
  - Uses the shared delegated evidence modal through data attributes; removed the duplicate modal, inline handlers, and unsafe inline feed renderer.
- `static/js/camera_stream.js`
  - Retained `/ws/camera/{id}/`, JSON/base64 JPEG canvas decoding, reconnect behavior, fullscreen support, and 10-second `/recognition/logs` polling.
  - Updated stream selectors for the new camera cards.
  - Builds external polling/WebSocket values with DOM methods and `textContent`; it does not use `innerHTML`.
  - Caps recognized and unknown feeds at the newest 30 nodes, uses one delegated action handler, and cleans up polling, sockets, and reconnect timers on unload.
- `static/css/main.css`
  - Added scoped AIRI live-console grid, camera, metric, evidence, pipeline, responsive, and empty-state styles using existing theme tokens.
- `tests/test_ui_pages.py`
  - Added isolated online/offline role-camera coverage, runtime metric and pipeline-error assertions, quality-best event/unknown evidence checks, the polling JSON contract, and safe/bounded transport source checks.
- `.self-eval-scores.jsonl`
  - Appended the required structured self-evaluation entry.
- `.superpowers/sdd/2026-08-25-airi-ui-migration/task-6-report.md`
  - This implementation and verification report.

## Test-first record

The three Task 6 tests were written before production changes. Their first focused run produced three expected failures:

- role camera labels/selectors and offline normalization were absent;
- quality-best event evidence and unknown polling activity were absent;
- the live implementation lacked a 30-node cap and still used `innerHTML`/inline handlers.

After implementation and the self-review correction that removed inferred dropped-frame/processed-FPS values, the focused run passed:

```text
============================= test session starts ==============================
platform linux -- Python 3.10.16, pytest-9.1.1, pluggy-1.6.0 -- /home/inomjon/anaconda3/envs/yolo/bin/python
cachedir: .pytest_cache
rootdir: /home/inomjon/projectAI/face_rec/face_recognition_airi
plugins: hydra-core-1.3.2, anyio-4.12.0
collecting ... collected 26 items / 23 deselected / 3 selected

tests/test_ui_pages.py::test_live_console_normalizes_role_cameras_and_missing_workers_as_offline PASSED [ 33%]
tests/test_ui_pages.py::test_live_console_and_logs_use_quality_best_evidence_with_stable_contract PASSED [ 66%]
tests/test_ui_pages.py::test_live_feed_script_is_safe_bounded_and_preserves_stream_contract PASSED [100%]

======================= 3 passed, 23 deselected in 9.68s =======================
```

## Required verification

Command:

```text
/home/inomjon/anaconda3/envs/yolo/bin/python -m pytest tests/test_ui_pages.py -v
```

Exact output:

```text
============================= test session starts ==============================
platform linux -- Python 3.10.16, pytest-9.1.1, pluggy-1.6.0 -- /home/inomjon/anaconda3/envs/yolo/bin/python
cachedir: .pytest_cache
rootdir: /home/inomjon/projectAI/face_rec/face_recognition_airi
plugins: hydra-core-1.3.2, anyio-4.12.0
collecting ... collected 26 items

tests/test_ui_pages.py::test_page_routes_return_html[/] PASSED           [  3%]
tests/test_ui_pages.py::test_page_routes_return_html[/employees] PASSED  [  7%]
tests/test_ui_pages.py::test_page_routes_return_html[/attendance] PASSED [ 11%]
tests/test_ui_pages.py::test_page_routes_return_html[/recognition] PASSED [ 15%]
tests/test_ui_pages.py::test_page_routes_return_html[/attendance/unknown] PASSED [ 19%]
tests/test_ui_pages.py::test_page_routes_return_html[/cameras] PASSED    [ 23%]
tests/test_ui_pages.py::test_dashboard_exposes_current_view_to_the_base_template PASSED [ 26%]
tests/test_ui_pages.py::test_employee_registration_uses_the_navigation_view_identifier PASSED [ 30%]
tests/test_ui_pages.py::test_shared_shell_exposes_localized_primary_navigation_once PASSED [ 34%]
tests/test_ui_pages.py::test_shared_shell_provides_theme_control_and_evidence_modal PASSED [ 38%]
tests/test_ui_pages.py::test_dashboard_renders_operational_sections_for_the_current_database_state PASSED [ 42%]
tests/test_ui_pages.py::test_dashboard_embeds_chart_data_and_never_exposes_debug_evidence_paths PASSED [ 46%]
tests/test_ui_pages.py::test_dashboard_excludes_direction_undetermined_rows_from_metrics_and_charts PASSED [ 50%]
tests/test_ui_pages.py::test_dashboard_table_is_limited_to_the_most_recent_recorded_rows PASSED [ 53%]
tests/test_ui_pages.py::test_employee_roster_filters_round_trip_and_uses_grouped_enrollment_counts PASSED [ 57%]
tests/test_ui_pages.py::test_employee_detail_uses_actual_enrollment_count_and_evidence_empty_states PASSED [ 61%]
tests/test_ui_pages.py::test_attendance_filters_round_trip_in_sql_once_and_render_evidence PASSED [ 65%]
tests/test_ui_pages.py::test_attendance_normalizes_an_inverted_date_range PASSED [ 69%]
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params0] PASSED [ 73%]
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params1] PASSED [ 76%]
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params2] PASSED [ 80%]
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params3] PASSED [ 84%]
tests/test_ui_pages.py::test_attendance_export_applies_the_visible_list_filters PASSED [ 88%]
tests/test_ui_pages.py::test_live_console_normalizes_role_cameras_and_missing_workers_as_offline PASSED [ 92%]
tests/test_ui_pages.py::test_live_console_and_logs_use_quality_best_evidence_with_stable_contract PASSED [ 96%]
tests/test_ui_pages.py::test_live_feed_script_is_safe_bounded_and_preserves_stream_contract PASSED [100%]

============================== 26 passed in 9.85s ==============================
```

Command:

```text
node --check static/js/camera_stream.js
```

Exact output: no stdout/stderr; exit code `0`.

Additional checks:

```text
/home/inomjon/anaconda3/envs/yolo/bin/python -m py_compile app/api/pages.py tests/test_ui_pages.py
```

No stdout/stderr; exit code `0`.

```text
/home/inomjon/anaconda3/envs/yolo/bin/python -m pytest tests -q
..........................                                               [100%]
26 passed in 8.90s
```

## Self-review

- Confirmed camera normalization is presentation-only and reads `runtime.workers`; no lifecycle, camera configuration, model, threshold, tracking, direction, recorder, or attendance-transition code changed.
- Confirmed missing workers and missing metrics use explicit offline/unavailable values. A first implementation inferred dropped frames from frames read minus frames processed; self-review removed that inference because queued/in-flight frames make it unreliable.
- Confirmed event and unknown snapshots use existing quality-best database/API paths and no `why_*` diagnostics appear in live HTML or polling JSON.
- Confirmed `/recognition/logs` retains exactly four top-level keys and existing employee/event fields while adding unknown activity and summary details backward-compatibly.
- Confirmed camera transport still uses `/ws/camera/{id}/`, parses the existing JSON frame envelope, decodes `data:image/jpeg;base64,...`, draws to canvas, reconnects after 3 seconds, and retains fullscreen behavior.
- Confirmed the polling cadence remains 10 seconds and all recent arrays/DOM render paths are capped at 30.
- Confirmed external text uses DOM construction and `textContent`, evidence URLs are restricted to same-origin HTTP(S)/root-relative paths, and shared delegated evidence controls work for dynamically created nodes.
- Confirmed listeners/timers are installed once per page load and polling, recognition socket, camera sockets, reconnect timeouts, and streams are cleaned up on unload.

## Self-evaluation

**Ambition:** Medium — this crossed server presentation adapters, Jinja, runtime diagnostics, WebSocket rendering, polling, security hardening, responsive CSS, and compatibility tests.

**Execution:** Adequate — automated contracts and syntax/regression checks are strong, but no physical two-camera stream or browser viewport was available for interactive validation.

**Devil's advocate:**

- Lower: the recognition core and transport protocol were intentionally unchanged, and browser/physical-camera behavior was not exercised.
- Higher: the work removed a real injection risk, preserved multiple existing contracts, handled missing runtime data honestly, and completed a demonstrated red-green cycle with 26 passing supported tests.
- Resolution: Medium/Adequate remains appropriate because the implementation is complete within automated scope, while live hardware and viewport validation remain meaningful gaps.

**Score: 3/5.**

## Concerns

- The current production `CameraWorker.stats()` does not publish processed FPS, dropped-frame count, or last-frame timestamp. Those fields therefore render as `Mavjud emas` until workers provide them; the page intentionally does not fabricate values. Camera FPS, resolution, processing latency (`timings.total`), state, and pipeline errors are available now.
- No physical two-camera/WebSocket or browser viewport pass was possible in this run. The transport decoder, cleanup, responsive selectors, and page rendering are covered statically/integration-wise, but Task 8 should still exercise real feeds at desktop/mobile sizes.
- Running `pytest -q` from the repository root is blocked during collection by pre-existing operational files outside `tests/`: `bench/test_attendance.py` executes a database scenario at import time, and `scripts/live_test.py` interprets pytest's `-q` argument as an integer. No Task 6 file imports them. The supported `pytest tests -q` suite passes 26/26; the out-of-scope collection configuration/scripts were not changed.

## Fix round 1/5

### Important findings addressed

1. `static/js/camera_stream.js`
   - A camera WebSocket opening no longer marks the camera online. It remains unavailable with `Kadr kutilmoqda` until a valid JPEG frame is decoded.
   - Frame metadata now inspects `stale`; stale frames mark the stream unavailable and do not update FPS, last-frame time, latency, or canvas content.
   - Added a monotonic frame sequence so an older asynchronous image decode cannot restore online state after a newer stale message.
2. `static/js/camera_stream.js`
   - Replaced the nonexistent `/ws/recognition/` live-console socket with the registered `/ws/attendance/` endpoint.
   - Added safe normalization for actual `{type: "event", ...}` worker envelopes, including transition-derived action and `/media/` normalization for relative quality-best snapshots.
   - Non-event envelopes are ignored. The existing 10-second `/recognition/logs` poll remains the fallback.
3. `app/api/pages.py`
   - Normalizes the real fresh `RtspSource.stats()` values `fps <= 0` and `resolution == "0x0"` to `None`, so templates render `Mavjud emas` instead of healthy-looking zero literals.
4. `app/web/viewmodels.py`
   - `EventVM.action_type` now maps `CHECK_IN` to `IN` and `CHECK_OUT` to `OUT`; camera role is used only when no attendance transition occurred.
5. `tests/test_ui_pages.py`
   - Added executable Node regression coverage for socket-open/stale behavior, out-of-order JPEG decode, safe actual attendance-event normalization, registered WebSocket URL, and polling fallback.
   - Added coverage against an actual unstarted `RtspSource.stats()` result.
   - Added role/transition mismatch coverage for both authoritative transitions and non-transition fallback.

### Test-first record

The new focused selection initially reported `5 failed, 1 passed`: the JavaScript was not test-exportable and still had the old socket behavior/endpoint, real RTSP zero values passed through, and both role/transition mismatch cases returned the camera role. Each finding was fixed and run independently. A strengthened asynchronous stale-frame race test then failed once and passed after adding frame sequencing.

Combined focused result before final verification:

```text
============================= test session starts ==============================
platform linux -- Python 3.10.16, pytest-9.1.1, pluggy-1.6.0 -- /home/inomjon/anaconda3/envs/yolo/bin/python
cachedir: .pytest_cache
rootdir: /home/inomjon/projectAI/face_rec/face_recognition_airi
plugins: hydra-core-1.3.2, anyio-4.12.0
collecting ... collected 32 items / 23 deselected / 9 selected

tests/test_ui_pages.py::test_live_console_normalizes_role_cameras_and_missing_workers_as_offline PASSED [ 11%]
tests/test_ui_pages.py::test_live_console_and_logs_use_quality_best_evidence_with_stable_contract PASSED [ 22%]
tests/test_ui_pages.py::test_live_feed_script_is_safe_bounded_and_preserves_stream_contract PASSED [ 33%]
tests/test_ui_pages.py::test_camera_socket_only_promotes_valid_non_stale_frames PASSED [ 44%]
tests/test_ui_pages.py::test_live_script_normalizes_actual_attendance_websocket_events PASSED [ 55%]
tests/test_ui_pages.py::test_live_camera_context_treats_real_zero_rtsp_stats_as_unavailable PASSED [ 66%]
tests/test_ui_pages.py::test_event_vm_action_prefers_attendance_transition_over_camera_role[OUT-CHECK_IN-IN] PASSED [ 77%]
tests/test_ui_pages.py::test_event_vm_action_prefers_attendance_transition_over_camera_role[IN-CHECK_OUT-OUT] PASSED [ 88%]
tests/test_ui_pages.py::test_event_vm_action_prefers_attendance_transition_over_camera_role[OUT-RE_SIGHTING-OUT] PASSED [100%]

====================== 9 passed, 23 deselected in 14.19s =======================
```

### Required verification

Command:

```text
/home/inomjon/anaconda3/envs/yolo/bin/python -m pytest tests/test_ui_pages.py -v
```

Exact output:

```text
============================= test session starts ==============================
platform linux -- Python 3.10.16, pytest-9.1.1, pluggy-1.6.0 -- /home/inomjon/anaconda3/envs/yolo/bin/python
cachedir: .pytest_cache
rootdir: /home/inomjon/projectAI/face_rec/face_recognition_airi
plugins: hydra-core-1.3.2, anyio-4.12.0
collecting ... collected 32 items

tests/test_ui_pages.py::test_page_routes_return_html[/] PASSED           [  3%]
tests/test_ui_pages.py::test_page_routes_return_html[/employees] PASSED  [  6%]
tests/test_ui_pages.py::test_page_routes_return_html[/attendance] PASSED [  9%]
tests/test_ui_pages.py::test_page_routes_return_html[/recognition] PASSED [ 12%]
tests/test_ui_pages.py::test_page_routes_return_html[/attendance/unknown] PASSED [ 15%]
tests/test_ui_pages.py::test_page_routes_return_html[/cameras] PASSED    [ 18%]
tests/test_ui_pages.py::test_dashboard_exposes_current_view_to_the_base_template PASSED [ 21%]
tests/test_ui_pages.py::test_employee_registration_uses_the_navigation_view_identifier PASSED [ 25%]
tests/test_ui_pages.py::test_shared_shell_exposes_localized_primary_navigation_once PASSED [ 28%]
tests/test_ui_pages.py::test_shared_shell_provides_theme_control_and_evidence_modal PASSED [ 31%]
tests/test_ui_pages.py::test_dashboard_renders_operational_sections_for_the_current_database_state PASSED [ 34%]
tests/test_ui_pages.py::test_dashboard_embeds_chart_data_and_never_exposes_debug_evidence_paths PASSED [ 37%]
tests/test_ui_pages.py::test_dashboard_excludes_direction_undetermined_rows_from_metrics_and_charts PASSED [ 40%]
tests/test_ui_pages.py::test_dashboard_table_is_limited_to_the_most_recent_recorded_rows PASSED [ 43%]
tests/test_ui_pages.py::test_employee_roster_filters_round_trip_and_uses_grouped_enrollment_counts PASSED [ 46%]
tests/test_ui_pages.py::test_employee_detail_uses_actual_enrollment_count_and_evidence_empty_states PASSED [ 50%]
tests/test_ui_pages.py::test_attendance_filters_round_trip_in_sql_once_and_render_evidence PASSED [ 53%]
tests/test_ui_pages.py::test_attendance_normalizes_an_inverted_date_range PASSED [ 56%]
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params0] PASSED [ 59%]
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params1] PASSED [ 62%]
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params2] PASSED [ 65%]
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params3] PASSED [ 68%]
tests/test_ui_pages.py::test_attendance_export_applies_the_visible_list_filters PASSED [ 71%]
tests/test_ui_pages.py::test_live_console_normalizes_role_cameras_and_missing_workers_as_offline PASSED [ 75%]
tests/test_ui_pages.py::test_live_console_and_logs_use_quality_best_evidence_with_stable_contract PASSED [ 78%]
tests/test_ui_pages.py::test_live_feed_script_is_safe_bounded_and_preserves_stream_contract PASSED [ 81%]
tests/test_ui_pages.py::test_camera_socket_only_promotes_valid_non_stale_frames PASSED [ 84%]
tests/test_ui_pages.py::test_live_script_normalizes_actual_attendance_websocket_events PASSED [ 87%]
tests/test_ui_pages.py::test_live_camera_context_treats_real_zero_rtsp_stats_as_unavailable PASSED [ 90%]
tests/test_ui_pages.py::test_event_vm_action_prefers_attendance_transition_over_camera_role[OUT-CHECK_IN-IN] PASSED [ 93%]
tests/test_ui_pages.py::test_event_vm_action_prefers_attendance_transition_over_camera_role[IN-CHECK_OUT-OUT] PASSED [ 96%]
tests/test_ui_pages.py::test_event_vm_action_prefers_attendance_transition_over_camera_role[OUT-RE_SIGHTING-OUT] PASSED [100%]

============================= 32 passed in 10.05s ==============================
```

Command:

```text
node --check static/js/camera_stream.js
```

Exact output: no stdout/stderr; exit code `0`.

Additional compile check:

```text
/home/inomjon/anaconda3/envs/yolo/bin/python -m py_compile app/api/pages.py app/web/viewmodels.py tests/test_ui_pages.py
```

No stdout/stderr; exit code `0`.

### Fix-round self-review

- Confirmed socket-open state is non-online, stale frames invalidate pending decodes, and only the latest valid non-stale decoded frame updates canvas/FPS/last-frame state.
- Confirmed the active Task 6 push client uses the router-registered `/ws/attendance/` endpoint and consumes the worker's exact event fields. Polling remains unchanged at 10 seconds and continues after socket failure.
- Confirmed event names and camera values still flow through the existing DOM construction/`textContent` renderer; relative snapshot paths become same-origin `/media/` paths before the evidence URL safety check.
- Confirmed real `RtspSource.stats()` zero initialization values become unavailable while positive runtime FPS/resolution values remain unchanged.
- Confirmed persisted `CHECK_IN`/`CHECK_OUT` transitions are authoritative on server-rendered and polled events, with role fallback only for non-transition events.
- Confirmed no recognition core, attendance service, worker lifecycle, thresholds, tracking, direction, recorder, camera configuration, polling cadence, or stream route/decoder changes were made.

### Remaining concerns

- Physical RTSP/WebSocket and browser viewport validation remains deferred to Task 8; this fix round adds executable client-state tests with controlled socket/frame ordering.
- `CameraWorker.stats()` still does not expose processed FPS, dropped frames, or last-frame timestamp; those metrics continue to render explicitly unavailable as documented above.
