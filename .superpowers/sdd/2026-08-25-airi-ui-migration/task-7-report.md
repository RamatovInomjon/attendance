# Task 7 Report: Enrollment, Unknown Review, and Camera Diagnostics

## Status

Completed.

## Implemented

- Restyled employee enrollment into Uzbek two-stage AIRI workflow cards for
  employee metadata and face capture. All pre-existing metadata names,
  hidden capture payload names, element IDs used by the registration script,
  capture source selection, WebSocket endpoint, and submit behavior remain
  unchanged.
- Added non-authoritative quality guidance for blur, face size, pose,
  alignment score, and duplicate risk. It explicitly says that it does not
  add backend validation. The existing capture completion state now explains
  that embedding is checked during saving.
- Replaced the unknown-attempt list with localized evidence cards. Cards use
  the existing `unknown_attempts` fields for the persisted best snapshot,
  camera ID, first/last seen values, and frame count. They open the shared
  evidence preview and expose a localized empty state. No identity assignment
  or delete mutation was added.
- Replaced camera write controls with read-only diagnostics. Each configured
  camera now has separate **Sozlangan**, **Kuzatilgan**, **Pipeline**, and
  **Drift** groups. Worker absence/staleness is rendered as unavailable,
  rather than as synthetic metrics.
- Added the Uzbek NVR ownership warning: encoder quality, bitrate, and GOP
  can be overwritten by the NVR and must be changed at the NVR; AIRI neither
  persists nor applies those settings.

## Files Changed

- `app/api/pages.py`
- `templates/employees/register.html`
- `templates/attendance/unknown.html`
- `templates/camera/settings.html`
- `static/css/main.css`
- `tests/test_ui_pages.py`
- `.superpowers/sdd/2026-08-25-airi-ui-migration/task-7-report.md`

## Test Evidence

Required command:

```text
/home/inomjon/anaconda3/envs/yolo/bin/python -m pytest tests/test_ui_pages.py -v
```

Exact final result:

```text
============================= test session starts ==============================
platform linux -- Python 3.10.16, pytest-9.1.1, pluggy-1.6.0 -- /home/inomjon/anaconda3/envs/yolo/bin/python
cachedir: .pytest_cache
rootdir: /home/inomjon/projectAI/face_rec/face_recognition_airi
plugins: hydra-core-1.3.2, anyio-4.12.0
collecting ... collected 35 items

tests/test_ui_pages.py::test_page_routes_return_html[/] PASSED
tests/test_ui_pages.py::test_page_routes_return_html[/employees] PASSED
tests/test_ui_pages.py::test_page_routes_return_html[/attendance] PASSED
tests/test_ui_pages.py::test_page_routes_return_html[/recognition] PASSED
tests/test_ui_pages.py::test_page_routes_return_html[/attendance/unknown] PASSED
tests/test_ui_pages.py::test_page_routes_return_html[/cameras] PASSED
tests/test_ui_pages.py::test_enrollment_page_keeps_submission_and_capture_contracts PASSED
tests/test_ui_pages.py::test_unknown_review_renders_evidence_and_localized_empty_state PASSED
tests/test_ui_pages.py::test_camera_diagnostics_separate_configuration_observation_pipeline_and_drift PASSED
tests/test_ui_pages.py::test_shared_shell_exposes_localized_primary_navigation_once PASSED
tests/test_ui_pages.py::test_shared_shell_provides_theme_control_and_evidence_modal PASSED
tests/test_ui_pages.py::test_dashboard_renders_operational_sections_for_the_current_database_state PASSED
tests/test_ui_pages.py::test_dashboard_embeds_chart_data_and_never_exposes_debug_evidence_paths PASSED
tests/test_ui_pages.py::test_dashboard_excludes_direction_undetermined_rows_from_metrics_and_charts PASSED
tests/test_ui_pages.py::test_dashboard_table_is_limited_to_the_most_recent_recorded_rows PASSED
tests/test_ui_pages.py::test_employee_roster_filters_round_trip_and_uses_grouped_enrollment_counts PASSED
tests/test_ui_pages.py::test_employee_detail_uses_actual_enrollment_count_and_evidence_empty_states PASSED
tests/test_ui_pages.py::test_attendance_filters_round_trip_in_sql_once_and_render_evidence PASSED
tests/test_ui_pages.py::test_attendance_normalizes_an_inverted_date_range PASSED
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params0] PASSED
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params1] PASSED
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params2] PASSED
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params3] PASSED
tests/test_ui_pages.py::test_attendance_export_applies_the_visible_list_filters PASSED
tests/test_ui_pages.py::test_live_console_normalizes_role_cameras_and_missing_workers_as_offline PASSED
tests/test_ui_pages.py::test_live_console_and_logs_use_quality_best_evidence_with_stable_contract PASSED
tests/test_ui_pages.py::test_live_feed_script_is_safe_bounded_and_preserves_stream_contract PASSED
tests/test_ui_pages.py::test_camera_socket_only_promotes_valid_non_stale_frames PASSED
tests/test_ui_pages.py::test_live_script_normalizes_actual_attendance_websocket_events PASSED
tests/test_ui_pages.py::test_live_camera_context_treats_real_zero_rtsp_stats_as_unavailable PASSED
tests/test_ui_pages.py::test_event_vm_action_prefers_attendance_transition_over_camera_role[OUT-CHECK_IN-IN] PASSED
tests/test_ui_pages.py::test_event_vm_action_prefers_attendance_transition_over_camera_role[IN-CHECK_OUT-OUT] PASSED
tests/test_ui_pages.py::test_event_vm_action_prefers_attendance_transition_over_camera_role[OUT-RE_SIGHTING-OUT] PASSED

============================== 35 passed in 5.05s ==============================
```

`/home/inomjon/anaconda3/envs/yolo/bin/python -m py_compile app/api/pages.py tests/test_ui_pages.py` also exited successfully.

## Self-Review

- Confirmed all existing enrollment field names and JavaScript-consumed IDs
  are present, including both captured-image payload inputs.
- Confirmed unknown and camera pages do not introduce submit, assignment, or
  delete controls.
- Confirmed unknown evidence remains the persisted snapshot path and empty
  evidence does not substitute a placeholder as a purported capture.
- Confirmed observed/pipeline metrics are explicitly unavailable when worker
  data is absent and that a valid `0` latency is not discarded.
- Confirmed no core recognition, model, threshold, schema, NVR, or camera
  configuration code changed.

## Concerns

- The workspace has no Git metadata, so no Git initialization or commit was
  performed, as required.
- The required FastAPI TestClient suite stalls inside the filesystem sandbox
  before its first route (the accepted Task 1 environment issue). The final
  required run completed outside that sandbox with all 35 tests passing.

## Fix Round 1/5

### Findings addressed

- The enrollment form had no explicit target, so browser submission defaulted
  to the GET-only `/employees/add` page. The form now targets the existing
  `POST /api/employees/` handler, and the v3 app mounts that existing router
  rather than adding a duplicate employee-creation implementation.
- A `worker.stats()` exception could previously abort the complete camera
  diagnostics page. Statistics are now read per worker with failure isolation;
  the affected configured camera renders as unavailable with the explicit
  `Ishchi statistikasi olinmadi` state.
- Each diagnostics item now carries a presentation-only `drift` object:
  `nvr-owned`, `unavailable` comparison, localized non-comparable label, and
  NVR ownership explanation. The template consumes this object and still
  performs no encoder writes.
- RTSP presentation now removes URL userinfo and query data while preserving
  the scheme, host, optional port, and path. A credential-bearing fixture
  proves that neither username nor password reaches rendered HTML.

### Added regression coverage

- Enrollment form action and all existing form/capture IDs.
- Existing employee API successful POST behavior with isolated dependency and
  metadata capture, plus confirmation that the v3 app mounts the same router.
- Per-worker stats failure rendering.
- Per-camera NVR-owned, non-comparable drift contract.
- RTSP username/password redaction with safe host/path display.

### Commands and exact final output

```text
/home/inomjon/anaconda3/envs/yolo/bin/python -m pytest tests/test_ui_pages.py -v
============================= test session starts ==============================
platform linux -- Python 3.10.16, pytest-9.1.1, pluggy-1.6.0 -- /home/inomjon/anaconda3/envs/yolo/bin/python
cachedir: .pytest_cache
rootdir: /home/inomjon/projectAI/face_rec/face_recognition_airi
plugins: hydra-core-1.3.2, anyio-4.12.0
collecting ... collected 38 items

tests/test_ui_pages.py::test_page_routes_return_html[/] PASSED           [  2%]
tests/test_ui_pages.py::test_page_routes_return_html[/employees] PASSED  [  5%]
tests/test_ui_pages.py::test_page_routes_return_html[/attendance] PASSED [  7%]
tests/test_ui_pages.py::test_page_routes_return_html[/recognition] PASSED [ 10%]
tests/test_ui_pages.py::test_page_routes_return_html[/attendance/unknown] PASSED [ 13%]
tests/test_ui_pages.py::test_page_routes_return_html[/cameras] PASSED    [ 15%]
tests/test_ui_pages.py::test_dashboard_exposes_current_view_to_the_base_template PASSED [ 18%]
tests/test_ui_pages.py::test_employee_registration_uses_the_navigation_view_identifier PASSED [ 21%]
tests/test_ui_pages.py::test_enrollment_page_keeps_submission_and_capture_contracts PASSED [ 23%]
tests/test_ui_pages.py::test_existing_employee_api_accepts_enrollment_form_post PASSED [ 26%]
tests/test_ui_pages.py::test_unknown_review_renders_evidence_and_localized_empty_state PASSED [ 28%]
tests/test_ui_pages.py::test_camera_diagnostics_separate_configuration_observation_pipeline_and_drift PASSED [ 31%]
tests/test_ui_pages.py::test_camera_diagnostics_survive_a_worker_stats_failure PASSED [ 34%]
tests/test_ui_pages.py::test_camera_diagnostics_exposes_nvr_owned_non_comparable_drift PASSED [ 36%]
tests/test_ui_pages.py::test_shared_shell_exposes_localized_primary_navigation_once PASSED [ 39%]
tests/test_ui_pages.py::test_shared_shell_provides_theme_control_and_evidence_modal PASSED [ 42%]
tests/test_ui_pages.py::test_dashboard_renders_operational_sections_for_the_current_database_state PASSED [ 44%]
tests/test_ui_pages.py::test_dashboard_embeds_chart_data_and_never_exposes_debug_evidence_paths PASSED [ 47%]
tests/test_ui_pages.py::test_dashboard_excludes_direction_undetermined_rows_from_metrics_and_charts PASSED [ 50%]
tests/test_ui_pages.py::test_dashboard_table_is_limited_to_the_most_recent_recorded_rows PASSED [ 52%]
tests/test_ui_pages.py::test_employee_roster_filters_round_trip_and_uses_grouped_enrollment_counts PASSED [ 55%]
tests/test_ui_pages.py::test_employee_detail_uses_actual_enrollment_count_and_evidence_empty_states PASSED [ 57%]
tests/test_ui_pages.py::test_attendance_filters_round_trip_in_sql_once_and_render_evidence PASSED [ 60%]
tests/test_ui_pages.py::test_attendance_normalizes_an_inverted_date_range PASSED [ 63%]
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params0] PASSED [ 65%]
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params1] PASSED [ 68%]
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params2] PASSED [ 71%]
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params3] PASSED [ 73%]
tests/test_ui_pages.py::test_attendance_export_applies_the_visible_list_filters PASSED [ 76%]
tests/test_ui_pages.py::test_live_console_normalizes_role_cameras_and_missing_workers_as_offline PASSED [ 78%]
tests/test_ui_pages.py::test_live_console_and_logs_use_quality_best_evidence_with_stable_contract PASSED [ 81%]
tests/test_ui_pages.py::test_live_feed_script_is_safe_bounded_and_preserves_stream_contract PASSED [ 84%]
tests/test_ui_pages.py::test_camera_socket_only_promotes_valid_non_stale_frames PASSED [ 86%]
tests/test_ui_pages.py::test_live_script_normalizes_actual_attendance_websocket_events PASSED [ 89%]
tests/test_ui_pages.py::test_live_camera_context_treats_real_zero_rtsp_stats_as_unavailable PASSED [ 92%]
tests/test_ui_pages.py::test_event_vm_action_prefers_attendance_transition_over_camera_role[OUT-CHECK_IN-IN] PASSED [ 94%]
tests/test_ui_pages.py::test_event_vm_action_prefers_attendance_transition_over_camera_role[IN-CHECK_OUT-OUT] PASSED [ 97%]
tests/test_ui_pages.py::test_event_vm_action_prefers_attendance_transition_over_camera_role[OUT-RE_SIGHTING-OUT] PASSED [100%]

============================== 38 passed in 5.32s ==============================
```

```text
/home/inomjon/anaconda3/envs/yolo/bin/python -m py_compile app/api/main.py app/api/pages.py tests/test_ui_pages.py
```

The compile command exited with status 0 and no output.

### Fix-round self-review

- The mounted employee router is the pre-existing handler; no `POST` route was
  added to `app/api/pages.py`.
- Worker failures are caught independently, so one failed camera cannot hide
  other diagnostics.
- The template reads `camera.drift` only and does not display invented encoder
  metrics or introduce configuration controls.
- Raw RTSP URLs never reach the diagnostics template; the presentation value
  is sanitized first.

## Fix Round 2

### Status

DONE_WITH_CONCERNS.

### Root cause and native v3 fix

- Confirmed that the Fix Round 1 form target was handled by
  `fast_api.routers.employees`: it wrote the legacy `db.sqlite3` tables and did
  not accept or persist either capture field. The v3 recognizer reads
  `Employee` and `FaceEmbedding` from `data/ematsy.db`, so successful legacy
  responses could never enroll a recognizable employee.
- Removed the legacy employee-router import and mount from `app/api/main.py`.
- Added the native `POST /api/employees/` handler to `app/api/pages.py`.
  Camera data URLs and multipart uploads now share the same
  `Enroller.embed_capture()` path: the existing detector, DFA aligner, active
  `assess()` quality thresholds, and AdaFace recognizer.
- Every valid image produces one float32 `FaceEmbedding` row using the active
  recognizer model name. The new v3 `Employee` and all accepted embeddings are
  flushed in one `session_scope()` transaction. A forced embedding-row
  constraint failure proves that the already-flushed employee is rolled back.
- Per-image no-face, alignment, blur, face-size, pose, corrupt-image, type,
  size, and count failures are returned as actionable Uzbek rejections. Valid
  images may still enroll; when zero valid embeddings remain, the handler
  returns 422 and creates no employee.
- After commit, the existing `runtime.reload_gallery()` atomically replaces the
  gallery reference used by each worker. A reload failure is logged and exposed
  as a restart warning without encouraging a duplicate enrollment retry.

### Image-upload addition

- Kept all existing camera fields, IDs, WebSocket behavior, multi-angle capture
  logic, and minimum camera-capture rule.
- Added an explicit multi-file `uploaded_images` control for JPEG, PNG, and
  WebP images. Server limits are 32 images total and 10 MB per image.
- Added accessible client-side validation, previews, per-image removal, and an
  `aria-live` result region. Removed files are also removed from fetch
  `FormData`; the multipart form remains usable as a controlled no-JS submit.
- Fetch submissions receive structured JSON and render success, warnings, and
  per-image errors in the page. Plain HTML success redirects to the employee
  detail when every image is accepted; partial success and all-failed results
  render accessible HTML with the relevant per-image details.

### Test replacement and additions

- Replaced the legacy CRUD mock that only proved metadata acceptance with
  temporary-v3-database integration tests.
- Stubbed only `Enroller.embed_capture()`, so request parsing, multipart/data-URL
  decoding, v3 SQLAlchemy writes, rollback, response negotiation, and gallery
  loading remain real. No detector, aligner, AdaFace, CUDA, or other GPU
  inference runs in tests.
- Added coverage for camera persistence, all-failed camera rollback, a forced
  `FaceEmbedding` persistence rollback, fetch JSON, no-JS redirect, multipart
  JPEG/PNG/WebP persistence, unsupported/oversize/count rejection, all-failed
  upload HTML, legacy-router absence, upload contracts, and rendered JavaScript
  parsing through Node.

### Files changed

- `app/api/main.py`
- `app/api/pages.py`
- `app/services/enrollment.py`
- `templates/employees/register.html`
- `static/css/main.css`
- `tests/test_ui_pages.py`
- `.superpowers/sdd/2026-08-25-airi-ui-migration/task-7-report.md`

### Commands and exact final output

Targeted enrollment tests:

```text
/home/inomjon/anaconda3/envs/yolo/bin/python -m pytest tests/test_ui_pages.py::test_enrollment_page_keeps_submission_and_capture_contracts tests/test_ui_pages.py::test_enrollment_inline_script_has_valid_javascript tests/test_ui_pages.py::test_native_enrollment_persists_v3_embeddings_and_returns_fetch_json tests/test_ui_pages.py::test_native_enrollment_rolls_back_and_renders_html_on_capture_rejection tests/test_ui_pages.py::test_native_enrollment_database_failure_rolls_back_employee_and_embeddings tests/test_ui_pages.py::test_native_enrollment_no_js_submission_redirects_to_v3_detail tests/test_ui_pages.py::test_multipart_uploads_persist_v3_embeddings_through_the_capture_service tests/test_ui_pages.py::test_multipart_uploads_report_unsupported_and_oversize_images_safely tests/test_ui_pages.py::test_multipart_uploads_render_all_rejections_and_leave_no_employee tests/test_ui_pages.py::test_v3_app_does_not_mount_the_legacy_employee_router -v
============================= test session starts ==============================
platform linux -- Python 3.10.16, pytest-9.1.1, pluggy-1.6.0 -- /home/inomjon/anaconda3/envs/yolo/bin/python
cachedir: .pytest_cache
rootdir: /home/inomjon/projectAI/face_rec/face_recognition_airi
plugins: hydra-core-1.3.2, anyio-4.12.0
collecting ... collected 10 items

tests/test_ui_pages.py::test_enrollment_page_keeps_submission_and_capture_contracts PASSED [ 10%]
tests/test_ui_pages.py::test_enrollment_inline_script_has_valid_javascript PASSED [ 20%]
tests/test_ui_pages.py::test_native_enrollment_persists_v3_embeddings_and_returns_fetch_json PASSED [ 30%]
tests/test_ui_pages.py::test_native_enrollment_rolls_back_and_renders_html_on_capture_rejection PASSED [ 40%]
tests/test_ui_pages.py::test_native_enrollment_database_failure_rolls_back_employee_and_embeddings PASSED [ 50%]
tests/test_ui_pages.py::test_native_enrollment_no_js_submission_redirects_to_v3_detail PASSED [ 60%]
tests/test_ui_pages.py::test_multipart_uploads_persist_v3_embeddings_through_the_capture_service PASSED [ 70%]
tests/test_ui_pages.py::test_multipart_uploads_report_unsupported_and_oversize_images_safely PASSED [ 80%]
tests/test_ui_pages.py::test_multipart_uploads_render_all_rejections_and_leave_no_employee PASSED [ 90%]
tests/test_ui_pages.py::test_v3_app_does_not_mount_the_legacy_employee_router PASSED [100%]

============================== 10 passed in 4.33s ==============================
```

Required full Task 7 UI suite:

```text
/home/inomjon/anaconda3/envs/yolo/bin/python -m pytest tests/test_ui_pages.py -v
============================= test session starts ==============================
platform linux -- Python 3.10.16, pytest-9.1.1, pluggy-1.6.0 -- /home/inomjon/anaconda3/envs/yolo/bin/python
cachedir: .pytest_cache
rootdir: /home/inomjon/projectAI/face_rec/face_recognition_airi
plugins: hydra-core-1.3.2, anyio-4.12.0
collecting ... collected 46 items

tests/test_ui_pages.py::test_page_routes_return_html[/] PASSED           [  2%]
tests/test_ui_pages.py::test_page_routes_return_html[/employees] PASSED  [  4%]
tests/test_ui_pages.py::test_page_routes_return_html[/attendance] PASSED [  6%]
tests/test_ui_pages.py::test_page_routes_return_html[/recognition] PASSED [  8%]
tests/test_ui_pages.py::test_page_routes_return_html[/attendance/unknown] PASSED [ 10%]
tests/test_ui_pages.py::test_page_routes_return_html[/cameras] PASSED    [ 13%]
tests/test_ui_pages.py::test_dashboard_exposes_current_view_to_the_base_template PASSED [ 15%]
tests/test_ui_pages.py::test_employee_registration_uses_the_navigation_view_identifier PASSED [ 17%]
tests/test_ui_pages.py::test_enrollment_page_keeps_submission_and_capture_contracts PASSED [ 19%]
tests/test_ui_pages.py::test_enrollment_inline_script_has_valid_javascript PASSED [ 21%]
tests/test_ui_pages.py::test_native_enrollment_persists_v3_embeddings_and_returns_fetch_json PASSED [ 23%]
tests/test_ui_pages.py::test_native_enrollment_rolls_back_and_renders_html_on_capture_rejection PASSED [ 26%]
tests/test_ui_pages.py::test_native_enrollment_database_failure_rolls_back_employee_and_embeddings PASSED [ 28%]
tests/test_ui_pages.py::test_native_enrollment_no_js_submission_redirects_to_v3_detail PASSED [ 30%]
tests/test_ui_pages.py::test_multipart_uploads_persist_v3_embeddings_through_the_capture_service PASSED [ 32%]
tests/test_ui_pages.py::test_multipart_uploads_report_unsupported_and_oversize_images_safely PASSED [ 34%]
tests/test_ui_pages.py::test_multipart_uploads_render_all_rejections_and_leave_no_employee PASSED [ 36%]
tests/test_ui_pages.py::test_v3_app_does_not_mount_the_legacy_employee_router PASSED [ 39%]
tests/test_ui_pages.py::test_unknown_review_renders_evidence_and_localized_empty_state PASSED [ 41%]
tests/test_ui_pages.py::test_camera_diagnostics_separate_configuration_observation_pipeline_and_drift PASSED [ 43%]
tests/test_ui_pages.py::test_camera_diagnostics_survive_a_worker_stats_failure PASSED [ 45%]
tests/test_ui_pages.py::test_camera_diagnostics_exposes_nvr_owned_non_comparable_drift PASSED [ 47%]
tests/test_ui_pages.py::test_shared_shell_exposes_localized_primary_navigation_once PASSED [ 50%]
tests/test_ui_pages.py::test_shared_shell_provides_theme_control_and_evidence_modal PASSED [ 52%]
tests/test_ui_pages.py::test_dashboard_renders_operational_sections_for_the_current_database_state PASSED [ 54%]
tests/test_ui_pages.py::test_dashboard_embeds_chart_data_and_never_exposes_debug_evidence_paths PASSED [ 56%]
tests/test_ui_pages.py::test_dashboard_excludes_direction_undetermined_rows_from_metrics_and_charts PASSED [ 58%]
tests/test_ui_pages.py::test_dashboard_table_is_limited_to_the_most_recent_recorded_rows PASSED [ 60%]
tests/test_ui_pages.py::test_employee_roster_filters_round_trip_and_uses_grouped_enrollment_counts PASSED [ 63%]
tests/test_ui_pages.py::test_employee_detail_uses_actual_enrollment_count_and_evidence_empty_states PASSED [ 65%]
tests/test_ui_pages.py::test_attendance_filters_round_trip_in_sql_once_and_render_evidence PASSED [ 67%]
tests/test_ui_pages.py::test_attendance_normalizes_an_inverted_date_range PASSED [ 69%]
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params0] PASSED [ 71%]
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params1] PASSED [ 73%]
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params2] PASSED [ 76%]
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params3] PASSED [ 78%]
tests/test_ui_pages.py::test_attendance_export_applies_the_visible_list_filters PASSED [ 80%]
tests/test_ui_pages.py::test_live_console_normalizes_role_cameras_and_missing_workers_as_offline PASSED [ 82%]
tests/test_ui_pages.py::test_live_console_and_logs_use_quality_best_evidence_with_stable_contract PASSED [ 84%]
tests/test_ui_pages.py::test_live_feed_script_is_safe_bounded_and_preserves_stream_contract PASSED [ 86%]
tests/test_ui_pages.py::test_camera_socket_only_promotes_valid_non_stale_frames PASSED [ 89%]
tests/test_ui_pages.py::test_live_script_normalizes_actual_attendance_websocket_events PASSED [ 91%]
tests/test_ui_pages.py::test_live_camera_context_treats_real_zero_rtsp_stats_as_unavailable PASSED [ 93%]
tests/test_ui_pages.py::test_event_vm_action_prefers_attendance_transition_over_camera_role[OUT-CHECK_IN-IN] PASSED [ 95%]
tests/test_ui_pages.py::test_event_vm_action_prefers_attendance_transition_over_camera_role[IN-CHECK_OUT-OUT] PASSED [ 97%]
tests/test_ui_pages.py::test_event_vm_action_prefers_attendance_transition_over_camera_role[OUT-RE_SIGHTING-OUT] PASSED [100%]

============================== 46 passed in 4.75s ==============================
```

Python compilation:

```text
/home/inomjon/anaconda3/envs/yolo/bin/python -m py_compile app/api/main.py app/api/pages.py app/services/enrollment.py tests/test_ui_pages.py
```

The command exited with status 0 and no output.

Direct inline JavaScript syntax check:

```text
sed -n '/^<script>$/,/^<\/script>$/p' templates/employees/register.html | sed '1d;$d;s/const useIpCamera = .*/const useIpCamera = false;/' | node --check -
```

The command exited with status 0 and no output.

### Concerns

- The v3 `Employee` schema has no `email` or `notes` columns. Those existing
  form names and submitted values remain accepted and are preserved when an
  HTML validation response re-renders, but no schema change was introduced to
  persist them because Task 7 requires the existing v3 models/schema.
- As documented in the prior rounds, this project's FastAPI `TestClient`
  stalls before route dispatch inside the filesystem sandbox. The exact
  TestClient commands above were therefore run outside that sandbox; all 46
  tests passed. Compilation and direct Node syntax checks ran normally inside.

## Fix Round 3

### Status

DONE.

### Findings addressed

1. Camera captures are still collected with the existing 24-frame/angle UI,
   but are now resized to a maximum 960-pixel edge, encoded as bounded JPEGs,
   converted to `Blob` objects, and submitted as `camera_images` file parts.
   The existing `captured_image` and `captured_images` IDs/names remain in the
   form, but production camera bytes are no longer serialized into Starlette's
   1 MiB-limited text parts. Fetch continues to receive structured JSON and
   render it in the accessible result region; plain upload forms remain usable
   without JavaScript.
2. `POST /api/employees/` now owns multipart parsing instead of allowing
   FastAPI parameter injection to parse first. The parser rejects more than 32
   file parts before creating the excess spool, rejects a file while it grows
   past 4 MiB, caps text parts at 64 KiB and fields at 12, checks
   `Content-Length` before parsing, and independently counts streamed bytes to
   a 10 MiB whole-request limit. Limit failures return controlled 413 JSON or
   accessible HTML.
3. Pillow inspects the bounded image header before OpenCV inference. The
   actual JPEG/PNG/WebP format and magic signature must agree with the declared
   MIME type and suffix. Images above 4096x4096 or 12,000,000 pixels are
   rejected before OpenCV decode/inference; Pillow verification and a
   post-decode dimension check provide additional corruption/bounds checks.
4. Unexpected extraction/database/persistence failures are logged internally
   at the route boundary and return one sanitized 500 message in HTML or JSON.
   Route tests force a `FaceEmbedding` constraint failure after employee flush
   and prove the transaction rolls back both tables and skips gallery reload.
5. Gallery reload remains post-commit. If it fails, fetch JSON includes the
   operator warning and normal HTML now renders a 201 success-with-warning page
   with `role="alert"` and a safe employee-detail link instead of silently
   redirecting.

The active `Enroller.embed_capture()` detector, aligner, quality assessment,
and AdaFace recognizer remain the only inference path. Tests stub only that
boundary and do not run GPU inference. The prior diagnostics, redaction,
legacy-router removal, v3 transaction, form names/IDs, upload previews/removal,
and partial-valid behavior remain covered by the full UI suite.

### Files changed in Fix Round 3

- `app/api/pages.py`
- `app/services/enrollment.py`
- `templates/employees/register.html`
- `tests/test_ui_pages.py`
- `.superpowers/sdd/2026-08-25-airi-ui-migration/task-7-report.md`

### Commands and exact final output

Targeted enrollment/security tests:

```text
/home/inomjon/anaconda3/envs/yolo/bin/python -m pytest tests/test_ui_pages.py -k 'enrollment or multipart or v3_app or gallery_reload or camera_captures' -q
.......................                                                  [100%]
23 passed, 34 deselected in 5.79s
```

Required full Task 7 UI suite:

```text
/home/inomjon/anaconda3/envs/yolo/bin/python -m pytest tests/test_ui_pages.py -v
============================= test session starts ==============================
platform linux -- Python 3.10.16, pytest-9.1.1, pluggy-1.6.0 -- /home/inomjon/anaconda3/envs/yolo/bin/python
cachedir: .pytest_cache
rootdir: /home/inomjon/projectAI/face_rec/face_recognition_airi
plugins: hydra-core-1.3.2, anyio-4.12.0
collecting ... collected 57 items

tests/test_ui_pages.py::test_page_routes_return_html[/] PASSED           [  1%]
tests/test_ui_pages.py::test_page_routes_return_html[/employees] PASSED  [  3%]
tests/test_ui_pages.py::test_page_routes_return_html[/attendance] PASSED [  5%]
tests/test_ui_pages.py::test_page_routes_return_html[/recognition] PASSED [  7%]
tests/test_ui_pages.py::test_page_routes_return_html[/attendance/unknown] PASSED [  8%]
tests/test_ui_pages.py::test_page_routes_return_html[/cameras] PASSED    [ 10%]
tests/test_ui_pages.py::test_dashboard_exposes_current_view_to_the_base_template PASSED [ 12%]
tests/test_ui_pages.py::test_employee_registration_uses_the_navigation_view_identifier PASSED [ 14%]
tests/test_ui_pages.py::test_enrollment_page_keeps_submission_and_capture_contracts PASSED [ 15%]
tests/test_ui_pages.py::test_enrollment_inline_script_has_valid_javascript PASSED [ 17%]
tests/test_ui_pages.py::test_native_enrollment_persists_v3_embeddings_and_returns_fetch_json PASSED [ 19%]
tests/test_ui_pages.py::test_native_enrollment_rolls_back_and_renders_html_on_capture_rejection PASSED [ 21%]
tests/test_ui_pages.py::test_native_enrollment_database_failure_rolls_back_employee_and_embeddings PASSED [ 22%]
tests/test_ui_pages.py::test_native_enrollment_no_js_submission_redirects_to_v3_detail PASSED [ 24%]
tests/test_ui_pages.py::test_multipart_uploads_persist_v3_embeddings_through_the_capture_service PASSED [ 26%]
tests/test_ui_pages.py::test_camera_captures_submit_24_realistic_jpeg_file_parts PASSED [ 28%]
tests/test_ui_pages.py::test_enrollment_parser_rejects_more_than_32_files_before_inference PASSED [ 29%]
tests/test_ui_pages.py::test_enrollment_parser_rejects_a_file_over_4_mib_with_accessible_html PASSED [ 31%]
tests/test_ui_pages.py::test_enrollment_parser_streaming_total_limit_without_content_length PASSED [ 33%]
tests/test_ui_pages.py::test_enrollment_content_length_over_10_mib_short_circuits_before_parsing PASSED [ 35%]
tests/test_ui_pages.py::test_enrollment_rejects_disguised_or_high_pixel_images_before_inference[disguised-tiff] PASSED [ 36%]
tests/test_ui_pages.py::test_enrollment_rejects_disguised_or_high_pixel_images_before_inference[high-pixel-png] PASSED [ 38%]
tests/test_ui_pages.py::test_enrollment_route_sanitizes_persistence_failure_and_skips_reload[application/json] PASSED [ 40%]
tests/test_ui_pages.py::test_enrollment_route_sanitizes_persistence_failure_and_skips_reload[text/html] PASSED [ 42%]
tests/test_ui_pages.py::test_gallery_reload_failure_is_visible_after_successful_commit[application/json] PASSED [ 43%]
tests/test_ui_pages.py::test_gallery_reload_failure_is_visible_after_successful_commit[text/html] PASSED [ 45%]
tests/test_ui_pages.py::test_multipart_uploads_report_unsupported_and_oversize_images_safely PASSED [ 47%]
tests/test_ui_pages.py::test_multipart_uploads_render_all_rejections_and_leave_no_employee PASSED [ 49%]
tests/test_ui_pages.py::test_v3_app_does_not_mount_the_legacy_employee_router PASSED [ 50%]
tests/test_ui_pages.py::test_unknown_review_renders_evidence_and_localized_empty_state PASSED [ 52%]
tests/test_ui_pages.py::test_camera_diagnostics_separate_configuration_observation_pipeline_and_drift PASSED [ 54%]
tests/test_ui_pages.py::test_camera_diagnostics_survive_a_worker_stats_failure PASSED [ 56%]
tests/test_ui_pages.py::test_camera_diagnostics_exposes_nvr_owned_non_comparable_drift PASSED [ 57%]
tests/test_ui_pages.py::test_shared_shell_exposes_localized_primary_navigation_once PASSED [ 59%]
tests/test_ui_pages.py::test_shared_shell_provides_theme_control_and_evidence_modal PASSED [ 61%]
tests/test_ui_pages.py::test_dashboard_renders_operational_sections_for_the_current_database_state PASSED [ 63%]
tests/test_ui_pages.py::test_dashboard_embeds_chart_data_and_never_exposes_debug_evidence_paths PASSED [ 64%]
tests/test_ui_pages.py::test_dashboard_excludes_direction_undetermined_rows_from_metrics_and_charts PASSED [ 66%]
tests/test_ui_pages.py::test_dashboard_table_is_limited_to_the_most_recent_recorded_rows PASSED [ 68%]
tests/test_ui_pages.py::test_employee_roster_filters_round_trip_and_uses_grouped_enrollment_counts PASSED [ 70%]
tests/test_ui_pages.py::test_employee_detail_uses_actual_enrollment_count_and_evidence_empty_states PASSED [ 71%]
tests/test_ui_pages.py::test_attendance_filters_round_trip_in_sql_once_and_render_evidence PASSED [ 73%]
tests/test_ui_pages.py::test_attendance_normalizes_an_inverted_date_range PASSED [ 75%]
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params0] PASSED [ 77%]
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params1] PASSED [ 78%]
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params2] PASSED [ 80%]
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params3] PASSED [ 82%]
tests/test_ui_pages.py::test_attendance_export_applies_the_visible_list_filters PASSED [ 84%]
tests/test_ui_pages.py::test_live_console_normalizes_role_cameras_and_missing_workers_as_offline PASSED [ 85%]
tests/test_ui_pages.py::test_live_console_and_logs_use_quality_best_evidence_with_stable_contract PASSED [ 87%]
tests/test_ui_pages.py::test_live_feed_script_is_safe_bounded_and_preserves_stream_contract PASSED [ 89%]
tests/test_ui_pages.py::test_camera_socket_only_promotes_valid_non_stale_frames PASSED [ 91%]
tests/test_ui_pages.py::test_live_script_normalizes_actual_attendance_websocket_events PASSED [ 92%]
tests/test_ui_pages.py::test_live_camera_context_treats_real_zero_rtsp_stats_as_unavailable PASSED [ 94%]
tests/test_ui_pages.py::test_event_vm_action_prefers_attendance_transition_over_camera_role[OUT-CHECK_IN-IN] PASSED [ 96%]
tests/test_ui_pages.py::test_event_vm_action_prefers_attendance_transition_over_camera_role[IN-CHECK_OUT-OUT] PASSED [ 98%]
tests/test_ui_pages.py::test_event_vm_action_prefers_attendance_transition_over_camera_role[OUT-RE_SIGHTING-OUT] PASSED [100%]

============================== 57 passed in 6.32s ==============================
```

Python compilation:

```text
/home/inomjon/anaconda3/envs/yolo/bin/python -m py_compile app/api/main.py app/api/pages.py app/services/enrollment.py tests/test_ui_pages.py
```

The command exited with status 0 and no output.

Direct inline JavaScript syntax check:

```text
sed -n '/^<script>$/,/^<\/script>$/p' templates/employees/register.html | sed '1d;$d;s/const useIpCamera = .*/const useIpCamera = false;/' | node --check -
```

The command exited with status 0 and no output.

### Concerns

- No unresolved Fix Round 3 finding remains.
- The pre-existing v3 schema still has no `email` or `notes` columns; the
  existing fields remain accepted/re-rendered but intentionally are not stored.
- As in Fix Round 2, FastAPI `TestClient` commands were run outside the
  filesystem sandbox because this environment stalls them before dispatch.
  The application code, temporary SQLite databases, and test isolation were
  unchanged; all targeted and full tests passed.
