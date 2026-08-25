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
