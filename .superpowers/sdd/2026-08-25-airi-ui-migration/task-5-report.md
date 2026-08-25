# Task 5 Report: Functional Attendance Filters and Evidence Review

## Status

Completed.

## Changed files

- `app/api/pages.py`
  - Added safe ISO-date parsing for `target_date`, `start_date`, and `end_date`, returning controlled HTTP 422 responses for invalid values.
  - Defaults an unfiltered attendance view to the selected business day, normalizes inverted ranges with a visible notice, and rejects inclusive ranges above 366 days.
  - Added `query`, `department`, and `status` filters. The attendance records are fetched with one `DailyAttendance`/`Employee` joined query, SQL predicates, deterministic newest-first ordering, and no per-row queries.
  - Provides active department values while retaining existing attendance route/context fields used by `/attendance` and `/attendance/history`.
- `templates/attendance/list.html`
  - Replaced the legacy attendance chart/status-modal page with Uzbek AIRI filters, reset and date-shortcut controls, query-preserving export URL, responsive shared data table, localized empty state, shared status chips, real IN/OUT evidence buttons, and employee detail links.
  - Removed duplicate evidence/status modals and all inline attendance handlers. Evidence uses escaped data attributes only.
- `static/js/main.js`
  - Added the attendance-scoped delegated evidence handler that calls `AiriUI.openEvidence`.
  - Added local-calendar 15-day and one-calendar-month shortcut behavior; each fills the native date fields and submits the filter form.
- `tests/test_ui_pages.py`
  - Added isolated SQLite page-handler coverage for SQL filtering by name/external ID, department, date, and status; retained form values, department options, deterministic one-join attendance query, query-preserving export URL, data-attribute evidence controls, and the absence of inline handlers.
  - Added coverage for normalized inverted dates and controlled 422 responses for malformed `start_date`, `end_date`, and `target_date`, plus ranges above 366 days.

## Test-first record

The attendance tests were added before the implementation. Their first focused run failed because `attendance_list` did not accept `query`, did not validate ranges, and the legacy template still used inline modal handlers. After the route, template, and shared-JS changes, the focused suite passed, followed by the complete required suite.

## Verification

`node --check static/js/main.js` completed successfully.

`/home/inomjon/anaconda3/envs/yolo/bin/python -m py_compile app/api/pages.py tests/test_ui_pages.py` completed successfully.

The required command completed successfully outside the filesystem sandbox; the SQLite-backed `TestClient` suite stalls inside that sandbox, as documented by accepted Tasks 1–4. Exact output:

```text
/home/inomjon/anaconda3/envs/yolo/bin/python -m pytest tests/test_ui_pages.py -v
============================= test session starts ==============================
platform linux -- Python 3.10.16, pytest-9.1.1, pluggy-1.6.0 -- /home/inomjon/anaconda3/envs/yolo/bin/python
cachedir: .pytest_cache
rootdir: /home/inomjon/projectAI/face_rec/face_recognition_airi
plugins: hydra-core-1.3.2, anyio-4.12.0
collecting ... collected 22 items

tests/test_ui_pages.py::test_page_routes_return_html[/] PASSED           [  4%]
tests/test_ui_pages.py::test_page_routes_return_html[/employees] PASSED  [  9%]
tests/test_ui_pages.py::test_page_routes_return_html[/attendance] PASSED [ 13%]
tests/test_ui_pages.py::test_page_routes_return_html[/recognition] PASSED [ 18%]
tests/test_ui_pages.py::test_page_routes_return_html[/attendance/unknown] PASSED [ 22%]
tests/test_ui_pages.py::test_page_routes_return_html[/cameras] PASSED    [ 27%]
tests/test_ui_pages.py::test_dashboard_exposes_current_view_to_the_base_template PASSED [ 31%]
tests/test_ui_pages.py::test_employee_registration_uses_the_navigation_view_identifier PASSED [ 36%]
tests/test_ui_pages.py::test_shared_shell_exposes_localized_primary_navigation_once PASSED [ 40%]
tests/test_ui_pages.py::test_shared_shell_provides_theme_control_and_evidence_modal PASSED [ 45%]
tests/test_ui_pages.py::test_dashboard_renders_operational_sections_for_the_current_database_state PASSED [ 50%]
tests/test_ui_pages.py::test_dashboard_embeds_chart_data_and_never_exposes_debug_evidence_paths PASSED [ 54%]
tests/test_ui_pages.py::test_dashboard_excludes_direction_undetermined_rows_from_metrics_and_charts PASSED [ 59%]
tests/test_ui_pages.py::test_dashboard_table_is_limited_to_the_most_recent_recorded_rows PASSED [ 63%]
tests/test_ui_pages.py::test_employee_roster_filters_round_trip_and_uses_grouped_enrollment_counts PASSED [ 68%]
tests/test_ui_pages.py::test_employee_detail_uses_actual_enrollment_count_and_evidence_empty_states PASSED [ 72%]
tests/test_ui_pages.py::test_attendance_filters_round_trip_in_sql_once_and_render_evidence PASSED [ 77%]
tests/test_ui_pages.py::test_attendance_normalizes_an_inverted_date_range PASSED [ 81%]
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params0] PASSED [ 86%]
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params1] PASSED [ 90%]
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params2] PASSED [ 95%]
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params3] PASSED [100%]

============================== 22 passed in 6.81s ==============================
```

## Self-review

- Confirmed malformed date parsing is constrained to the attendance query boundary and raises HTTP 422 rather than `ValueError`/500.
- Confirmed the range calculation permits at most 366 inclusive calendar days, swaps inverted dates before querying, and exposes the normalized values and notice.
- Confirmed all attendance filters are predicates on the single joined attendance query, with newest-first ties broken by attendance ID. The separate department query is not row-dependent.
- Confirmed the table uses the existing shared `.data-table` horizontal-scroll behavior and shared status-chip styles; no fabricated attendance metrics or evidence were added.
- Confirmed the page does not create a second evidence modal and contains no `onclick`, `statusDetailsModal`, or snapshot modal code. The attendance-only delegated handler prevents duplicate listeners on accepted dashboard/profile evidence controls.
- Confirmed `/attendance/history` still delegates to the attendance list and the CSV endpoint/URL contract was not changed; the export link retains the active query string exactly.
- Confirmed no attendance core, transition, model, or schema code changed.

## Concerns

- Browser viewport interaction was not automated. JavaScript syntax was checked and the shared responsive table/evidence patterns were exercised through rendered-page tests, but a manual visual/accessibility pass remains useful.
- CSV export now accepts the active list filters while retaining the established single-day `day` contract; no remaining functional export concern was found in this round.

## Fix round 1/5

### Changes

- `app/api/pages.py`
  - Extracted the joined attendance predicate builder and deterministic newest-first ordering into shared helpers used by both the HTML list and API/CSV consumers.
- `app/api/main.py`
  - Extended `/api/attendance` and `/api/attendance/export` with backward-compatible optional `query`, `department`, `start_date`, `end_date`, and `status` parameters.
  - Preserved existing `day` behavior while exporting the same SQL-filtered rows and ordering as the visible attendance list. Range CSV rows retain each record's actual business date; a single-day export retains its existing filename format.
- `templates/attendance/list.html`
  - Added selectable `NO_CHECKIN` (`Kelish qayd etilmagan`) and the corresponding localized warning status chip.
- `tests/test_ui_pages.py`
  - Added multiple matching attendance rows and asserts newest-first rendering order, three joined attendance queries plus three department queries across three isolated renders, and no per-employee selects.
  - Added a real CSV endpoint test proving all visible-list filters affect the file and that legacy `day` clients still receive the single-day CSV contract.

### Test-first record

The focused regressions first failed after fixture corrections: the rendered status select did not contain `NO_CHECKIN`, and the CSV included records excluded by the list's department/status predicates. The shared query implementation made both focused regressions pass before the full suite.

### Verification

`/home/inomjon/anaconda3/envs/yolo/bin/python -m py_compile app/api/main.py app/api/pages.py tests/test_ui_pages.py` completed successfully.

The required suite ran outside the filesystem sandbox because SQLite-backed `TestClient` requests stall inside it. Exact output:

```text
/home/inomjon/anaconda3/envs/yolo/bin/python -m pytest tests/test_ui_pages.py -v
============================= test session starts ==============================
platform linux -- Python 3.10.16, pytest-9.1.1, pluggy-1.6.0 -- /home/inomjon/anaconda3/envs/yolo/bin/python
cachedir: .pytest_cache
rootdir: /home/inomjon/projectAI/face_rec/face_recognition_airi
plugins: hydra-core-1.3.2, anyio-4.12.0
collecting ... collected 23 items

tests/test_ui_pages.py::test_page_routes_return_html[/] PASSED           [  4%]
tests/test_ui_pages.py::test_page_routes_return_html[/employees] PASSED  [  8%]
tests/test_ui_pages.py::test_page_routes_return_html[/attendance] PASSED [ 13%]
tests/test_ui_pages.py::test_page_routes_return_html[/recognition] PASSED [ 17%]
tests/test_ui_pages.py::test_page_routes_return_html[/attendance/unknown] PASSED [ 21%]
tests/test_ui_pages.py::test_page_routes_return_html[/cameras] PASSED    [ 26%]
tests/test_ui_pages.py::test_dashboard_exposes_current_view_to_the_base_template PASSED [ 30%]
tests/test_ui_pages.py::test_employee_registration_uses_the_navigation_view_identifier PASSED [ 34%]
tests/test_ui_pages.py::test_shared_shell_exposes_localized_primary_navigation_once PASSED [ 39%]
tests/test_ui_pages.py::test_shared_shell_provides_theme_control_and_evidence_modal PASSED [ 43%]
tests/test_ui_pages.py::test_dashboard_renders_operational_sections_for_the_current_database_state PASSED [ 47%]
tests/test_ui_pages.py::test_dashboard_embeds_chart_data_and_never_exposes_debug_evidence_paths PASSED [ 52%]
tests/test_ui_pages.py::test_dashboard_excludes_direction_undetermined_rows_from_metrics_and_charts PASSED [ 56%]
tests/test_ui_pages.py::test_dashboard_table_is_limited_to_the_most_recent_recorded_rows PASSED [ 60%]
tests/test_ui_pages.py::test_employee_roster_filters_round_trip_and_uses_grouped_enrollment_counts PASSED [ 65%]
tests/test_ui_pages.py::test_employee_detail_uses_actual_enrollment_count_and_evidence_empty_states PASSED [ 69%]
tests/test_ui_pages.py::test_attendance_filters_round_trip_in_sql_once_and_render_evidence PASSED [ 73%]
tests/test_ui_pages.py::test_attendance_normalizes_an_inverted_date_range PASSED [ 78%]
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params0] PASSED [ 82%]
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params1] PASSED [ 86%]
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params2] PASSED [ 91%]
tests/test_ui_pages.py::test_attendance_rejects_invalid_or_overlong_date_ranges[params3] PASSED [ 95%]
tests/test_ui_pages.py::test_attendance_export_applies_the_visible_list_filters PASSED [100%]

============================= 23 passed in 11.00s =============================
```

### Review

- The HTML list and CSV/API call the same joined predicate builder and deterministic ordering helper; no dynamic SQL strings are built.
- The multi-row regression asserts a fixed two SQL statements per page render, not a query per employee.
- CSV filtering is validated at the response boundary, while the legacy one-day filename and response remain covered.
- No deferred minor or attendance-transition/model work was changed.
