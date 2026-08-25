# Task 3 Report: AIRI Operational Dashboard

## Status

Completed. The dashboard now renders real database attendance aggregates, recent recognition evidence, configured-camera/runtime health, and server-fed weekly/monthly charts on the accepted shared shell.

## Changed files

- `app/api/pages.py`
  - Added bounded grouped queries for seven daily and six monthly attendance aggregates, filling empty dates/months with zero-valued real aggregate results.
  - Supplies `on_time_today`, `late_arrivals`, `absent_today`, `camera_health`, `recent_events`, `weekly_chart_data`, and `monthly_chart_data` to the dashboard.
  - Applies the existing local 09:00 late rule (`check_in > 09:00`) and calculates daily metrics from active employees, consistent with the total-employee denominator.
  - Merges enabled DB cameras with read-only `runtime.workers` statistics. A configured camera without a worker is explicitly `Offline` / `Ishchi mavjud emas`; a worker with an unavailable stream is explicitly `Mavjud emas`.
  - Does not start, stop, reload, or otherwise mutate runtime workers.
- `templates/dashboard/index.html`
  - Rebuilt the dashboard with Uzbek welcome, daily metrics, quick links, recent recognition cards, today’s IN/OUT evidence table, system health, and weekly/monthly charts.
  - Uses `EventVM.snapshot` and attendance snapshots only for evidence. Debug score-best paths are neither constructed nor displayed.
  - Adds localized empty/unavailable states, including `Hozircha tanishlar yo'q`.
  - Embeds chart payloads only with `json_script`; Chart.js reads CSS variables and updates its existing instances on `airi:themechange` without creating duplicates.
- `static/css/main.css`
  - Added dashboard-only welcome, metric, evidence-preview, and chart sizing helpers that compose with the accepted shared design system.
- `tests/test_ui_pages.py`
  - Added current-database dashboard rendering contracts for all required section headings and evidence-or-empty behavior.
  - Added checks for both JSON chart payloads and the absence of debug evidence path references.

## Test-first record

Added the two dashboard response-contract tests before implementation. The focused test run then failed only because the old dashboard lacked the required Uzbek sections and JSON chart elements. After implementation, the same focused suite passed.

## Verification

The required command was run outside the filesystem sandbox because the project’s SQLite-backed TestClient requests stall inside it (the same accepted Task 2 environment limitation). Exact clean output:

```text
/home/inomjon/anaconda3/envs/yolo/bin/python -m pytest tests/test_ui_pages.py -v
============================= test session starts ==============================
platform linux -- Python 3.10.16, pytest-9.1.1, pluggy-1.6.0 -- /home/inomjon/anaconda3/envs/yolo/bin/python
cachedir: .pytest_cache
rootdir: /home/inomjon/projectAI/face_rec/face_recognition_airi
plugins: hydra-core-1.3.2, anyio-4.12.0
collecting ... collected 12 items

tests/test_ui_pages.py::test_page_routes_return_html[/] PASSED           [  8%]
tests/test_ui_pages.py::test_page_routes_return_html[/employees] PASSED  [ 16%]
tests/test_ui_pages.py::test_page_routes_return_html[/attendance] PASSED [ 25%]
tests/test_ui_pages.py::test_page_routes_return_html[/recognition] PASSED [ 33%]
tests/test_ui_pages.py::test_page_routes_return_html[/attendance/unknown] PASSED [ 41%]
tests/test_ui_pages.py::test_page_routes_return_html[/cameras] PASSED    [ 50%]
tests/test_ui_pages.py::test_dashboard_exposes_current_view_to_the_base_template PASSED [ 58%]
tests/test_ui_pages.py::test_employee_registration_uses_the_navigation_view_identifier PASSED [ 66%]
tests/test_ui_pages.py::test_shared_shell_exposes_localized_primary_navigation_once PASSED [ 75%]
tests/test_ui_pages.py::test_shared_shell_provides_theme_control_and_evidence_modal PASSED [ 83%]
tests/test_ui_pages.py::test_dashboard_renders_operational_sections_for_the_current_database_state PASSED [ 91%]
tests/test_ui_pages.py::test_dashboard_embeds_chart_data_and_never_exposes_debug_evidence_paths PASSED [100%]

============================== 12 passed in 6.23s ==============================
```

Additional checks: Python compilation and Jinja template parsing passed. The rendered-page contracts confirm the weekly/monthly JSON scripts and no `why_` reference.

## Self-review

- Checked that chart data comes from exactly one grouped daily query and one grouped monthly query, rather than a query per label.
- Checked that time values are localized by the existing `DailyVM`/`EventVM` adapters before template formatting.
- Checked that active employee attendance is the only data used for the summary metric denominator and numerators.
- Checked that camera health reads `runtime.workers` only and explicitly represents missing workers/streams.
- Checked that all dashboard evidence URLs come from quality-best event or attendance snapshots, with no debug evidence construction.
- Checked chart initialization stores one instance per chart and color refreshes call `update()` on those instances.
- Preserved Tasks 1–2 tests and did not initialize Git or dispatch subagents.

## Concerns

- Focused route/UI tests passed, but this Task 3 scope did not include a browser-driven visual viewport pass; that remains part of the migration’s later full-verification task.
- The accepted environment requires the focused SQLite-backed pytest command to run outside the filesystem sandbox; its clean result is recorded above.

## Fix round 1/5: recorded presence and dashboard row bound

### Root cause

The dashboard counted every `DailyAttendance` row for metrics and both chart queries. A row created before direction/attendance transition resolution can have neither `check_in_time` nor `check_out_time`, so it was incorrectly counted as presence. The dashboard also called the shared `_daily_rows` helper without a limit; changing that helper’s default would have changed the full `/attendance` page.

### Fix details

- Added `_recorded_presence_clause()` as the shared SQL predicate:
  `check_in_time IS NOT NULL OR check_out_time IS NOT NULL`.
- Applied that predicate to dashboard daily rows plus the bounded weekly and monthly grouped aggregate queries. Dashboard metrics use a separate all-day active-record query with that same predicate, so the present/on-time/late/absent definition is consistent with the charts without being constrained by the table limit.
- Extended `_daily_rows` with opt-in `recorded_only`, `limit`, and `recent_first` parameters. Its original default behavior remains unchanged for attendance pages.
- Added documented `DASHBOARD_DAILY_ROWS_LIMIT = 50` and `_dashboard_daily_rows()`, which selects only the 50 newest recorded rows ordered by `COALESCE(check_out_time, check_in_time) DESC, id DESC`. The existing dashboard attendance link continues to expose the full attendance page.
- Added isolated in-memory integration coverage that renders the real dashboard handler and proves: (1) transition-undetermined rows are excluded from the displayed attendance ratio and both chart payloads; (2) a busy dashboard table contains only the 50 newest recorded rows.

### Verification

The required command was run outside the filesystem sandbox because its SQLite-backed TestClient requests stall inside the sandbox. Exact clean output:

```text
/home/inomjon/anaconda3/envs/yolo/bin/python -m pytest tests/test_ui_pages.py -v
============================= test session starts ==============================
platform linux -- Python 3.10.16, pytest-9.1.1, pluggy-1.6.0 -- /home/inomjon/anaconda3/envs/yolo/bin/python
cachedir: .pytest_cache
rootdir: /home/inomjon/projectAI/face_rec/face_recognition_airi
plugins: hydra-core-1.3.2, anyio-4.12.0
collecting ... collected 14 items

tests/test_ui_pages.py::test_page_routes_return_html[/] PASSED           [  7%]
tests/test_ui_pages.py::test_page_routes_return_html[/employees] PASSED  [ 14%]
tests/test_ui_pages.py::test_page_routes_return_html[/attendance] PASSED [ 21%]
tests/test_ui_pages.py::test_page_routes_return_html[/recognition] PASSED [ 28%]
tests/test_ui_pages.py::test_page_routes_return_html[/attendance/unknown] PASSED [ 35%]
tests/test_ui_pages.py::test_page_routes_return_html[/cameras] PASSED    [ 42%]
tests/test_ui_pages.py::test_dashboard_exposes_current_view_to_the_base_template PASSED [ 50%]
tests/test_ui_pages.py::test_employee_registration_uses_the_navigation_view_identifier PASSED [ 57%]
tests/test_ui_pages.py::test_shared_shell_exposes_localized_primary_navigation_once PASSED [ 64%]
tests/test_ui_pages.py::test_shared_shell_provides_theme_control_and_evidence_modal PASSED [ 71%]
tests/test_ui_pages.py::test_dashboard_renders_operational_sections_for_the_current_database_state PASSED [ 78%]
tests/test_ui_pages.py::test_dashboard_embeds_chart_data_and_never_exposes_debug_evidence_paths PASSED [ 85%]
tests/test_ui_pages.py::test_dashboard_excludes_direction_undetermined_rows_from_metrics_and_charts PASSED [ 92%]
tests/test_ui_pages.py::test_dashboard_table_is_limited_to_the_most_recent_recorded_rows PASSED [100%]

============================== 14 passed in 7.11s ==============================
```

### Self-review

- Confirmed the shared presence predicate is used by both aggregate queries and the dashboard-only daily rows query.
- Confirmed dashboard metrics and charts use the shared predicate across all active records, independently from the table slice.
- Confirmed `/attendance` still calls `_daily_rows(s, day)` with its unbounded default behavior.
- Confirmed recent ordering has a timestamp fallback and `id DESC` tie-breaker.

### Metric/table separation correction

During the first fix-round self-review, the bounded table was found to be feeding the summary metrics. That would undercount a day with more than 50 recorded employees. Added `_dashboard_attendance_metrics()` to count all active recorded rows and retrieve only all-day check-in timestamps needed for the localized 09:00 late calculation. The 50-row table remains display-only. The busy-day integration test now asserts both `Davomat: 51/51 (100%)` and exactly 50 newest table rows.

Final required command/output after this correction:

```text
/home/inomjon/anaconda3/envs/yolo/bin/python -m pytest tests/test_ui_pages.py -v
============================= test session starts ==============================
platform linux -- Python 3.10.16, pytest-9.1.1, pluggy-1.6.0 -- /home/inomjon/anaconda3/envs/yolo/bin/python
cachedir: .pytest_cache
rootdir: /home/inomjon/projectAI/face_rec/face_recognition_airi
plugins: hydra-core-1.3.2, anyio-4.12.0
collecting ... collected 14 items

tests/test_ui_pages.py::test_page_routes_return_html[/] PASSED           [  7%]
tests/test_ui_pages.py::test_page_routes_return_html[/employees] PASSED  [ 14%]
tests/test_ui_pages.py::test_page_routes_return_html[/attendance] PASSED [ 21%]
tests/test_ui_pages.py::test_page_routes_return_html[/recognition] PASSED [ 28%]
tests/test_ui_pages.py::test_page_routes_return_html[/attendance/unknown] PASSED [ 35%]
tests/test_ui_pages.py::test_page_routes_return_html[/cameras] PASSED    [ 42%]
tests/test_ui_pages.py::test_dashboard_exposes_current_view_to_the_base_template PASSED [ 50%]
tests/test_ui_pages.py::test_employee_registration_uses_the_navigation_view_identifier PASSED [ 57%]
tests/test_ui_pages.py::test_shared_shell_exposes_localized_primary_navigation_once PASSED [ 64%]
tests/test_ui_pages.py::test_shared_shell_provides_theme_control_and_evidence_modal PASSED [ 71%]
tests/test_ui_pages.py::test_dashboard_renders_operational_sections_for_the_current_database_state PASSED [ 78%]
tests/test_ui_pages.py::test_dashboard_embeds_chart_data_and_never_exposes_debug_evidence_paths PASSED [ 85%]
tests/test_ui_pages.py::test_dashboard_excludes_direction_undetermined_rows_from_metrics_and_charts PASSED [ 92%]
tests/test_ui_pages.py::test_dashboard_table_is_limited_to_the_most_recent_recorded_rows PASSED [100%]

============================== 14 passed in 5.43s ==============================
```
