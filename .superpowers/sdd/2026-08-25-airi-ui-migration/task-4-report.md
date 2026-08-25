# Task 4 Report: Employee Roster and Detail Experience

## Status

Completed.

## Changed files

- `app/api/pages.py`
  - Added a single grouped `FaceEmbedding` count subquery to the active employee roster query; this prevents per-row enrollment count queries.
  - Added name-or-external-ID query matching and retained the selected department filter and its active department options.
  - Corrected the detail embedding count to count `face_embedding` rows for the selected employee.
- `app/web/viewmodels.py`
  - Added backwards-compatible `enrollment_count: int = 0` and `enrollment_state: str = "not_enrolled"` fields.
  - Maps a positive real count to `enrolled`, retaining existing `EmployeeVM.of(employee)` callers.
- `templates/employees/list.html`
  - Rebuilt the roster in Uzbek with search, department select, reset action, responsive AIRI data table, localized empty state, enrollment status/count, and view-only action.
  - Removed unsupported edit/delete controls.
- `templates/employees/detail.html`
  - Renders identity metadata, real enrollment state/count, the latest 30 attendance summaries, IN/OUT evidence buttons using `data-evidence-*` and `AiriUI.openEvidence`, plus explicit missing-image/evidence states.
  - Removed unsupported edit/delete controls.
- `static/css/main.css`
  - Added small reusable employee avatar, profile placeholder, and metadata helpers that compose with the accepted AIRI card/table styles.
- `tests/test_ui_pages.py`
  - Added isolated in-memory page-handler coverage for filter round trips, full-name/external-ID search behavior, grouped roster counts, real detail counts, evidence controls, empty states, and absent delete controls.
- `.self-eval-scores.jsonl`
  - Appended the task self-evaluation required by the workspace workflow.

## Test-first record

The two new focused tests were added before implementation. Their first run failed because the accepted roster did not have a department select/enrollment counts and the detail page did not render true enrollment/evidence states. After the query, view-model, template, and CSS changes, the same focused tests passed.

## Verification

`/home/inomjon/anaconda3/envs/yolo/bin/python -m py_compile app/api/pages.py app/web/viewmodels.py tests/test_ui_pages.py` completed successfully.

The required command stalls at the first SQLite-backed `TestClient` request inside the filesystem sandbox. It was rerun outside that sandbox and completed successfully. Exact output:

```text
/home/inomjon/anaconda3/envs/yolo/bin/python -m pytest tests/test_ui_pages.py -v
============================= test session starts ==============================
platform linux -- Python 3.10.16, pytest-9.1.1, pluggy-1.6.0 -- /home/inomjon/anaconda3/envs/yolo/bin/python
cachedir: .pytest_cache
rootdir: /home/inomjon/projectAI/face_rec/face_recognition_airi
plugins: hydra-core-1.3.2, anyio-4.12.0
collecting ... collected 16 items

tests/test_ui_pages.py::test_page_routes_return_html[/] PASSED           [  6%]
tests/test_ui_pages.py::test_page_routes_return_html[/employees] PASSED  [ 12%]
tests/test_ui_pages.py::test_page_routes_return_html[/attendance] PASSED [ 18%]
tests/test_ui_pages.py::test_page_routes_return_html[/recognition] PASSED [ 25%]
tests/test_ui_pages.py::test_page_routes_return_html[/attendance/unknown] PASSED [ 31%]
tests/test_ui_pages.py::test_page_routes_return_html[/cameras] PASSED    [ 37%]
tests/test_ui_pages.py::test_dashboard_exposes_current_view_to_the_base_template PASSED [ 43%]
tests/test_ui_pages.py::test_employee_registration_uses_the_navigation_view_identifier PASSED [ 50%]
tests/test_ui_pages.py::test_shared_shell_exposes_localized_primary_navigation_once PASSED [ 56%]
tests/test_ui_pages.py::test_shared_shell_provides_theme_control_and_evidence_modal PASSED [ 62%]
tests/test_ui_pages.py::test_dashboard_renders_operational_sections_for_the_current_database_state PASSED [ 68%]
tests/test_ui_pages.py::test_dashboard_embeds_chart_data_and_never_exposes_debug_evidence_paths PASSED [ 75%]
tests/test_ui_pages.py::test_dashboard_excludes_direction_undetermined_rows_from_metrics_and_charts PASSED [ 81%]
tests/test_ui_pages.py::test_dashboard_table_is_limited_to_the_most_recent_recorded_rows PASSED [ 87%]
tests/test_ui_pages.py::test_employee_roster_filters_round_trip_and_uses_grouped_enrollment_counts PASSED [ 93%]
tests/test_ui_pages.py::test_employee_detail_uses_actual_enrollment_count_and_evidence_empty_states PASSED [100%]

============================== 16 passed in 5.40s ==============================
```

## Self-review

- Confirmed the roster selects counts from one grouped `face_embedding` subquery, with a zero-count `COALESCE`, rather than counting embeddings per rendered employee.
- Confirmed detail counts `FaceEmbedding.id` for only the requested employee and continues to provide the legacy `embedding_count`, `records`, and `attendance_history` context fields.
- Confirmed `EmployeeVM` defaults preserve callers that do not supply enrollment data.
- Confirmed query and department values round-trip through the rendered form; query matching covers both full name and external ID using isolated data.
- Confirmed both templates use explicit missing image/evidence messages and contain no delete link or trash icon.
- Confirmed the detail evidence buttons use the shared `data-evidence-src` convention and invoke `window.AiriUI.openEvidence`.
- Confirmed no schema migration, CV/enrollment algorithm, route, or attendance-service code changed.

## Self-evaluation

**Task:** Implemented the AIRI employee roster and detail experience with grouped enrollment counts and evidence states.

**Ambition:** Medium — This was a bounded feature across the presentation query, compatibility adapter, two templates, and integration-style UI tests.

**Execution:** Strong — The implementation follows the real embedding schema, retains compatibility contracts, verifies the aggregate behavior with isolated data, and passes the complete required suite.

**Devil's Advocate:**

- Lower: The existing page patterns and shared evidence modal made the structure straightforward, and the scope did not require browser-driven visual testing.
- Higher: The work corrected a real erroneous database count while preventing an N+1 regression and retaining legacy template context at the same time.
- Resolution: Medium ambition with strong execution is appropriate; the lack of a browser viewport pass is a bounded verification concern, not a functional gap in the specified test scope.

**Score: 4/5** — Meaningful, well-verified presentation/data integration work with one environment-limited visual verification gap.

## Concerns

- The required suite passes outside the filesystem sandbox; inside it, the first SQLite-backed `TestClient` request stalls before a result, the same environment limitation documented by accepted Tasks 1–3.
- This task did not include a browser viewport/accessibility audit; responsive horizontal scrolling relies on the accepted shared `.data-table` interface.
