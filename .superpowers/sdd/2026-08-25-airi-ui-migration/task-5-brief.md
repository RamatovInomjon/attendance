# Task 5: Functional Attendance Filters and Evidence Review

Work in `/home/inomjon/projectAI/face_rec/face_recognition_airi`. Build on accepted Tasks 1-4.

## Files

Modify `app/api/pages.py`, `templates/attendance/list.html`, `static/js/main.js`, `tests/test_ui_pages.py`, and necessary shared CSS only.

## Route/Data Requirements

`attendance_list` must accept `query`, `department`, `start_date`, `end_date`, and `status`; retain values and provide `departments`. Parse ISO dates safely. Default to the selected business day when no range is supplied. Handle inverted ranges consistently with visible validation or normalization. Reject/cap ranges above 366 days. Query `DailyAttendance` joined with `Employee` once and apply name/external-ID, department, status, and date filters in SQL. Use deterministic newest-first ordering. Avoid N+1 queries.

Preserve `/attendance`, `/attendance/history`, export URL/API contract, and existing view-model behavior. Malformed dates must produce controlled 422/validation output, never 500.

## UI Requirements

Render Uzbek AIRI-style heading and controls: search, department, start/end dates, status, reset, 15-day and one-month shortcuts, and export link preserving active query parameters. Table columns: date, employee, department, check-in, check-out, worked time, status, IN/OUT evidence, details. Use shared status chips and `AiriUI.openEvidence` via safe data attributes/delegated handlers—no inline JS string interpolation. Use responsive horizontal scrolling and localized empty state.

Date shortcut buttons calculate local calendar dates, populate native date fields, and submit. Do not duplicate the generic evidence modal.

Tests must verify filter values round-trip, SQL filtering with isolated data/helper tests where practical, malformed/inverted/overlong ranges, query-string-preserving export, no inline `onclick` evidence handlers, and existing suite.

Run `/home/inomjon/anaconda3/envs/yolo/bin/python -m pytest tests/test_ui_pages.py -v`; exact output in report.

## Constraints

- No attendance transition/core/model/schema changes.
- No fake evidence or statistics.
- Do not dispatch subagents.
- Self-review and write `.superpowers/sdd/2026-08-25-airi-ui-migration/task-5-report.md`.

