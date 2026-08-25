# Task 4: Employee Roster and Detail Experience

Work in `/home/inomjon/projectAI/face_rec/face_recognition_airi`. Build on accepted shared shell/dashboard. Do not implement attendance/live/enrollment form redesigns.

## Files

Modify `app/api/pages.py`, `app/web/viewmodels.py`, `templates/employees/list.html`, `templates/employees/detail.html`, `tests/test_ui_pages.py`, and only necessary reusable CSS additions.

## Requirements

Extend `EmployeeVM` with backwards-compatible defaults `enrollment_count: int = 0` and `enrollment_state: str = "not_enrolled"`. Inspect `app/db/models.py` for the real face embedding table/relationship. Employee list must aggregate enrollment counts for all listed employees in one grouped query or joined subquery—no per-row count queries. Detail must compute the selected employee's actual embedding count, fixing any existing incorrect count.

`/employees` must support and retain `query` (matching full name or external ID) and `department`. Render Uzbek AIRI-style roster heading `Xodimlar ro'yxati`, search, department select, reset, columns for identity, department, position, phone, enrollment status, and view action. Use responsive horizontal scrolling and a localized empty state. Do not render unsupported delete controls.

Employee detail must show identity metadata, real enrollment count/state, recent attendance history, and check-in/check-out evidence using shared `AiriUI.openEvidence` data attributes. Missing images must have explicit empty states. Preserve existing detail URL and view-model field compatibility.

Tests must verify filter values round-trip, name/external-ID query behavior with isolated data where practical, enrollment fields/count logic, no delete action, detail evidence/empty state, and all existing tests. Avoid mutating production data: use transaction rollback or focused helper/unit tests.

Run `/home/inomjon/anaconda3/envs/yolo/bin/python -m pytest tests/test_ui_pages.py -v` and report exact output.

## Constraints

- No N+1 query.
- No core CV, schema migration, enrollment algorithm, route, or attendance-service changes.
- No fake profile/enrollment values.
- Do not dispatch subagents.
- Write `.superpowers/sdd/2026-08-25-airi-ui-migration/task-4-report.md` and self-review.

