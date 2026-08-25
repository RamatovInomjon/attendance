# Task 1: UI Contract Tests and Shared Context

Read `docs/AIRI_UI_MIGRATION_SPEC.md` only for binding project-wide constraints. Do not implement later plan tasks.

## Requirements

Create `tests/test_ui_pages.py`; modify `app/api/pages.py` and, only if required, `app/web/django_compat.py`.

Produce `render(name: str, *, request=None, current_view: str = "", **ctx) -> HTMLResponse`. Every page must receive `current_view`, `request`, and `today` without breaking current routes.

Add a FastAPI TestClient route-smoke test covering `/`, `/employees`, `/attendance`, `/recognition`, `/attendance/unknown`, and `/cameras`; each must return 200 and HTML. Add a dashboard assertion for `data-current-view="dashboard:home"`. If the current base template lacks this attribute, add the smallest compatible attribute required for this Task 1 contract; do not redesign the shell yet.

Pass explicit identifiers from page routes: `dashboard:home`, `employees:list`, `attendance:list`, `recognition:live`, `attendance:unknown`, and `camera:settings`.

Run `/home/inomjon/anaconda3/envs/yolo/bin/python -m pytest tests/test_ui_pages.py -v` and report exact results.

## Constraints

- Do not modify recognition/core pipeline behavior.
- Do not redesign UI in this task.
- Do not dispatch subagents.
- Self-review the diff and write the report to `.superpowers/sdd/2026-08-25-airi-ui-migration/task-1-report.md`.
- Report status as DONE, DONE_WITH_CONCERNS, NEEDS_CONTEXT, or BLOCKED; list files changed and test summary.

