# Task 3: AIRI Operational Dashboard

Work in `/home/inomjon/projectAI/face_rec/face_recognition_airi`. Build the dashboard on the accepted shared shell/design system. Do not implement later employee/attendance/live page redesigns.

## Files

Modify `app/api/pages.py`, `templates/dashboard/index.html`, and `tests/test_ui_pages.py`. Shared CSS may receive dashboard-only reusable component additions only when necessary; prefer existing `.airi-card`, `.metric-card`, `.status-chip`, `.data-table`.

## Data Contract

The dashboard must receive real values: `on_time_today: int`, `late_arrivals: int`, `absent_today: int`, `camera_health: list[dict]`, `recent_events: list[EventVM]`, `weekly_chart_data: list[dict]`, `monthly_chart_data: list[dict]`.

Use localized timestamps and the existing 09:00 lateness rule. Query seven daily attendance aggregates and six monthly aggregates with bounded aggregate queries, not one query per day/month if avoidable. Build camera health by merging enabled DB cameras with `runtime.workers` stats; absent workers must render explicit offline/unavailable state. Do not start/stop workers or mutate runtime.

## UI Contract

Use Uzbek headings: `Kunlik davomat`, `Tezkor amallar`, `So'nggi tanishlar`, and `Tizim holati`. Include:

- welcome/date card with total employees and attendance ratio;
- daily present/on-time/late/absent metrics;
- quick links to attendance, employees, live monitoring, and export/report where a real URL exists;
- recent recognition cards with employee name, camera, IN/OUT, time, confidence, and quality-best `EventVM.snapshot` evidence; never construct or show `why_*` paths;
- today's attendance table with IN/OUT times and evidence;
- weekly and monthly Chart.js charts fed only by `json_script` data;
- compact camera/pipeline status.

All values must be real or explicitly unavailable. Add localized empty states such as `Hozircha tanishlar yo'q`. Charts must use CSS-derived colors and respond to `airi:themechange` without duplicating chart instances.

Tests must cover section headings, page rendering with current real/empty DB state, evidence markup or localized empty state, chart JSON presence, and no `why_` reference. Preserve Task 1/2 tests.

Run `/home/inomjon/anaconda3/envs/yolo/bin/python -m pytest tests/test_ui_pages.py -v` and report exact clean output.

## Constraints

- No changes to core CV, attendance transition services, models, routes, or API contracts.
- Do not show fake statistics.
- Do not dispatch subagents.
- Self-review and write `.superpowers/sdd/2026-08-25-airi-ui-migration/task-3-report.md`.

