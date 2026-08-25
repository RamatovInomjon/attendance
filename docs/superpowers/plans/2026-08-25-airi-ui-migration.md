# AIRI UI/UX Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver an Uzbek-first AIRI-style administration interface for the two-camera attendance system without changing its computer-vision or attendance algorithms.

**Architecture:** Keep FastAPI, Jinja, Bootstrap, the existing database, and presentation adapters. Centralize visual behavior in shared CSS/JS; page routes provide presentation-ready data and templates remain server-rendered, with the current live JSON polling retained.

**Tech Stack:** Python, FastAPI, SQLAlchemy, Jinja2, Bootstrap 5.3, Bootstrap Icons, Chart.js, vanilla JavaScript, pytest, FastAPI TestClient.

**Spec:** `docs/AIRI_UI_MIGRATION_SPEC.md`

## Global Constraints

- Do not modify recognition thresholds, tracking, direction, attendance transitions, inference, recording, debug capture, gallery embeddings, or `app/core/`.
- Dashboard and live pages show quality-best evidence; `why_*` score-best files remain debug-only.
- Show real database/runtime values or explicit unavailable states—never fake live values.
- Do not copy private code, credentials, personal records, or proprietary assets from the reference platform.
- Support widths from 320 px upward without navigation/content overlap.
- Preserve current URLs and API contracts.
- There is no `.git` in this directory; use verification checkpoints instead of commits.

## File Responsibilities

- `templates/base.html`: shell, localized navigation, theme control, global evidence modal.
- `static/css/main.css`: theme tokens, components, responsive behavior.
- `static/js/main.js`: theme persistence, mobile navigation, evidence modal, date shortcuts.
- `app/api/pages.py`: page queries, filters, active-route context, health summaries.
- `app/web/viewmodels.py`: presentation-only evidence and enrollment fields.
- Page templates: dashboard, employees, attendance, live, enrollment, unknowns, cameras.
- `tests/test_ui_pages.py`: route, filter, empty-state, and UI-contract regression tests.

---

### Task 1: UI Contract Tests and Shared Context

**Files:** Create `tests/test_ui_pages.py`; modify `app/api/pages.py`, `app/web/django_compat.py`.

**Interfaces:** Produce `render(name: str, *, request=None, current_view: str = "", **ctx) -> HTMLResponse`; every page receives `current_view`, `request`, and `today`.

- [ ] Add the baseline test:

```python
from fastapi.testclient import TestClient
from app.api.main import app
client = TestClient(app)

def test_primary_ui_routes_render():
    for path in ("/", "/employees", "/attendance", "/recognition", "/attendance/unknown", "/cameras"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert "<!DOCTYPE html>" in response.text
```

- [ ] Run `/home/inomjon/anaconda3/envs/yolo/bin/python -m pytest tests/test_ui_pages.py -v`; record baseline failures.
- [ ] Make `render` own shared context and pass explicit identifiers (`dashboard:home`, `employees:list`, `attendance:list`, `recognition:live`, `attendance:unknown`, `camera:settings`) from routes.
- [ ] Assert the dashboard emits `data-current-view="dashboard:home"`.
- [ ] Re-run the test; expect PASS. Record checkpoint.

### Task 2: Design System and Responsive Shell

**Files:** Modify `templates/base.html`, `static/css/main.css`, `static/js/main.js`, `tests/test_ui_pages.py`.

**Interfaces:** Produce `[data-theme]`, `.app-nav`, `.airi-card`, `.metric-card`, `.status-chip`, `.data-table`, and `window.AiriUI.openEvidence({src,title,subtitle})`.

- [ ] Test for localized links `Boshqaruv`, `Xodimlar`, `Davomat`, `Jonli kuzatuv`, `Kameralar`, plus `id="themeToggle"`.
- [ ] Replace the navbar with semantic desktop/mobile markup, neutral local branding, explicit active links, and pipeline-health link.
- [ ] Define dark/light tokens for background, surfaces, border, text, muted, accent, semantic colors, radius, and shadow; replace hard-coded white component backgrounds.
- [ ] Add responsive rules at 991.98, 767.98, and 479.98 px. The collapsed menu must remain in document flow.
- [ ] Persist `airi-theme` in localStorage, update `aria-pressed`, dispatch `airi:themechange`, close mobile navigation after selection, and implement the shared evidence modal.
- [ ] Run UI tests and `rg -n 'style=".*(margin|background)' templates/base.html`; expect no layout-critical inline shell styles.

### Task 3: AIRI Operational Dashboard

**Files:** Modify `app/api/pages.py`, `templates/dashboard/index.html`, `tests/test_ui_pages.py`.

**Interfaces:** Produce `on_time_today`, `late_arrivals`, `absent_today`, `camera_health`, `recent_events`, `weekly_chart_data`, and `monthly_chart_data`.

- [ ] Test headings `Kunlik davomat`, `Tezkor amallar`, `So'nggi tanishlar`, and `Tizim holati`; require evidence markup or a localized empty state.
- [ ] Query seven daily and six monthly attendance aggregates. Build camera health from database cameras plus `runtime.workers`; missing workers are explicitly offline.
- [ ] Render welcome metrics, daily attendance, quick actions, recent quality-best evidence, today's attendance table, charts, and pipeline health using shared components.
- [ ] Feed Chart.js only through `json_script`; update chart colors after `airi:themechange`.
- [ ] Run UI tests with populated and empty data states; expect PASS.

### Task 4: Employee Roster and Detail

**Files:** Modify `app/api/pages.py`, `app/web/viewmodels.py`, `templates/employees/list.html`, `templates/employees/detail.html`, `tests/test_ui_pages.py`.

**Interfaces:** Extend `EmployeeVM` with `enrollment_count: int = 0` and `enrollment_state: str = "not_enrolled"`; preserve `query` and `department` filters.

- [ ] Test `/employees?query=Inomjon&department=AI` returns controls with both selected values and `Xodimlar ro'yxati`.
- [ ] Inspect the embedding table in `app/db/models.py`; aggregate counts by employee in one grouped query (no N+1 query).
- [ ] Build the searchable/filterable roster with enrollment badges, view actions, horizontal mobile scrolling, and localized empty state. Do not add unsupported deletion.
- [ ] Build employee detail with metadata, enrollment status, attendance history, and IN/OUT evidence previews.
- [ ] Run tests and inspect SQL logging for N+1 behavior; expect PASS.

### Task 5: Functional Attendance and Evidence

**Files:** Modify `app/api/pages.py`, `templates/attendance/list.html`, `static/js/main.js`, `tests/test_ui_pages.py`.

**Interfaces:** `attendance_list` accepts `query`, `department`, `start_date`, `end_date`, `status`; produces departments, filtered records, and retained filter values.

- [ ] Test a request with `start_date=2026-08-01`, `end_date=2026-08-25`, `query=Inomjon`; assert values round-trip and status is 200.
- [ ] Parse ISO dates, cap ranges at 366 days, handle inverted ranges consistently, join Employee once, and apply filters in SQL.
- [ ] Render reference-style filters, 15-day/month shortcuts, export preserving query string, responsive evidence table, status chips, and empty state.
- [ ] Use data attributes plus delegated events for the shared evidence modal; remove inline JavaScript arguments.
- [ ] Run tests including malformed dates; expect controlled 422/validation output, never HTTP 500.

### Task 6: Two-Camera Live Console

**Files:** Modify `app/api/pages.py`, `templates/recognition/live.html`, `static/js/camera_stream.js`, `tests/test_ui_pages.py`.

**Interfaces:** Camera cards expose `data-camera-id`, `data-camera-role`, `data-stream-canvas`, and `data-stream-state`; retain `/recognition/logs` and current stream transports.

- [ ] Test for `Kirish kamerasi`, `Chiqish kamerasi`, `Kamera FPS`, and `Algoritm FPS`.
- [ ] Normalize each camera's role, stream FPS, processed FPS, latency, drops, resolution, state, and last-frame time; unavailable workers become offline.
- [ ] Render responsive IN/OUT cards, camera metrics, attendance summary, quality-best recent recognitions, unknown activity, and pipeline errors.
- [ ] Change only JavaScript selectors/render targets needed by the markup; retain decoder/poll intervals, cap feed at 30 nodes, and use delegated evidence clicks.
- [ ] Run UI tests and `curl -fsS http://127.0.0.1:8000/recognition/logs`; require `employees`, `events`, `unknown_attempts`, and `stats`.

### Task 7: Enrollment, Unknowns, and Camera Diagnostics

**Files:** Modify `templates/employees/register.html`, `templates/attendance/unknown.html`, `templates/camera/settings.html`, `app/api/pages.py`, `tests/test_ui_pages.py`.

**Interfaces:** Preserve all enrollment input names/endpoints. Unknown cards consume current sighting fields. Camera context separates `configured`, `observed`, `pipeline`, and `drift`.

- [ ] Test specialized pages contain localized enrollment, unknown, and NVR-warning labels.
- [ ] Restyle enrollment as identity metadata then face capture/quality validation while preserving every existing submission contract and element needed by current scripts.
- [ ] Render unknown best snapshot, camera, first/last seen, frame count, preview, and empty state; do not invent assignment mutations.
- [ ] Present configured camera identity separately from observed RTSP/pipeline metrics and explain that NVR-owned encoder values may drift.
- [ ] Run UI tests with cameras online/offline; expect PASS.

### Task 8: Full Verification and Documentation

**Files:** Modify `docs/UI.md`, `docs/OPERATIONS.md`; verify all changed files.

**Interfaces:** Produce documented URLs, evidence semantics, filters, themes, health indicators, and NVR limitation.

- [ ] Run `/home/inomjon/anaconda3/envs/yolo/bin/python -m pytest -v`; expect PASS.
- [ ] Capture `curl -fsS http://127.0.0.1:8000/api/health` before/after restart; pipeline errors must not increase because of UI changes.
- [ ] Browser-check `/`, `/employees`, `/attendance`, `/recognition`, `/attendance/unknown`, `/employees/add`, and `/cameras` at approximately 1440×900.
- [ ] Check the same pages at 390×844 and 320×700: menu in flow, one-column cards, horizontally scrollable tables, fitting dialogs, reachable actions.
- [ ] Toggle themes, reload to prove persistence, test reduced-motion, and inspect browser console for errors.
- [ ] Verify a real recent event uses the same quality-best evidence on dashboard/live and never a `why_*` file.
- [ ] Document operator behavior and NVR ownership in `docs/UI.md` and `docs/OPERATIONS.md`.
- [ ] Record final changed-file list, tests, viewport results, health response, and limitations. If Git is later initialized, commit as `feat(ui): add AIRI-style attendance interface`.

## Execution Notes

- Existing templates contain large local CSS/JS blocks. Move reusable rules into shared files incrementally; avoid an unrelated wholesale rewrite.
- Before enrollment edits, inventory current IDs, field names, and endpoints and retain compatibility.
- Before changing `camera_stream.js`, capture current stream and `/recognition/logs` behavior as a baseline.

