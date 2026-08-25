# AIRI UI/UX Migration Specification

Date: 2026-08-25  
Status: Approved design, pending implementation

## Objective

Refactor the local FastAPI/Jinja interface so it follows the visual language and workflows of `faceid.airi.uz`, while preserving the local project's improved two-camera recognition, tracking, evidence, and diagnostics features.

The migration is a presentation-layer change. It must not alter recognition thresholds, tracking, direction decisions, attendance state transitions, model inference, recording, or debug capture.

## Design Direction

The interface will use an original implementation inspired by the reference platform:

- dark navy application background;
- translucent blue cards with restrained borders and shadows;
- violet/blue active-navigation treatment;
- compact summary cards and colored status tiles;
- rounded filters, tables, buttons, badges, and image previews;
- Uzbek-first interface text;
- user-selectable dark and light modes, persisted locally;
- responsive behavior that fixes the reference site's mobile navigation overlap.

No private source code or proprietary assets from the reference platform will be copied. Existing local assets will be reused where suitable, and otherwise the implementation will use CSS, Bootstrap Icons, and neutral branding.

## Information Architecture

The primary navigation will contain:

1. **Boshqaruv** — dashboard and daily operational summary.
2. **Xodimlar** — employee roster and employee details.
3. **Davomat** — attendance records, date filters, evidence, and exports.
4. **Jonli kuzatuv** — the local system's two-camera live recognition page.
5. **Ro'yxatdan o'tkazish** — employee creation and face enrollment.
6. **Noma'lumlar** — unknown-person review queue.
7. **Kameralar** — RTSP configuration and pipeline health.

Unsupported reference-only areas such as doctorants, consultants, visitors, project members, departments, and administrator management will not be presented as working links. They may be added later when corresponding domain models and routes exist.

## Shared Application Shell

`templates/base.html` will become the single AIRI-style shell:

- compact desktop navigation with a clearly highlighted active route;
- collapsible mobile menu that participates in document flow and never overlays page content;
- neutral local product logo and Uzbek product title;
- dark/light-mode control;
- pipeline-health indicator linking to live monitoring;
- consistent maximum content width and page spacing;
- accessible focus states and keyboard-operable menus.

The current route name must be supplied by each page so active navigation is deterministic rather than inferred from fragile URL text.

## Dashboard

The dashboard will preserve the reference hierarchy and add operational recognition data:

- welcome card with date, total employees, and today's attendance ratio;
- daily attendance card showing present, on-time, late, and absent counts;
- quick-action cards for attendance, employees, daily report/export, and live monitoring;
- recent recognition events with the best-quality aligned face, name, camera, IN/OUT action, confidence, and event time;
- today's attendance table with check-in/check-out evidence;
- weekly and monthly charts;
- compact camera/pipeline health summary.

All dashboard values must come from the existing database and runtime workers. No placeholder statistics will be shown as real values.

## Employees

The employee list will use the reference platform's searchable, filterable table pattern while retaining the local schema:

- full name and external ID search;
- department filter;
- name, identifier, department, position, phone, enrollment state, and actions;
- responsive horizontal scrolling on small screens;
- clear empty state;
- employee detail page with profile data, enrollment information, attendance history, and evidence images.

Destructive actions will not be added unless supported by an authenticated backend route and explicit confirmation flow.

## Attendance

The attendance page will provide:

- employee/name search when supported by the route;
- department and date-range filters;
- quick ranges for 15 days and one month;
- export action using the existing export API;
- table columns for date, employee, department, check-in, check-out, net work time, status, evidence, and details;
- image preview modal showing the best display-quality face for IN and OUT events;
- unambiguous IN/OUT badges and confidence metadata where available.

The backend query will be extended only where required to make visible filters functional. Direction and attendance decision logic remains unchanged.

## Live Recognition

The existing recognition page remains a first-class local extension to the reference UX:

- side-by-side entrance and exit camera cards on wide screens;
- stacked camera cards on narrow screens;
- connection state, camera FPS, processed FPS, latency, and dropped-frame indicators;
- recent recognized people with best-quality aligned face and original evidence frame;
- confidence, quality, pose, camera role, and IN/OUT decision;
- unknown-person activity and pipeline error state;
- polling that updates content without page reload and without unbounded DOM growth.

The dashboard and live page must always display the quality-best snapshot. Score-best diagnostic frames remain available only in debug evidence.

## Enrollment

Registration will follow a clear two-stage workflow:

1. identity and employee metadata;
2. face capture/import, quality validation, embedding creation, and enrollment result.

The page will explain rejected captures using actionable quality reasons such as blur, face size, pose, alignment score, or duplicate identity risk. Existing enrollment APIs and the active AdaFace model remain authoritative.

## Unknown Review and Cameras

Unknown review will show grouped sightings with camera, time range, frame count, and best snapshot. Camera settings will distinguish:

- configuration reported by the camera;
- runtime stream statistics;
- recognition pipeline throughput;
- configuration drift warnings caused by an external NVR.

The UI must not imply that local software controls encoder settings that the NVR later overwrites.

## Responsive and Accessibility Requirements

- No overlap between navigation and content at widths from 320 px upward.
- Cards collapse to one column on phones.
- Tables remain usable through horizontal scrolling rather than compressed unreadable columns.
- Tap targets are at least approximately 44 px where practical.
- Text and status colors meet readable contrast in both themes.
- Status meaning is conveyed by text/icon as well as color.
- Images include useful alternative text and lazy loading where appropriate.
- Animations respect reduced-motion preferences.

## Implementation Boundaries

Primary files expected to change:

- `templates/base.html`
- `templates/dashboard/index.html`
- `templates/employees/list.html`
- `templates/employees/detail.html`
- `templates/employees/register.html`
- `templates/attendance/list.html`
- `templates/attendance/unknown.html`
- `templates/recognition/live.html`
- `templates/camera/settings.html`
- `static/css/main.css`
- `static/js/main.js`
- `app/api/pages.py`
- `app/web/viewmodels.py`

Recognition-core modules under `app/core/`, model files, gallery embeddings, camera readers, recording, and attendance transition services are out of scope.

## Verification

Implementation is complete only when:

1. all affected routes return HTTP 200 with real database data;
2. the dashboard, employees, attendance, live, unknown, registration, and camera pages render without browser console errors;
3. navigation works at desktop and mobile widths;
4. dark/light mode persists and does not flash an incorrect theme excessively;
5. evidence images load and dashboard/live views use the quality-best snapshot;
6. live polling continues to update without breaking cards or duplicating handlers;
7. the recognition service health and existing pipeline tests remain unchanged/passing;
8. templates handle empty databases, unavailable cameras, and missing snapshots gracefully;
9. no reference-platform credentials, data, or private assets are written into the repository.

## Delivery Sequence

1. Build shared tokens, shell, navigation, and theme behavior.
2. Rebuild dashboard and connect real operational data.
3. Rebuild employee and attendance workflows.
4. Rebuild live recognition, unknown review, registration, and camera pages.
5. Validate responsive behavior, route rendering, live updates, and recognition-pipeline isolation.

