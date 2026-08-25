# Task 2: AIRI Design System and Responsive Application Shell

This task builds the shared visual foundation used by all later page tasks. Work only in `/home/inomjon/projectAI/face_rec/face_recognition_airi`.

## Requirements

Modify `templates/base.html`, `static/css/main.css`, `static/js/main.js`, and `tests/test_ui_pages.py`.

The shell must provide Uzbek navigation labels: `Boshqaruv`, `Xodimlar`, `Davomat`, `Jonli kuzatuv`, `Ro'yxatdan o'tkazish`, `Noma'lumlar`, and `Kameralar`. Use current URLs and `current_view` for active state. Add `id="themeToggle"`, a neutral local logo/product title, and a compact pipeline-health link to `/recognition`.

Produce reusable interfaces: `[data-theme]`, `.app-nav`, `.airi-card`, `.metric-card`, `.status-chip`, `.data-table`, `.evidence-modal`, and `window.AiriUI.openEvidence({src,title,subtitle})`.

Dark/light theme tokens must cover background, surface, elevated surface, border, text, muted text, accent, success, warning, danger, radius, and shadow. Existing shared components must use tokens instead of hard-coded white backgrounds where touched. Persist theme under localStorage key `airi-theme`, update toggle `aria-pressed`/label/icon, and dispatch `airi:themechange`.

Use a semantic Bootstrap desktop/mobile navigation. At <=991.98 px it must collapse in document flow, never overlap content, and close after a nav selection. Add responsive rules at 991.98, 767.98, and 479.98 px. Add accessible focus-visible styles, status text/icons beyond color, 44px practical tap targets, and reduced-motion handling.

Add one shared Bootstrap evidence modal in the base shell. `AiriUI.openEvidence` populates image src/alt, title, subtitle, handles missing src safely, and opens the modal without inline event strings.

Retain CDN dependencies and all Jinja URL helpers. Do not change page-specific templates, recognition/core behavior, routes, or API contracts in this task. Preserve the existing `data-current-view` contract.

Tests must assert localized links/URLs, theme toggle, shared modal, current-view attribute, and no duplicate primary navigation. Run `/home/inomjon/anaconda3/envs/yolo/bin/python -m pytest tests/test_ui_pages.py -v` and report exact output.

## Constraints

- Implement original styling; do not copy private/proprietary source or assets.
- Support 320px upward with no shell overlap.
- Do not dispatch subagents.
- Self-review and write `.superpowers/sdd/2026-08-25-airi-ui-migration/task-2-report.md`.
- Return status, files changed, one-line tests, concerns.

