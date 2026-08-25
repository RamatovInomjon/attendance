# Task 2 Report: AIRI Design System and Responsive Application Shell

## Status

Completed the shared shell/design-system scope only. No page-specific template, route, recognition-core, or API-contract changes were made.

## Changed files

- `templates/base.html`
  - Replaced the old primary navigation with one semantic Bootstrap navigation landmark (`#airiPrimaryNavigation`) using the requested Uzbek labels and existing Jinja URL helpers.
  - Preserved the body `data-current-view` contract and applied active state from `current_view`.
  - Added neutral AIRI branding, the compact `/recognition` pipeline-health link, accessible theme control (`#themeToggle`), and the shared `#evidenceModal`.
  - Kept Bootstrap and Chart.js CDN dependencies intact.
- `static/css/main.css`
  - Added light/dark `[data-theme]` tokens for background, surface, elevated surface, border, text, muted text, accent, success, warning, danger, radius, and shadow.
  - Added reusable `.app-nav`, `.airi-card`, `.metric-card`, `.status-chip`, `.data-table`, and `.evidence-modal` interfaces.
  - Added responsive shell rules at 991.98px, 767.98px, and 479.98px, focus-visible styling, practical 44px shell controls, textual/icon status support, and reduced-motion overrides.
- `static/js/main.js`
  - Added persisted `airi-theme` theme behavior, updated toggle state/label/icon, and `airi:themechange` dispatch.
  - Added `window.AiriUI.openEvidence({src, title, subtitle})`, including safe missing-image handling and Bootstrap modal display without inline event handlers.
  - Closes the mobile navigation after selection; preserved the optional recognition WebSocket behavior.
- `tests/test_ui_pages.py`
  - Added rendered-shell assertions for the localized links and URLs, one primary navigation landmark, theme toggle, shared evidence modal, and retained current-view tests.

## Test-first record

Added the shared-shell response contracts before the base-shell implementation. The first sandboxed run reached the new shell test but could not complete a database-backed request under the sandbox. The base template was then rendered directly to validate its Jinja syntax before the full suite was rerun outside the sandbox.

## Verification

Additional checks passed:

- `base template renders`
- `node --check static/js/main.js`

Required command, run outside the sandbox because the sandboxed SQLite request stalled before UI assertions:

```text
/home/inomjon/anaconda3/envs/yolo/bin/python -m pytest tests/test_ui_pages.py -v
============================= test session starts ==============================
platform linux -- Python 3.10.16, pytest-9.1.1, pluggy-1.6.0 -- /home/inomjon/anaconda3/envs/yolo/bin/python
cachedir: .pytest_cache
rootdir: /home/inomjon/projectAI/face_rec/face_recognition_airi
plugins: hydra-core-1.3.2, anyio-4.12.0
collecting ... collected 10 items

tests/test_ui_pages.py::test_page_routes_return_html[/] PASSED           [ 10%]
tests/test_ui_pages.py::test_page_routes_return_html[/employees] PASSED  [ 20%]
tests/test_ui_pages.py::test_page_routes_return_html[/attendance] PASSED [ 30%]
tests/test_ui_pages.py::test_page_routes_return_html[/recognition] PASSED [ 40%]
tests/test_ui_pages.py::test_page_routes_return_html[/attendance/unknown] PASSED [ 50%]
tests/test_ui_pages.py::test_page_routes_return_html[/cameras] PASSED    [ 60%]
tests/test_ui_pages.py::test_dashboard_exposes_current_view_to_the_base_template PASSED [ 70%]
tests/test_ui_pages.py::test_employee_registration_uses_the_navigation_view_identifier PASSED [ 80%]
tests/test_ui_pages.py::test_shared_shell_exposes_localized_primary_navigation_once PASSED [ 90%]
tests/test_ui_pages.py::test_shared_shell_provides_theme_control_and_evidence_modal PASSED [100%]

============================== 10 passed in 3.39s ==============================
```

## Self-review

- Confirmed only Task 2’s four requested implementation/test files plus this report changed.
- Confirmed all seven localized links use the existing URL helpers and one primary navigation landmark is rendered.
- Confirmed `data-current-view` remains on the body.
- Confirmed the modal has a single shared shell instance and the JS API removes the image source when input is absent.
- Confirmed the shell collapse remains in normal document flow at mobile widths and only closes through Bootstrap after a navigation selection.
- Confirmed the project has no Git repository; none was initialized.

## Concerns

The sandboxed pytest process hangs in the first database-backed dashboard request (the project SQLite connection uses a 30-second busy timeout), before any UI assertion. The exact required suite passed when run outside the sandbox in 3.39 seconds. This is an execution-environment lock limitation, not a test or shell failure.

## Fix round 1/5: evidence-modal close contrast

### Fix details

The dark theme uses `--airi-surface: #182233` for the evidence modal, while Bootstrap's default `.btn-close` keeps a black SVG background image. The existing `.evidence-modal .btn-close` rule set only the 44px practical target and did not alter the glyph, producing insufficient contrast.

Added the following dark-theme-scoped override to `static/css/main.css`:

```css
[data-theme="dark"] .evidence-modal .btn-close {
    filter: invert(1) grayscale(100%) brightness(200%);
    opacity: 1;
}
```

This turns the Bootstrap close glyph white at full opacity on the dark evidence-modal surface. It is scoped to `[data-theme="dark"] .evidence-modal .btn-close`, so the light-theme close icon is unchanged. The existing rendered-shell test now also asserts that the shared modal exposes its accessible close control (`data-bs-dismiss="modal"` and `aria-label="Yopish"`).

### Verification

Required command, run outside the sandbox because the sandboxed SQLite request stalls before UI assertions:

```text
/home/inomjon/anaconda3/envs/yolo/bin/python -m pytest tests/test_ui_pages.py -v
============================= test session starts ==============================
platform linux -- Python 3.10.16, pytest-9.1.1, pluggy-1.6.0 -- /home/inomjon/anaconda3/envs/yolo/bin/python
cachedir: .pytest_cache
rootdir: /home/inomjon/projectAI/face_rec/face_recognition_airi
plugins: hydra-core-1.3.2, anyio-4.12.0
collecting ... collected 10 items

tests/test_ui_pages.py::test_page_routes_return_html[/] PASSED           [ 10%]
tests/test_ui_pages.py::test_page_routes_return_html[/employees] PASSED  [ 20%]
tests/test_ui_pages.py::test_page_routes_return_html[/attendance] PASSED [ 30%]
tests/test_ui_pages.py::test_page_routes_return_html[/recognition] PASSED [ 40%]
tests/test_ui_pages.py::test_page_routes_return_html[/attendance/unknown] PASSED [ 50%]
tests/test_ui_pages.py::test_page_routes_return_html[/cameras] PASSED    [ 60%]
tests/test_ui_pages.py::test_dashboard_exposes_current_view_to_the_base_template PASSED [ 70%]
tests/test_ui_pages.py::test_employee_registration_uses_the_navigation_view_identifier PASSED [ 80%]
tests/test_ui_pages.py::test_shared_shell_exposes_localized_primary_navigation_once PASSED [ 90%]
tests/test_ui_pages.py::test_shared_shell_provides_theme_control_and_evidence_modal PASSED [100%]

============================== 10 passed in 5.21s ==============================
```
