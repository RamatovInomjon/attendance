# Task 1 Report: UI Contract Tests and Shared Context

## Status

DONE

## Implemented

- Added `render(name: str, *, request=None, current_view: str = "", **ctx) -> HTMLResponse`.
  It supplies `request`, `current_view`, and `today` for every rendered page,
  while preserving any explicitly supplied `today` value.
- Passed explicit page identifiers from the target routes:
  `dashboard:home`, `employees:list`, `attendance:list`, `recognition:live`,
  `attendance:unknown`, and `camera:settings`.
- Added `data-current-view="{{ current_view }}"` to the existing base-template
  body element without redesigning the shell.
- Added FastAPI `TestClient` smoke coverage for `/`, `/employees`,
  `/attendance`, `/recognition`, `/attendance/unknown`, and `/cameras`, plus
  the dashboard current-view assertion.

## Files Changed

- `app/api/pages.py`
- `templates/base.html`
- `tests/test_ui_pages.py`
- `.superpowers/sdd/2026-08-25-airi-ui-migration/task-1-report.md`

## Test Evidence

Required command, run after implementation:

```text
/home/inomjon/anaconda3/envs/yolo/bin/python -m pytest tests/test_ui_pages.py -v
```

Exact observed result: pytest started, collected 7 items, then stalled at
`tests/test_ui_pages.py::test_page_routes_return_html[/]` without a result
within the 30-second execution window.

The cause is external to the page implementation: the specified interpreter's
AnyIO blocking portal hangs even for a minimal FastAPI `TestClient` request
before an application handler runs. Installing the missing `pytest` and
`httpx2` packages allowed collection but did not resolve that portal hang.

Supplementary direct verification passed:

```text
direct route rendering: PASS (6 routes)
```

This invoked all six target handlers with a complete ASGI request scope,
verified HTTP 200 and `text/html` responses, and confirmed the dashboard body
contains `data-current-view="dashboard:home"`. `py_compile` also completed
successfully for `app/api/pages.py` and `tests/test_ui_pages.py`.

## Self-Review

- The shared renderer adds only the requested context contract.
- Existing explicit dashboard date data remains intact.
- The base-template edit is the smallest attribute addition required by Task 1.
- No files under `app/core/` or recognition/attendance decision logic changed.

## Concerns

None for Task 1. The initial TestClient issue is resolved in Fix Round 1.

## Fix Round 1

### Changes

- Kept the required FastAPI `TestClient` contract and added a regression test
  for `/employees/add`, verifying the registration navigation identifier.
- Corrected the registration route identifier from `recognition:register` to
  `employees:register`, matching the existing base navigation.
- Declared and pinned the reproducible standard TestClient dependency set in
  `requirements.txt`: `fastapi==0.141.1`, `starlette==0.50.0`,
  `anyio==4.12.0`, `httpx==0.28.1`, and `pytest==9.1.1`.
- Removed the obsolete `httpx2` and `httpcore2` packages from the specified
  interpreter. The focused test module filters only the unrelated upstream
  `thop.profile` deprecation warning so the requested command is clean.

### Dependency Diagnosis

The original manifest omitted the FastAPI test dependencies. Its unpinned
Starlette installation resolved to 1.6.0, whose TestClient selected the
experimental `httpx2` transport, while FastAPI documents the standard `httpx`
test-client dependency. The project now pins a compatible Starlette/AnyIO/
standard-httpx stack.

The remaining observed hang was not application or worker startup: it occurred
inside `AnyIO`'s cross-thread blocking portal under the Codex filesystem
sandbox, before the test app or route handler started. The suite uses an
isolated `FastAPI()` instance with the page router only, so its TestClient
context manager never invokes the production application's `runtime.start()`
lifespan and cannot start camera or model workers. Running the same required
command outside that sandbox completes normally.

### Command and Exact Output

```text
/home/inomjon/anaconda3/envs/yolo/bin/python -m pytest tests/test_ui_pages.py -v
============================= test session starts ==============================
platform linux -- Python 3.10.16, pytest-9.1.1, pluggy-1.6.0 -- /home/inomjon/anaconda3/envs/yolo/bin/python
cachedir: .pytest_cache
rootdir: /home/inomjon/projectAI/face_rec/face_recognition_airi
plugins: hydra-core-1.3.2, anyio-4.12.0
collecting ... collected 8 items

tests/test_ui_pages.py::test_page_routes_return_html[/] PASSED           [ 12%]
tests/test_ui_pages.py::test_page_routes_return_html[/employees] PASSED  [ 25%]
tests/test_ui_pages.py::test_page_routes_return_html[/attendance] PASSED [ 37%]
tests/test_ui_pages.py::test_page_routes_return_html[/recognition] PASSED [ 50%]
tests/test_ui_pages.py::test_page_routes_return_html[/attendance/unknown] PASSED [ 62%]
tests/test_ui_pages.py::test_page_routes_return_html[/cameras] PASSED    [ 75%]
tests/test_ui_pages.py::test_dashboard_exposes_current_view_to_the_base_template PASSED [ 87%]
tests/test_ui_pages.py::test_employee_registration_uses_the_navigation_view_identifier PASSED [100%]

============================== 8 passed in 4.52s ===============================
```
