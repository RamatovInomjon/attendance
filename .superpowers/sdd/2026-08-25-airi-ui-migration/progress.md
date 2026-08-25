# SDD ledger — plan: docs/superpowers/plans/2026-08-25-airi-ui-migration.md

Ruling: Work sequentially in the current workspace because the project has no Git metadata and `sdd-workspace` cannot create a worktree. Reviews use explicit file snapshots/diffs instead of commits. Cost if wrong: changes are less isolated and rollback is manual, mitigated by one writer at a time and per-task review.

## Preflight dependency scan

| Tasks | Shared file/interface | Finding |
|---|---|---|
| 1 → 2 | `render`, `current_view`, `base.html` | Clean: Task 1 produces active-route context consumed by shell. |
| 1 → 3–7 | `app/api/pages.py` | Clean: later tasks extend page contexts without changing Task 1 contract. |
| 2 → 3–7 | shared CSS classes, `AiriUI` | Clean: later templates consume the shared design system. |
| 2 → 5 | `static/js/main.js`, evidence modal | Clean: Task 5 extends existing generic behaviors. |
| 3 → 8 | dashboard and evidence semantics | Clean: final verification checks quality-best evidence. |
| 4 → 5 | `EmployeeVM`, attendance employee fields | Clean: added defaults preserve existing callers. |
| 5 → 8 | attendance filters and evidence | Clean: final verification exercises the contract. |
| 6 → 8 | live selectors and polling | Clean: final verification checks both streams and evidence. |
| 7 → 8 | specialized pages and NVR warning | Clean: final docs capture the operating limitation. |
| Task 1 | tests vs. code | Clean. |
| Task 2 | tests vs. shell behavior | Clean. |
| Task 3 | aggregates vs. dashboard markup | Clean. |
| Task 4 | enrollment aggregate vs. schema inspection | Clean. |
| Task 5 | date/filter contract | Clean. |
| Task 6 | stable selectors vs. existing transport | Clean. |
| Task 7 | styling vs. preserved submission contracts | Clean. |
| Task 8 | automated/browser verification | Clean. |

Task 1: fix round 1/5 (3 addressed, 0 open — TestClient harness, declared dependencies, registration current_view)
Task 1: complete (review clean; 8 tests passed)
Task 2: minor (deferred): navigation tests do not bind each localized label to its exact href or verify representative active links.
Task 2: minor (deferred): pre-existing recognition WebSocket fields are interpolated through innerHTML in static/js/main.js; final review should triage sanitization.
Task 2: fix round 1/5 (1 addressed, 0 open — dark evidence-modal close contrast)
Task 2: complete (review clean; 10 tests passed; 2 deferred minors)
Task 3: minor (deferred): tests do not isolate timestamp-less records, 09:00/09:01 boundary, and missing-worker offline state.
Task 3: fix round 1/5 (2 addressed, 0 open — presence predicate and bounded dashboard rows)
Task 3: complete (review clean; 14 tests passed; 1 deferred minor)
Task 4: complete (review clean; 16 tests passed)
Task 5: minor (deferred): date shortcut JavaScript lacks DOM/browser coverage for month-end/leap and delegated click behavior.
Task 5: minor (deferred): legacy run.sh still points at fast_api.main while documented operator path uses scripts/run.py.
Task 5: fix round 1/5 (3 addressed, 0 open — filtered CSV, NO_CHECKIN, query/order/export tests)
Task 5: complete (review clean; 23 tests passed; 2 deferred minors)
Task 6: Ruling: the worker's bounded design gate is satisfied by the user's approved specification and implementation plan; resume without another user prompt. Cost if wrong: Task 6 could require rework if the approved design was interpreted too broadly, mitigated by task review.
Task 6: minor (deferred): recognition-log polling lacks an in-flight guard and can overlap on slow responses.
Task 6: minor (deferred): transport tests are source-string checks rather than DOM/WebSocket behavior tests.
Task 6: fix round 1/5 (4 addressed, 0 open — stale state, attendance WS, unavailable stats, transition action)
Task 6: complete (review clean; 32 tests reported passed; 2 deferred minors)
Task 7: fix round 1/5 (3 addressed, 2 open — v3 face enrollment remains disconnected; native submit ends on JSON)
Task 7: Ruling: Fix Round 2's native v3 transactional enrollment design is within the already approved spec and plan; no second user approval is required. Cost if wrong: enrollment integration may need rework, mitigated by v3 persistence/rollback tests and scoped review.
