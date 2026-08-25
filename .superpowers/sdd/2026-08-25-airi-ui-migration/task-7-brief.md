# Task 7: Enrollment, Unknown Review, and Camera Diagnostics

Work in `/home/inomjon/projectAI/face_rec/face_recognition_airi`. Preserve accepted Tasks 1-6 and all existing submission/stream contracts.

## Files

Modify `templates/employees/register.html`, `templates/attendance/unknown.html`, `templates/camera/settings.html`, `app/api/pages.py`, `tests/test_ui_pages.py`, and necessary shared CSS. Inspect existing registration JavaScript/IDs/endpoints before editing and preserve compatibility.

## Enrollment Requirements

Restyle the existing registration into a clear Uzbek two-stage visual workflow: (1) employee identity/metadata; (2) face capture/import, quality validation, embedding result. Preserve every existing form field name, element ID consumed by scripts, capture source, endpoint, and submission behavior. Show actionable quality-rejection space/guidance for blur, face size, pose, alignment score, and duplicate risk; do not invent unsupported backend validation. Use AIRI components and responsive layout.

## Unknown Review Requirements

Render unknown sighting cards/table with best snapshot, camera, first/last seen, frame count, shared evidence preview, and localized empty state. Consume existing fields exactly. Do not add identity-assignment/delete mutations without backend routes.

## Camera Diagnostics Requirements

Separate each camera into `configured`, `observed`, `pipeline`, and `drift` presentation groups. Show camera id/name/role/IP/configuration; observed resolution/FPS/connection; algorithm FPS/latency/drops/errors when available; explicit unavailable states. Include a clear Uzbek NVR ownership/drift warning: encoder quality/bitrate/GOP may be overwritten by the NVR and must be changed at the NVR. Do not imply this app persists those settings and do not apply camera configuration.

Tests must verify specialized page headings/states, registration contract IDs/names retained, unknown empty/populated markup, NVR warning, configured-vs-observed groups, and online/offline camera rendering. Preserve all tests.

Run `/home/inomjon/anaconda3/envs/yolo/bin/python -m pytest tests/test_ui_pages.py -v` and report exact output.

## Constraints

- No registration API, model, alignment/recognition quality threshold, schema, camera config, NVR, or core changes.
- No unsupported destructive controls.
- Do not dispatch subagents.
- Self-review and write `.superpowers/sdd/2026-08-25-airi-ui-migration/task-7-report.md`.

