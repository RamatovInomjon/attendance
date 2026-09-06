# UI

The interface is the one the project shipped with — Bootstrap 5.3, Uzbek
labels, the blue glass navbar — served unchanged from `templates/` and fed by
the v3 recognition core. It was never deleted; v3 simply had not been wired to
it yet.

## How Django templates run under FastAPI

`templates/` was written for Django. Rather than rewrite ~6,000 lines of working
markup, two small modules bridge the gap:

**`app/web/django_compat.py`** builds a Jinja2 environment providing what the
markup depends on: a no-op `{% load %}` tag, a `url()` global backed by a name →
path map, and the filters actually used — `date`, `time`, `default`,
`duration_hm`, `floatformat`, `pluralize`, `escapejs`, `json_script`, `yesno`,
`add_class`, `media_url`, `clean_phone`, `length`, `safe`, `add`, `get_item`.

**`app/web/viewmodels.py`** translates the v3 schema into the field names the
templates read. The schema deliberately uses `business_date`, `worked_seconds`
and `phone`; the templates read `record.date`, `record.working_hours` (a
timedelta) and `employee.phone_number`. Rather than rename the schema to suit
the markup or edit the markup to suit the schema, the mapping lives at the
presentation boundary. Local-time conversion happens here too, so templates
never handle timezones.

## Pages

| route | template | source |
|---|---|---|
| `/` | `dashboard/index.html` | daily rows, event feed, KPI counts |
| `/recognition` | `recognition/live.html` | live workers + today's attendance |
| `/employees`, `/employees/{id}`, `/employees/add` | `employees/*.html` | employee table |
| `/attendance` | `attendance/list.html` | daily rows + 7-day chart |
| `/attendance/unknown` | `attendance/unknown.html` | unknown sightings |
| `/cameras` | `camera/settings.html` | camera table + live stream stats |
| `/login` | `auth/login.html` | renders; **auth is not implemented** |

`/recognition/logs` returns JSON and is polled by the live page to refresh its
side panels without a reload.

## The live console

`recognition/live.html` already contained canvas-based WebSocket streaming, so
`app/api/ws.py` serves the contract it expects rather than the template being
changed:

```
/ws/camera/{id}/          {"type":"frame","data":<base64 jpeg>,"fps":…,"stale":…}
/ws/camera/primary/       first enabled camera
/ws/attendance/           recognition events as they happen
```

Ten frames a second, each carrying the annotated frame — green boxes for
recognized people, orange for tracked-but-unidentified with the gate that
rejected them. `{id}` also accepts a camera name or role (`entrance`, `out`).

**Each connection renders its own copy** from the worker's latest processed
frame. The previous implementation had every viewer calling `.get()` on one
shared queue, so two open tabs each received roughly half the frames.

MJPEG remains available at `/video/{camera_id}` for anything that cannot use a
WebSocket.

## Two fixes applied to the templates

- `recognition/live.html:655` built its feed cards with `< div ... data - employee=`
  — malformed tags the browser rendered as literal text rather than markup. The
  Recognition Feed had never displayed correctly.
- `/employees/add` had to be declared before `/employees/{employee_id}`, or the
  path parameter swallows `add` and fails int conversion with a 422.

## Not done

There is **no authentication**. `/login` renders but posts nowhere, and
`django_compat` supplies a stub user so the templates' `user.is_authenticated`
checks pass. Every page and API route is open — fine on the LAN, not for
exposure beyond it.
