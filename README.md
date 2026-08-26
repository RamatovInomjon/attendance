# EmAtSy — Face Recognition Attendance

Two RTSP cameras at a corridor: one means **arrived**, one means **left**.
Faces are detected, tracked, aligned and identified on the GPU; a state machine
turns role-tagged sightings into check-in / check-out times.

```
YOLOv8n-face  →  ByteTrack  →  CVLFace DFA  →  AdaFace IR-101  →  state machine
  detection      tracking       alignment       recognition        attendance
```

| | |
|---|---|
| Enrolled | 54 people, 268 embeddings |
| Gallery separation | d′ = 11.33, rank-1 100.00%, 0 false accepts in 35,775 pairs |
| Throughput | 20–43 fps/camera sustained (6 fps needed) |
| Cameras | 2 × Hikvision DS-2CD2083G2-I, 4K H.264 @12fps |

## Quick start

```bash
python scripts/enroll.py          # build the gallery from face_id_users/
python scripts/seed_cameras.py    # register cameras + their IN/OUT roles
python scripts/run.py             # workers + web UI on :8000
```

Open <http://localhost:8000>. The UI is the project's original Bootstrap
interface, served from `templates/` and fed by the v3 recognition core.

| page | what |
|---|---|
| `/` | dashboard — KPIs, today's attendance, real-time log |
| `/recognition` | **Live Recognition Console** — both camera feeds over WebSocket, live stats, recognition feed |
| `/employees` | roster, search, per-person detail and history |
| `/attendance` | daily attendance, weekly chart, CSV export |
| `/attendance/unknown` | unrecognized sightings |
| `/cameras` | camera config and live stream health |

## Docs

| file | what |
|---|---|
| [docs/ALGORITHM.md](docs/ALGORITHM.md) | **start here** - how the algorithm works, end to end |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | shipping to another machine: licence, encrypted models, bundle |
| [docs/DEPLOY_AISCAN.md](docs/DEPLOY_AISCAN.md) | the AIRI GPU-server deployment: hosts, ports, proxy, traps |
| [docs/PLAN.md](docs/PLAN.md) | the implementation plan and the decisions behind it |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | how the pipeline works, module by module |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | running, deploying, tuning, troubleshooting |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | every measurement, and how to reproduce it |
| [docs/AUDIT.md](docs/AUDIT.md) | review of the removed `fast_api/` implementation, kept for its rationale |
| [docs/UI.md](docs/UI.md) | how the original templates are served from the v3 core |
| [docs/IMAGE_QUALITY.md](docs/IMAGE_QUALITY.md) | why frames were blurry, what was measured, what was changed |
| [docs/SCHEDULED_RUN.md](docs/SCHEDULED_RUN.md) | the 2026-08-25 07:00 capture run: what starts, what is saved, retention |
| [docs/DIRECTION.md](docs/DIRECTION.md) | how direction of travel decides check-in vs check-out |

## Layout

```
app/
  config.py            every tunable, one place
  core/                geometry, detector, aligner, recognizer,
                       quality, tracker, gallery, stream, pipeline
  services/            enrollment, attendance, worker
  db/                  models, session
  api/main.py          FastAPI app, video, JSON, CSV
  api/pages.py         HTML routes rendering the original templates
  api/ws.py            WebSocket camera streaming for the live console
  web/django_compat.py Django template constructs for Jinja2
  web/viewmodels.py    v3 schema -> template field names
templates/             the original Bootstrap UI (unchanged)
static/                its CSS and JS
models/                runtime weights (139 MB) + MANIFEST.sha256 + README
                       _archive/ holds the fp32 master, never shipped
scripts/               enroll, seed_cameras, run, live_test, diagnose_live
bench/                 the benchmark and validation scripts
face_id_users/         enrolment export (55 people, biometric - not in git)
data/                  recordings, debug captures, logs, ematsy.db (not in git)
```

`fast_api/` (the Django-era carry-over) was removed on 2026-08-25 once the last
importer went away: enrolment now runs natively in `app/`. Its findings live on
in [docs/AUDIT.md](docs/AUDIT.md), and a guard test asserts the legacy employee
router is never remounted.
