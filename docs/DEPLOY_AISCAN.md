# Deploying to aiscan.airi.uz/faceid

Concrete record of the deployment onto the AIRI GPU server, and how to repeat,
update or roll it back. The general vendor/customer flow is in
[DEPLOYMENT.md](DEPLOYMENT.md); this is the specifics of this one site.

---

## Topology

```
browser
  → https://aiscan.airi.uz/faceid/           (public, TLS)
  → edge proxy: `mamograf-app` container on 10.10.0.75, /app/app/main.py
  → http://10.10.0.72:8021/faceid/...        (GPU server, this app)
```

Three machines, and it matters which is which:

| host | role | access |
|---|---|---|
| 195.158.1.149 | public TLS endpoint for `aiscan.airi.uz` | **no credentials** |
| 10.10.0.75 (`ai`) | runs the edge-proxy container; no GPU | ssh, password |
| **10.10.0.72 (`gpu6`)** | **RTX 3090 — this app runs here** | ssh, password |

**The proxy does not strip the path.** A request to `/faceid/x` arrives at the
backend as `/faceid/x`, not `/x`. That is why `settings.url_prefix` exists and
why FastAPI's `root_path` alone is not enough — every route, mount, generated
link and WebSocket URL has to carry the prefix.

---

## The GPU server

```
Ubuntu 24.04.2 · x86_64 · 24 cores · 31 GB RAM · 812 GB free
NVIDIA RTX 3090 24 GB · driver 590.48.01
GPU UUID  GPU-e4726460-6a7a-9ba6-2017-7ac4920d500b
licence fingerprint  f1e32d3ad3ee3c201473d5949f3b827f
```

Shared with four other projects. **Do not disturb them:**

| port | project |
|---|---|
| 8020 | ppe_guard |
| 8077, 8079, 8085, 8086 | mamograf_train, imgstyle, reid_server |
| 11434 | Ollama |
| **8021** | **faceid (ours)** |

The cameras **are** reachable from this server — `192.168.1.2` and
`192.168.1.64` both answer ping and RTSP/554 — so the full pipeline can run
here, not merely the console.

---

## Python 3.10 is mandatory

The compiled core ships as `cpython-310-x86_64-linux-gnu.so`. That is ABI-locked:
on the system Python 3.12 it does not import at all, and no configuration fixes
it. A `venv` does **not** help — `python3 -m venv` inherits the interpreter it
was created from.

So a private interpreter was installed, touching nothing system-wide and no
sudo required:

```bash
curl -sSL -o /tmp/mc.sh \
  https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh
bash /tmp/mc.sh -b -p ~/faceid/miniconda      # -> Python 3.10.13
~/faceid/miniconda/bin/python -m venv ~/faceid/venv
```

---

## Deploy from scratch

**1 — vendor side, on the build machine**

```bash
# licence bound to the server's GPU; reuse existing key material so the same
# encrypted models keep working
python scripts/make_license.py --issue \
    --fingerprint f1e32d3ad3ee3c201473d5949f3b827f \
    --customer "AIRI aiscan gpu6" --days 365 \
    --key-material "<base64 from the previous licence>" \
    --out licence_server.key

python scripts/encrypt_models.py --licence licence_server.key
python scripts/encrypt_models.py --licence licence_server.key --verify
python scripts/build_native.py
python scripts/package_release.py --out dist --licence licence_server.key --tar
```

**2 — upload and unpack**

```bash
scp dist/ematsy-release.tar.gz gpu6@10.10.0.72:~/faceid/
ssh gpu6@10.10.0.72 'cd ~/faceid && tar xzf ematsy-release.tar.gz'
```

**3 — install dependencies (ONE install at a time)**

```bash
cd ~/faceid/ematsy
~/faceid/venv/bin/pip install -r requirements.txt
# the CPU onnxruntime is pulled in as a transitive dep and shadows the GPU
# build, costing ~10x; remove it and install the GPU one WITHOUT deps
~/faceid/venv/bin/pip uninstall -y onnxruntime
~/faceid/venv/bin/pip install --no-deps onnxruntime-gpu==1.23.2
~/faceid/venv/bin/python -c "import onnxruntime as o; print(o.get_available_providers())"
# must list CUDAExecutionProvider
```

> Run it **detached and only once**: `setsid nohup ./install.sh > install.log &`.
> Two concurrent pip installs into the same venv race and leave half-written
> packages — that happened here and the venv had to be rebuilt. Torch pulls
> ~10 GB of CUDA wheels, so an SSH timeout mid-install is easy to hit.

**4 — configure**

```bash
cd ~/faceid/ematsy
cat >> .env <<'ENV'
url_prefix=/faceid
port=8021
host=0.0.0.0          # NOT 127.0.0.1 - the proxy connects from another host
ENV
~/faceid/venv/bin/python scripts/seed_admin.py     # signing key + default admin
~/faceid/venv/bin/python scripts/seed_cameras.py   # camera URLs and geometry
~/faceid/venv/bin/python scripts/enroll.py         # gallery from face_id_users/
```

**5 — run**

```bash
cd ~/faceid/ematsy && ~/faceid/venv/bin/python scripts/run.py
curl http://127.0.0.1:8021/faceid/health          # -> {"status":"ok"}
```

For a persistent service, `deploy/ematsy.service` is a systemd template —
installing it needs sudo, which this account does not have. Without sudo, use
`setsid nohup … &` and re-run after a reboot.

---

## The edge proxy (10.10.0.75) — installed

This container fronts `aiscan.airi.uz` for **every** project. A mistake here
takes down `/ppe/`, `/manim/` and the rest, not just ours.

**Back up first:**

```bash
ssh ai@10.10.0.75
docker cp mamograf-app:/app/app/main.py ~/main.py.bak-$(date +%Y%m%d-%H%M)
```

The file already had a **prefix-preserving** proxy helper, `_extra_proxy`, used
by `/iqttalim` and `/harakat`. That is exactly the shape we need — it forwards
`/faceid/x` to the backend as `/faceid/x` — so the change is three small edits
rather than a new mechanism.

**1 — register the backend** (`_EXTRA_PROXIES`, ~line 6395):

```python
"faceid": os.environ.get("FACEID_URL", "http://10.10.0.72:8021").rstrip("/"),
```

**2 — the HTTP routes**, modelled on `/harakat`:

```python
@app.get("/faceid", include_in_schema=False)
def _faceid_root_redirect():
    return _MentorRedirect(url="/faceid/")

@app.api_route("/faceid/{path:path}", include_in_schema=False,
               methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def _faceid_proxy(path: str, request: _MentorRequest):
    return await _extra_proxy("faceid", path, request)
```

**3 — let our app send its own CSP/X-Frame headers** (~line 415): add
`"/faceid"` to the `startswith((...))` tuple, or the edge stamps
`X-Frame-Options: DENY` over ours.

**4 — the WebSocket bridge, which `_extra_proxy` cannot do.**

`_extra_proxy` is an httpx request/response bridge. It has no upgrade support,
so a WebSocket handshake through it returns **500** — the live camera view and
the live attendance feed simply never connect. That needs a real websocket
route:

```python
@app.websocket("/faceid/ws/{path:path}")
async def _faceid_ws_proxy(websocket: WebSocket, path: str):
    import asyncio
    from websockets.asyncio.client import connect as _ws_connect   # inside the
    # function ON PURPOSE: an import failure here must not stop the container,
    # which serves every other project on the domain.
    ...
```

Two details that matter:

- **Forward the `Cookie` header.** The backend's auth middleware checks the
  session on the handshake; without it the upgrade is answered with a 303 to
  the login page and the socket never opens.
- **Pump both directions** and cancel the surviving task when either side
  closes, or the bridge leaks a task per disconnected viewer.

Then syntax-check before shipping it back:

```bash
python3 -m py_compile /tmp/main.py            # do not skip this
docker cp /tmp/main.py mamograf-app:/app/app/main.py
docker restart mamograf-app
```

**Rollback:**

```bash
docker cp ~/main.py.bak-20260826-140814 mamograf-app:/app/app/main.py
docker restart mamograf-app
```

A known-good copy with our block already applied is kept alongside it as
`~/main.py.faceid-working-20260826-1420`.

> The container publishes **host 8081 → container 8000**. `curl 127.0.0.1:8000`
> on the host returns nothing and does not mean the proxy is down.

---

## The server owns capture (handover, 2026-08-26)

Capture moved from the laptop to the GPU server. Only ONE host may capture:
both writing means two independent attendance databases for the same people and
double the RTSP load on the cameras.

```bash
# 1 - stop the laptop cleanly, so its workers flush
kill <pid of scripts/run.py>          # SIGTERM; wait for "worker stopped"

# 2 - hand the database over. It is WAL, so cp corrupts it - use the backup API
sqlite3 data/ematsy.db ".backup '/tmp/handover.db'"
sqlite3 /tmp/handover.db "pragma integrity_check;"     # must say ok
scp /tmp/handover.db gpu6@10.10.0.72:/tmp/

# 3 - on the server: STOP the service first, it holds the db open
cp data/ematsy.db data/ematsy.db.pre-handover-$(date +%Y%m%d-%H%M)
rm -f data/ematsy.db-wal data/ematsy.db-shm     # stale WAL of the old db
cp /tmp/handover.db data/ematsy.db
```

**Do the media copy AFTER the final database backup, not before.** Copying
snapshots first and the database second leaves the database referencing images
that were written in between, and every one of them 404s on the dashboard.

**Carry the whole database, not just the gallery.** Anyone still checked in
(`presence='INSIDE'`) has an open session; without their row the next
check-out has nothing to close and lands as `NO_CHECKIN`. Eleven people were
open at this handover.

### The timezone is already handled — pin it only for readable logs

The GPU server runs **UTC**; the cameras and operators are **Asia/Tashkent**.
That sounds alarming and is not, because the application never reads the host
clock's zone:

- `settings.timezone` is fixed at `"Asia/Tashkent"` in `app/config.py`;
- every column goes through `UtcDateTime`, which stores and returns aware UTC;
- `business_date()` converts with `ts.astimezone(settings.tz)` and applies the
  04:00 day boundary.

`app/db/models.py` and `app/services/attendance.py` both carry docstrings about
the earlier versions of exactly this bug (`utcnow().date()` filing early
arrivals under the previous day, a 09:02 check-in rendering as 04:02). It is
fixed at the source, so a UTC host writes the same rows a Tashkent host would.

What the host zone *does* affect is Python's `logging`, which formats in local
time — so on an unpinned UTC server the log reads five hours behind the console
and behind what operators report. `~/faceid/start.sh` pins the zone for that
reason alone:

```bash
#!/bin/bash
export TZ=Asia/Tashkent          # log readability only; storage is UTC either way
cd /home/gpu6/faceid/ematsy
exec /home/gpu6/faceid/venv/bin/python scripts/run.py
```

Do not "fix" a five-hour gap between stored timestamps and the log by shifting
what gets written. Storage is UTC on purpose; `/faceid/api/attendance` renders
it as `+05:00`, which is what you should check.

### Known limitation: the MJPEG endpoint

`_extra_proxy` buffers a whole response (`rr.content`) before returning it, so
`/faceid/video/<id>` — an endless multipart stream — never completes and pins a
proxy worker for its 900 s timeout. No page embeds it (the live view uses the
WebSocket bridge instead), so this is latent; do not link to it from a template
without first making the proxy stream, or it will affect every project on the
domain.

---

## Verify

```bash
curl -sk https://aiscan.airi.uz/faceid/health   # {"status":"ok"}
# and prove nothing else broke:
curl -sk -o /dev/null -w "%{http_code}\n" https://aiscan.airi.uz/ppe/
curl -sk -o /dev/null -w "%{http_code}\n" https://aiscan.airi.uz/manim/
```

`curl` alone is **not enough**, and did not catch the two real bugs here. Both
were found only by driving the site in a browser:

- curl followed an explicit `next=/faceid/`, so it never exercised the default
  and missed that login redirected to the domain root;
- curl fetched pages, not their sub-resources, so it missed every snapshot
  404ing.

So load it in a browser, sign in from `/faceid/login` **without** a `next`
parameter, and check the console is clean:

```js
[...document.querySelectorAll('img')]
  .filter(i => !(i.complete && i.naturalWidth > 0)).map(i => i.src)      // broken
[...document.querySelectorAll('img')].map(i => i.getAttribute('src'))
  .filter(s => s.startsWith('/') && !s.startsWith('/faceid/'))           // unprefixed
```

Both must be empty (`evidenceModalImage`, a hidden modal placeholder with
`src=""`, is the one legitimate exception).

`/iqttalim/` and `/harakat/` answer **502** — their backends on 10.10.0.75:8093
and :8094 are not running. That predates this work; confirm with the backup
file, which points at the same ports.

Then sign in at `https://aiscan.airi.uz/faceid/login` with `inomjon` / `123456`
and **change that password immediately** at `/faceid/users`.

---

## Updating

The core is compiled, so an update is a rebuild plus re-upload:

```bash
# build machine
python scripts/build_native.py
python scripts/package_release.py --out dist --licence licence_server.key --tar
scp dist/ematsy-release.tar.gz gpu6@10.10.0.72:~/faceid/
# server
cd ~/faceid && tar xzf ematsy-release.tar.gz && <restart the process>
```

Interpreted files only — `app/api`, `app/web`, `app/config.py`, `templates/` —
can go as a small delta tarball instead; that is how the URL-prefix change was
shipped here.

---

## Traps hit during this deployment

| symptom | cause |
|---|---|
| `ImportError` on a `.so` | server Python is 3.12; the core is cp310. Use `~/faceid/miniconda` |
| `autobahn==25.10.2` unsatisfiable | `requirements.txt` still listed the removed Django stack; autobahn needs ≥3.11 |
| `curl` SSL failures on the server | its CA store cannot verify pypi — **pip works anyway**, it bundles its own certs |
| venv half-populated | two pip installs raced; kill both, delete the venv, install once |
| ssh refuses to connect | the host offers only `ssh-rsa` (SHA-1). `-o HostKeyAlgorithms=+ssh-rsa`. Worth fixing on the server |
| 502 at the public URL | service down, or bound to `127.0.0.1` instead of `0.0.0.0` |
| static 404 under `/faceid/` | a template emitting `/static/...` without `{{ PREFIX }}` |
| GPU unused | CPU `onnxruntime` shadowing the GPU build — check providers |
| login lands on the domain root | `next` defaulted to `/`, which is same-site and so passed the old guard. `_safe_next` now confines it to the prefix |
| every snapshot 404s | a `/media/...` URL built without the prefix; all of them now go through `media_path()` |
| WebSocket 500 through the proxy | `_extra_proxy` is HTTP-only; needs the `@app.websocket` bridge |
| proxy "down" on `127.0.0.1:8000` | wrong port — the container publishes **8081**→8000 |
| `pkill -f "scripts/run.py"` kills the SSH session | the pattern matches the remote shell's own command string. Kill by PID |
| `AssertionError: scope["type"] == "http"` | a WebSocket request reaching a StaticFiles mount — a WS path with no WS route |

---

## Status (2026-08-26)

Live and reachable at **https://aiscan.airi.uz/faceid/**.

```
backend     10.10.0.72:8021, providers ['Tensorrt', 'CUDA', 'CPU']
licence     'AIRI aiscan gpu6', 365 days
models      6.2 / 2.0 / 130.4 MB, decrypted in memory from .enc
gallery     268 embeddings / 54 people
media       617 snapshots (2.8 MB)
```

Verified end to end through the public domain:

| check | result |
|---|---|
| `/faceid/health` | 200 `{"status":"ok"}` (public probe) |
| `/faceid/` anonymous | 303 → `/faceid/login` |
| login → lands on | `/faceid/` |
| `/faceid/{,attendance,users,employees}` | 200 |
| `/faceid/api/{health,attendance}` | 200 authed, 401 anonymous |
| `/faceid/media/...` | 200 authed, 303 anonymous |
| WebSocket `/faceid/ws/attendance/` | **101 Switching Protocols** |
| WebSocket `/faceid/ws/camera/entrance` | **101 Switching Protocols** |
| broken / unprefixed assets | none |
| `/`, `/ppe/`, `/manim/`, `/iclaude/` | unchanged from baseline |

Media requires a session by design — the snapshots are biometric data, so they
must not be fetchable from the shared domain without one.

Install took 51 minutes, almost all of it fetching torch's CUDA wheels
(~13 GB cached, venv ~6 GB). `torch` is needed only for ENROLMENT — the live
recognition path is pure ONNX — so a recognition-only deployment could drop
`ultralytics`/`torch` and shrink to well under 1 GB.

## Open items

- **No systemd unit** — `gpu6` has no sudo, so the service runs under
  `setsid nohup ~/faceid/start.sh &` and does **not** survive a reboot.
- **The default password is still `123456`.** Change it at `/faceid/users`.
- 59 snapshots from 24-25 Aug are referenced by the database but were deleted
  in an earlier cleanup, so those rows show a placeholder. Today's are intact.
- Unchanged pipeline items, not specific to this deployment: suppressing
  emission when the direction verdict is UNKNOWN, transition-keyed debounce,
  honest `direction_reason` logging, and `NO_CHECKIN` not clearing on a later
  check-in (visible in the migrated data).
