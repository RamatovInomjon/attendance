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

## The edge proxy (10.10.0.75) — the risky step

This container fronts `aiscan.airi.uz` for **every** project. A mistake here
takes down `/ppe/`, `/manim/` and the rest, not just ours.

**Back up first:**

```bash
ssh ai@10.10.0.75
docker cp mamograf-app:/app/app/main.py ~/main.py.bak-$(date +%Y%m%d-%H%M)
```

Edit a local copy, adding a `/faceid` block modelled on the existing `/ppe`
one, and add `/faceid` to the CSP middleware list:

```python
_FACEID_URL = os.environ.get("FACEID_URL", "http://10.10.0.72:8021").rstrip("/")

@app.get("/faceid", include_in_schema=False)
def _faceid_root_redirect():
    return RedirectResponse(url="/faceid/")

@app.api_route("/faceid/{path:path}", include_in_schema=False,
               methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def _faceid_proxy(path: str, request: Request):
    target = f"{_FACEID_URL}/faceid/{path}"     # path forwarded UNCHANGED
    ...
```

Then syntax-check before shipping it back:

```bash
python3 -m py_compile /tmp/main.py            # do not skip this
docker cp /tmp/main.py mamograf-app:/app/app/main.py
docker restart mamograf-app
```

**Rollback:**

```bash
docker cp ~/main.py.bak-<time> mamograf-app:/app/app/main.py
docker restart mamograf-app
```

---

## Verify

```bash
curl -sI https://aiscan.airi.uz/faceid          # 307/308 -> /faceid/
curl -s  https://aiscan.airi.uz/faceid/health   # {"status":"ok"}
# and prove nothing else broke:
curl -sI https://aiscan.airi.uz/ppe/ https://aiscan.airi.uz/manim/ | grep HTTP
```

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

---

## Open items

- **The proxy block is not installed yet.** Until it is, the app answers only on
  `http://10.10.0.72:8021/faceid/` inside the network.
- **No systemd unit** — needs sudo. The process does not survive a reboot.
- **Cameras**: reachable from the server, but this laptop is currently capturing
  from the same two cameras. Running both writes two independent attendance
  databases for the same people and doubles the RTSP load. Decide which host
  owns capture before enabling the workers here.
