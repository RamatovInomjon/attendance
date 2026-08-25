#!/bin/bash
# EmAtSy - start the capture service (2 camera workers + web UI on :8000).
#
# This used to launch `uvicorn fast_api.main:app`. That package was the
# Django-era carry-over and is gone; the service is app.api.main, started
# through scripts/run.py so the camera workers come up with it.
#
# The interpreter is pinned deliberately: the system python3 lacks the CUDA
# stack, and a scheduled run once crash-looped 370 times because systemd
# resolved a different python than an interactive shell did.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-/home/inomjon/anaconda3/envs/yolo/bin/python3}"
[ -x "$PY" ] || { echo "interpreter not found: $PY" >&2; exit 1; }

# A CPU-only onnxruntime silently shadows onnxruntime-gpu and costs ~10x.
"$PY" - <<'PYCHK'
from app.core.onnx_env import available_providers
p = available_providers()
print("onnxruntime providers:", p)
if "CUDAExecutionProvider" not in p:
    raise SystemExit("CUDA provider missing - see docs/OPERATIONS.md")
PYCHK

echo "Starting EmAtSy on http://127.0.0.1:8000 ..."
exec "$PY" scripts/run.py
