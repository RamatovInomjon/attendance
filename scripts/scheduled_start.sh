#!/usr/bin/env bash
# Scheduled start for the capture run. Invoked by a systemd user timer.
#
# Waits for the GPU before starting: a model was training when this was
# scheduled, and starting on a full GPU would fail in a way that is tedious to
# diagnose after the fact (onnxruntime falls back to CPU silently rather than
# erroring, which looks like the pipeline simply being slow).
set -uo pipefail
cd /home/inomjon/projectAI/face_rec/face_recognition_airi
export TZ=Asia/Tashkent

# The absolute interpreter, not `python3`. systemd runs with a clean PATH that
# does not include the conda environment, so a bare `python3` picks the system
# one and every import fails. This cost three hours of a capture window.
PY=/home/inomjon/anaconda3/envs/yolo/bin/python3
if [ ! -x "$PY" ]; then
  echo "FATAL: interpreter $PY missing"; exit 78
fi

LOG_DIR=data/logs; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"
exec >>"$LOG" 2>&1

echo "=== scheduled start $(date '+%F %T %Z') ==="

need_mb=2500
for i in $(seq 1 120); do          # up to 60 minutes
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  free=${free:-0}
  if [ "$free" -ge "$need_mb" ]; then
    echo "GPU has ${free} MiB free (need ${need_mb}) - starting"
    break
  fi
  echo "waiting for GPU: ${free} MiB free, need ${need_mb} (attempt $i/120)"
  sleep 30
done

if [ "${free:-0}" -lt "$need_mb" ]; then
  echo "GPU still busy after 60 min - starting anyway; check for CPU fallback"
fi

# Fail loudly if the CPU-only onnxruntime has crept back in: it shadows the GPU
# build and silently drops everything to CPU. This has happened three times.
"$PY" - <<'PYCHK'
import sys
sys.path.insert(0, ".")
from app.core.onnx_env import preload_cuda_libs
preload_cuda_libs()
import onnxruntime as ort
p = ort.get_available_providers()
print("onnxruntime providers:", p)
if "CUDAExecutionProvider" not in p:
    print("WARNING: CUDA provider missing - run:")
    print("  pip uninstall -y onnxruntime onnxruntime-gpu && pip install --no-deps onnxruntime-gpu==1.23.2")
    raise SystemExit(75)
PYCHK

# --check, not --apply. Enforcing on every start silently overwrote settings
# changed from the camera's own web UI, which looked like the camera reverting
# on its own. Drift is now reported, not corrected; run --apply by hand when you
# actually want the recorded values enforced.
"$PY" scripts/camera_config.py --check || echo "camera config check failed - continuing"

echo "launching..."
exec "$PY" scripts/run.py
