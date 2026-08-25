"""Save full-resolution 4K frames that contain a face, for offline experiments."""
import sys, time, threading
sys.path.insert(0, "/home/inomjon/projectAI/face_rec/face_recognition_airi")
from pathlib import Path
import cv2, numpy as np
from sqlalchemy import select
from app.config import settings
from app.core.detector import build_detector
from app.core.stream import RtspSource
from app.db.models import Camera
from app.db.session import session_scope

OUT = Path("data/faceframes"); OUT.mkdir(parents=True, exist_ok=True)
SECS = int(sys.argv[1]) if len(sys.argv) > 1 else 180
MAX = 60
saved = {"n": 0}
lock = threading.Lock()

def watch(name, url):
    det = build_detector("yolo", settings.model_path(settings.detector_model),
                         imgsz=settings.detect_width, conf=0.3)
    src = RtspSource(url, name=name, transport="tcp", queue_size=2).start()
    t0 = time.time()
    while not src.connected and time.time()-t0 < 20: time.sleep(0.3)
    t0, nth = time.time(), 0
    while time.time()-t0 < SECS:
        f = src.read(timeout=2.0)
        if f is None: continue
        nth += 1
        if nth % 3: continue
        h, w = f.image.shape[:2]
        sc = settings.detect_width / w
        d = det.detect(cv2.resize(f.image, (settings.detect_width, int(h*sc))))
        if not d: continue
        with lock:
            if saved["n"] >= MAX: break
            saved["n"] += 1
            n = saved["n"]
        cv2.imwrite(str(OUT/f"{name}_{n:03d}.jpg"), f.image, [cv2.IMWRITE_JPEG_QUALITY, 97])
        print(f"  saved {name}_{n:03d}.jpg  ({len(d)} face(s))")
    src.stop()

with session_scope() as s:
    cams = [(c.name, c.rtsp_url) for c in
            s.execute(select(Camera).where(Camera.enabled.is_(True))).scalars()]
ts = [threading.Thread(target=watch, args=c, daemon=True) for c in cams]
for t in ts: t.start()
for t in ts: t.join()
print(f"\n{saved['n']} full-resolution frames with faces -> {OUT}/")
