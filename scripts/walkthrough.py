#!/usr/bin/env python3
"""Watch BOTH cameras at once and log every face with full diagnostics.

Run this, then walk past both cameras. Prints per face: size, sharpness, pose,
which gate passed/failed, the best gallery match and the runner-up. Writes the
aligned crops so you can see exactly what the recognizer was given.

    python scripts/walkthrough.py 180
"""
import logging, sys, threading, time
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2, numpy as np
from sqlalchemy import select

from app.config import settings
from app.core.aligner import FaceAligner
from app.core.detector import build_detector
from app.core.geometry import aligned_to_uint8
from app.core.quality import assess
from app.core.recognizer import FaceRecognizer
from app.core.stream import RtspSource
from app.core.tracker import FaceTracker
from app.db.models import Camera
from app.db.session import session_scope
from app.services.enrollment import load_gallery

logging.basicConfig(level=logging.WARNING, format="%(message)s")
SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 180
OUT = Path("data/walkthrough"); OUT.mkdir(parents=True, exist_ok=True)
for f in OUT.glob("*.jpg"): f.unlink()

g = load_gallery()
rows, lock, stop = [], threading.Lock(), threading.Event()


def watch(cid, name, role, url):
    det = build_detector("yolo", settings.model_path(settings.detector_model),
                         imgsz=settings.detect_width, conf=0.25)
    ali = FaceAligner(settings.model_path(settings.aligner_model),
                      crop_size=settings.align_crop_size, margin=settings.align_margin)
    rec = FaceRecognizer(settings.model_path(settings.recognizer_model))
    trk = FaceTracker(frame_rate=12)

    src = RtspSource(url, name=name, transport="tcp", queue_size=2).start()
    t0 = time.time()
    while not src.connected and time.time() - t0 < 20:
        time.sleep(0.3)
    if not src.connected:
        print(f"!! {name} could not connect"); return

    print(f"   {name} [{role}] watching…")
    n, nth = 0, 0
    t0 = time.time()
    while not stop.is_set() and time.time() - t0 < SECONDS:
        f = src.read(timeout=2.0)
        if f is None: continue
        nth += 1
        if nth % 2: continue

        h, w = f.image.shape[:2]
        sc = settings.detect_width / w
        dets = det.detect(cv2.resize(f.image, (settings.detect_width, int(h*sc))))
        if not dets:
            trk.update(np.zeros((0,4),np.float32), np.zeros((0,),np.float32)); continue

        boxes = np.stack([d.box for d in dets]) / sc
        scores = np.array([d.score for d in dets], np.float32)
        tracked = trk.update(boxes, scores)
        if not tracked: continue

        rgb = cv2.cvtColor(f.image, cv2.COLOR_BGR2RGB)
        tb = [b for _t, b, _s in tracked]
        faces = ali.align(rgb, tb)
        if not faces: continue
        embs = rec.embed(np.stack([x.aligned for x in faces]))

        for (tid, b, ds), af, emb in zip(tracked, faces, embs):
            q = assess(b, af.aligned, af.landmarks, af.score,
                       min_face_px=settings.min_face_px, min_laplacian_var=settings.min_laplacian_var,
                       min_aligner_score=settings.min_aligner_score,
                       max_yaw_deg=settings.max_yaw_deg, max_pitch_deg=settings.max_pitch_deg)
            sims = g.M @ (emb / (np.linalg.norm(emb)+1e-12))
            per = {}
            for o, s_ in zip(g.owner, sims):
                per[int(o)] = max(per.get(int(o), -2), float(s_))
            rank = sorted(per.items(), key=lambda kv: -kv[1])
            bid, best = rank[0]
            second = rank[1][1] if len(rank) > 1 else -1.0
            n += 1
            with lock:
                rows.append(dict(cam=name, tid=tid, face_px=q.face_px, sharp=q.sharpness,
                                 yaw=q.yaw, pitch=q.pitch, alsc=af.score,
                                 gate=q.reason or "PASS", best=best, second=second,
                                 who=g.name(bid)))
                print(f"{name[:8]:<8} t{tid:<3} {q.face_px:5.0f}px sh{q.sharpness:5.0f} "
                      f"yaw{q.yaw:6.1f} pit{q.pitch:6.1f} al{af.score:4.2f} "
                      f"{(q.reason or 'PASS'):>12} best={best:6.3f} 2nd={second:6.3f} {g.name(bid)[:20]}")
            if n <= 60:
                stem = f"{name}_t{tid}_{n:03d}_{best:.2f}"
                # The aligned 112x112 is what the recognizer actually sees.
                cv2.imwrite(str(OUT/f"{stem}_aligned.jpg"),
                            cv2.cvtColor(aligned_to_uint8(af.aligned), cv2.COLOR_RGB2BGR))
                # The native crop it was warped from, at full sensor resolution:
                # the only way to tell a bad frame from a bad alignment.
                x1, y1, x2, y2 = [int(v) for v in b]
                side = max(x2 - x1, y2 - y1) * settings.align_margin / 2.0
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                H, W = f.image.shape[:2]
                nx1, ny1 = max(0, int(cx - side)), max(0, int(cy - side))
                nx2, ny2 = min(W, int(cx + side)), min(H, int(cy + side))
                nat = f.image[ny1:ny2, nx1:nx2]
                if nat.size:
                    cv2.imwrite(str(OUT/f"{stem}_native.jpg"), nat,
                                [cv2.IMWRITE_JPEG_QUALITY, 96])
    src.stop()
    print(f"   {name}: {n} face observations")


with session_scope() as s:
    cams = [(c.id, c.name, c.role.value, c.rtsp_url)
            for c in s.execute(select(Camera).where(Camera.enabled.is_(True))).scalars()]

print("="*100)
print(f"WALKTHROUGH  {SECONDS}s   gallery {len(g)} emb / {g.n_people} people   "
      f"threshold={settings.recognition_threshold}")
print("="*100)
ts = [threading.Thread(target=watch, args=c, daemon=True) for c in cams]
for t in ts: t.start()
try:
    for t in ts: t.join()
except KeyboardInterrupt:
    stop.set()

print("\n" + "="*100); print("SUMMARY"); print("="*100)
if not rows:
    print("No faces observed. Nobody walked past, or detection failed.")
    sys.exit(0)

passed = [r for r in rows if r["gate"] == "PASS"]
print(f"observations={len(rows)}   passed gates={len(passed)}   gated={len(rows)-len(passed)}")
for k, v in Counter(r["gate"] for r in rows).most_common():
    print(f"    {k:>14}: {v}")

for label, sub in (("ALL", rows), ("GATE-PASSING", passed)):
    if not sub: continue
    b = np.array([r["best"] for r in sub]); fp = np.array([r["face_px"] for r in sub])
    sh = np.array([r["sharp"] for r in sub])
    print(f"\n{label}  (n={len(sub)})")
    print(f"  face_px   p5={np.percentile(fp,5):6.0f}  median={np.median(fp):6.0f}  p95={np.percentile(fp,95):6.0f}")
    print(f"  sharpness p5={np.percentile(sh,5):6.0f}  median={np.median(sh):6.0f}")
    print(f"  best      min={b.min():.3f}  median={np.median(b):.3f}  p95={np.percentile(b,95):.3f}  max={b.max():.3f}")
    print("  fraction clearing threshold:")
    for t in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
        print(f"      >= {t:.2f}: {int((b>=t).sum()):4d}/{len(b)}  ({(b>=t).mean()*100:5.1f}%)")

print("\n  top matches seen:")
for who, c in Counter(r["who"] for r in rows if r["best"] >= 0.25).most_common(8):
    sc = [r["best"] for r in rows if r["who"] == who]
    print(f"    {who[:28]:<28} n={c:<4} best={max(sc):.3f}")
print(f"\ncrops -> {OUT}/")
