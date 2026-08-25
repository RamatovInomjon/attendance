#!/usr/bin/env python3
"""Watch live streams and report, per detected face, exactly why it was or was
not recognized.  This is the tool for closing the gap between gallery-photo
scores and real corridor scores."""
import logging, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2, numpy as np
from sqlalchemy import select

from app.config import settings
from app.core.aligner import FaceAligner
from app.core.detector import build_detector
from app.core.quality import assess
from app.core.recognizer import FaceRecognizer
from app.core.stream import RtspSource
from app.db.models import Camera
from app.db.session import session_scope
from app.services.enrollment import load_gallery

logging.basicConfig(level=logging.WARNING, format="%(message)s")
SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
OUT = Path("data/diagnose"); OUT.mkdir(parents=True, exist_ok=True)

g = load_gallery()
det = build_detector("yolo", settings.model_path(settings.detector_model),
                     imgsz=settings.detect_width, conf=0.25)   # lower, to see near-misses
ali = FaceAligner(settings.model_path(settings.aligner_model),
                  crop_size=settings.align_crop_size, margin=settings.align_margin)
rec = FaceRecognizer(settings.model_path(settings.recognizer_model))

with session_scope() as s:
    cams = [(c.id, c.name, c.role.value, c.rtsp_url)
            for c in s.execute(select(Camera).where(Camera.enabled.is_(True))).scalars()]

print(f"gallery {len(g)} emb / {g.n_people} people   threshold={settings.recognition_threshold}")
print(f"watching {SECONDS}s per camera\n")

rows = []
for cid, name, role, url in cams:
    src = RtspSource(url, name=name, transport="tcp", queue_size=2).start()
    t0 = time.time()
    while not src.connected and time.time()-t0 < 20: time.sleep(0.3)
    if not src.connected:
        print(f"{name}: could not connect"); src.stop(); continue

    print("="*96); print(f"{name}  [{role}]"); print("="*96)
    print(f"{'t':>5} {'facepx':>7} {'sharp':>7} {'yaw':>6} {'pitch':>6} {'alsc':>5} "
          f"{'gate':>12} {'best':>6} {'2nd':>6} {'match':<22}")
    print("-"*96)
    t0 = time.time(); nth = 0; nface = 0
    while time.time()-t0 < SECONDS:
        f = src.read(timeout=2.0)
        if f is None: continue
        nth += 1
        if nth % 3: continue
        dets = det.detect(cv2.resize(f.image,(settings.detect_width,
                int(f.image.shape[0]*settings.detect_width/f.image.shape[1]))))
        if not dets: continue
        scale = f.image.shape[1]/settings.detect_width
        boxes = [d.box*scale for d in dets]
        rgb = cv2.cvtColor(f.image, cv2.COLOR_BGR2RGB)
        faces = ali.align(rgb, boxes)
        if not faces: continue
        embs = rec.embed(np.stack([x.aligned for x in faces]))
        for b, af, emb in zip(boxes, faces, embs):
            q = assess(b, af.aligned, af.landmarks, af.score,
                       min_face_px=settings.min_face_px, min_laplacian_var=settings.min_laplacian_var,
                       min_aligner_score=settings.min_aligner_score,
                       max_yaw_deg=settings.max_yaw_deg, max_pitch_deg=settings.max_pitch_deg)
            sims = g.M @ (emb/np.linalg.norm(emb))
            per = {}
            for o,s_ in zip(g.owner, sims): per[int(o)] = max(per.get(int(o),-2), float(s_))
            rank = sorted(per.items(), key=lambda kv:-kv[1])
            best_id, best = rank[0]; second = rank[1][1] if len(rank)>1 else -1
            nface += 1
            rows.append(dict(cam=name, face_px=q.face_px, sharp=q.sharpness, yaw=q.yaw,
                             pitch=q.pitch, alsc=af.score, gate=q.reason or "PASS",
                             best=best, second=second, who=g.name(best_id)))
            print(f"{time.time()-t0:5.1f} {q.face_px:7.0f} {q.sharpness:7.0f} {q.yaw:6.1f} {q.pitch:6.1f} "
                  f"{af.score:5.2f} {(q.reason or 'PASS'):>12} {best:6.3f} {second:6.3f} {g.name(best_id)[:22]:<22}")
            if nface <= 40:
                from app.core.geometry import aligned_to_uint8
                cv2.imwrite(str(OUT/f"{name}_{nface:03d}_{best:.2f}.jpg"),
                            cv2.cvtColor(aligned_to_uint8(af.aligned), cv2.COLOR_RGB2BGR))
    src.stop()
    print(f"\n{name}: {nface} faces observed\n")

if rows:
    import statistics as st
    print("="*96); print("SUMMARY"); print("="*96)
    p = [r for r in rows if r["gate"]=="PASS"]
    print(f"faces={len(rows)}  passed gates={len(p)}  gated={len(rows)-len(p)}")
    from collections import Counter
    for k,v in Counter(r["gate"] for r in rows).most_common(): print(f"   {k:>14}: {v}")
    for label, sub in (("all", rows), ("gate-passing", p)):
        if not sub: continue
        b=[r["best"] for r in sub]; fp=[r["face_px"] for r in sub]
        print(f"\n{label}: face_px p5={np.percentile(fp,5):.0f} median={np.median(fp):.0f} p95={np.percentile(fp,95):.0f}")
        print(f"{label}: best-score min={min(b):.3f} median={np.median(b):.3f} max={max(b):.3f}")
        for t in (0.25,0.30,0.35,0.40,0.45):
            print(f"    >= {t:.2f}: {sum(1 for x in b if x>=t)}/{len(b)}")
    print(f"\ncrops written to {OUT}/")
