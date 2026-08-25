"""Sweep the aligner crop margin and measure separation on the real gallery.

margin is a multiplier on the detector box: 1.0 = tight box, 1.5 = 25% padding
per side, 2.0 = 50% per side.
"""
import sys, glob, os, time
sys.path.insert(0, "/home/inomjon/projectAI/face_rec/face_recognition_airi")
import numpy as np, cv2

from app.config import settings
from app.core.detector import build_detector
from app.core.aligner import FaceAligner
from app.core.recognizer import FaceRecognizer

ROOT = "/home/inomjon/projectAI/face_rec/face_recognition_airi"
GAL = f"{ROOT}/face_id_users"
det = build_detector("yolo", f"{ROOT}/models/yolov8n-face.pt", imgsz=1280, conf=0.35)
rec = FaceRecognizer(f"{ROOT}/models/adaface_ir101_webface12m_fp16.onnx")

people = sorted(d for d in os.listdir(GAL) if os.path.isdir(f"{GAL}/{d}") and not d.startswith("_"))
imgs = []
for pid, person in enumerate(people):
    for p in sorted(glob.glob(f"{GAL}/{person}/*.png")):
        imgs.append((pid, p))
print(f"{len(imgs)} images / {len(people)} people\n")

# cache detections once
cache = []
for pid, p in imgs:
    bgr = cv2.imread(p)
    d = det.detect(bgr)
    if not d: continue
    b = max(d, key=lambda x: (x.box[2]-x.box[0])*(x.box[3]-x.box[1])).box
    cache.append((pid, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), b))
print(f"cached {len(cache)} detections\n")

print(f"{'margin':>7} {'pad/side':>9} {'aligned':>8} {'gen mean':>9} {'imp max':>8} {'d-prime':>8} {'rank-1':>8} {'min gen':>8}")
print("-"*72)
for margin in [1.0, 1.15, 1.3, 1.5, 1.6, 1.8, 2.0, 2.3]:
    ali = FaceAligner(f"{ROOT}/models/dfa_mobilenet_aligner.onnx",
                      crop_size=settings.align_crop_size, margin=margin)
    E, L = [], []
    for pid, rgb, b in cache:
        f = ali.align(rgb, [b])
        if not f: continue
        E.append(rec.embed(f[0].aligned[None])[0]); L.append(pid)
    if len(E) < 10: print(f"{margin:7.2f}  align failed"); continue
    E = np.stack(E); L = np.array(L)
    S = E @ E.T
    iu = np.triu_indices(len(E), k=1)
    same = (L[:,None]==L[None,:])[iu]; sims = S[iu]
    gen, imp = sims[same], sims[~same]
    dp = (gen.mean()-imp.mean())/np.sqrt((gen.var()+imp.var())/2)
    correct = 0
    for i in range(len(E)):
        s = S[i].copy(); s[i] = -2
        best = {}
        for j in range(len(E)):
            if j==i: continue
            best[L[j]] = max(best.get(L[j],-2), s[j])
        if max(best, key=best.get) == L[i]: correct += 1
    pad = (margin-1)/2*100
    print(f"{margin:7.2f} {pad:8.0f}% {len(E):8d} {gen.mean():9.4f} {imp.max():8.4f} "
          f"{dp:8.2f} {correct/len(E)*100:7.2f}% {gen.min():8.4f}")
