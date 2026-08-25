"""Why 224? And can we do better?

Theory
------
The DFA graph resizes its input to 160x160 internally and the reference warps
the 112x112 crop out of THAT tensor.  So with margin m, the face occupies
160/m pixels inside the aligner, and the warp resamples it to 112:

    m = 1.0  -> face 160 px -> 112   (downsample, full detail)
    m = 1.3  -> face 123 px -> 112   (slight downsample)
    m = 1.43 -> face 112 px -> 112   (1:1 - the break-even point)
    m = 1.8  -> face  89 px -> 112   (UPSAMPLE - inventing detail)
    m = 2.3  -> face  70 px -> 112   (heavy upsample)

Prediction: quality should be flat up to ~1.43 and fall off after.  Test it,
and test whether the intermediate crop_size matters at all.
"""
import sys, glob, os
sys.path.insert(0, "/home/inomjon/projectAI/face_rec/face_recognition_airi")
import numpy as np, cv2

from app.config import settings
from app.core.aligner import FaceAligner
from app.core.detector import build_detector
from app.core.recognizer import FaceRecognizer

ROOT = "/home/inomjon/projectAI/face_rec/face_recognition_airi"
GAL = f"{ROOT}/face_id_users"
det = build_detector("yolo", settings.model_path(settings.detector_model), imgsz=1280, conf=0.35)
rec = FaceRecognizer(settings.model_path(settings.recognizer_model))

people = sorted(d for d in os.listdir(GAL) if os.path.isdir(f"{GAL}/{d}") and not d.startswith("_"))
cache = []
for pid, person in enumerate(people):
    for p in sorted(glob.glob(f"{GAL}/{person}/*.png")):
        bgr = cv2.imread(p)
        d = det.detect(bgr)
        if not d: continue
        b = max(d, key=lambda x:(x.box[2]-x.box[0])*(x.box[3]-x.box[1])).box
        cache.append((pid, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), b))
print(f"{len(cache)} faces cached / {len(people)} people\n")

def evaluate(margin, crop_size):
    ali = FaceAligner(settings.model_path(settings.aligner_model),
                      crop_size=crop_size, margin=margin)
    E, L = [], []
    for pid, rgb, b in cache:
        f = ali.align(rgb, [b])
        if not f: continue
        E.append(rec.embed(f[0].aligned[None])[0]); L.append(pid)
    E = np.stack(E); L = np.array(L)
    S = E @ E.T
    iu = np.triu_indices(len(E), k=1)
    same = (L[:,None]==L[None,:])[iu]; sims = S[iu]
    gen, imp = sims[same], sims[~same]
    dp = (gen.mean()-imp.mean())/np.sqrt((gen.var()+imp.var())/2)
    corr = 0
    for i in range(len(E)):
        s = S[i].copy(); s[i] = -2
        best = {}
        for j in range(len(E)):
            if j == i: continue
            best[L[j]] = max(best.get(L[j],-2), s[j])
        if max(best, key=best.get) == L[i]: corr += 1
    return dp, corr/len(E)*100, gen.mean(), imp.max(), gen.min()

print("A) crop_size sweep at margin 1.30  (does the intermediate size matter?)")
print(f"{'crop_size':>10} {'face px in 160':>15} {'d-prime':>9} {'rank-1':>8} {'gen mean':>9} {'min gen':>8}")
print("-"*68)
for cs in (144, 160, 192, 224, 256, 320):
    dp, r1, gm, im_, gmin = evaluate(1.30, cs)
    print(f"{cs:10d} {160/1.30:15.0f} {dp:9.2f} {r1:7.2f}% {gm:9.4f} {gmin:8.4f}")

print("\nB) margin sweep at the best crop_size  (theory: break-even at 1.43)")
print(f"{'margin':>8} {'pad/side':>9} {'face px in 160':>15} {'d-prime':>9} {'rank-1':>8} {'gen mean':>9} {'min gen':>8}")
print("-"*80)
for m in (1.0, 1.1, 1.2, 1.25, 1.30, 1.35, 1.43, 1.5, 1.6, 1.8, 2.0):
    dp, r1, gm, im_, gmin = evaluate(m, 224)
    flag = "  <- 1:1" if abs(m-1.43) < 0.01 else ""
    print(f"{m:8.2f} {(m-1)/2*100:8.0f}% {160/m:15.0f} {dp:9.2f} {r1:7.2f}% {gm:9.4f} {gmin:8.4f}{flag}")
