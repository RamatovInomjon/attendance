"""Isolate the cause of low live scores.

Takes gallery images of KNOWN people, degrades them to match the live corridor
conditions we measured (face ~92px, sharpness ~36, steep downward pitch), and
re-matches. If degraded known faces still score high, the live low scores mean
the walker was not enrolled. If they collapse, it is the image conditions.
"""
import sys, glob
sys.path.insert(0, "/home/inomjon/projectAI/face_rec/face_recognition_airi")
import numpy as np, cv2

from app.config import settings
from app.core.aligner import FaceAligner
from app.core.detector import build_detector
from app.core.quality import sharpness_of
from app.core.recognizer import FaceRecognizer
from app.services.enrollment import load_gallery

g = load_gallery()
det = build_detector("yolo", settings.model_path(settings.detector_model), imgsz=1280, conf=0.3)
ali = FaceAligner(settings.model_path(settings.aligner_model),
                  crop_size=settings.align_crop_size, margin=settings.align_margin)
rec = FaceRecognizer(settings.model_path(settings.recognizer_model))

def match(rgb, box):
    f = ali.align(rgb, [box])
    if not f: return None, 0.0, 0.0
    e = rec.embed(f[0].aligned[None])[0]
    sims = g.M @ (e/np.linalg.norm(e))
    per = {}
    for o,s in zip(g.owner, sims): per[int(o)] = max(per.get(int(o),-2), float(s))
    r = sorted(per.items(), key=lambda kv:-kv[1])
    return g.name(r[0][0]), r[0][1], sharpness_of(f[0].aligned)

tests = ["001_Oybek", "054_Inomjon", "011_Komoliddin"]
print(f"{'person':<18} {'condition':<28} {'face px':>8} {'sharp':>7} {'score':>7}  top match")
print("-"*92)
for person in tests:
    p = sorted(glob.glob(f"face_id_users/{person}/*.png"))[0]
    orig = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    d = det.detect(cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))
    if not d: print(f"{person}: no face"); continue
    b = max(d, key=lambda x:(x.box[2]-x.box[0])*(x.box[3]-x.box[1])).box
    fw = b[2]-b[0]

    variants = [("original (enrolment)", orig, b)]

    # shrink so the face is ~92px, like the corridor median
    for target in (130, 92, 60):
        s = target/fw
        small = cv2.resize(orig, (int(orig.shape[1]*s), int(orig.shape[0]*s)), interpolation=cv2.INTER_AREA)
        variants.append((f"face resized to {target}px", small, b*s))

    # motion blur, matching the measured live sharpness
    s = 92/fw
    small = cv2.resize(orig,(int(orig.shape[1]*s),int(orig.shape[0]*s)),interpolation=cv2.INTER_AREA)
    for k in (3,5,9):
        kern = np.zeros((k,k),np.float32); kern[k//2,:]=1.0/k     # horizontal motion
        variants.append((f"92px + motion blur k={k}", cv2.filter2D(small,-1,kern), b*s))

    # steep downward pitch, like a 3m ceiling mount
    for deg in (20, 35):
        h,w = small.shape[:2]
        f_ = 1.0
        src = np.float32([[0,0],[w,0],[w,h],[0,h]])
        d_ = np.tan(np.radians(deg))*h*0.35
        dst = np.float32([[d_,0],[w-d_,0],[w,h],[0,h]])
        M = cv2.getPerspectiveTransform(src,dst)
        warped = cv2.warpPerspective(small,M,(w,h))
        variants.append((f"92px + {deg}deg pitch", warped, b*s))

    for i,(label,img,box) in enumerate(variants):
        who, sc, sh = match(img, box)
        fpx = box[2]-box[0]
        flag = "" if who==None else ("  <-- correct" if person.split('_',1)[1].lower() in (who or "").lower() else f"  MISMATCH")
        print(f"{person if i==0 else '':<18} {label:<28} {fpx:8.0f} {sh:7.0f} {sc:7.3f}  {who}{flag}")
    print()
