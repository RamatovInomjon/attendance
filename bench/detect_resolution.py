"""At what size does the detector start missing faces?

This decides whether a detection ROI is worth building.  The pipeline currently
downscales 3840 -> 1280 before detecting, a 3x reduction: a face measuring 97 px
in the 4K frame arrives at the detector as ~32 px.  If that is near the
detector's floor, restricting detection to the walkway polygon and running it at
native resolution would find faces the current path misses.
"""
import sys, glob
sys.path.insert(0, "/home/inomjon/projectAI/face_rec/face_recognition_airi")
import cv2, numpy as np
from app.config import settings
from app.core.detector import build_detector

det = build_detector("yolo", settings.model_path(settings.detector_model), imgsz=1280, conf=0.35)

imgs = sorted(glob.glob("face_id_users/*/image_01.png"))[:30]
print(f"{len(imgs)} gallery faces, rescaled to simulate distance\n")

print(f"{'face px in detector input':>26} {'detected':>10} {'recall':>8}")
print("-"*48)
recall = {}
for target in (12, 16, 20, 24, 28, 32, 40, 48, 64, 96):
    found = 0
    for p in imgs:
        im = cv2.imread(p)
        d0 = det.detect(im)
        if not d0: continue
        b = max(d0, key=lambda x:(x.box[2]-x.box[0])*(x.box[3]-x.box[1])).box
        fw = b[2]-b[0]
        s = target/fw
        small = cv2.resize(im, (max(32,int(im.shape[1]*s)), max(32,int(im.shape[0]*s))),
                           interpolation=cv2.INTER_AREA)
        # pad onto a 1280-wide canvas so imgsz is held constant
        canvas = np.zeros((720,1280,3), np.uint8)
        hh, ww = small.shape[:2]
        if hh<=720 and ww<=1280:
            canvas[(720-hh)//2:(720-hh)//2+hh, (1280-ww)//2:(1280-ww)//2+ww] = small
        else:
            canvas = cv2.resize(small,(1280,720))
        if det.detect(canvas): found += 1
    recall[target] = found/len(imgs)*100
    print(f"{target:>26} {found:>7}/{len(imgs)} {recall[target]:>7.0f}%")

print("\n--- what this means for the corridor ---")
print("A face measuring N px in the 4K frame arrives at the detector as N/3 px")
print("(3840 -> 1280).  Measured corridor faces:\n")
for name, px4k in (("p5  (distant)", 56), ("median", 97), ("p95 (close)", 201)):
    at1280 = px4k/3
    nearest = min(recall, key=lambda k: abs(k-at1280))
    print(f"  {name:<16} {px4k:>4} px in 4K  ->  {at1280:>5.0f} px at the detector"
          f"  ~{recall[nearest]:>3.0f}% recall")
print("\nWith a detection ROI at native resolution those same faces reach the")
print("detector at their FULL 4K size, moving each row far up this table.")
