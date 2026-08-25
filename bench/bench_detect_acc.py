"""Detection rate + face size on the REAL gallery (270 imgs, 1280x720)."""
import os, glob, time, logging
import numpy as np, cv2
logging.getLogger("ultralytics").setLevel(logging.ERROR)

MODELS="/home/inomjon/projectAI/face_rec/face_recognition_airi/models"
GAL="/home/inomjon/projectAI/face_rec/face_recognition_airi/face_id_users"
imgs=sorted(glob.glob(f"{GAL}/*/*.png"))
print(f"gallery images: {len(imgs)}")

def biggest(boxes):
    if len(boxes)==0: return None
    a=[(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    return boxes[int(np.argmax(a))]

# --- YuNet ---
res={}
det=cv2.FaceDetectorYN.create(f"{MODELS}/face_detection_yunet_2023mar.onnx","",(1280,720),0.6,0.3,5000)
w=[];miss=[];t0=time.perf_counter()
for p in imgs:
    im=cv2.imread(p)
    if im.shape[:2]!=(720,1280): im=cv2.resize(im,(1280,720))
    det.setInputSize((im.shape[1],im.shape[0]))
    _,f=det.detect(im)
    if f is None or len(f)==0: miss.append(p); continue
    f=f[np.argmax(f[:,2]*f[:,3])]
    w.append(f[2])
res['YuNet']=(np.array(w),miss,time.perf_counter()-t0)

# --- YOLOv8n-face ---
from ultralytics import YOLO
m=YOLO(f"{MODELS}/yolov8n-face.pt"); m.to("cuda")
w=[];miss=[];t0=time.perf_counter()
for p in imgs:
    im=cv2.imread(p)
    if im.shape[:2]!=(720,1280): im=cv2.resize(im,(1280,720))
    r=m.predict(im,verbose=False,device=0,imgsz=1280,conf=0.4)[0]
    b=r.boxes.xyxy.cpu().numpy()
    bb=biggest(b)
    if bb is None: miss.append(p); continue
    w.append(bb[2]-bb[0])
res['YOLOv8n-face']=(np.array(w),miss,time.perf_counter()-t0)

print(f"\n{'detector':16} {'found':>10} {'missed':>7} {'face w: p5':>11} {'median':>8} {'p95':>7} {'wall':>8}")
print("-"*72)
for k,(w,miss,el) in res.items():
    print(f"{k:16} {len(w):5d}/{len(imgs):<4} {len(miss):7d} {np.percentile(w,5):11.1f} {np.median(w):8.1f} {np.percentile(w,95):7.1f} {el:7.1f}s")
for k,(w,miss,el) in res.items():
    if miss: print(f"\n{k} missed ({len(miss)}): " + ", ".join('/'.join(p.split('/')[-2:]) for p in miss[:8]))
