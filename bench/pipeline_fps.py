"""Measure real end-to-end pipeline throughput at realistic face counts.

Answers: how many frames per second can one camera worker sustain, and where
does the time go, with 0/1/2/3 faces in frame.
"""
import sys, time
sys.path.insert(0, "/home/inomjon/projectAI/face_rec/face_recognition_airi")
import numpy as np, cv2, glob
from app.config import settings
from app.core.detector import build_detector
from app.core.aligner import FaceAligner
from app.core.recognizer import FaceRecognizer
from app.services.enrollment import load_gallery

g = load_gallery()
det = build_detector("yolo", settings.model_path(settings.detector_model),
                     imgsz=settings.detect_width, conf=settings.detect_conf)
ali = FaceAligner(settings.model_path(settings.aligner_model), crop_size=settings.align_crop_size,
                  margin=settings.align_margin, mode=settings.align_mode)
rec = FaceRecognizer(settings.model_path(settings.recognizer_model), batch_size=settings.embed_batch)
print(f"model {settings.recognizer_model}   detect_width {settings.detect_width}   "
      f"every {settings.process_every_nth} frame(s)\n")

# a real 4K frame, with real faces pasted in at corridor scale
base = cv2.imread("/tmp/claude-1000/-home-inomjon-projectAI-face-rec-face-recognition-airi/5ed61b02-fc69-491a-b559-68baef899b6f/scratchpad/probe/snap_in.jpg")
if base is None: base = np.random.randint(0,255,(2160,3840,3),np.uint8)
faces = [cv2.imread(p) for p in sorted(glob.glob("face_id_users/*/image_01.png"))[:3]]

def frame_with(n):
    f = base.copy()
    for k in range(n):
        src = faces[k % len(faces)]
        d = det.detect(src)
        if not d: continue
        b = max(d, key=lambda x:(x.box[2]-x.box[0])*(x.box[3]-x.box[1])).box
        x1,y1,x2,y2 = [int(v) for v in b]
        pad = int((x2-x1)*0.5)
        crop = src[max(0,y1-pad):y2+pad, max(0,x1-pad):x2+pad]
        if crop.size == 0: continue
        s = 190/max(crop.shape[:2])          # ~100 px face, matching measurements
        crop = cv2.resize(crop,(int(crop.shape[1]*s),int(crop.shape[0]*s)))
        oy, ox = 900, 700 + k*420
        f[oy:oy+crop.shape[0], ox:ox+crop.shape[1]] = crop
    return f

def timeit(fn, n=25):
    for _ in range(4): fn()
    t=time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter()-t)/n*1000

def full(frame):
    h,w = frame.shape[:2]
    sc = settings.detect_width/w
    small = cv2.resize(frame,(settings.detect_width,int(h*sc)),interpolation=cv2.INTER_LINEAR)
    dets = det.detect(small)
    if not dets: return 0
    boxes = np.stack([d.box for d in dets])/sc
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    af = ali.align(rgb, boxes)
    if not af: return 0
    e = rec.embed(np.stack([x.aligned for x in af]))
    for v in e: g.match(v, settings.recognition_threshold, settings.second_best_margin)
    return len(af)

print(f"{'faces':>6} {'resize':>8} {'detect':>8} {'align':>8} {'embed':>8} {'match':>8} {'TOTAL':>9} {'max fps':>9}")
print("-"*74)
for n in (0,1,2,3):
    fr = frame_with(n)
    h,w = fr.shape[:2]; sc = settings.detect_width/w
    t_res = timeit(lambda: cv2.resize(fr,(settings.detect_width,int(h*sc)),interpolation=cv2.INTER_LINEAR))
    small = cv2.resize(fr,(settings.detect_width,int(h*sc)),interpolation=cv2.INTER_LINEAR)
    t_det = timeit(lambda: det.detect(small))
    dets = det.detect(small)
    nf = len(dets)
    if nf:
        boxes = np.stack([d.box for d in dets])/sc
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        t_ali = timeit(lambda: ali.align(rgb, boxes), 15)
        af = ali.align(rgb, boxes)
        arr = np.stack([x.aligned for x in af])
        t_emb = timeit(lambda: rec.embed(arr), 15)
        e = rec.embed(arr)
        t_mat = timeit(lambda: [g.match(v, settings.recognition_threshold,
                                        settings.second_best_margin) for v in e], 200)
    else:
        t_ali=t_emb=t_mat=0.0
    tot = t_res+t_det+t_ali+t_emb+t_mat
    print(f"{nf:>6} {t_res:>7.1f}m {t_det:>7.1f}m {t_ali:>7.1f}m {t_emb:>7.1f}m {t_mat:>7.2f}m "
          f"{tot:>8.1f}m {1000/tot:>8.1f}")

print(f"\nTwo cameras share one GPU, so per-camera budget is half the single-camera figure.")
