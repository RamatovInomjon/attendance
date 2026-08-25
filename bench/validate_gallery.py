"""Embed the whole enrolment gallery and measure genuine/impostor separation.

This is the measurement that sets RECOGNITION_THRESHOLD.  Everything else in the
pipeline is tuning; this is the number that decides whether an attendance row is
right or wrong.
"""
import sys, glob, json, time, os
sys.path.insert(0, "/home/inomjon/projectAI/face_rec/face_recognition_airi")
import numpy as np, cv2

from app.core.detector import YoloFaceDetector
from app.core.aligner import FaceAligner
from app.core.recognizer import FaceRecognizer

ROOT = "/home/inomjon/projectAI/face_rec/face_recognition_airi"
M = f"{ROOT}/models"
GAL = f"{ROOT}/face_id_users"
REC = sys.argv[1] if len(sys.argv) > 1 else "adaface_ir101_webface12m_fp16.onnx"

det = YoloFaceDetector(f"{M}/yolov8n-face.pt", imgsz=1280, conf=0.35)
ali = FaceAligner(f"{M}/dfa_mobilenet_aligner.onnx")
rec = FaceRecognizer(f"{M}/{REC}")
print(f"recognizer={REC}  aligner={ali.provider}  recog={rec.provider}")

people = sorted(d for d in os.listdir(GAL) if os.path.isdir(f"{GAL}/{d}"))
embs, labels, paths, ascore = [], [], [], []
t0 = time.perf_counter()
for pid, person in enumerate(people):
    for p in sorted(glob.glob(f"{GAL}/{person}/*.png")):
        bgr = cv2.imread(p)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dets = det.detect(bgr)
        if not dets:
            print("  no face:", p); continue
        d = max(dets, key=lambda d: (d.box[2]-d.box[0])*(d.box[3]-d.box[1]))
        af = ali.align(rgb, [d.box])
        if not af: print("  no align:", p); continue
        e = rec.embed(af[0].aligned[None])[0]
        embs.append(e); labels.append(pid); paths.append(p); ascore.append(af[0].score)
el = time.perf_counter() - t0

E = np.stack(embs); L = np.array(labels)
print(f"\nembedded {len(E)} images / {len(people)} people in {el:.1f}s ({el/len(E)*1000:.0f} ms/img)")
print(f"aligner face score: min={np.min(ascore):.3f} median={np.median(ascore):.3f}")

S = E @ E.T
iu = np.triu_indices(len(E), k=1)
same = (L[:, None] == L[None, :])[iu]
sims = S[iu]
gen, imp = sims[same], sims[~same]

print(f"\n{'':12} {'n':>7} {'mean':>8} {'std':>7} {'min':>8} {'p1':>8} {'p50':>8} {'p99':>8} {'max':>8}")
for nm, a in (("genuine", gen), ("impostor", imp)):
    print(f"{nm:12} {len(a):7d} {a.mean():8.4f} {a.std():7.4f} {a.min():8.4f} "
          f"{np.percentile(a,1):8.4f} {np.percentile(a,50):8.4f} {np.percentile(a,99):8.4f} {a.max():8.4f}")

print(f"\nd-prime = {(gen.mean()-imp.mean())/np.sqrt((gen.var()+imp.var())/2):.2f}")
print(f"\n{'thresh':>8} {'FAR':>10} {'FRR':>9} {'TAR':>8}")
for t in np.arange(0.15, 0.71, 0.05):
    far = (imp >= t).mean(); frr = (gen < t).mean()
    print(f"{t:8.2f} {far:10.5f} {frr:9.4f} {1-frr:8.4f}")

for target in (1e-3, 1e-4, 0.0):
    cand = [t for t in np.arange(0.10, 0.95, 0.005) if (imp >= t).mean() <= target]
    if cand:
        t = min(cand); print(f"\nFAR<={target:g}: threshold={t:.3f}  FRR={(gen<t).mean():.4f}  TAR={1-(gen<t).mean():.4f}")

# 1:N closed-set identification, leave-one-out
correct = rank = 0
for i in range(len(E)):
    s = S[i].copy(); s[i] = -2
    best = {}
    for j in range(len(E)):
        if j == i: continue
        best[L[j]] = max(best.get(L[j], -2), s[j])
    order = sorted(best, key=best.get, reverse=True)
    if order[0] == L[i]: correct += 1
    rank += order.index(L[i]) + 1
print(f"\n1:N leave-one-out  rank-1 accuracy = {correct}/{len(E)} = {correct/len(E)*100:.2f}%   mean rank = {rank/len(E):.2f}")
np.savez(f"{ROOT}/bench/gallery_{REC.replace('.onnx','')}.npz", E=E, L=L, paths=np.array(paths))
