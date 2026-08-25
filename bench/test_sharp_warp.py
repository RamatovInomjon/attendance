"""Can we beat the reference warp?

The reference warps the 112x112 crop out of the aligner's internal 160x160
tensor.  That means a 292 px crop is downsampled to 160, then resampled to 112 -
and a small 126 px crop is UPsampled to 160 and then back down to 112.

Alternative: use the aligner only for landmarks, and warp from the original
full-resolution crop straight to 112.  One resample instead of two.

This deviates from the reference PyTorch numerics (which the ONNX export was
verified against), so it is only legitimate if enrolment and inference both use
it - which they do, since we control both. Test whether it actually helps.
"""
import sys, glob, os
sys.path.insert(0, "/home/inomjon/projectAI/face_rec/face_recognition_airi")
import numpy as np, cv2

from app.config import settings
from app.core import geometry as G
from app.core.aligner import FaceAligner
from app.core.detector import build_detector
from app.core.quality import sharpness_of
from app.core.recognizer import FaceRecognizer

ROOT = "/home/inomjon/projectAI/face_rec/face_recognition_airi"
GAL = f"{ROOT}/face_id_users"
det = build_detector("yolo", settings.model_path(settings.detector_model), imgsz=1280, conf=0.35)
rec = FaceRecognizer(settings.model_path(settings.recognizer_model))


class SharpAligner(FaceAligner):
    """Same landmarks, but warp from the native-resolution crop."""
    def align_sharp(self, frame_rgb, boxes):
        crops, chw = [], []
        for b in boxes:
            c, cb = self._cut(frame_rgb, b)
            if c.size == 0: continue
            c = cv2.resize(c, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
            crops.append(c); chw.append(G.to_normalized_chw(c))
        if not crops: return []
        batch = np.stack(chw).astype(np.float32)
        ldmk, _b, score, _resized = self.session.run(None, {"image": batch})
        # theta computed in the NATIVE crop frame, not the 160 frame
        thetas = np.stack([G.landmarks_to_theta(ldmk[i], self.crop_size, G.OUTPUT_SIZE)
                           for i in range(len(crops))])
        aligned = G.warp_batch(batch, thetas, G.OUTPUT_SIZE)
        return [(aligned[i], float(score[i][0])) for i in range(len(crops))]


people = sorted(d for d in os.listdir(GAL) if os.path.isdir(f"{GAL}/{d}") and not d.startswith("_"))
cache = []
for pid, person in enumerate(people):
    for p in sorted(glob.glob(f"{GAL}/{person}/*.png")):
        bgr = cv2.imread(p); d = det.detect(bgr)
        if not d: continue
        b = max(d, key=lambda x:(x.box[2]-x.box[0])*(x.box[3]-x.box[1])).box
        cache.append((pid, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), b))

def stats(E, L):
    E = np.stack(E); L = np.array(L); S = E @ E.T
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
    return dp, corr/len(E)*100, gen.mean(), gen.min(), imp.max()

print(f"{'mode':<34} {'sharp':>7} {'d-prime':>9} {'rank-1':>8} {'gen mean':>9} {'min gen':>8} {'imp max':>8}")
print("-"*88)
for margin in (1.30,):
    for cs in (160, 256, 384):
        ali = SharpAligner(settings.model_path(settings.aligner_model), crop_size=cs, margin=margin)
        # reference path
        E,L,sh = [],[],[]
        for pid, rgb, b in cache:
            f = ali.align(rgb, [b])
            if not f: continue
            E.append(rec.embed(f[0].aligned[None])[0]); L.append(pid); sh.append(sharpness_of(f[0].aligned))
        d = stats(E,L)
        print(f"{'reference (warp from 160)  cs='+str(cs):<34} {np.mean(sh):7.0f} {d[0]:9.2f} {d[1]:7.2f}% {d[2]:9.4f} {d[3]:8.4f} {d[4]:8.4f}")
        # sharp path
        E,L,sh = [],[],[]
        for pid, rgb, b in cache:
            f = ali.align_sharp(rgb, [b])
            if not f: continue
            E.append(rec.embed(f[0][0][None])[0]); L.append(pid); sh.append(sharpness_of(f[0][0]))
        d = stats(E,L)
        print(f"{'SHARP     (warp from crop) cs='+str(cs):<34} {np.mean(sh):7.0f} {d[0]:9.2f} {d[1]:7.2f}% {d[2]:9.4f} {d[3]:8.4f} {d[4]:8.4f}")
        print()
