#!/usr/bin/env python3
"""Domain adaptation: add real corridor crops to the gallery.

The single biggest remaining accuracy lever.  The gallery is built from clean
1280x720 enrolment photos; the cameras see small, angled, motion-blurred faces
from 3 m up.  That domain gap is why live scores peak around 0.5 while
gallery-to-gallery scores sit at 0.90.

Adding a handful of *confirmed* corridor crops per person pulls the gallery into
the deployment domain and typically lifts live scores substantially.

Two-phase by design, because auto-enrolling whatever the matcher believes would
let one wrong match poison an identity permanently:

    phase 1   collect      capture high-confidence crops to data/live_enroll/
    phase 2   confirm      you review the folders, delete anything wrong
    phase 3   commit       --commit adds what survived to the gallery, tagged
                           `live:` and floored like any other corridor crop
                           (see app/services/augment.py)

Usage:
    python scripts/enroll_from_live.py 300        # collect for 5 minutes
    #  ... review data/live_enroll/<person>/ and delete bad crops ...
    python scripts/enroll_from_live.py --commit
"""
import logging, shutil, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2, numpy as np
from sqlalchemy import select

from app.config import settings
from app.core.aligner import FaceAligner
from app.core.detector import build_detector
from app.core.geometry import aligned_to_uint8, to_normalized_chw
from app.core.quality import assess
from app.core.recognizer import FaceRecognizer
from app.core.stream import RtspSource
from app.core.tracker import FaceTracker
from app.db.models import Camera, Employee, FaceEmbedding
from app.db.session import session_scope
from app.services import augment
from app.services.augment import TAG
from app.services.enrollment import load_gallery

logging.basicConfig(level=logging.WARNING, format="%(message)s")
OUT = Path("data/live_enroll")
COLLECT_MIN_SCORE = 0.32      # above the measured impostor ceiling (0.184)
COLLECT_MIN_MARGIN = 0.10
MAX_PER_PERSON = 12


def commit():
    """Add reviewed crops to the gallery as corridor references.

    Written the way app/services/augment.py writes one: tagged `live:` so the
    review page can list and remove them, and floored at
    max(threshold, augment_live_floor) so a corridor face answers to the same
    higher bar as every other. This used to insert untagged rows with no floor
    - judged at the global threshold, invisible to the review page, and exempt
    from the one policy that makes holding corridor crops safe.
    """
    rec = FaceRecognizer(settings.model_path(settings.recognizer_model))
    floor = max(settings.threshold_for(settings.recognizer_model),
                settings.augment_live_floor)
    with session_scope() as s:
        known = {ext: (eid, full) for eid, ext, full in s.execute(
            select(Employee.id, Employee.external_id, Employee.full_name)).all()}

    # Embed first, write second: the batch embed must not run inside the
    # write transaction while the capture threads wait on it.
    batches = []
    for folder in sorted(p for p in OUT.iterdir() if p.is_dir()):
        if folder.name not in known:
            print(f"  ?? no employee '{folder.name}' - skipped"); continue
        crops = sorted(folder.glob("*.png"))
        if not crops:
            continue
        # crops are already the canonical 112x112 aligned output
        batch = np.stack([to_normalized_chw(
            cv2.cvtColor(cv2.imread(str(c)), cv2.COLOR_BGR2RGB)) for c in crops])
        batches.append((folder.name, crops, rec.embed(batch)))

    added = 0
    with session_scope() as s:
        for ext, crops, embs in batches:
            emp_id, full = known[ext]
            for c, e in zip(crops, embs):
                s.add(FaceEmbedding(
                    employee_id=emp_id, source_file=f"{TAG}live_enroll/{ext}/{c.name}",
                    vector=e.astype(np.float32).tobytes(), dim=int(e.shape[0]),
                    model_name=settings.recognizer_model, quality=1.0,
                    threshold=floor,
                ))
                added += 1
            print(f"  + {full:<30} {len(crops)} live crops  (floor {floor:.3f})")
    print(f"\nadded {added} live embeddings")
    if added:
        # What augment.add() does after an insert: every live row's floor
        # re-checked against the corridor probes, and any lookalike logged.
        augment.recalibrate()
    g = load_gallery()
    print(f"gallery now {len(g)} embeddings / {g.n_people} people")
    print("POST /api/gallery/reload (or restart) to pick this up")


def collect(seconds):
    g = load_gallery()
    counts, lock = {}, threading.Lock()

    def watch(cid, name, role, url):
        det = build_detector("yolo", settings.model_path(settings.detector_model),
                             imgsz=settings.detect_width, conf=settings.detect_conf)
        ali = FaceAligner(settings.model_path(settings.aligner_model),
                          crop_size=settings.align_crop_size, margin=settings.align_margin)
        rec = FaceRecognizer(settings.model_path(settings.recognizer_model))
        trk = FaceTracker(frame_rate=12)
        src = RtspSource(url, name=name, transport="tcp", queue_size=2).start()
        t0 = time.time()
        while not src.connected and time.time() - t0 < 20:
            time.sleep(0.3)
        if not src.connected:
            print(f"!! {name} no connection"); return

        t0, nth = time.time(), 0
        while time.time() - t0 < seconds:
            f = src.read(timeout=2.0)
            if f is None: continue
            nth += 1
            if nth % 2: continue
            h, w = f.image.shape[:2]
            sc = settings.detect_width / w
            dets = det.detect(cv2.resize(f.image, (settings.detect_width, int(h * sc))))
            if not dets:
                trk.update(np.zeros((0,4),np.float32), np.zeros((0,),np.float32)); continue
            boxes = np.stack([d.box for d in dets]) / sc
            tracked = trk.update(boxes, np.array([d.score for d in dets], np.float32))
            if not tracked: continue
            rgb = cv2.cvtColor(f.image, cv2.COLOR_BGR2RGB)
            faces = ali.align(rgb, [b for _t, b, _s in tracked])
            if not faces: continue
            embs = rec.embed(np.stack([x.aligned for x in faces]))
            for (tid, b, _s), af, emb in zip(tracked, faces, embs):
                q = assess(b, af.aligned, af.landmarks, af.score,
                           min_face_px=settings.min_face_px,
                           min_laplacian_var=settings.min_laplacian_var,
                           min_aligner_score=settings.min_aligner_score,
                           max_yaw_deg=settings.max_yaw_deg, max_pitch_deg=settings.max_pitch_deg)
                if not q.ok: continue
                m = g.match(emb, COLLECT_MIN_SCORE, COLLECT_MIN_MARGIN)
                if m.employee_id is None: continue
                with session_scope() as s:
                    emp = s.get(Employee, m.employee_id)
                    ext, full = emp.external_id, emp.full_name
                with lock:
                    if counts.get(ext, 0) >= MAX_PER_PERSON: continue
                    counts[ext] = counts.get(ext, 0) + 1
                    d = OUT / ext; d.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(d / f"{name}_{counts[ext]:02d}_{m.score:.3f}.png"),
                                cv2.cvtColor(aligned_to_uint8(af.aligned), cv2.COLOR_RGB2BGR))
                    print(f"  {full:<30} {m.score:.3f} (margin {m.margin:.3f})  [{counts[ext]}]")
        src.stop()

    with session_scope() as s:
        cams = [(c.id, c.name, c.role.value, c.rtsp_url)
                for c in s.execute(select(Camera).where(Camera.enabled.is_(True))).scalars()]
    print(f"collecting for {seconds}s  (min score {COLLECT_MIN_SCORE}, margin {COLLECT_MIN_MARGIN})\n")
    ts = [threading.Thread(target=watch, args=c, daemon=True) for c in cams]
    for t in ts: t.start()
    for t in ts: t.join()

    print(f"\ncollected into {OUT}/ :")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<24} {v}")
    print("\nNOW REVIEW THE FOLDERS and delete any crop that is the wrong person.")
    print("Then:  python scripts/enroll_from_live.py --commit")


if __name__ == "__main__":
    if "--commit" in sys.argv:
        commit()
    else:
        secs = next((int(a) for a in sys.argv[1:] if a.isdigit()), 300)
        if OUT.exists(): shutil.rmtree(OUT)
        OUT.mkdir(parents=True)
        collect(secs)
