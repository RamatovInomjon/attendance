#!/usr/bin/env python3
"""Set the recognition threshold from corridor evidence, per recognizer.

    python bench/calibrate_threshold.py --label 054_Inomjon
    python bench/calibrate_threshold.py --label 054_Inomjon \
        --recognizer <other_model.onnx>

The previous version of this file parsed two walkthrough LOG files, hardcoded
the quality gates as they stood in August (`al>=0.40` against today's 0.900) and
simulated the retired "first 3 of the last 5" vote. It could not answer the
question any more, so it is kept as `calibrate_threshold_logs.py.old` and this
replaces it: replay real clips through the real gates, and measure.

WHY THIS CANNOT BE SKIPPED WHEN CHANGING RECOGNIZER
---------------------------------------------------
Thresholds do not transfer between models: each puts its impostor
distribution on a different scale, so a value that is safe for one can accept
very nearly anybody on another. Nothing in the resulting data would show it -
the attendance rows all look normal, they just name the wrong people.

It is also worth re-running for the SAME model whenever the cameras, lighting
or mounting change, because what it measures is the corridor, not the model.

WHAT IS MEASURED
----------------
Every gate-passing face in the labelled clips is scored against each enrolled
person (best image per person, exactly as `Gallery.match` does):

  genuine  - the score against the person the clip actually shows
  impostor - the best score against ANY OTHER enrolled person

The reported operating point is the highest impostor observed, plus a margin.
A false accept records one person as another and nothing in the data reveals
it; a miss costs nothing, because the person is seen again on their next pass.
So the threshold sits above every impostor ever seen, not at some equal-error
compromise between the two.

WHY 4K CLIPS AND NOT `data/recordings/`
---------------------------------------
Those are written at `record_width=1920`, so every face in them is half its live
size. A threshold calibrated there is calibrated for a domain the cameras do not
operate in - the same mistake that made the face-size gate look binding when at
true 4K it rejects 4%.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/recordings_4k")
    ap.add_argument("--label", required=True,
                    help="enrolment folder of the person in the clips, "
                         "e.g. 054_Inomjon")
    ap.add_argument("--match", default=None,
                    help="comma-separated substrings; keep clips matching any")
    ap.add_argument("--recognizer", default=None)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from app.config import settings
    if args.recognizer:
        settings.recognizer_model = args.recognizer

    from app.core.onnx_env import preload_cuda_libs
    preload_cuda_libs()
    import cv2
    from app.core.aligner import FaceAligner
    from app.core.head_detector import HeadDetector, CLS_HEAD, CLS_PERSON
    from app.core.quality import assess
    from app.core.recognizer import FaceRecognizer
    from app.db.models import Employee
    from app.db.session import session_scope
    from app.services.enrollment import load_gallery
    from sqlalchemy import select

    gallery = load_gallery()
    with session_scope() as s:
        who = s.execute(select(Employee).where(
            Employee.folder == args.label)).scalar_one_or_none()
    if who is None:
        raise SystemExit(f"  no employee with folder {args.label!r}")
    truth = who.id
    print(f"  recognizer {Path(settings.recognizer_model).name}")
    print(f"  labelled   {who.full_name} (employee {truth})")
    print(f"  gallery    {len(gallery)} embeddings / {gallery.n_people} people\n")

    det = HeadDetector(settings.model_path(settings.head_model),
                       size=settings.head_input, conf=settings.head_conf)
    ali = FaceAligner(settings.model_path(settings.aligner_model),
                      crop_size=settings.align_crop_size,
                      margin=settings.align_margin, mode=settings.align_mode)
    rec = FaceRecognizer(settings.model_path(settings.recognizer_model),
                         batch_size=settings.embed_batch)

    clips = sorted(glob.glob(str(ROOT / args.dir / "*" / "*.mp4")))
    if args.match:
        wanted = [m.strip() for m in args.match.split(",") if m.strip()]
        clips = [c for c in clips if any(m in Path(c).name for m in wanted)]
    if not clips:
        raise SystemExit("  no clips matched")

    # Which gallery rows belong to the labelled person.
    mine = gallery.owner == truth
    if not mine.any():
        raise SystemExit(f"  {who.full_name} has no embeddings in the gallery")

    gen, imp, gated = [], [], 0
    for path in clips:
        cap = cv2.VideoCapture(path)
        n = 0
        while n < args.frames:
            ok, img = cap.read()
            if not ok:
                break
            n += 1
            h, w = img.shape[:2]
            sc = settings.detect_width / w
            small = cv2.resize(img, (settings.detect_width, int(round(h * sc))),
                               interpolation=cv2.INTER_LINEAR)
            heads = [d for d in det.detect(small, want=None) if d.cls == CLS_HEAD]
            boxes = [np.array(d.box, np.float32) / sc for d in heads]
            boxes = [b for b in boxes
                     if max(b[2] - b[0], b[3] - b[1]) >= settings.min_face_px
                     and min(b[2], w) > max(b[0], 0) and min(b[3], h) > max(b[1], 0)]
            if not boxes:
                continue
            faces = ali.align(img, boxes, is_bgr=True)
            keep = []
            for b, f in zip(boxes, faces):
                q = assess(b, f.aligned, f.landmarks, f.score,
                           min_face_px=settings.min_face_px,
                           min_laplacian_var=settings.min_laplacian_var,
                           min_aligner_score=settings.min_aligner_score,
                           max_yaw_deg=settings.max_yaw_deg,
                           max_pitch_deg=settings.max_pitch_deg)
                if q.ok:
                    keep.append(f)
                else:
                    gated += 1
            if not keep:
                continue
            embs = rec.embed(np.stack([f.aligned for f in keep]))
            sims = embs @ gallery.M.T                       # (K, N images)
            for row in sims:
                gen.append(float(row[mine].max()))
                imp.append(float(row[~mine].max()))
        cap.release()

    if not gen:
        raise SystemExit("  no gate-passing faces found")

    g, i = np.array(gen), np.array(imp)
    print(f"  {len(clips)} clips -> {len(g)} gate-passing faces "
          f"({gated} rejected by the quality gates)\n")
    for name, a in (("genuine ", g), ("impostor", i)):
        print(f"  {name}  min={a.min():.3f}  p05={np.percentile(a,5):.3f}  "
              f"median={np.median(a):.3f}  p95={np.percentile(a,95):.3f}  "
              f"max={a.max():.3f}")

    worst_imp, weakest_gen = float(i.max()), float(g.min())
    sep = weakest_gen - worst_imp
    print(f"\n  worst impostor  {worst_imp:.3f}")
    print(f"  weakest genuine {weakest_gen:.3f}")
    print(f"  separation      {sep:+.3f}")

    # A false accept is the failure that matters, so sit above every impostor
    # ever observed. 20% of the gap when there is one; otherwise say so plainly.
    if sep > 0:
        rec_thr = worst_imp + 0.20 * sep
        note = "above every observed impostor"
    else:
        rec_thr = worst_imp + 0.02
        note = ("OVERLAPPING - no threshold separates them on this data; "
                "this value still refuses every observed impostor but will "
                "also refuse some genuine frames")
    print(f"\n  recommended threshold {rec_thr:.3f}  ({note})")
    cur = settings.threshold_for(settings.recognizer_model)
    print(f"  currently configured  {cur:.3f}"
          + ("   <-- ACCEPTS OBSERVED IMPOSTORS" if cur <= worst_imp else "   ok"))
    print(f"\n  at {rec_thr:.3f}: {(g >= rec_thr).mean()*100:.0f}% of genuine "
          f"frames still match, {(i >= rec_thr).sum()} impostor frames accepted")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "recognizer": Path(settings.recognizer_model).name,
            "label": args.label, "employee_id": truth,
            "clips": len(clips), "faces": len(g), "gated": gated,
            "genuine": {"min": weakest_gen, "median": float(np.median(g)),
                        "max": float(g.max())},
            "impostor": {"median": float(np.median(i)), "max": worst_imp},
            "separation": sep, "recommended_threshold": rec_thr,
            "configured_threshold": cur,
        }, indent=2))
        print(f"\n  written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
