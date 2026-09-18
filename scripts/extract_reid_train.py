#!/usr/bin/env python3
"""Extract body crops PER PASS with the face embedding, for ReID training.

    python scripts/extract_reid_train.py --dir data/recordings --top 30

Difference from `extract_persons.py`, and why:

* One folder per PASS, not per person. Identity is assigned later by
  clustering the face embeddings, so the extractor must not pre-commit to
  a name. Unknown passes are the point here, not a leftover.
* The face embedding of each pass is written out. That is what lets two
  passes of the same unenrolled person be linked afterwards, using a
  modality independent of the body - the same principle that makes the
  existing test set trustworthy.
* More crops per pass (--top 30). Three is right for human review;
  training wants every usable view of the pass.

WHY NOT JUST TRAIN ON THE NAMED PEOPLE
--------------------------------------
The evaluation set (data/test_reid, 45 people) was built from these same
recordings. Training on the enrolled population would put the test
identities in the training set. The unenrolled passes are disjoint from
it by construction, so they are the ones that can be trained on.

Reads recordings, writes images. Never touches the attendance database.
"""
from __future__ import annotations

import argparse
import glob
import heapq
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="data/recordings")
    ap.add_argument("--out", default="data/reid_passes")
    ap.add_argument("--top", type=int, default=30,
                    help="highest-scoring body crops kept per pass")
    ap.add_argument("--frames", type=int, default=4000)
    ap.add_argument("--width", type=int, default=0,
                    help="body crop width; 0 = keep native crop size. Native is "
                         "the point: the existing ReID set is all 128x256, which "
                         "made the input-resolution axis unmeasurable.")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from app.config import settings
    from app.core.onnx_env import preload_cuda_libs
    preload_cuda_libs()
    import cv2
    from app.core.direction import config_from_camera
    from app.core.pipeline import CameraPipeline, _body_crop
    from app.core.stream import Frame
    from app.db.models import Camera
    from app.db.session import session_scope
    from app.services.enrollment import load_gallery
    from sqlalchemy import select

    gallery = load_gallery()
    with session_scope() as s:
        cams = {c.name: config_from_camera(c)
                for c in s.execute(select(Camera)).scalars()}
    thr = settings.threshold_for(settings.recognizer_model)

    clips = sorted(glob.glob(str(ROOT / args.dir / "*" / "*.mp4")))
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        raise SystemExit(f"  no clips under {args.dir}")
    out_root = ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    ledger = out_root / "_progress.json"
    done: set = set()
    if args.resume and ledger.is_file():
        done = set(json.loads(ledger.read_text()).get("clips_done", []))
        print(f"  resuming: {len(done)} clip(s) already done")

    print(f"  recognizer {Path(settings.recognizer_model).name}  threshold {thr:.3f}")
    print(f"  gallery    {len(gallery)} embeddings / {gallery.n_people} people")
    print(f"  {len(clips)} clip(s) -> {out_root}   top {args.top}/pass"
          f"   width {'native' if not args.width else args.width}\n", flush=True)

    prev_all = settings.save_all_frames
    settings.save_all_frames = True
    tie = itertools.count()
    pipes: dict = {}
    n_pass = n_named = n_crop = 0

    def flush(ct, tops, clip_stem, cam):
        nonlocal n_pass, n_named, n_crop
        n_pass += 1
        n_named += ct.employee_id is not None
        best = sorted(tops.get(ct.track_id, []), key=lambda x: -x[0])[: args.top]
        if not best:
            return
        d = out_root / f"{clip_stem}_t{ct.track_id}"
        d.mkdir(parents=True, exist_ok=True)
        for score, _seq, body in best:
            cv2.imwrite(str(d / f"{clip_stem}_t{ct.track_id}_s{score:.3f}.jpg"),
                        body, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            n_crop += 1
        vec = None
        if getattr(ct, "vector", None) is not None:
            v = np.asarray(ct.vector, dtype=np.float32).ravel()
            n = float(np.linalg.norm(v))
            vec = (v / n).tolist() if n > 0 else None
        (d / "_pass.json").write_text(json.dumps({
            "clip": f"{clip_stem}.mp4", "camera": cam, "track_id": ct.track_id,
            "name": ct.name or None, "employee_id": ct.employee_id,
            "nearest_employee_id": getattr(ct, "nearest_employee_id", None),
            "score": round(float(ct.best_score), 4),
            "margin": round(float(ct.best_margin), 4),
            "embedded_frames": ct.embedded_frames, "face_px": ct.face_px,
            "direction": ct.direction, "duration_s": round(ct.duration_s, 1),
            "crops": len(best), "face_vector": vec,
        }, ensure_ascii=False), encoding="utf-8")

    try:
        for n, path in enumerate(clips, 1):
            cam, clip_stem = Path(path).parent.name, Path(path).stem
            if clip_stem in done:
                continue
            if cam not in pipes:
                p = CameraPipeline(cam, gallery, direction_cfg=cams.get(cam))
                p.threshold = thr
                pipes[cam] = p
            pipe = pipes[cam]
            pipe.tracks.clear()
            pipe.tracker.reset()
            tops: dict[int, list] = {}
            cap = cv2.VideoCapture(path)
            t0, i = 1_700_000_000.0, 0
            try:
                while i < args.frames:
                    ok, img = cap.read()
                    if not ok:
                        break
                    res = pipe.process(Frame(img, t0 + i / 20.0, i))
                    i += 1
                    live = {t.track_id: t for t in res.tracks}
                    for c in res.candidates:
                        st = live.get(c.track_id)
                        if st is None or st.person_box is None:
                            continue
                        body = (_body_crop(img, st.person_box, max_w=args.width)
                                if args.width else _body_crop(img, st.person_box))
                        if body is None:
                            continue
                        h = tops.setdefault(c.track_id, [])
                        heapq.heappush(h, (float(c.score), next(tie), body))
                        if len(h) > args.top:
                            heapq.heappop(h)
                    for ct in res.completed:
                        flush(ct, tops, clip_stem, cam)
                        tops.pop(ct.track_id, None)
            finally:
                cap.release()
            for ct in pipe.flush():
                flush(ct, tops, clip_stem, cam)
            done.add(clip_stem)
            if n % 25 == 0 or n == len(clips):
                ledger.write_text(json.dumps(
                    {"dir": args.dir, "clips_done": sorted(done)}, indent=2))
                print(f"  [{n:4d}/{len(clips)}] {clip_stem:34s} "
                      f"{n_named}/{n_pass} named, {n_crop} crops", flush=True)
    finally:
        settings.save_all_frames = prev_all
        ledger.write_text(json.dumps(
            {"dir": args.dir, "clips_done": sorted(done)}, indent=2))

    print(f"\n  {n_pass} pass(es), {n_named} named, {n_pass - n_named} unenrolled")
    print(f"  {n_crop} crops -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
