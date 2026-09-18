#!/usr/bin/env python3
"""Replay native-4K corridor clips and align every usable face THREE ways.

    python bench/extract_native_faces.py --clips <dir>/Entrance/20260907_1140*.mp4 \
        --out /media/inomjon/T7/face_eval_20260915

WHY THIS EXISTS
---------------
`bench/compare_recognizers.py` scored production's saved crops. Those crops were
aligned by THIS pipeline - DFA-mobilenet landmarks and a grid_sample warp - and
the challenger it was comparing (`ir50S3v2s_sr10final`) was trained on crops
from a different aligner: RetinaFace, DFA-ResNet50 landmarks and a cv2
getAffineMatrix warp (projectAI/1/data_aug/vb30k/gpu_align003.py). A recognizer
judged on another model's alignment can lose for reasons that have nothing to do
with its weights, so this produces, from the SAME native frame and the SAME
head box, all three:

  pipe   what production feeds the recognizer today (FaceAligner, sharp mode)
  d50    same native square crop, DFA-ResNet50 landmarks, ArcFace similarity
         warp with cv2.warpAffine - the challenger's training landmarks and warp
  r003   the challenger's full training aligner on that crop, RetinaFace included

Detection, tracking and the quality gates are production's own (head+person
detector at detect_width, BYTETracker on person boxes, `quality.assess`), so a
face is kept here exactly when production would have embedded it. No recognizer
runs in this script at all: which faces exist, and what they are labelled, must
not depend on either model under test.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ALIGN003 = Path("/home/inomjon/projectAI/1/data_aug/vb30k")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=0, help="smoke test")
    ap.add_argument("--stride", type=int, default=2,
                    help="process every Nth frame (tracker rate is adjusted to match)")
    args = ap.parse_args()

    from app.config import settings
    from app.core.onnx_env import best_providers, preload_cuda_libs
    preload_cuda_libs()
    from app.core.aligner import FaceAligner
    from app.core.geometry import aligned_to_uint8
    from app.core.head_detector import CLS_HEAD, CLS_PERSON, HeadDetector
    from app.core.pipeline import _head_in
    from app.core.quality import assess
    from app.core.tracker import FaceTracker

    sys.path.insert(0, str(ALIGN003))
    from gpu_align003 import Pipeline003
    from gpu_align import align112

    tz = settings.tz
    det = HeadDetector(settings.model_path(settings.head_model),
                       size=settings.head_input, conf=settings.head_conf)
    ali = FaceAligner(settings.model_path(settings.aligner_model),
                      crop_size=settings.align_crop_size,
                      margin=settings.align_margin, mode=settings.align_mode)
    p003 = Pipeline003(str(ALIGN003 / "cfg003"), best_providers())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for clip in args.clips:
        clip = Path(clip)
        cam = clip.parent.name
        dest = out / f"{cam}_{clip.stem}.npz"
        if dest.exists():
            print(f"  skip {dest.name} (exists)", flush=True)
            continue
        start = datetime.strptime(clip.stem[:15], "%Y%m%d_%H%M%S").replace(tzinfo=tz)
        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            # A missing or unreadable clip used to yield "0 frames, 0 faces" and
            # an empty npz that looked like a quiet corridor.
            raise SystemExit(f"  cannot open {clip}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        tracker = FaceTracker(
            frame_rate=max(1, int(round(fps / args.stride))),
            track_high_thresh=settings.track_high_thresh,
            track_low_thresh=settings.track_low_thresh,
            match_thresh=settings.track_match_thresh,
            track_buffer=settings.track_buffer)

        rows = {k: [] for k in ("frame", "t", "track", "box", "face_px", "sharp",
                                "yaw", "pitch", "ascore", "pipe", "d50", "d50_score",
                                "r003", "r003_ok")}
        idx, t0, kept = -1, time.time(), 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            idx += 1
            if args.max_frames and idx >= args.max_frames:
                break
            if idx % args.stride:
                continue
            ok, img = cap.retrieve()
            if not ok:
                break
            h, w = img.shape[:2]
            sc = settings.detect_width / w
            small = cv2.resize(img, (settings.detect_width, int(round(h * sc))),
                               interpolation=cv2.INTER_LINEAR)
            dets = det.detect(small, want=None)
            heads = [d for d in dets if d.cls == CLS_HEAD and d.score >= settings.head_conf]
            persons = [d for d in dets if d.cls == CLS_PERSON]
            hb = (np.stack([d.box for d in heads]).astype(np.float32) / sc
                  if heads else np.zeros((0, 4), np.float32))
            src = persons if persons else heads
            tb = (np.stack([d.box for d in src]).astype(np.float32) / sc
                  if src else np.zeros((0, 4), np.float32))
            ts = (np.array([d.score for d in src], np.float32)
                  if src else np.zeros((0,), np.float32))
            tracked = tracker.update(tb, ts)

            pending = []
            for tid, box, _score in tracked:
                head = _head_in(box, hb) if persons else box
                if head is None:
                    continue
                x1, y1, x2, y2 = head
                face_px = float(max(x2 - x1, y2 - y1))
                on = min(x2, w) > max(x1, 0) and min(y2, h) > max(y1, 0)
                if face_px >= settings.min_face_px and on:
                    pending.append((tid, np.asarray(head, np.float32)))
            if not pending:
                continue

            faces = ali.align(img, [b for _t, b in pending], is_bgr=True)
            if len(faces) != len(pending):
                continue
            for (tid, box), f in zip(pending, faces):
                q = assess(box, f.aligned, f.landmarks, f.score,
                           min_face_px=settings.min_face_px,
                           min_laplacian_var=settings.min_laplacian_var,
                           min_aligner_score=settings.min_aligner_score,
                           max_yaw_deg=settings.max_yaw_deg,
                           max_pitch_deg=settings.max_pitch_deg)
                if not q.ok:
                    continue
                # The native square crop production's aligner reads, in BGR.
                native, _cb = ali._cut(img, box)
                pts, s50 = p003.dfa_landmarks(native)
                d50 = cv2.cvtColor(align112(native, pts), cv2.COLOR_BGR2RGB)
                r003, _ds, _fs = p003.align(native)
                rows["frame"].append(idx)
                rows["t"].append((start + timedelta(seconds=idx / fps)).timestamp())
                rows["track"].append(tid)
                rows["box"].append(box)
                rows["face_px"].append(q.face_px)
                rows["sharp"].append(q.sharpness)
                rows["yaw"].append(q.yaw)
                rows["pitch"].append(q.pitch)
                rows["ascore"].append(q.aligner_score)
                rows["pipe"].append(aligned_to_uint8(f.aligned))
                rows["d50"].append(d50)
                rows["d50_score"].append(s50)
                rows["r003"].append(cv2.cvtColor(r003, cv2.COLOR_BGR2RGB)
                                    if r003 is not None else np.zeros((112, 112, 3), np.uint8))
                rows["r003_ok"].append(r003 is not None)
                kept += 1
        cap.release()
        np.savez_compressed(dest, **{k: np.asarray(v) for k, v in rows.items()},
                            camera=cam, clip=str(clip), fps=fps)
        print(f"  {dest.name}: {idx + 1} frames, {kept} faces, "
              f"{len(set(rows['track']))} tracks, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
