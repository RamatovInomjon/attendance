#!/usr/bin/env python3
"""Replay native-4K clips through the real pipeline and report every pass.

    python bench/eval_4k.py
    python bench/eval_4k.py --dir data/recordings_4k --recognizer <other.onnx>

Why this and not scripts/benchmark.py: that one replays `data/recordings/`,
which is written at `record_width=1920`. Every head box in those clips is HALF
its live size, so face_px, the size gate and every score derived from them are
measured in a domain the cameras do not operate in. It is the mistake that made
the size gate look like the binding constraint when at true 4K it rejects 4%.

This reports, per completed pass: who the consensus named, the committed score,
how many frames agreed, and the head size actually seen - so a threshold can be
set from corridor evidence instead of scaled off the enrolment gallery.
"""
from __future__ import annotations

import argparse
import collections
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
    ap.add_argument("--recognizer", default=None,
                    help="override settings.recognizer_model")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from app.config import settings
    if args.recognizer:
        settings.recognizer_model = args.recognizer

    from app.core.onnx_env import preload_cuda_libs
    preload_cuda_libs()
    import cv2
    from app.core.direction import config_from_camera
    from app.core.pipeline import CameraPipeline
    from app.core.stream import Frame
    from app.db.models import Camera
    from app.db.session import session_scope
    from app.services.enrollment import load_gallery
    from sqlalchemy import select

    gallery = load_gallery()
    with session_scope() as s:
        cams = {c.name: config_from_camera(c)
                for c in s.execute(select(Camera)).scalars()}

    thr = args.threshold if args.threshold is not None \
        else settings.threshold_for(settings.recognizer_model)
    print(f"  recognizer {Path(settings.recognizer_model).name}  threshold {thr:.3f}")
    print(f"  gallery    {len(gallery)} embeddings / {gallery.n_people} people\n")

    clips = sorted(glob.glob(str(ROOT / args.dir / "*" / "*.mp4")))
    if not clips:
        raise SystemExit(f"  no clips under {args.dir}")

    passes, all_px, all_scores = [], [], []
    pipes: dict = {}
    for path in clips:
        cam = Path(path).parent.name
        if cam not in pipes:
            p = CameraPipeline(cam, gallery, direction_cfg=cams.get(cam))
            p.threshold = thr
            pipes[cam] = p
        pipe = pipes[cam]
        pipe.tracks.clear()

        cap = cv2.VideoCapture(path)
        t = 1_700_000_000.0
        n = 0
        while n < args.frames:
            ok, img = cap.read()
            if not ok:
                break
            res = pipe.process(Frame(img, t + n / 20.0, n))
            n += 1
            for ct in res.completed:
                passes.append((cam, Path(path).name, ct))
        cap.release()
        for ct in pipe.flush():
            passes.append((cam, Path(path).name, ct))

    named = [p for p in passes if p[2].employee_id is not None]
    print(f"  {len(clips)} clips -> {len(passes)} passes, {len(named)} recognised "
          f"({len(named)/max(1,len(passes))*100:.0f}%)\n")

    print(f"  {'clip':34s} {'who':22s} {'score':>6s} {'emb':>4s} {'px':>4s} {'dir':>7s}")
    for cam, clip, ct in passes:
        who = ct.name if ct.employee_id else f"(miss, near {ct.best_score:.3f})"
        print(f"  {clip[:34]:34s} {who[:22]:22s} {ct.best_score:6.3f} "
              f"{ct.embedded_frames:4d} {ct.face_px:4d} {ct.direction:>7s}")
        all_px.append(ct.face_px)
        all_scores.append((ct.employee_id, ct.best_score, ct.embedded_frames))

    tally = collections.Counter(ct.name for _c, _f, ct in named)
    print(f"\n  identities: {dict(tally)}")
    if all_px:
        a = np.array(all_px)
        print(f"  face_px    min={a.min()} median={np.median(a):.0f} max={a.max()}")
    if named:
        sc = np.array([ct.best_score for _c, _f, ct in named])
        print(f"  scores     min={sc.min():.3f} median={np.median(sc):.3f} max={sc.max():.3f}")
        emb = np.array([ct.embedded_frames for _c, _f, ct in named])
        print(f"  emb frames min={emb.min()} median={np.median(emb):.0f} max={emb.max()}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "recognizer": Path(settings.recognizer_model).name,
            "threshold": thr, "clips": len(clips),
            "passes": len(passes), "recognised": len(named),
            "identities": dict(tally),
            "rows": [{"clip": f, "camera": c, "name": ct.name,
                      "employee_id": ct.employee_id, "score": ct.best_score,
                      "margin": ct.best_margin, "emb": ct.embedded_frames,
                      "face_px": ct.face_px, "direction": ct.direction}
                     for c, f, ct in passes],
        }, indent=2))
        print(f"\n  written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
