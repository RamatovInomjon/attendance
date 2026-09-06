#!/usr/bin/env python3
"""Replay clips through the real pipeline and keep every track's trajectory.

    python bench/dump_trajectories.py --dir data/recordings_4k --out data/bench/traj_4k

Writes, per clip, `<camera>/<clip>.json` holding one record per completed
track - identity, the pipeline's own direction verdict, and EVERY trajectory
point as [t, cx, cy, area, synth, edge] - plus `<clip>_strip.jpg`, one frame
per second with the tracked boxes and the tripwire drawn, so a human can label
what each track actually did.

bench/direction_eval.py then re-decides those points offline with any version
of the direction algorithm and scores it against the labels. That is the
experiment loop: the expensive part (decode, detect, track, recognise) runs
once; the cheap part (the verdict) can be changed and re-run in seconds.

Why the whole pipeline and not just detector+tracker: the tracks must be the
ones the pipeline would really produce - same association, same synthesised
heads, same pruning - or the evaluation measures a different system.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _clip_start(path: Path) -> float:
    try:
        return datetime.strptime("_".join(path.stem.split("_")[:2]), "%Y%m%d_%H%M%S").timestamp()
    except ValueError:
        return 1_700_000_000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/recordings_4k")
    ap.add_argument("--out", default="data/bench/traj_4k")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--limit", type=int, default=0, help="clips per camera (0 = all)")
    ap.add_argument("--strip-every", type=float, default=1.0, help="seconds between strip frames")
    ap.add_argument("--only", default="", help="substring filter on clip name")
    args = ap.parse_args()

    from app.config import settings
    from app.core.onnx_env import preload_cuda_libs
    preload_cuda_libs()
    import cv2
    from sqlalchemy import select
    from app.core.direction import config_from_camera
    from app.core.pipeline import CameraPipeline
    from app.core.stream import Frame
    from app.db.models import Camera
    from app.db.session import session_scope
    from app.services.enrollment import load_gallery

    gallery = load_gallery()
    with session_scope() as s:
        cams = {c.name: config_from_camera(c)
                for c in s.execute(select(Camera)).scalars()}

    class Dumping(CameraPipeline):
        """Keeps each dead track's trajectory next to its CompletedTrack."""

        def _prune(self, now):
            dying = {tid: t for tid, t in self.tracks.items()
                     if now - t.last_seen > settings.track_max_age_s}
            out = super()._prune(now)
            for ct in out:
                st = dying.get(ct.track_id)
                ct.trajectory = st.trajectory if st is not None else None
                ct.first_seen_ts = st.first_seen if st is not None else 0.0
                ct.last_seen_ts = st.last_seen if st is not None else 0.0
            return out

    out_root = ROOT / args.out
    cam_dirs = sorted(p for p in (ROOT / args.dir).iterdir() if p.is_dir())
    total_tracks = 0
    for cam_dir in cam_dirs:
        cam = cam_dir.name
        cfg = cams.get(cam)
        if cfg is None:
            print(f"  {cam}: no camera row with this name, skipped")
            continue
        pipe = Dumping(cam, gallery, direction_cfg=cfg)
        clips = sorted(cam_dir.glob("*.mp4"))
        if args.only:
            clips = [c for c in clips if args.only in c.name]
        if args.limit:
            clips = clips[: args.limit]
        (out_root / cam).mkdir(parents=True, exist_ok=True)
        print(f"  {cam}: {len(clips)} clip(s)  line={cfg.line} inside={cfg.inside_side} "
              f"depth_grows_inward={cfg.depth_grows_inward}")

        for path in clips:
            target = out_root / cam / (path.stem + ".json")
            if target.exists():
                continue
            pipe.tracks.clear()
            pipe.tracker.reset()
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                print(f"    cannot open {path.name}")
                continue
            t0 = _clip_start(path)
            n = 0
            completed = []
            strip, next_strip = [], 0.0
            w = h = 0
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                h, w = img.shape[:2]
                ts = t0 + n / args.fps
                res = pipe.process(Frame(img, ts, n))
                completed.extend(res.completed)
                if ts - t0 >= next_strip:
                    next_strip += args.strip_every
                    small = cv2.resize(img, (640, int(h * 640 / w)), interpolation=cv2.INTER_AREA)
                    sc = 640.0 / w
                    H, W = small.shape[:2]
                    if cfg.line:
                        (x1, y1), (x2, y2) = cfg.line
                        cv2.line(small, (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H)),
                                 (0, 255, 255), 1)
                    for st in res.tracks:
                        if ts - st.last_seen > 0.2:
                            continue
                        b = (st.box * sc).astype(int)
                        cv2.rectangle(small, (b[0], b[1]), (b[2], b[3]), (0, 200, 0), 1)
                        if st.face_box is not None:
                            fb = (st.face_box * sc).astype(int)
                            cv2.rectangle(small, (fb[0], fb[1]), (fb[2], fb[3]), (0, 0, 255), 1)
                        cv2.putText(small, f"t{st.track_id} {st.name[:10]}", (b[0], max(10, b[1] - 3)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(small, f"{ts - t0:5.1f}s", (4, H - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
                    strip.append(small)
                n += 1
            cap.release()
            completed.extend(pipe.flush())

            records = []
            for ct in completed:
                traj = getattr(ct, "trajectory", None)
                pts = (traj._buf[:traj.n].round(6).tolist() if traj is not None else [])
                records.append({
                    "camera": cam, "clip": path.name, "track_id": ct.track_id,
                    "first_seen": getattr(ct, "first_seen_ts", 0.0) - t0,
                    "last_seen": getattr(ct, "last_seen_ts", 0.0) - t0,
                    "duration_s": ct.duration_s,
                    "employee_id": ct.employee_id, "name": ct.name,
                    "score": round(float(ct.best_score), 4),
                    "embedded": ct.embedded_frames, "gated": ct.gated_frames,
                    "face_px": ct.face_px,
                    "direction": ct.direction, "direction_reason": ct.direction_reason,
                    "frame_w": w, "frame_h": h,
                    "points": pts,
                })
            total_tracks += len(records)
            target.write_text(json.dumps({
                "camera": cam, "clip": path.name, "fps": args.fps, "frames": n,
                "line": cfg.line, "inside_side": cfg.inside_side,
                "depth_grows_inward": cfg.depth_grows_inward,
                "tracks": records,
            }))
            if strip:
                cols = 6
                rows = [strip[i:i + cols] for i in range(0, len(strip), cols)]
                Hs, Ws = strip[0].shape[:2]
                canvas = np.zeros((len(rows) * Hs, cols * Ws, 3), np.uint8)
                for r, row in enumerate(rows):
                    for c, im in enumerate(row):
                        canvas[r * Hs:(r + 1) * Hs, c * Ws:(c + 1) * Ws] = im
                cv2.imwrite(str(out_root / cam / (path.stem + "_strip.jpg")), canvas,
                            [cv2.IMWRITE_JPEG_QUALITY, 80])
            who = ", ".join(f"t{r['track_id']}:{r['name'] or '?'}:{r['direction'][:2]}"
                            for r in records)
            print(f"    {path.name}: {n} frames, {len(records)} track(s)  {who}")
    print(f"  done: {total_tracks} tracks under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
