#!/usr/bin/env python3
"""Can tracklets from THIS camera be linked to each other by face?

    python bench/cctv_face_probe.py --video /media/.../D23_...mp4 --start 900 --seconds 240

WHY THIS RUNS BEFORE ANY EXTRACTION
-----------------------------------
Cross-tracklet identity linking is what turns a pile of tracklets into a ReID
training set: without it every tracklet is its own identity, and a model learns
"same clip, same person" rather than "same person". The linking key CANNOT be
the body, because the body model is the thing being trained - using it to make
its own labels teaches it whatever it already believes. On the corridor the key
was the face, and it worked because people walk toward the camera.

This camera is a wide-angle unit looking down on a cafe terrace. Faces are
angled, often turned to a table or a phone, and the lens distorts. Whether a
face template can be built at all - and at what inter-pupil distance - decides
whether cross-tracklet linking is possible here or whether the honest answer is
tracklet-as-identity.

So: run the real pipeline over a busy window and report, per completed track,
how many frames cleared the face gates and the best IPD achieved. The corridor
needs IPD >= 20 px for a face to link rather than mislink; that same bar is the
one to judge these against.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=int, default=0, help="skip N frames first")
    ap.add_argument("--seconds", type=int, default=240)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    import cv2
    from app.config import settings
    from app.core.onnx_env import preload_cuda_libs
    preload_cuda_libs()
    from app.core.pipeline import CameraPipeline
    from app.core.stream import Frame
    from app.services.enrollment import load_gallery

    gallery = load_gallery()
    pipe = CameraPipeline("cctv", gallery)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"  cannot open {args.video}")

    for _ in range(args.start):
        if not cap.grab():
            break
    want = int(args.seconds * args.fps)
    done, t0, i = [], time.time(), 0
    body_crops = 0
    while i < want:
        ok, img = cap.read()
        if not ok:
            break
        res = pipe.process(Frame(img, 1_700_000_000.0 + i / args.fps, i))
        body_crops += len(res.body_crops)
        done.extend(res.completed)
        i += 1
        if i % 500 == 0:
            print(f"    {i}/{want} frames, {len(done)} tracks closed, "
                  f"{body_crops} body crops", end="\r", flush=True)
    cap.release()
    done.extend(pipe.flush())
    dt = time.time() - t0
    print(f"\n  {i} frames in {dt:.0f}s ({i / max(dt, 1e-9):.1f} fps), "
          f"{len(done)} completed track(s), {body_crops} body crops\n")

    if not done:
        print("  no tracks - nothing to judge")
        return 0

    rows = []
    for t in done:
        rows.append({
            "track": int(t.track_id),
            "duration_s": round(float(t.duration_s), 1),
            "body_crops_possible": int(t.embedded_frames),
            "face_frames": int(getattr(t, "face_frames", 0) or 0),
            "best_ipd": round(float(getattr(t, "best_ipd", 0.0) or 0.0), 1),
            "has_template": getattr(t, "face_template", None) is not None,
            "travel": round(float(t.travel), 3),
        })
    ipd_min = settings.pseudo_face_ipd_min
    usable = [r for r in rows if r["has_template"] and r["best_ipd"] >= ipd_min]
    withface = [r for r in rows if r["has_template"]]
    longt = [r for r in rows if r["duration_s"] >= 3.0]

    print(f"  tracks total                    : {len(rows)}")
    print(f"  ...lasting >= 3 s               : {len(longt)}")
    print(f"  ...with ANY face template       : {len(withface)}"
          f"  ({100 * len(withface) / len(rows):.0f}%)")
    print(f"  ...with IPD >= {ipd_min:.0f} px (linkable) : {len(usable)}"
          f"  ({100 * len(usable) / len(rows):.0f}%)   <- the ceiling on "
          f"cross-tracklet linking")
    if withface:
        ipds = np.array([r["best_ipd"] for r in withface])
        print(f"\n  best IPD among tracks that have a face: "
              f"p25={np.percentile(ipds, 25):.1f} median={np.median(ipds):.1f} "
              f"p75={np.percentile(ipds, 75):.1f} max={ipds.max():.1f}")
    mv = np.array([r["travel"] for r in rows])
    print(f"  travel (fraction of frame): median={np.median(mv):.3f} "
          f"p90={np.percentile(mv, 90):.3f}   "
          f"{int((mv < 0.05).sum())} tracks barely moved (seated)")
    print(f"\n  longest tracks:")
    for r in sorted(rows, key=lambda x: -x["duration_s"])[:8]:
        print(f"    t{r['track']:<5d} {r['duration_s']:6.1f}s  "
              f"faces={r['face_frames']:3d} ipd={r['best_ipd']:5.1f} "
              f"travel={r['travel']:.3f}")

    if args.json:
        p = ROOT / args.json
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"video": args.video, "frames": i,
                                 "fps_processed": i / max(dt, 1e-9),
                                 "tracks": rows}, indent=2))
        print(f"\n  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
