#!/usr/bin/env python3
"""How much of this footage actually contains people?

    python bench/footage_density.py --dir /media/inomjon/T7/cctv --every 300

WHY MEASURE BEFORE PROCESSING
-----------------------------
Running the pipeline over footage is the expensive step: 8.8 hours at 15 fps is
475,000 frames, and a 2560x1440 frame costs roughly what a 4K one does once it
is downscaled for detection. If the camera watches an empty terrace for most of
the day, most of that spend produces nothing - and which HOURS are worth
processing is not guessable from the file names.

So this samples one frame every `--every` frames, runs only the head/person
detector on it, and reports the share of samples containing a person and how
they are distributed across each file. It is two orders of magnitude cheaper
than a real pass and answers the only question that matters first: what is
worth a real pass.

It deliberately reads SEQUENTIALLY and skips with grab(), rather than seeking.
These files are HEVC with open GOPs; `set(CAP_PROP_POS_FRAMES)` on them makes
the decoder complain ("Could not find ref with POC") and can hand back a
half-reconstructed frame, which would be scored as an empty one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--every", type=int, default=300, help="sample 1 frame in N")
    ap.add_argument("--limit", type=int, default=0, help="files, 0 = all")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    import cv2
    from app.config import settings
    from app.core.onnx_env import preload_cuda_libs
    preload_cuda_libs()
    from app.core.head_detector import CLS_HEAD, CLS_PERSON, HeadDetector

    hd = HeadDetector(settings.model_path(settings.head_model),
                      size=settings.head_input,
                      conf=min(settings.head_conf, settings.track_low_thresh))
    files = sorted(Path(args.dir).glob("*.mp4"))
    if args.limit:
        files = files[: args.limit]
    print(f"  {len(files)} file(s), sampling 1 frame in {args.every}\n")
    print(f"  {'file':32s} {'sampled':>8s} {'with a person':>14s} {'busiest run':>28s}")

    out = []
    for f in files:
        cap = cv2.VideoCapture(str(f))
        if not cap.isOpened():
            print(f"  {f.name:32s}  cannot open")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        n = seen = withp = 0
        hits = []          # (seconds, persons)
        while True:
            ok = cap.grab()
            if not ok:
                break
            n += 1
            if n % args.every:
                continue
            ok, img = cap.retrieve()
            if not ok or img is None:
                continue
            seen += 1
            dets = hd.detect(img)
            p = sum(1 for d in dets
                    if d.cls == CLS_PERSON and d.score >= settings.head_conf)
            h = sum(1 for d in dets
                    if d.cls == CLS_HEAD and d.score >= settings.head_conf)
            if p or h:
                withp += 1
                hits.append((n / fps, p, h))
        cap.release()
        # the densest ten-minute window, which is where a real pass should start
        best, best_c = None, 0
        for t0 in range(0, int(n / fps) + 1, 300):
            c = sum(1 for s, _p, _h in hits if t0 <= s < t0 + 600)
            if c > best_c:
                best, best_c = t0, c
        share = 100 * withp / max(seen, 1)
        span = (f"{best // 60:02d}:{best % 60:02d}-{(best + 600) // 60:02d}:"
                f"{(best + 600) % 60:02d} ({best_c} hits)") if best is not None else "-"
        print(f"  {f.name:32s} {seen:8d} {withp:6d} ({share:4.1f}%) {span:>28s}")
        out.append({"file": f.name, "frames": n, "sampled": seen,
                    "with_person": withp, "share_pct": share,
                    "max_persons": max((p for _s, p, _h in hits), default=0),
                    "hits": [[round(s, 1), p, h] for s, p, h in hits]})

    tot_s = sum(o["sampled"] for o in out)
    tot_w = sum(o["with_person"] for o in out)
    print(f"\n  overall: {tot_w}/{tot_s} sampled frames contain a person "
          f"({100 * tot_w / max(tot_s, 1):.1f}%)")
    print(f"  most people in one sampled frame: "
          f"{max((o['max_persons'] for o in out), default=0)}")
    if args.json:
        p = ROOT / args.json
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
