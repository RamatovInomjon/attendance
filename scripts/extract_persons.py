#!/usr/bin/env python3
"""Run recordings through the pipeline and file each person's body crops by name.

    python scripts/extract_persons.py --dir data/recordings_4k
    python scripts/extract_persons.py --dir data/recordings --top 3
    python scripts/extract_persons.py --dir data/recordings --no-unknown

Output, one folder per person, both cameras together:

    data/persons/
        Inomjon_Ramatov/
            20260831_114832_Entrance_t1_s0.518.jpg      <- best of the pass
            20260831_114832_Entrance_t1_s0.503.jpg      <- 2nd
            20260831_114832_Entrance_t1_s0.491.jpg      <- 3rd
            _passes.json
        unknown_t7_20260826_131218_Exit/                <- one pass nobody matched
            20260826_131218_Exit_t7_s0.164.jpg
            ...

WHY BODY CROPS AND NOT FACES
----------------------------
A person is recognisable to a human by build, clothing and posture; a 112x112
aligned face frequently is not - least of all on the crops that turn out to be
wrong, which is exactly when somebody needs to look. These folders are for human
review and for labelling, so they hold the image a human can actually judge.

WHY THREE PER PASS AND NOT ONE
------------------------------
One frame can flatter or libel a pass. Three spread across the approach show
whether the identity held up as the person got closer, and give a labeller
something to disagree with.

UNKNOWN PASSES
--------------
Filed under their own folder named for the track, so each is one person-pass
that can be reviewed and named. The track id alone is not unique - it restarts
per clip - so the clip is part of the folder name.

This reads recordings and writes images. It never touches the attendance
database, so it is safe to run against a live install.
"""
from __future__ import annotations

import argparse
import glob
import heapq
import itertools
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _slug(name: str) -> str:
    """A folder name that survives every filesystem, still readable."""
    s = re.sub(r"[^\w\s-]", "", str(name), flags=re.UNICODE).strip()
    return re.sub(r"\s+", "_", s) or "Unnamed"


CROP = re.compile(r"^(?P<clip>\d{8}_\d{6})_(?P<cam>[A-Za-z]+)_t(?P<track>\d+)"
                  r"_s(?P<score>[0-9.]+)\.jpg$")


def summarise(out_root: Path) -> None:
    """Report from the images on disk, not from what this run happens to hold.

    The crop filename carries the clip, the camera and the score, so a summary
    survives an interrupted run - which matters, because the first full pass
    over 760 clips was killed by a reboot at 57% and the per-folder JSON was
    only written at the very end. Everything below is recoverable from the
    files themselves.
    """
    people, unknown_passes = {}, 0
    for d in sorted(out_root.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith("unknown_t"):
            unknown_passes += 1
            continue
        seen = {}
        for f in d.glob("*.jpg"):
            m = CROP.match(f.name)
            if not m:
                continue
            cam = m.group("cam")
            key = (m.group("clip"), cam, m.group("track"))
            seen.setdefault(cam, set()).add(key)
        if seen:
            people[d.name] = seen

    if not people:
        print("  nothing on disk yet")
        return
    cams = sorted({c for v in people.values() for c in v})
    print(f"\n  {'person':34s} " + "  ".join(f"{c[:9]:>9s}" for c in cams)
          + "   both?")
    both = 0
    for who in sorted(people, key=lambda k: -sum(len(v) for v in people[k].values())):
        v = people[who]
        cells = "  ".join(f"{len(v.get(c, ())):9d}" for c in cams)
        ok = all(v.get(c) for c in cams)
        both += ok
        print(f"  {who[:34]:34s} {cells}   {'yes' if ok else 'NO'}")
    print(f"\n  {both} of {len(people)} people seen on BOTH cameras; "
          f"{len(people) - both} on only one")
    print(f"  {unknown_passes} unrecognised pass(es) in their own folders")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="data/recordings_4k",
                    help="replay root: <dir>/<CameraName>/*.mp4")
    ap.add_argument("--out", default="data/persons")
    ap.add_argument("--top", type=int, default=3,
                    help="highest-scoring body crops to keep per pass")
    ap.add_argument("--frames", type=int, default=4000, help="max frames per clip")
    ap.add_argument("--width", type=int, default=320, help="body crop width")
    ap.add_argument("--no-unknown", action="store_true",
                    help="skip passes that matched nobody")
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--limit", type=int, default=0, help="stop after N clips")
    ap.add_argument("--resume", action="store_true",
                    help="skip clips already recorded in <out>/_progress.json. "
                         "A full run is ~an hour; a laptop reboot should not "
                         "cost all of it.")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the summary from what is already on disk and "
                         "exit, without running the pipeline")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the calibrated recognition threshold. Below "
                         "the calibrated value this WILL admit false accepts - "
                         "it is a coverage probe, not a configuration change.")
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

    thr = (args.threshold if args.threshold is not None
           else settings.threshold_for(settings.recognizer_model))
    calibrated = settings.threshold_for(settings.recognizer_model)
    clips = sorted(glob.glob(str(ROOT / args.dir / "*" / "*.mp4")))
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        raise SystemExit(f"  no clips under {args.dir}")
    out_root = ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        summarise(out_root)
        return 0

    ledger_path = out_root / "_progress.json"
    done: set = set()
    if args.resume and ledger_path.is_file():
        done = set(json.loads(ledger_path.read_text()).get("clips_done", []))
        print(f"  resuming: {len(done)} clip(s) already processed")

    print(f"  recognizer {Path(settings.recognizer_model).name}  threshold {thr:.3f}"
          + ("" if abs(thr - calibrated) < 1e-9 else
             f"   (CALIBRATED IS {calibrated:.3f} - this run admits false accepts)"))
    print(f"  gallery    {len(gallery)} embeddings / {gallery.n_people} people")
    print(f"  {len(clips)} clip(s) from {args.dir} -> {out_root}")
    print(f"  keeping the top {args.top} body crop(s) per pass\n")

    # Candidates carry a per-frame score; this is what makes a top-N possible
    # without changing how the live vote keeps its single best.
    prev_all = settings.save_all_frames
    settings.save_all_frames = True

    saved, records = defaultdict(int), defaultdict(list)
    n_pass = n_named = n_nobody = 0
    tie = itertools.count()          # keeps heapq off the ndarray on score ties
    pipes: dict = {}

    def flush_pass(ct, tops, clip_stem, cam):
        nonlocal n_pass, n_named, n_nobody
        n_pass += 1
        named = ct.employee_id is not None
        n_named += named
        if not named and args.no_unknown:
            return
        folder = (_slug(ct.name) if named
                  else f"unknown_t{ct.track_id}_{clip_stem}")
        best = sorted(tops.get(ct.track_id, []), key=lambda x: -x[0])[: args.top]
        if not best:
            n_nobody += 1
        d = out_root / folder
        d.mkdir(parents=True, exist_ok=True)
        for score, _seq, body in best:
            cv2.imwrite(str(d / f"{clip_stem}_t{ct.track_id}_s{score:.3f}.jpg"),
                        body, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            saved[folder] += 1
        records[folder].append({
            "clip": f"{clip_stem}.mp4", "camera": cam, "track_id": ct.track_id,
            "name": ct.name or None, "employee_id": ct.employee_id,
            "score": round(float(ct.best_score), 4),
            "margin": round(float(ct.best_margin), 4),
            "embedded_frames": ct.embedded_frames, "face_px": ct.face_px,
            "direction": ct.direction, "duration_s": round(ct.duration_s, 1),
            "crops": len(best),
        })

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
            # Track ids restart per clip, and so must the crops held against them.
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
                        body = _body_crop(img, st.person_box, max_w=args.width)
                        if body is None:
                            continue
                        h = tops.setdefault(c.track_id, [])
                        heapq.heappush(h, (float(c.score), next(tie), body))
                        if len(h) > args.top:
                            heapq.heappop(h)      # drop the weakest kept so far

                    for ct in res.completed:
                        flush_pass(ct, tops, clip_stem, cam)
                        tops.pop(ct.track_id, None)
            finally:
                cap.release()
            for ct in pipe.flush():
                flush_pass(ct, tops, clip_stem, cam)

            done.add(clip_stem)
            if n % 25 == 0 or n == len(clips):
                # Checkpointed, not written once at the end: an hour of work
                # should not evaporate because the machine went to sleep.
                ledger_path.write_text(json.dumps(
                    {"dir": args.dir, "threshold": thr,
                     "clips_done": sorted(done)}, indent=2))
                # flush=True: stdout is block-buffered when redirected to a
                # file, so a long run shows nothing at all until it finishes -
                # which is exactly when progress stops being useful.
                print(f"  [{n:4d}/{len(clips)}] {clip_stem:34s} "
                      f"{n_named}/{n_pass} recognised, {sum(saved.values())} crops",
                      flush=True)
    finally:
        settings.save_all_frames = prev_all

    ledger_path.write_text(json.dumps(
        {"dir": args.dir, "threshold": thr, "clips_done": sorted(done)}, indent=2))
    for who, rows in records.items():
        f = out_root / who / "_passes.json"
        prior = json.loads(f.read_text()) if f.is_file() else []
        f.write_text(json.dumps(prior + rows, indent=2, ensure_ascii=False))

    # Who was seen on WHICH camera. The question a day's crops should answer is
    # whether the same people appear on both - an entrance without a matching
    # exit is either a miss or somebody still inside.
    by_cam: dict = {}
    for who, rows in records.items():
        if who.startswith("unknown_t"):
            continue
        c = by_cam.setdefault(who, {})
        for r in rows:
            c[r["camera"]] = c.get(r["camera"], 0) + 1

    known = {k: v for k, v in saved.items() if not k.startswith("unknown_t")}
    unk = {k: v for k, v in saved.items() if k.startswith("unknown_t")}
    print(f"\n  {'=' * 66}")
    print(f"  {n_named}/{n_pass} passes recognised "
          f"({n_named / max(n_pass, 1) * 100:.0f}%)")
    cams_seen = sorted({c for v in by_cam.values() for c in v})
    hdr = "  ".join(f"{c[:9]:>9s}" for c in cams_seen)
    print(f"    {'person':34s} {hdr}   both?")
    both = one = 0
    for who in sorted(by_cam, key=lambda k: -sum(by_cam[k].values())):
        v = by_cam[who]
        cells = "  ".join(f"{v.get(c, 0):9d}" for c in cams_seen)
        ok = all(v.get(c, 0) > 0 for c in cams_seen)
        both += ok
        one += not ok
        print(f"    {who[:34]:34s} {cells}   {'yes' if ok else 'NO'}")
    print(f"\n    {both} person(s) seen on BOTH cameras, {one} on only one")
    if unk:
        print(f"    {'unknown (one folder per pass)':38s} {sum(unk.values()):5d} "
              f"crop(s)  {len(unk)} pass(es)")
    print(f"  {'=' * 66}")
    if n_nobody:
        print(f"  NOTE: {n_nobody} pass(es) yielded no body crop - the head was "
              f"tracked with no associated person box, so nothing to crop.")
    print(f"  written to {out_root}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
