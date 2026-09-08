#!/usr/bin/env python3
"""Score the whole-pass face re-match against the per-frame vote, on real video.

    python bench/tracklet_recheck_eval.py --dir data/recordings --limit 60
    python bench/tracklet_recheck_eval.py --dir data/recordings --sweep
    python bench/tracklet_recheck_eval.py --dir data/recordings --json out.json

WHY THIS RUNS THE PIPELINE RATHER THAN READING STORED CROPS
-----------------------------------------------------------
The research measurement behind `tracklet_face_threshold` was made on the
320-px BODY crops the ReID collector stores, with the live quality gates
relaxed. That answers "could a whole-pass template recover these people", and
it is the reason the setting exists at all.

It is NOT what the deployed rule does. `CameraWorker._second_chance` combines
the embeddings that ALREADY PASSED the live gates, computed from full-frame
faces on the capture thread - a different, smaller and better set of frames
than the offline experiment used. Quoting the offline number as if it were this
rule's would be quoting a measurement of something else.

So this drives the real `CameraPipeline` over recorded clips and applies the
real rule to the `CompletedTrack`s that come out. Everything measured here is
what the deployment does.

WHAT IS AND IS NOT GROUND TRUTH
-------------------------------
There is no label for a pass the face path missed - that is what makes it a
miss. Two things can still be measured honestly, and they answer different
questions:

* **AGREEMENT** (precision evidence). On passes the live vote DID name, does
  the whole-pass template independently reach the same person? A disagreement
  is a real error by one of the two rules, and every one of them is printed.
  This is the closest thing to truth available, and it is the number to watch.

* **RECOVERY** (the gain, unlabelled). On passes the live vote left unnamed,
  how many does the template name? Nothing here proves those are right. What
  the sweep shows is how fast the count grows as the threshold falls, which is
  what a threshold sitting in a safe place looks like: recovery that rises
  smoothly while agreement stays at 100%.

A recovery whose person was never seen anywhere else in the corpus is flagged
separately (`isolated`), because a name that appears exactly once and nowhere
near another sighting of the same person is the shape a false accept takes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def clips(root: Path, camera: str, limit: int, stride: int = 1) -> list[Path]:
    """Clips to run, in recording order.

    `stride` samples ACROSS the corpus rather than taking a prefix. Clip names
    are timestamps, so `--limit 100` alone would measure the first two hours of
    one morning - and this corridor's traffic, lighting and who walks through
    it all vary across a day. A stride keeps the sample cheap without making it
    a sample of one time of day.
    """
    d = root / camera
    if not d.is_dir():
        return []
    files = sorted(d.glob("*.mp4"))[::max(1, stride)]
    return files[:limit] if limit else files


def run_camera(name: str, files: list[Path], gallery) -> list:
    """Every completed track in these clips, through the production pipeline."""
    import cv2
    from app.core.pipeline import CameraPipeline
    from app.core.stream import Frame
    from app.config import settings

    pipe = CameraPipeline(name, gallery)
    out, idx = [], 0
    for n, f in enumerate(files, 1):
        cap = cv2.VideoCapture(str(f))
        if not cap.isOpened():
            continue
        # A clip is a discontinuity from the previous one: reset the tracker so
        # a track cannot span two recordings made minutes apart. The live
        # worker does exactly this on a stream gap.
        out.extend(pipe.flush())
        pipe.tracker.reset()
        nth, t0 = 0, time.time()
        while True:
            ok, img = cap.read()
            if not ok:
                break
            nth += 1
            if nth % settings.process_every_nth:
                continue
            idx += 1
            res = pipe.process(Frame(image=img, ts=t0 + idx * 0.05, index=idx))
            out.extend(res.completed)
        cap.release()
        print(f"    [{name}] {n}/{len(files)} {f.name}  tracks so far {len(out)}",
              end="\r", flush=True)
    out.extend(pipe.flush())
    print(f"    [{name}] {len(files)} clip(s), {len(out)} completed track(s)"
          + " " * 20)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="data/recordings")
    ap.add_argument("--cameras", default="Entrance,Exit")
    ap.add_argument("--limit", type=int, default=60,
                    help="clips per camera after striding; 0 for all")
    ap.add_argument("--stride", type=int, default=1,
                    help="take every Nth clip, so a limited run still spans "
                         "the whole recording day")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from app.config import settings
    from app.services.enrollment import load_gallery

    root = ROOT / args.dir
    gallery = load_gallery()
    print(f"  gallery {len(gallery)} embeddings / {gallery.n_people} people")
    print(f"  live threshold {settings.threshold_for(settings.recognizer_model):.3f}"
          f"  margin {settings.second_best_margin:.3f}"
          f"  consensus {settings.vote_consensus:.2f}"
          f"  min {settings.vote_min_recognitions}")
    print(f"  tracklet threshold {settings.tracklet_face_threshold:.3f}"
          f"  margin {settings.tracklet_face_margin:.3f}"
          f"  min frames {settings.tracklet_min_face_frames}\n")

    tracks = []
    for cam in [c.strip() for c in args.cameras.split(",") if c.strip()]:
        files = clips(root, cam, args.limit, args.stride)
        if not files:
            print(f"    [{cam}] no clips under {root / cam}")
            continue
        tracks.extend((cam, t) for t in run_camera(cam, files, gallery))

    if not tracks:
        raise SystemExit("  no completed tracks; nothing to score")

    # Who the live path named anywhere in the corpus. A recovery of somebody
    # who appears nowhere else is the shape a false accept takes.
    live_people = {t.employee_id for _c, t in tracks if t.employee_id is not None}

    def judge(thr: float, mg: float, minf: int) -> dict:
        agree = disagree = no_call = 0
        recovered = isolated = 0
        rec_by_person: Counter = Counter()
        wrong: list = []
        for cam, t in tracks:
            if t.face_template is None or t.face_frames < minf:
                if t.employee_id is not None:
                    no_call += 1
                continue
            m = gallery.match(t.face_template, thr, mg)
            if t.employee_id is not None:
                if m.employee_id is None:
                    no_call += 1
                elif m.employee_id == t.employee_id:
                    agree += 1
                else:
                    disagree += 1
                    wrong.append({
                        "camera": cam, "track": t.track_id,
                        "live": gallery.name(t.employee_id),
                        "tracklet": gallery.name(m.employee_id),
                        "live_score": round(float(t.best_score), 3),
                        "tracklet_score": round(float(m.score), 3),
                        "frames": int(t.face_frames),
                        "ipd": round(float(t.best_ipd), 1)})
            elif m.employee_id is not None:
                recovered += 1
                rec_by_person[m.employee_id] += 1
                if m.employee_id not in live_people:
                    isolated += 1
        judged = agree + disagree
        return {"threshold": thr, "margin": mg, "min_frames": minf,
                "agree": agree, "disagree": disagree, "no_call": no_call,
                "agreement": (agree / judged) if judged else float("nan"),
                "recovered": recovered, "isolated": isolated,
                "recovered_people": len(rec_by_person), "wrong": wrong}

    named = sum(1 for _c, t in tracks if t.employee_id is not None)
    unnamed = len(tracks) - named
    with_face = sum(1 for _c, t in tracks if t.face_template is not None)
    print(f"\n  {len(tracks)} passes: {named} named by the live vote, "
          f"{unnamed} unnamed; {with_face} carry a face template")

    r = judge(settings.tracklet_face_threshold, settings.tracklet_face_margin,
              settings.tracklet_min_face_frames)
    print(f"\n  AT THE CONFIGURED SETTING")
    print(f"    agreement on named passes : {r['agree']}/{r['agree'] + r['disagree']}"
          f"  ({r['agreement'] * 100:.1f}%)   {r['no_call']} not re-called")
    print(f"    recovered from unnamed    : {r['recovered']}"
          f"  ({r['recovered'] / unnamed * 100:.1f}% of {unnamed})"
          f"  across {r['recovered_people']} people")
    print(f"    ...of which ISOLATED      : {r['isolated']}"
          f"   (named nobody the live path ever saw)")
    if named:
        print(f"    attendance passes        : {named} -> "
              f"{named + r['recovered']}  (+{r['recovered'] / named * 100:.0f}%)")
    for w in r["wrong"][:10]:
        print(f"      DISAGREE {w['camera']} t{w['track']}: live {w['live']} "
              f"({w['live_score']:.3f}) vs tracklet {w['tracklet']} "
              f"({w['tracklet_score']:.3f}), {w['frames']} frames, ipd {w['ipd']}")

    rows = [r]
    if args.sweep:
        print(f"\n  {'thr':>6s} {'margin':>7s} {'agree':>7s} {'disagr':>7s} "
              f"{'agree%':>7s} {'recov':>7s} {'isolated':>9s}")
        for thr in (0.20, 0.24, 0.28, 0.30, 0.32, 0.36, 0.40):
            for mg in (0.0, settings.tracklet_face_margin):
                x = judge(thr, mg, settings.tracklet_min_face_frames)
                rows.append(x)
                print(f"  {thr:6.2f} {mg:7.3f} {x['agree']:7d} {x['disagree']:7d} "
                      f"{x['agreement'] * 100:6.1f}% {x['recovered']:7d} "
                      f"{x['isolated']:9d}")

    if args.json:
        out = ROOT / args.json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"passes": len(tracks), "named": named, "unnamed": unnamed,
             "with_face": with_face, "results": rows}, indent=2))
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
