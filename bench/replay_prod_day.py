#!/usr/bin/env python3
"""Replay production footage through the new pipeline and compare with the day.

    python bench/replay_prod_day.py --clips data/gpu6_prod_20260907/clips \
        --db data/gpu6_prod_20260907/ematsy_prod.db --day 2026-09-07
    ... --json data/bench/replay_prod_20260907.json

WHAT MAKES THIS DIFFERENT FROM bench/tracklet_recheck_eval.py
--------------------------------------------------------------
That one scores the second-chance rule against ITSELF - does the whole-pass
template agree with the per-frame vote. Useful, and blind to one thing: whether
the replay resembles what production actually did.

This one has the production database for the same day. Every clip is a known
five-minute window of wall-clock time, so the events production recorded inside
that window are what the DEPLOYED system saw on the DEPLOYED hardware from the
live stream. That is the baseline, and it is not a reconstruction.

Three numbers per clip, and the comparison is between the second and third:

    prod   distinct employees production recorded in this window
    live   distinct employees this replay's per-frame vote named
    +trk   ... plus the ones the whole-pass template recovered

`prod` vs `live` is the FAITHFULNESS check. They should be close; where they
are not, the reasons are known and are listed in the output rather than hidden:
a strided clip set cuts tracks at its edges, the replay starts each clip with a
clean tracker, and a person walking across a clip boundary is one pass live and
two here - or none, if neither half reaches the vote's five-frame floor.

`live` vs `+trk` is the ANSWER: what the change is worth, measured on the same
frames, with the same gallery, on one real day.

RECOVERED NAMES ARE CHECKED, NOT COUNTED
----------------------------------------
A recovered name is not automatically right. Each one is classified:

    corroborated  production ALSO recorded that person that day - the replay
                  found them in a window where the live vote did not, which is
                  what "recovered a missed pass" looks like
    novel         that person appears nowhere in production's whole day. This
                  is the shape a false accept takes, and it is reported
                  separately and never folded into the gain.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLIP_S = 300.0          # session.json: segment_s
# Thresholds the recovery curve is drawn at, in one pass over the footage.
SWEEP = (0.15, 0.18, 0.215, 0.24, 0.27, 0.30, 0.33)


def clip_start(path: Path, tz) -> datetime | None:
    """`20260907_094036.mp4` -> local datetime. The recorder names by start."""
    try:
        return datetime.strptime(path.stem[:15], "%Y%m%d_%H%M%S").replace(tzinfo=tz)
    except ValueError:
        return None


def prod_events(db: Path, day: str) -> list[tuple[datetime, int, str]]:
    """(utc ts, employee_id, transition) for every standing event of the day."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "select ts, employee_id, transition from recognition_event "
        "where business_date = ? and employee_id is not null "
        "and voided_at is null", (day,)).fetchall()
    con.close()
    out = []
    for ts, emp, tr in rows:
        t = str(ts).replace("T", " ")[:26]
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                out.append((datetime.strptime(t, fmt).replace(
                    tzinfo=ZoneInfo("UTC")), int(emp), str(tr)))
                break
            except ValueError:
                continue
    return out


def run_clip(pipe, path: Path, every: int, reid=None) -> list:
    """Every completed track in one clip, through the production pipeline.

    When `reid` is given, the pipeline's periodic BODY CROPS are collected and
    aggregated per pass exactly as `ReidWorker` does, so each completed track
    carries the same body feature the deployment would store for it. That is
    what makes the grouping measurement below a measurement of production and
    not of a reimplementation.
    """
    import cv2
    from app.core.reid import aggregate, sharpness_of
    from app.core.stream import Frame
    from app.config import settings
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    out, idx, n = [], 0, 0
    crops: dict = {}
    t0 = time.time()
    while True:
        ok, img = cap.read()
        if not ok:
            break
        n += 1
        if n % every:
            continue
        idx += 1
        res = pipe.process(Frame(image=img, ts=t0 + idx * 0.05, index=idx))
        if reid is not None:
            for c in res.body_crops:
                crops.setdefault((c.track_id, round(c.first_seen, 3)),
                                 []).append((c.image, c.score))
        out.extend(res.completed)
    cap.release()
    out.extend(pipe.flush())

    if reid is not None:
        keep = settings.body_crop_keep_per_pass
        for t in out:
            got = crops.get((t.track_id, round(t.first_seen, 3)))
            t.body_feature = None
            if not got or len(got) < settings.reid_min_crops:
                continue
            # `_spread` picks crops as far apart in the pass as possible; here
            # the list is already in capture order, so an even stride over it
            # is the same selection.
            step = max(1, len(got) // keep)
            sel = got[::step][:keep]
            imgs = [g[0] for g in sel]
            t.body_feature = aggregate(reid.embed(imgs), [g[1] for g in sel],
                                       [sharpness_of(i) for i in imgs])
    # DROP THE PICTURES. A CompletedTrack carries the aligned face, the native
    # crop, the body crop and a 1400-px context frame - tens of megabytes per
    # clip, and this holds every pass of the day at once to compare them. The
    # scoring below reads only the identity, the template and the counters.
    for t in out:
        t.native = t.context = t.crop = t.score_crop = t.person_crop = None
        t.quality = None
    return out


GROUP_GRID = ((0.35, 0.75), (0.30, 0.60), (0.30, 0.75), (0.35, 0.60),
              (0.40, 0.60), (0.40, 0.75), (0.45, 0.60))


def group_unknowns(tracks, day: str, face_thr=None, body_thr=None) -> dict:
    """Run the production pseudo-gallery over the passes nobody named.

    The features here are the DEPLOYED ones: a face template built from
    full-4K frames by the live gates, and a body feature aggregated from the
    same periodic crops the ReID worker stores. The corresponding measurement
    in `bench/pseudo_group_eval.py` runs on 320-px stored crops instead, so its
    face features are degraded and its thresholds cannot be read across.
    """
    from datetime import date as _date
    from app.config import settings
    from app.db.models import ReidPass
    from app.db.session import session_scope
    from app.services.pseudo_gallery import PseudoGallery

    y, m, d = (int(x) for x in day.split("-"))
    bday = _date(y, m, d)
    base = datetime(y, m, d, tzinfo=ZoneInfo("UTC"))
    items = [(cam, t) for cam, t in tracks if t.employee_id is None]
    items.sort(key=lambda x: x[1].first_seen)

    face_ok = [t for _c, t in items
               if t.face_template is not None
               and t.best_ipd >= settings.pseudo_face_ipd_min]
    out = {"unnamed": len(items), "with_usable_face": len(face_ok)}
    if len(face_ok) >= 2:
        F = np.stack([t.face_template for t in face_ok])
        out["truth_people"] = len(set(truth_clusters(F, 0.45).tolist()))

    of, ob = settings.pseudo_face_threshold, settings.pseudo_body_threshold
    if face_thr is not None:
        settings.pseudo_face_threshold = face_thr
    if body_thr is not None:
        settings.pseudo_body_threshold = body_thr
    with session_scope() as s:
        g = PseudoGallery()
        labels = []
        for i, (cam, t) in enumerate(items):
            body = getattr(t, "body_feature", None)
            row = ReidPass(camera_id=1 if cam == "Entrance" else 2,
                           camera_name=cam, track_id=i, first_seen=base,
                           last_seen=base, business_date=bday,
                           direction="UNKNOWN",
                           dim=0 if body is None else len(body),
                           model_name=settings.reid_model)
            s.add(row)
            s.flush()
            p = g.place(s, row, face=t.face_template, body=body,
                        face_ipd=float(t.best_ipd),
                        body_model=settings.reid_model)
            labels.append(p.id if p is not None else -(i + 1))
        out.update(g.stats())
        out["groups"] = len(set(labels))
        out["face_threshold"] = settings.pseudo_face_threshold
        out["body_threshold"] = settings.pseudo_body_threshold
        # Pair precision/recall against the face-cluster truth. It is circular
        # with respect to the face threshold - see bench/group_calibrate.py -
        # so it is reported and NOT selected on. What it is good for here is
        # the comparison the crops corpus cannot make: these templates come
        # from 4K frames, and 72% of passes carry a usable face against 23.5%
        # on the stored 320-px crops.
        if len(face_ok) >= 2:
            idx = [i for i, (_c, t) in enumerate(items)
                   if t.face_template is not None
                   and t.best_ipd >= settings.pseudo_face_ipd_min]
            F = np.stack([items[i][1].face_template for i in idx])
            truth = truth_clusters(F, 0.45)
            lab = np.asarray(labels)[idx]
            same_p = lab[:, None] == lab[None, :]
            same_t = truth[:, None] == truth[None, :]
            iu = np.triu_indices(len(idx), k=1)
            pp, tt = same_p[iu], same_t[iu]
            tp, fp, fn = int((pp & tt).sum()), int((pp & ~tt).sum()), int((~pp & tt).sum())
            out["precision"] = tp / (tp + fp) if tp + fp else float("nan")
            out["recall"] = tp / (tp + fn) if tp + fn else float("nan")
            out["groups_on_covered"] = len(set(lab.tolist()))
        s.rollback()
    settings.pseudo_face_threshold, settings.pseudo_body_threshold = of, ob
    return out


def truth_clusters(F: np.ndarray, threshold: float) -> np.ndarray:
    """Connected components of "these two faces are the same person"."""
    n = len(F)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    S = F @ F.T
    for i in range(n):
        for j in np.where(S[i] >= threshold)[0]:
            if j > i:
                a, b = find(i), find(int(j))
                if a != b:
                    parent[a] = b
    lab, out = {}, []
    for i in range(n):
        r = find(i)
        out.append(lab.setdefault(r, len(lab)))
    return np.asarray(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True, help="dir with <Camera>/*.mp4")
    ap.add_argument("--db", required=True, help="production database snapshot")
    ap.add_argument("--day", required=True, help="business date, YYYY-MM-DD")
    ap.add_argument("--cameras", default="Entrance,Exit")
    ap.add_argument("--every", type=int, default=0,
                    help="process every Nth frame; 0 uses settings.process_every_nth")
    ap.add_argument("--limit", type=int, default=0, help="clips per camera")
    ap.add_argument("--no-group", action="store_true",
                    help="skip the pseudo-person grouping measurement")
    ap.add_argument("--save-templates", default=None,
                    help="npz of every pass's face template, body feature and "
                         "the identity the live vote gave it. THE NAMED ones "
                         "are the point: they carry a real label AND 4K "
                         "quality at once, which is the only way to calibrate "
                         "a face threshold both representatively and "
                         "non-circularly. Without this a threshold can only be "
                         "checked against clusters of the same feature, which "
                         "scores 100% by construction at the clustering "
                         "threshold and settles nothing.")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from app.config import settings
    from app.core.onnx_env import preload_cuda_libs
    from app.core.pipeline import CameraPipeline
    from app.services.enrollment import load_gallery
    preload_cuda_libs()
    from app.core.reid import PersonReID

    every = args.every or settings.process_every_nth
    tz = settings.tz
    gallery = load_gallery()
    names = gallery.names
    print(f"  gallery {len(gallery)} embeddings / {gallery.n_people} people "
          f"(from {Path(args.db).name})")
    print(f"  live thr {settings.threshold_for(settings.recognizer_model):.3f} "
          f"margin {settings.second_best_margin:.3f} consensus "
          f"{settings.vote_consensus} min {settings.vote_min_recognitions}")
    print(f"  tracklet thr {settings.tracklet_face_threshold:.3f} "
          f"margin {settings.tracklet_face_margin:.3f} "
          f"minframes {settings.tracklet_min_face_frames}")
    print(f"  frame stride {every}\n")

    reid = None if args.no_group else PersonReID(
        settings.model_path(settings.reid_model), batch_size=settings.embed_batch)
    if reid is not None:
        print(f"  reid {reid.model_name} {reid.dim}-d\n")

    events = prod_events(Path(args.db), args.day)
    prod_day_people = {e for _t, e, _tr in events}
    print(f"  production recorded {len(events)} standing event(s) that day "
          f"over {len(prod_day_people)} people\n")

    per_clip, all_tracks = [], []
    for cam in [c.strip() for c in args.cameras.split(",") if c.strip()]:
        files = sorted((Path(args.clips) / cam).glob("*.mp4"))
        if args.limit:
            files = files[: args.limit]
        if not files:
            print(f"    [{cam}] no clips")
            continue
        pipe = CameraPipeline(cam, gallery)
        for i, f in enumerate(files, 1):
            t0 = time.time()
            # Each clip is separated from the next by the stride, so it IS a
            # discontinuity: start it with a clean tracker, exactly as the live
            # worker does after a gap in the stream.
            pipe.flush()
            pipe.tracker.reset()
            tracks = run_clip(pipe, f, every, reid)
            start = clip_start(f, tz)
            end = start + timedelta(seconds=CLIP_S) if start else None
            prod_here = {e for t, e, _tr in events
                         if start and start <= t.astimezone(tz) < end}
            live = {t.employee_id for t in tracks if t.employee_id is not None}
            # Recovery is computed at a RANGE of thresholds, not just the
            # configured one. A replay of this corpus costs two hours, and
            # re-running it per threshold to draw one curve is the kind of
            # measurement nobody repeats - so the curve comes out of the run
            # that was already paid for. `rec` is the configured point.
            rec, curve = {}, {}
            for thr in SWEEP:
                got = {}
                for t in tracks:
                    if t.employee_id is not None or t.face_template is None:
                        continue
                    if t.face_frames < settings.tracklet_min_face_frames:
                        continue
                    m = gallery.match(t.face_template, thr,
                                      settings.tracklet_face_margin)
                    if m.employee_id is not None:
                        got[m.employee_id] = max(got.get(m.employee_id, 0.0),
                                                 float(m.score))
                curve[thr] = {int(k): round(v, 3) for k, v in got.items()}
                if abs(thr - settings.tracklet_face_threshold) < 1e-9:
                    rec = curve[thr]
            if not rec:
                for t in tracks:
                    if t.employee_id is not None or t.face_template is None:
                        continue
                    if t.face_frames < settings.tracklet_min_face_frames:
                        continue
                    m = gallery.match(t.face_template,
                                      settings.tracklet_face_threshold,
                                      settings.tracklet_face_margin)
                    if m.employee_id is not None:
                        rec[m.employee_id] = max(rec.get(m.employee_id, 0.0),
                                                 float(m.score))
            per_clip.append({
                "camera": cam, "clip": f.name,
                "start": start.isoformat() if start else None,
                "passes": len(tracks),
                "unnamed": sum(1 for t in tracks if t.employee_id is None),
                "prod": sorted(prod_here), "live": sorted(live),
                "recovered": {int(k): round(v, 3) for k, v in rec.items()},
                "curve": {str(k): sorted(v) for k, v in curve.items()},
            })
            all_tracks.extend((cam, t) for t in tracks)
            print(f"    [{cam}] {i}/{len(files)} {f.name} "
                  f"{len(tracks):3d} passes  prod {len(prod_here):2d}  "
                  f"live {len(live):2d}  +trk {len(set(rec) - live):2d}  "
                  f"({time.time() - t0:.0f}s)")

    if not per_clip:
        raise SystemExit("  nothing replayed")

    # ---- aggregate ------------------------------------------------------
    passes = sum(c["passes"] for c in per_clip)
    unnamed = sum(c["unnamed"] for c in per_clip)
    live_pairs = {(c["camera"], c["clip"], e) for c in per_clip for e in c["live"]}
    rec_pairs = {(c["camera"], c["clip"], e) for c in per_clip
                 for e in c["recovered"] if e not in c["live"]}
    prod_pairs = {(c["camera"], c["clip"], e) for c in per_clip for e in c["prod"]}

    corroborated = {p for p in rec_pairs if p[2] in prod_day_people}
    novel = rec_pairs - corroborated

    print(f"\n  {'-' * 68}")
    print(f"  {passes} passes replayed, {unnamed} left unnamed by the live vote")
    print(f"\n  DISTINCT (clip, person) sightings")
    print(f"    production recorded in these windows : {len(prod_pairs)}")
    print(f"    replay, live vote only              : {len(live_pairs)}")
    print(f"    replay, + whole-pass template       : "
          f"{len(live_pairs) + len(rec_pairs)}"
          f"   (+{len(rec_pairs)}, "
          f"{len(rec_pairs) / max(len(live_pairs), 1) * 100:.0f}%)")
    print(f"\n  RECOVERY CURVE (distinct clip-person sightings the live vote "
          f"missed)")
    print(f"    {'thr':>6s} {'recovered':>10s} {'corroborated':>13s} {'novel':>7s}")
    curve_rows = []
    for thr in SWEEP:
        pairs = {(c["camera"], c["clip"], e) for c in per_clip
                 for e in c.get("curve", {}).get(str(thr), [])
                 if e not in c["live"]}
        cor = {p for p in pairs if p[2] in prod_day_people}
        curve_rows.append({"threshold": thr, "recovered": len(pairs),
                           "corroborated": len(cor), "novel": len(pairs - cor)})
        mark = "  <- configured" if abs(thr - settings.tracklet_face_threshold) < 1e-9 else ""
        print(f"    {thr:6.3f} {len(pairs):10d} {len(cor):13d} "
              f"{len(pairs - cor):7d}{mark}")

    print(f"\n  the {len(rec_pairs)} recovered sighting(s) at the configured "
          f"threshold:")
    print(f"    corroborated (person seen elsewhere in production's day): "
          f"{len(corroborated)}")
    print(f"    novel        (person absent from the whole production day): "
          f"{len(novel)}")
    if novel:
        for cam, clip, e in sorted(novel):
            sc = next(c["recovered"][e] for c in per_clip
                      if c["clip"] == clip and c["camera"] == cam)
            print(f"      NOVEL {cam} {clip} -> {names.get(e, e)} ({sc:.3f})")

    missed_by_replay = prod_pairs - live_pairs
    found_by_replay = live_pairs - prod_pairs
    print(f"\n  faithfulness of the replay (live vote vs production):")
    print(f"    in both                       : {len(prod_pairs & live_pairs)}")
    print(f"    production only               : {len(missed_by_replay)}"
          f"   (clip edges, and 4K frames the live host processed at speed)")
    print(f"    replay only                   : {len(found_by_replay)}")
    recovered_of_missed = {p for p in rec_pairs if p in prod_pairs}
    print(f"    of the production-only ones, recovered by the template: "
          f"{len(recovered_of_missed)}")

    if args.save_templates:
        out = ROOT / args.save_templates
        out.parent.mkdir(parents=True, exist_ok=True)
        keep = [(c, t) for c, t in all_tracks if t.face_template is not None]
        if keep:
            np.savez_compressed(
                out,
                face=np.stack([t.face_template for _c, t in keep]),
                body=np.stack([
                    getattr(t, "body_feature", None)
                    if getattr(t, "body_feature", None) is not None
                    else np.zeros(512, np.float32) for _c, t in keep]),
                employee_id=np.array([t.employee_id if t.employee_id is not None
                                      else -1 for _c, t in keep], np.int64),
                ipd=np.array([t.best_ipd for _c, t in keep], np.float32),
                frames=np.array([t.face_frames for _c, t in keep], np.int32),
                camera=np.array([c for c, _t in keep]),
                first_seen=np.array([t.first_seen for _c, t in keep], np.float64))
            named = sum(1 for _c, t in keep if t.employee_id is not None)
            print(f"\n  saved {len(keep)} template(s) ({named} labelled) -> {out}")

    grouping = None
    group_sweep = []
    if not args.no_group:
        print(f"\n  GROUPING SWEEP on 4K-quality features")
        print(f"    {'face':>5s} {'body':>5s} {'groups':>7s} {'by face':>8s} "
              f"{'by body':>8s} {'prec':>7s} {'recall':>7s} {'vs truth':>9s}")
        for ft, bt in GROUP_GRID:
            r = group_unknowns(all_tracks, args.day, ft, bt)
            group_sweep.append(r)
            mark = ("  <- configured"
                    if (abs(ft - settings.pseudo_face_threshold) < 1e-9
                        and abs(bt - settings.pseudo_body_threshold) < 1e-9) else "")
            print(f"    {ft:5.2f} {bt:5.2f} {r['groups']:7d} {r['by_face']:8d} "
                  f"{r['by_body']:8d} {r.get('precision', float('nan')) * 100:6.1f}% "
                  f"{r.get('recall', float('nan')) * 100:6.1f}% "
                  f"{r.get('groups_on_covered', 0):4d}/{r.get('truth_people', 0):<4d}{mark}")
        grouping = group_unknowns(all_tracks, args.day)
        n_t = grouping.get("truth_people")
        print(f"\n  UNIQUE VISITORS among the {grouping['unnamed']} unnamed passes")
        print(f"    passes with a usable face : {grouping['with_usable_face']} "
              f"({grouping['with_usable_face'] / max(grouping['unnamed'], 1) * 100:.0f}%)"
              f"   <- the ceiling on regrouping")
        print(f"    pseudo-people             : {grouping['groups']}"
              f"   (linked by face {grouping['by_face']}, by body "
              f"{grouping['by_body']}, new {grouping['created']})")
        if n_t:
            print(f"    face-truth people among the usable-face passes: {n_t}")
        print(f"    WITHOUT grouping the day reports {grouping['unnamed']} "
              f"separate unknowns; with it, {grouping['groups']} "
              f"({(1 - grouping['groups'] / max(grouping['unnamed'], 1)) * 100:.0f}% fewer)")

    if args.json:
        out = ROOT / args.json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "day": args.day, "clips": len(per_clip), "passes": passes,
            "unnamed": unnamed, "frame_stride": every,
            "prod_sightings": len(prod_pairs), "live_sightings": len(live_pairs),
            "recovered": len(rec_pairs), "corroborated": len(corroborated),
            "novel": len(novel), "grouping": grouping,
            "group_sweep": group_sweep, "recovery_curve": curve_rows,
            "per_clip": per_clip}, indent=2))
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
