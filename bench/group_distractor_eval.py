#!/usr/bin/env python3
"""Score grouping on real identities, with the day's real crowd as distractors.

    python bench/group_distractor_eval.py --persons <tree> --days 20260902,20260904,20260905

WHY THIS EXISTS
---------------
`bench/group_calibrate.py` scores the `known/` corpus on its own: about 200
passes belonging to 30-odd people. That is the only non-circular truth
available, and on it a body threshold of 0.60 looks excellent.

It is also a population that does not exist. A real day is ~1400 passes over
several hundred people, so the pool a pass is matched against is seven times
larger and far sparser. Matching takes the MAXIMUM similarity over that pool,
and the maximum of a larger sample of impostors is higher - so a threshold
calibrated on the small pool will make more false links on the real one. This
is the same base-rate argument that keeps body similarity from ever naming an
employee (see app/services/pseudo_gallery.py); it applies to grouping too, just
less brutally.

So: the gallery is filled with EVERY pass of the day, known and unknown, in
time order. Pair precision and recall are then computed only over the `known/`
passes, where identity is real. The unknown passes contribute nothing to the
score and everything to the difficulty - which is exactly their role in
production.

A pass that the day's crowd pulls away from its own person shows up as lost
recall; one that the crowd wrongly attaches shows up as lost precision. Neither
is visible when the known corpus is scored alone.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from group_calibrate import day_features            # noqa: E402
from pseudo_group_eval import pair_scores           # noqa: E402


def run_mixed(known, unknown, day: str, settings, face_thr, body_thr):
    """Fill one gallery with the whole day; score only the labelled part."""
    from app.db.models import ReidPass
    from app.db.session import session_scope
    from app.services.pseudo_gallery import PseudoGallery

    y, m, d = int(day[:4]), int(day[4:6]), int(day[6:])
    bday, when = date(y, m, d), datetime(y, m, d, tzinfo=timezone.utc)
    tagged = [(t, True) for t in known] + [(t, False) for t in unknown]
    tagged.sort(key=lambda x: x[0]["stamp"])        # chronological, as it happens

    of, ob = settings.pseudo_face_threshold, settings.pseudo_body_threshold
    settings.pseudo_face_threshold, settings.pseudo_body_threshold = face_thr, body_thr
    labels, is_known = [], []
    try:
        with session_scope() as s:
            g = PseudoGallery()
            for i, (t, k) in enumerate(tagged):
                row = ReidPass(camera_id=1 if t["camera"] == "Entrance" else 2,
                               camera_name=t["camera"], track_id=i,
                               first_seen=when, last_seen=when,
                               business_date=bday, direction="UNKNOWN",
                               dim=len(t["body"]), model_name=settings.reid_model)
                s.add(row)
                s.flush()
                p = g.place(s, row, face=t["face"], body=t["body"],
                            face_ipd=t["ipd"], body_model=settings.reid_model)
                labels.append(p.id if p is not None else -(i + 1))
                is_known.append(k)
            s.rollback()
    finally:
        settings.pseudo_face_threshold, settings.pseudo_body_threshold = of, ob

    labels = np.asarray(labels)
    is_known = np.asarray(is_known)
    names = [t["name"] for t, k in tagged if k]
    lab: dict = {}
    truth = np.array([lab.setdefault(n, len(lab)) for n in names])
    prec, rec, tp, fp = pair_scores(labels[is_known], truth)
    n_truth = len(lab)
    n_pred = len(set(labels[is_known].tolist()))
    return {"precision": prec, "recall": rec,
            "f1": (2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
            "truth_people": n_truth, "pred_people": n_pred,
            "count_err": (n_pred - n_truth) / max(n_truth, 1) * 100,
            "groups_all": len(set(labels.tolist())), "passes": len(tagged)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--persons", required=True)
    ap.add_argument("--days", required=True)
    ap.add_argument("--cache-dir", default="data/bench/feat_cache")
    ap.add_argument("--label", default="calibration")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from app.config import settings
    root, cache = Path(args.persons), ROOT / args.cache_dir
    days = [d.strip() for d in args.days.split(",") if d.strip()]

    data = {}
    for d in days:
        data[d] = (day_features(root, d, "known", cache, None, 10),
                   day_features(root, d, "unknown", cache, None, 10))
        print(f"  {d}: {len(data[d][0])} labelled + {len(data[d][1])} unlabelled "
              f"= {len(data[d][0]) + len(data[d][1])} passes")

    print(f"\n  {args.label} days, KNOWN scored INSIDE the day's real crowd")
    print(f"  {'face':>5s} {'body':>5s} {'prec':>7s} {'recall':>7s} {'F1':>7s} "
          f"{'count err':>10s} {'groups/day':>11s}")
    rows = []
    for ft in (0.24, 0.26, 0.28, 0.30, 0.32, 0.35, 0.40):
        for bt in (0.60, 0.65, 0.70, 0.75, 0.80, 1.01):
            per = [run_mixed(data[d][0], data[d][1], d, settings, ft, bt)
                   for d in days]
            a = {k: float(np.mean([p[k] for p in per]))
                 for k in ("precision", "recall", "f1", "count_err")}
            a |= {"face": ft, "body": bt,
                  "groups_all": int(np.mean([p["groups_all"] for p in per])),
                  "passes": int(np.mean([p["passes"] for p in per]))}
            rows.append(a)
            print(f"  {ft:5.2f} {bt:5.2f} {a['precision'] * 100:6.1f}% "
                  f"{a['recall'] * 100:6.1f}% {a['f1'] * 100:6.1f}% "
                  f"{a['count_err']:+9.0f}% {a['groups_all']:11d}")

    best = max(rows, key=lambda r: r["f1"])
    print(f"\n  best F1 with distractors: face {best['face']:.2f} body "
          f"{best['body']:.2f}  P {best['precision'] * 100:.1f} "
          f"R {best['recall'] * 100:.1f} F1 {best['f1'] * 100:.1f} "
          f"count {best['count_err']:+.0f}%")
    if args.json:
        out = ROOT / args.json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"days": days, "rows": rows, "best": best},
                                  indent=2, default=float))
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
