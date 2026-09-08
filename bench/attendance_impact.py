#!/usr/bin/env python3
"""What would the second-chance rule have done to LAST WEEK'S attendance?

    python bench/attendance_impact.py --persons <tree> --db <prod.db> \
        --days 20260902,...,20260907

WHY NOT COUNT RECOVERED PASSES
------------------------------
"The template names 128 more passes" is not an attendance number, and quoting
it as one overstates the gain by about 10x. A recovered pass only changes a
timesheet if it survives three filters that a stored `persons/unknown/` folder
says nothing about:

 1. IT MUST REACH THE RULE. `_second_chance` runs inside
    `CameraWorker._persist_completed`, on CompletedTracks. Only 34% of stored
    unknown passes came from one - the rest were flushed by the ReID worker's
    stale-pass sweep, never became a completed track, and are invisible to the
    face path entirely. `_pass.json` carries `face_score` exactly when a
    CompletedTrack existed, which is how they are told apart here.

 2. IT MUST CARRY A DIRECTION. Both cameras have tripwire geometry configured,
    so `AttendanceService.record` is called with `require_direction=True` and a
    pass whose direction is UNKNOWN is written as NO_DIRECTION and moves no
    state. 17% of the passes that reach the rule are in that position.

 3. IT MUST LAND ON A HOLE. A check-out recovered for somebody who already has
    one changes nothing an operator would notice.

So this counts the third filter, having applied the first two: distinct
employee-days whose MISSING transition a recovered pass supplies.

It is an upper bound on the good and a lower bound on the harm: the recovered
passes include an estimated false-accept share (see bench/tracklet_far_eval.py),
and a false accept that fills a hole fills it with the wrong person.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from group_calibrate import day_features            # noqa: E402


def holes_of(db: Path, days) -> dict:
    """(day, employee) -> which transitions the attendance row is missing."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    lo, hi = min(days), max(days)
    fmt = lambda d: f"{d[:4]}-{d[4:6]}-{d[6:]}"                     # noqa: E731
    out = {}
    for d, e, ci, co, st in con.execute(
            "select business_date, employee_id, check_in_time, check_out_time, "
            "status from daily_attendance where business_date between ? and ?",
            (fmt(lo), fmt(hi))):
        need = [k for k, missing in (("IN", ci is None),
                                     ("OUT", co is None or st == "NO_CHECKOUT"))
                if missing]
        if need:
            out[(str(d).replace("-", ""), int(e))] = need
    con.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--persons", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--days", required=True)
    ap.add_argument("--cache-dir", default="data/bench/feat_cache")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from app.config import settings
    from app.services.enrollment import load_gallery

    g = load_gallery()
    days = [d.strip() for d in args.days.split(",") if d.strip()]
    holes = holes_of(Path(args.db), days)
    root, cache = Path(args.persons), ROOT / args.cache_dir
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    n_gal, n_live = con.execute(
        "select count(*), sum(threshold is not null) from face_embedding").fetchone()
    named_now = con.execute(
        "select count(*) from reid_pass where business_date between ? and ? "
        "and employee_id is not null",
        (f"{days[0][:4]}-{days[0][4:6]}-{days[0][6:]}",
         f"{days[-1][:4]}-{days[-1][4:6]}-{days[-1][6:]}")).fetchone()[0]
    con.close()

    print(f"  gallery {n_gal} rows ({n_live or 0} CCTV) / {g.n_people} people")
    print(f"  threshold {settings.tracklet_face_threshold} "
          f"margin {settings.tracklet_face_margin}")
    print(f"  baseline: {named_now} passes named by the live path, "
          f"{len(holes)} incomplete attendance rows\n")

    named = reaches = directed = 0
    fills, detail = set(), []
    for d in days:
        for t in day_features(root, d, "unknown", cache, None, 10):
            if t["face"] is None:
                continue
            m = g.match(t["face"], settings.tracklet_face_threshold,
                        settings.tracklet_face_margin)
            if m.employee_id is None:
                continue
            named += 1
            meta = {}
            f = t["dir"] / "_pass.json"
            if f.is_file():
                try:
                    meta = json.loads(f.read_text())
                except Exception:
                    pass
            if meta.get("face_score") is None:
                continue                       # no CompletedTrack: never reaches the rule
            reaches += 1
            direction = str(meta.get("direction") or "UNKNOWN")
            if direction not in ("ENTER", "EXIT"):
                continue                       # NO_DIRECTION: moves no state
            directed += 1
            side = "IN" if direction == "ENTER" else "OUT"
            need = holes.get((d, m.employee_id))
            if need and side in need:
                if (d, m.employee_id, side) not in fills:
                    detail.append((d, g.name(m.employee_id), side,
                                   round(float(m.score), 3)))
                fills.add((d, m.employee_id, side))

    print(f"  template names, over all stored unknown passes : {named}")
    print(f"  ...came from a CompletedTrack (reaches rule)   : {reaches}"
          f"   ({100 * reaches / max(named, 1):.0f}%)")
    print(f"  ...carries ENTER/EXIT (can move attendance)    : {directed}"
          f"   ({100 * directed / max(named, 1):.0f}%)")
    print(f"\n  ATTENDANCE ROWS COMPLETED: {len(fills)} of {len(holes)} "
          f"({100 * len(fills) / max(len(holes), 1):.0f}%)")
    for d, n, side, sc in sorted(detail):
        print(f"    {d}  {n[:30]:30s} {side:3s}  score {sc}")
    print(f"\n  named passes {named_now} -> {named_now + reaches} "
          f"(+{100 * reaches / max(named_now, 1):.1f}%)")
    if args.json:
        out = ROOT / args.json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"days": days, "gallery_rows": n_gal, "gallery_cctv": n_live,
             "threshold": settings.tracklet_face_threshold,
             "named_baseline": named_now, "template_named": named,
             "reaches_rule": reaches, "directed": directed,
             "holes": len(holes), "holes_filled": len(fills),
             "detail": detail}, indent=2, default=float))
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
