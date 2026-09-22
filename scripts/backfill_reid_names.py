#!/usr/bin/env python3
"""Give back the names that the reid_pass UNIQUE collisions threw away.

Until the merge in `ReidWorker._flush`, a pass whose body crops paused past the
stale limit was written as UNKNOWN, and the completed track that arrived later -
with the name, direction and face template - either collided with that row and
was dropped (`UNIQUE constraint failed`), or found nothing open and was dropped
silently. 5.1% of named recognitions from 8 to 22 September were left with an
unnamed, directionless reid_pass row.

The name was never actually lost: `recognition_event` recorded the same
completed track, from the same camera and track, when the face path named it.
So each such row can be matched to its event and relabelled - from the face
path's own decision, not from any new inference.

MATCHING
--------
`recognition_event` stores camera, track and time, not the pass's first_seen.
The owning row is the pass on that camera and track that was RUNNING at the
event's time: the latest `first_seen` at or before it, no more than
`--max-pass-minutes` earlier. Track ids repeat only after a tracker reset, so a
bound is what keeps a reused id from being confused with the right pass. A row
already carrying a DIFFERENT name is reported and left alone.

Only reid_pass rows change - employee_id, name, and direction where the row's
was UNKNOWN. Attendance is untouched; it never depended on this table.

    python scripts/backfill_reid_names.py            # dry run
    python scripts/backfill_reid_names.py --apply
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=None)
    ap.add_argument("--since", default="2026-09-01")
    ap.add_argument("--max-pass-minutes", type=float, default=30.0)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    path = args.db or settings.database_url_sync.split("///", 1)[1]
    db = sqlite3.connect(path, timeout=30)
    db.execute("PRAGMA busy_timeout=30000")
    names = dict(db.execute("select id, full_name from employee"))

    events = db.execute("""
        select id, employee_id, camera_id, track_id, business_date, ts, direction
        from recognition_event
        where employee_id is not null and voided_at is null and source = 'live'
          and business_date >= ?""", (args.since,)).fetchall()

    fixes, conflicts, already, unmatched = [], [], 0, 0
    for ev_id, emp, cam, trk, bdate, ts, direction in events:
        row = db.execute("""
            select id, employee_id, direction from reid_pass
            where camera_id = ? and track_id = ? and business_date = ?
              and first_seen <= ?
              and (julianday(?) - julianday(first_seen)) * 1440 <= ?
            order by first_seen desc limit 1""",
            (cam, trk, bdate, ts, ts, args.max_pass_minutes)).fetchone()
        if row is None:
            unmatched += 1
            continue
        rid, rem, rdir = row
        if rem == emp:
            already += 1
        elif rem is None:
            new_dir = direction if (direction in ("ENTER", "EXIT")
                                    and rdir in (None, "", "UNKNOWN")) else rdir
            fixes.append((rid, emp, names.get(emp, ""), new_dir, ev_id))
        else:
            conflicts.append((rid, rem, emp, ev_id))

    # One reid row can be claimed by two events (a pass and its debounced
    # repeat). Keep it only if every claim names the same person.
    by_row = {}
    for f in fixes:
        by_row.setdefault(f[0], []).append(f)
    clean = [v[0] for v in by_row.values() if len({x[1] for x in v}) == 1]
    split = [k for k, v in by_row.items() if len({x[1] for x in v}) > 1]

    print(f"named live recognitions since {args.since}: {len(events)}")
    print(f"   reid row already carries the name : {already}")
    print(f"   reid row UNNAMED -> will be named : {len(clean)}")
    print(f"   reid row names SOMEONE ELSE       : {len(conflicts)}  (left alone)")
    print(f"   claimed by two different people   : {len(split)}  (left alone)")
    print(f"   no running pass found             : {unmatched}")
    for rid, emp, name, d, ev in clean[:10]:
        print(f"      reid_pass {rid} <- {name}  direction {d}   (event {ev})")
    if len(clean) > 10:
        print(f"      ... and {len(clean) - 10} more")

    if not args.apply:
        print("\ndry run - nothing written. Re-run with --apply.")
        return 0
    db.executemany("update reid_pass set employee_id = ?, name = ?, direction = ? "
                   "where id = ? and employee_id is null",
                   [(emp, name, d, rid) for rid, emp, name, d, _ev in clean])
    db.commit()
    print(f"\nnamed {len(clean)} reid_pass row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
