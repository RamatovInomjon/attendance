#!/usr/bin/env python3
"""Clear the attendance history and start collecting again from zero.

    python scripts/reset_attendance.py                 # dry run, prints only
    python scripts/reset_attendance.py --apply
    python scripts/reset_attendance.py --apply --media  # also delete the images

WHAT IS KEPT, AND WHY
---------------------
  employee         who exists. Re-creating these means re-enrolling everybody.
  face_embedding   the gallery. 268 vectors that took a run of scripts/enroll.py
                   and the original photographs to produce.
  camera           URLs, roles, and the direction geometry, which was
                   CALIBRATED on site and cannot be recovered from the database.
  app_user         logins.

WHAT IS CLEARED
---------------
  recognition_event   every sighting ever recorded
  daily_attendance    every check-in, check-out and worked-seconds total
  unknown_sighting    unrecognised passes
  reid_pass           body-ReID features and their cross-camera links

This is the whole observed history. It is not recoverable except from the backup
this script takes first, so the default is a dry run and `--apply` is required.

`--media` additionally deletes the stored images: media/snapshots (evidence
thumbnails the dashboard links to), data/persons (ReID body crops) and
data/debug (forensic captures). Without it those files stay on disk but nothing
references them any more, and the retention sweep will remove them in its own
time. With a fresh database they are orphans from the first minute, so on a
genuine restart you usually do want --media.
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# (table, what it holds) - cleared in this order, children before parents.
CLEAR = [
    ("recognition_event", "every sighting ever recorded"),
    ("daily_attendance", "check-ins, check-outs, worked seconds"),
    ("unknown_sighting", "unrecognised passes"),
    ("reid_pass", "body-ReID features and cross-camera links"),
]
KEEP = [
    ("employee", "who exists"),
    ("face_embedding", "the enrolment gallery"),
    ("camera", "URLs, roles, calibrated direction geometry"),
    ("app_user", "logins"),
]
MEDIA = [
    ("media/snapshots", "evidence thumbnails"),
    ("data/persons", "ReID body crops"),
    ("data/debug", "forensic captures"),
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="actually delete; without it nothing is written")
    ap.add_argument("--media", action="store_true",
                    help="also delete stored images, not just the rows")
    args = ap.parse_args()

    from app.config import settings
    db = Path(settings.database_url_sync.replace("sqlite:///", ""))
    if not db.is_file():
        raise SystemExit(f"  no database at {db}")

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    have = {r[0] for r in con.execute(
        "select name from sqlite_master where type='table'")}

    print(f"  database {db}\n")
    print("  KEPT")
    for t, why in KEEP:
        n = con.execute(f"select count(*) from {t}").fetchone()[0] if t in have else 0
        print(f"    {t:20s} {n:8d}   {why}")
    print("\n  CLEARED")
    total = 0
    for t, why in CLEAR:
        n = con.execute(f"select count(*) from {t}").fetchone()[0] if t in have else 0
        total += n
        print(f"    {t:20s} {n:8d}   {why}")
    if args.media:
        print("\n  DELETED FROM DISK")
        for rel, why in MEDIA:
            d = ROOT / rel
            n = sum(1 for _ in d.rglob("*") if _.is_file()) if d.is_dir() else 0
            mb = (sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6
                  if d.is_dir() else 0)
            print(f"    {rel:20s} {n:8d}   {mb:8.0f} MB   {why}")
    con.close()

    if not args.apply:
        print(f"\n  {total} row(s) would be deleted. Nothing written.")
        print("  Re-run with --apply (add --media to delete the images too).")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    backup = db.with_name(f"{db.stem}.pre-reset-{stamp}.db")
    src = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    dst = sqlite3.connect(str(backup))
    with dst:
        src.backup(dst)
    src.close()
    ok = sqlite3.connect(str(backup)).execute("pragma integrity_check").fetchone()[0]
    print(f"\n  backup {backup.name}  ({backup.stat().st_size/1e6:.1f} MB, {ok})")
    if ok != "ok":
        raise SystemExit("  backup failed its integrity check - refusing to delete")

    con = sqlite3.connect(str(db))
    with con:
        for t, _ in CLEAR:
            if t in have:
                con.execute(f"delete from {t}")
        con.execute("vacuum")
    con.close()
    print(f"  cleared {total} row(s)")

    if args.media:
        for rel, _ in MEDIA:
            d = ROOT / rel
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
                d.mkdir(parents=True, exist_ok=True)
                print(f"  emptied {rel}")

    print("\n  Done. Restart the service; it starts recording from now.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
