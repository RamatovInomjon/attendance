#!/usr/bin/env python3
"""Additive schema migration, index maintenance, and attendance repair.

    python scripts/migrate.py                          # schema + indexes
    python scripts/migrate.py --repair-attendance      # dry run, prints only
    python scripts/migrate.py --repair-attendance --apply

Every schema change here is an additive ADD COLUMN, which SQLite applies without
rewriting the table and which leaves existing rows intact. Anything destructive
must wait for a real migration tool.

The repair pass is separate, defaults to a dry run, and takes a backup before it
writes. It exists because three fixed bugs left damage behind that new code
cannot correct on its own:

  * `NO_CHECKIN` was a one-way latch, so 21 rows carry the flag despite having a
    check-in time - 70% false positives on the flag operators review.
  * The end-of-day sweep never ran, so rows sit at presence=INSIDE on business
    dates days old, claiming those people are still in the building.
  * Retention deleted snapshots that daily_attendance still references, because
    the keep-set was built from recognition_event alone.
"""
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import inspect, select, text
from app.config import settings
from app.db.session import engine, init_db, session_scope

ADDITIONS = {
    "camera": [
        ("line_x1", "FLOAT"), ("line_y1", "FLOAT"),
        ("line_x2", "FLOAT"), ("line_y2", "FLOAT"),
        ("inside_side", "INTEGER DEFAULT 1"),
        ("depth_grows_inward", "BOOLEAN DEFAULT 1"),
        ("min_travel", "FLOAT DEFAULT 0.06"),
    ],
    "recognition_event": [
        ("direction", "VARCHAR(16) DEFAULT 'UNKNOWN'"),
        ("direction_reason", "VARCHAR(96) DEFAULT ''"),
        # Admin corrections. NULL means "standing", which is every existing row.
        ("voided_at", "DATETIME"), ("voided_by", "VARCHAR(64)"),
        ("void_reason", "VARCHAR(160)"),
    ],
    "unknown_sighting": [
        ("resolved_employee_id", "INTEGER"), ("resolved_kind", "VARCHAR(16)"),
        ("resolved_by", "VARCHAR(64)"), ("resolved_at", "DATETIME"),
    ],
    # NULL, not a default: an enrolment photograph has no floor of its own and
    # is judged against the global threshold. Only augmented corridor crops
    # carry a value, so backfilling one here would silently re-threshold the
    # whole gallery.
    "face_embedding": [("threshold", "FLOAT")],
    # Backfilled below from is_admin, not defaulted here: a NULL default would
    # be indistinguishable from "role not chosen yet" and every existing
    # account would silently lose or gain powers on the first restart.
    "app_user": [("role", "VARCHAR(16)")],
}

# (name, table, columns) - created if absent.
INDEXES_WANTED = [
    # The cross-camera ReID lookup: unmatched passes from the other camera on
    # the same day.
    ("ix_reid_date_cam_matched", "reid_pass",
     "business_date, camera_id, matched_pass_id"),
    # The per-pass debounce lookup, which runs on the capture thread inside a
    # write transaction. Without the camera_id/ts tail it scanned every event
    # the employee ever produced and sorted them in a temp B-tree.
    ("ix_event_emp_cam_ts", "recognition_event", "employee_id, camera_id, ts"),
    # /attendance/unknown orders by last_seen; that was a full table scan.
    ("ix_unknown_sighting_last_seen", "unknown_sighting", "last_seen"),
]

# Indexes that cost a write on every insert and serve no query: strict prefixes
# of a composite index, or ordered on a column nothing orders on.
INDEXES_REDUNDANT = [
    "ix_recognition_event_business_date",   # prefix of ix_event_date_ts
    "ix_recognition_event_employee_id",     # prefix of ix_event_emp_date
    "ix_unknown_sighting_first_seen",       # no query touches first_seen
]


def migrate_schema():
    init_db()
    insp = inspect(engine)
    with engine.begin() as conn:
        for table, cols in ADDITIONS.items():
            if table not in insp.get_table_names():
                print(f"  {table}: not present, created by init_db")
                continue
            have = {c["name"] for c in insp.get_columns(table)}
            for name, decl in cols:
                if name in have:
                    continue
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {decl}"))
                print(f"  {table}.{name} added")

    # Existing accounts predate roles. An admin becomes "admin" and everyone
    # else "viewer", which is exactly what they could do yesterday: the new
    # "operator" is opt-in and is never assigned by a migration.
    insp = inspect(engine)
    if "app_user" in insp.get_table_names():
        with engine.begin() as conn:
            n = conn.execute(text(
                "UPDATE app_user SET role = CASE WHEN is_admin THEN 'admin' "
                "ELSE 'viewer' END WHERE role IS NULL OR role = ''")).rowcount
            if n:
                print(f"  app_user.role backfilled for {n} account(s)")

    insp = inspect(engine)
    tables = set(insp.get_table_names())
    with engine.begin() as conn:
        for name, table, cols in INDEXES_WANTED:
            if table not in tables:
                continue
            if name in {i["name"] for i in insp.get_indexes(table)}:
                continue
            conn.execute(text(f"CREATE INDEX {name} ON {table} ({cols})"))
            print(f"  index {name} created")
        for name in INDEXES_REDUNDANT:
            conn.execute(text(f"DROP INDEX IF EXISTS {name}"))
        print(f"  {len(INDEXES_REDUNDANT)} redundant index(es) dropped if present")
    print("migration complete")


def _backup() -> Path | None:
    db = Path(settings.database_url_sync.replace("sqlite:///", ""))
    if not db.exists():
        return None
    dest = db.with_name(f"{db.stem}.pre-repair-{datetime.now():%Y%m%d_%H%M%S}.db")
    # The live database is WAL, so a plain copy can be torn. Use the backup API.
    import sqlite3
    src = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    dst = sqlite3.connect(str(dest))
    with dst:
        src.backup(dst)
    src.close(); dst.close()
    return dest


def repair_attendance(apply: bool) -> int:
    from app.db.models import DailyAttendance, RecognitionEvent, UnknownSighting
    from app.services.attendance import business_date

    # The BUSINESS date, not the calendar one: between midnight and 04:00 the
    # current business day is yesterday's calendar date, and date.today()
    # would flag every row of a shift still in progress as unfinished.
    today = business_date(datetime.now(settings.tz))
    changes = 0
    with session_scope() as s:
        rows = s.execute(select(DailyAttendance)).scalars().all()
        for r in rows:
            # 1. status recomputed from state. The latch could set NO_CHECKIN
            #    but never clear it, so a later genuine check-in left it flagged.
            want = None
            if r.status == "NO_CHECKIN" and r.check_in_time is not None:
                want = "PRESENT"
            if want and want != r.status:
                print(f"  daily {r.id:4d} emp {r.employee_id:3d} {r.business_date}  "
                      f"status {r.status} -> {want}  (check_in {r.check_in_time})")
                changes += 1
                if apply:
                    r.status = want

            # 2. Rows still INSIDE on a finished day. Flagged, never closed with
            #    an invented time - a plausible-looking check-out is the one
            #    thing that would destroy trust in the report.
            if r.presence is not None and str(r.presence).endswith("INSIDE") \
                    and r.business_date < today:
                print(f"  daily {r.id:4d} emp {r.employee_id:3d} {r.business_date}  "
                      f"presence INSIDE on a finished day -> NO_CHECKOUT")
                changes += 1
                if apply:
                    if settings.open_interval_policy == "close_at_eod":
                        from app.db.models import PresenceStatus
                        r.presence = PresenceStatus.OUTSIDE
                        r.entered_at = None
                    if r.status != "NO_CHECKIN":
                        r.status = "NO_CHECKOUT"

        # 3. Snapshot paths pointing at files retention already deleted.
        media = settings.media_dir
        dangling = 0
        for r in rows:
            for attr in ("check_in_snapshot", "check_out_snapshot"):
                v = getattr(r, attr)
                if v and not (media / v).exists():
                    dangling += 1
                    if apply:
                        setattr(r, attr, None)
        for model, attr in ((RecognitionEvent, "snapshot"),
                            (UnknownSighting, "snapshot")):
            for obj in s.execute(select(model)).scalars().all():
                v = getattr(obj, attr)
                if v and not (media / v).exists():
                    dangling += 1
                    if apply:
                        setattr(obj, attr, None)
        if dangling:
            print(f"  {dangling} snapshot path(s) point at files that no longer "
                  f"exist -> cleared")
            changes += dangling

        if not apply:
            s.rollback()
    return changes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair-attendance", action="store_true")
    ap.add_argument("--apply", action="store_true",
                    help="actually write; without it the repair is a dry run")
    ap.add_argument("--skip-schema", action="store_true")
    args = ap.parse_args()

    if not args.skip_schema:
        migrate_schema()

    if args.repair_attendance:
        print("\n  attendance repair " + ("(APPLYING)" if args.apply else "(dry run)"))
        print("  " + "-" * 66)
        if args.apply:
            b = _backup()
            print(f"  backup: {b}" if b else "  no database file to back up")
        n = repair_attendance(args.apply)
        print("  " + "-" * 66)
        print(f"  {n} change(s) " + ("applied" if args.apply else
                                     "would be applied; re-run with --apply"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
