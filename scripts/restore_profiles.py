#!/usr/bin/env python3
"""Put back employee names, departments, positions and phones from a backup.

WHY THIS EXISTS
---------------
`scripts/enroll.py` used to assign these four fields unconditionally from a
folder's `metadata.json`. A folder WITHOUT one therefore did not leave the
record alone - it overwrote it with the folder suffix and three empty strings.
On gpu6 the enrolment export had been copied as photographs only, so the
2026-09-18 gallery rebuild rewrote all 54 full names ("Xamdamov Rustam" ->
"Rustam", from `013_Rustam`) and blanked every department, position and phone.

Nothing else was touched: no employee was created or deleted, every id survived
and the attendance history still joined correctly, which is exactly why the
damage was invisible until somebody read the names on a page.

`app/services/enrollment.py` no longer does this - a folder with no metadata now
leaves an existing record alone. This script repairs a database that was already
rebuilt by the old behaviour.

USAGE
-----
    python scripts/restore_profiles.py --from data/ematsy.pre-restart-YYYYMMDD_HHMM.db
    python scripts/restore_profiles.py --from <backup> --apply

It prints what it would change and exits without writing unless `--apply` is
given. Employees are matched BY ID, never by name: the names are the thing
that is wrong, so matching on them would be circular.

It is safe to run against the live database while the service is up - four
columns of one table, in one short transaction - but there is no reason not to
stop it first if you can.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings

FIELDS = ("full_name", "department", "position", "phone")


def _rows(conn: sqlite3.Connection) -> dict[int, tuple]:
    cols = ", ".join(FIELDS)
    return {r[0]: tuple(r[1:]) for r in conn.execute(
        f"SELECT id, {cols} FROM employee")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="source", required=True,
                    help="the backup database to read the good values from")
    ap.add_argument("--db", default=None,
                    help="the database to repair (default: the configured one)")
    ap.add_argument("--apply", action="store_true",
                    help="actually write; without it this is a dry run")
    args = ap.parse_args()

    src_path = Path(args.source)
    if not src_path.exists():
        print(f"no such backup: {src_path}", file=sys.stderr)
        return 2

    db_path = Path(args.db) if args.db else Path(
        settings.database_url_sync.split("///", 1)[1])
    if not db_path.exists():
        print(f"no such database: {db_path}", file=sys.stderr)
        return 2

    src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
    good = _rows(src)
    src.close()

    dst = sqlite3.connect(str(db_path), timeout=30)
    dst.execute("PRAGMA busy_timeout=30000")
    current = _rows(dst)

    changes = [(i, current[i], good[i]) for i in sorted(good)
               if i in current and current[i] != good[i]]

    if not changes:
        print(f"nothing to restore: {db_path.name} already matches "
              f"{src_path.name} on {', '.join(FIELDS)}")
        dst.close()
        return 0

    print(f"{len(changes)} employee(s) differ between {db_path.name} and "
          f"{src_path.name}:\n")
    for i, now, was in changes:
        print(f"  employee {i}")
        for field, a, b in zip(FIELDS, now, was):
            if a != b:
                print(f"      {field:<11} {a!r}  ->  {b!r}")

    if not args.apply:
        print(f"\ndry run - nothing written. Re-run with --apply to restore.")
        dst.close()
        return 0

    sets = ", ".join(f"{f}=?" for f in FIELDS)
    dst.executemany(f"UPDATE employee SET {sets} WHERE id=?",
                    [(*was, i) for i, _now, was in changes])
    dst.commit()
    dst.close()
    print(f"\nrestored {len(changes)} employee(s) in {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
