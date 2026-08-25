#!/usr/bin/env python3
"""Daily maintenance: flag open intervals, prune old snapshots and unknowns.

Run from cron/systemd timer shortly after the day boundary (04:00 local):
    5 4 * * *  cd /path/to/project && python scripts/maintenance.py
"""
import logging, sys, time
from datetime import datetime, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import delete, select

from app.config import settings
from app.db.models import RecognitionEvent, UnknownSighting
from app.db.session import session_scope
from app.services.attendance import AttendanceService, business_date

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("maintenance")


def main():
    now = datetime.now(settings.tz)
    yesterday = business_date(now - timedelta(days=1))

    with session_scope() as s:
        n = AttendanceService().close_open_intervals(s, yesterday)
        log.info("flagged %d open interval(s) for %s", n, yesterday)

        cutoff = business_date(now - timedelta(days=settings.snapshot_retention_days))
        gone = s.execute(delete(RecognitionEvent).where(
            RecognitionEvent.business_date < cutoff)).rowcount
        gone_u = s.execute(delete(UnknownSighting).where(
            UnknownSighting.business_date < cutoff)).rowcount
        log.info("deleted %d event(s), %d unknown(s) older than %s", gone, gone_u, cutoff)

        keep = {r for (r,) in s.execute(
            select(RecognitionEvent.snapshot).where(RecognitionEvent.snapshot.isnot(None))) }

    snap_dir = settings.media_dir / "snapshots"
    removed = freed = 0
    age_limit = time.time() - settings.snapshot_retention_days * 86400
    for f in snap_dir.glob("*.jpg"):
        rel = f"snapshots/{f.name}"
        if rel in keep:
            continue
        if f.stat().st_mtime < age_limit:
            freed += f.stat().st_size
            f.unlink()
            removed += 1
    log.info("removed %d orphan snapshot(s), freed %.1f MB", removed, freed / 1e6)


if __name__ == "__main__":
    main()
