#!/usr/bin/env python3
"""Re-decide past recognitions against the current gallery, and fix the record.

    python scripts/recheck_history.py                       # dry run, prints only
    python scripts/recheck_history.py --since 2026-09-01
    python scripts/recheck_history.py --sheet data/debug/_recheck
    python scripts/recheck_history.py --apply

A recognition is a decision taken once, under whatever the gallery and the
threshold were at that instant. Change either and the record does not follow: a
pass accepted through an unfloored corridor crop keeps naming the wrong person
long after the rule that admitted it is gone.

This replays every stored pass we still hold the evidence for, and voids the
ones the current gallery would refuse. Each void runs through the same code the
admin button uses, so the day is rebuilt by replay - a promoted check-out, a
recomputed `worked_seconds`, and the crop that caused it removed.

It only ever removes a name. It never invents one: a recognition the current
gallery would make but the old one did not never happened, and writing
attendance for a walk nobody observed is fabrication however good the sums.

LOOK BEFORE YOU APPLY. `--sheet` writes a contact sheet of every face it means
to void. Whether a refusal is a false accept removed or a genuine recognition
lost is a judgement only somebody who knows these faces can make, and --apply
without that look is trusting a threshold you have not checked.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _backup() -> Path | None:
    """`.backup`, never cp: the database is WAL and a plain copy can tear."""
    import sqlite3
    from app.config import settings
    from datetime import datetime
    url = settings.database_url_sync
    if "///" not in url:
        return None
    src = Path(url.split("///", 1)[1])
    if not src.is_file():
        return None
    dst = src.with_name(f"{src.stem}.pre-recheck-{datetime.now():%Y%m%d_%H%M}.db")
    with sqlite3.connect(src) as a, sqlite3.connect(dst) as b:
        a.backup(b)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default=None, metavar="YYYY-MM-DD",
                    help="only business dates from here on")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sheet", default=None, metavar="DIR",
                    help="write a contact sheet of the faces to be voided")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--by", default="recheck")
    args = ap.parse_args()

    from app.config import settings
    from app.core.onnx_env import preload_cuda_libs
    from app.services import recheck

    since = date.fromisoformat(args.since) if args.since else None
    thr = settings.threshold_for(settings.recognizer_model)
    print(f"  gallery in force : threshold {thr:.3f}, corridor floor "
          f"{max(thr, settings.augment_live_floor):.3f}")
    preload_cuda_libs()
    rep = recheck.run(since=since, limit=args.limit)
    refused = rep.refused
    already = sum(1 for v in rep.verdicts if v.already_voided)

    print(f"  passes re-decided: {rep.captures}")
    print(f"    still stand    : {sum(1 for v in rep.verdicts if v.still_named)}")
    print(f"    already voided : {already}")
    print(f"    TO BE VOIDED   : {len(refused)}")
    if rep.unjudged:
        print(f"  {rep.unjudged} stored event(s) have no capture left and are LEFT ALONE.")
        print(f"     A decision that cannot be re-derived is not one to overturn.")
    if not refused:
        print("\n  Nothing to correct.")
        return 0

    print()
    refused.sort(key=lambda v: (str(v.business_date), v.stem))
    for v in refused:
        print(f"    {v.business_date}  {v.stem[9:15]}  {v.transition:<12} "
              f"{v.name[:24]:24s} {v.score_then:.3f} -> {v.score_now:.3f}  "
              f"(best studio photo {v.enrolment_best:.3f})")

    if args.sheet:
        sys.path.insert(0, str(ROOT / "bench"))
        from floor_impact import _sheet
        out = Path(args.sheet); out.mkdir(parents=True, exist_ok=True)
        rows = [(v.face if Path(v.face).exists() else "",
                 f"{v.name[:20]}\n{v.score_then:.3f}->{v.score_now:.3f}")
                for v in refused]
        rows = [r for r in rows if r[0]]
        n = _sheet(rows, out / "to_be_voided.jpg")
        print(f"\n  {out / 'to_be_voided.jpg'}  ({n} faces) - LOOK AT THESE")

    if not args.apply:
        print(f"\n  Dry run. Nothing written. Add --apply to void these "
              f"{len(refused)} pass(es).")
        return 0

    bk = _backup()
    if bk:
        print(f"\n  backed up {bk.name}")
    out = recheck.apply(rep, by=args.by)
    print(f"  voided {out['voided']} pass(es); {out['days']} person-day(s) rebuilt")
    for eid, why in out["failed"]:
        print(f"    event {eid} not voided: {why}")
    print("  Reload the gallery (POST /api/gallery/reload) or restart, so the "
          "crops removed by these voids leave the running gallery too.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
