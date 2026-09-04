#!/usr/bin/env python3
"""Measure what corridor crops in the gallery actually cost in false accepts.

    python bench/far_live_rows.py                 # against the live database
    python bench/far_live_rows.py --days 30       # a wider probe window
    python bench/far_live_rows.py --split 2026-09-02   # calibrate/evaluate split

WHY THIS EXISTS
---------------
A corridor crop added to the gallery is judged against a higher floor than an
enrolment photograph (`augment_live_floor`). That number is a POLICY, chosen
from a measurement on one site over three days - not a constant - and the only
honest way to keep it is to be able to re-take the measurement. This is that
measurement, so nobody has to trust a figure in a comment.

THE PROBE SET, AND ITS ONE LIMITATION
-------------------------------------
`unknown_sighting.vector` holds the face embedding of somebody the pipeline
named nobody. Hundreds accumulate daily at no cost, and unlike the enrolment
gallery they are real corridor faces - the population a live query is drawn
from. Scoring them against the gallery answers the question that matters: how
often does a face that should stay anonymous get given a name?

The limitation is that "unknown" is not "not an employee". Some of these are
enrolled people the studio photograph missed, and naming those is the feature
WORKING. So the rate below is an upper bound on false accepts, not a
measurement of them, and the two cannot be separated without labels. Marking
sightings "not an employee" in the review UI is what turns this into a real
FAR; until then, read it as a comparison - a corridor crop should be no more
dangerous than the enrolment photograph beside it, and that comparison is sound
because both are scored against the same probes.

Anything printed here is aggregate. No vector or image leaves the machine.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _probes(day_from: date, day_to: date | None):
    from sqlalchemy import select
    from app.db.models import UnknownSighting
    from app.db.session import session_scope
    with session_scope() as s:
        q = select(UnknownSighting.vector, UnknownSighting.nearest_employee_id,
                   UnknownSighting.business_date).where(
            UnknownSighting.vector.is_not(None),
            UnknownSighting.business_date >= day_from)
        if day_to is not None:
            q = q.where(UnknownSighting.business_date <= day_to)
        rows = s.execute(q).all()
    if not rows:
        return np.zeros((0, 512), np.float32), np.zeros((0,), np.int64)
    P = np.stack([np.frombuffer(r[0], np.float32) for r in rows])
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
    return P, np.array([r[1] if r[1] is not None else -1 for r in rows], np.int64)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=14,
                    help="how far back to take corridor probes (default 14)")
    ap.add_argument("--split", default=None, metavar="YYYY-MM-DD",
                    help="report probes before and from this date separately. "
                         "Use the date the crops were ADDED: probes older than "
                         "that were judged unknown without them, so they are a "
                         "genuinely out-of-sample test.")
    args = ap.parse_args()

    from app.config import settings
    from app.services.augment import TAG, _gallery_rows

    thr = settings.threshold_for(settings.recognizer_model)
    floor = max(thr, float(settings.augment_live_floor))
    M, owner, ids, tags, names = _gallery_rows()
    live = np.array([t.startswith(TAG) for t in tags])
    if not len(M):
        print("  the gallery is empty"); return 1
    print(f"  gallery      {len(M)} rows: {(~live).sum()} enrolment, {live.sum()} corridor")
    print(f"  threshold    {thr:.3f}      live floor {floor:.3f}")
    if not live.any():
        print("\n  No corridor crops in the gallery - nothing to measure.")
        return 0

    since = date.today() - timedelta(days=args.days)
    split = date.fromisoformat(args.split) if args.split else None
    sets = [("all probes", since, None)]
    if split:
        sets = [(f"before {split} (out of sample)", since, split - timedelta(days=1)),
                (f"from {split}", split, None)]

    for tag, a, b in sets:
        P, near = _probes(a, b)
        if not len(P):
            print(f"\n  {tag}: no stored vectors"); continue
        S = P @ M.T
        e_best = S[:, ~live].max(1)
        l_best = S[:, live].max(1)
        # A probe the pipeline already thought was person X is very likely X
        # walking past again; counting it against X's own crop would punish the
        # crop for resembling its own subject.
        own = np.where(near[:, None] == owner[None, :], S, -2.0)[:, live].max(1)
        blind = own < l_best
        print(f"\n  {tag}: {len(P)} corridor faces the pipeline named nobody")
        print(f"     via an ENROLMENT photo : {(e_best >= thr).sum():4d} "
              f"({100*(e_best >= thr).mean():5.2f}%)  max {e_best.max():.3f}")
        print(f"     via a CORRIDOR crop    : {(l_best >= floor).sum():4d} "
              f"({100*(l_best >= floor).mean():5.2f}%)  max {l_best.max():.3f}"
              f"   [no floor: {(l_best >= thr).sum()} / {100*(l_best >= thr).mean():.2f}%]")
        verdict = ("SAFER than an enrolment photo"
                   if (l_best >= floor).mean() <= (e_best >= thr).mean()
                   else "MORE DANGEROUS than an enrolment photo")
        print(f"     -> a corridor crop is currently {verdict}")
        print(f"     floor sweep:")
        for f in (thr, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
            if f < thr:
                continue
            print(f"        {f:.2f}: {(l_best >= f).sum():4d} of {len(P)} named "
                  f"({100*(l_best >= f).mean():5.2f}%)"
                  f"{'   <- in force' if abs(f - floor) < 1e-9 else ''}")
        print(f"     probes whose nearest match was NOT the crop's own person: "
              f"{blind.sum()} of {len(P)}")
    print("\n  Read the rates as a comparison, not as FAR: some of these are")
    print("  enrolled people the studio photo missed, and naming those is the")
    print("  point. The bar a crop must clear is the enrolment photo's own rate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
