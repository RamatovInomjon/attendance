#!/usr/bin/env python3
"""What does the second-chance rule do to somebody who is NOT enrolled?

    python bench/tracklet_far_eval.py --persons <tree> \
        --days 20260902,20260903,20260904,20260905,20260906,20260907

THE QUESTION THIS ANSWERS, AND THE TWO NUMBERS THAT DO NOT
----------------------------------------------------------
Lowering `tracklet_face_threshold` recovers more passes. The obvious worry is
that it also hands employee names to visitors, and two numbers were quoted
earlier that DO NOT measure that:

  * `wrong` on the known corpus - one enrolled person matched as another
    enrolled person. That is a misidentification, not a false accept: both
    people are in the gallery, and the margin rule is what guards it.
  * `novel` - a recovered name whose person appears nowhere else in
    production's day. Suggestive, and not evidence: an employee who really did
    pass once, in the pass the live vote missed, is "novel" by construction.

FAR here means the thing that actually corrupts attendance: a person who is not
in the gallery at all being given a name. Measuring it needs impostor probes -
faces known not to be enrolled - and production has almost none labelled (three
`resolved_kind='visitor'` rows in the whole database).

LEAVE-ONE-PERSON-OUT MAKES IMPOSTORS OUT OF EMPLOYEES
-----------------------------------------------------
So they are manufactured, by the standard open-set protocol: take every
labelled pass of person X, remove X ENTIRELY from the gallery, and match. X is
now genuinely not enrolled, so any name the rule returns is a true false
accept, and the probe is a real CCTV pass through this corridor rather than a
synthetic one.

This is stricter than production in one way and softer in another, and both are
worth stating:

  * STRICTER: an employee who walks past daily is a harder impostor than a
    stranger, because the gallery contains their colleagues, their uniform and
    their lighting. Real visitors are more often nobody-like.
  * SOFTER: the gallery is one person smaller, so the runner-up margin has one
    fewer candidate to clear. With 56 people the effect is small, and it biases
    toward MORE false accepts, not fewer - the safe direction for a safety
    measurement.

FPIR is reported as the fraction of impostor probes that receive ANY name,
which is what an attendance system suffers from - not per-identity accuracy.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from group_calibrate import day_features            # noqa: E402


def gallery_rows(db: Path):
    """Every active enrolment row: vector, owner, per-row floor, name."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    vecs, owner, floor, names = [], [], [], {}
    for eid, blob, thr, full in con.execute(
            "select f.employee_id, f.vector, f.threshold, e.full_name "
            "from face_embedding f join employee e on e.id = f.employee_id "
            "where e.is_active = 1"):
        v = np.frombuffer(blob, np.float32)
        vecs.append(v / (np.linalg.norm(v) + 1e-12))
        owner.append(int(eid))
        floor.append(float(thr) if thr is not None else 0.0)
        names[int(eid)] = full
    con.close()
    return (np.stack(vecs), np.asarray(owner), np.asarray(floor, np.float32),
            names)


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
    from app.core.gallery import Gallery
    from app.services.reid_worker import slug

    M, owner, floor, names = gallery_rows(Path(args.db))
    slug2id = {slug(v): k for k, v in names.items()}
    print(f"  gallery {len(M)} rows / {len(set(owner.tolist()))} people; "
          f"{int((floor > 0).sum())} rows carry a CCTV floor")

    root, cache = Path(args.persons), ROOT / args.cache_dir
    days = [d.strip() for d in args.days.split(",") if d.strip()]
    probes = []          # (employee_id, template)
    for d in days:
        for t in day_features(root, d, "known", cache, None, 10):
            eid = slug2id.get(t["name"])
            if t["face"] is not None and eid is not None:
                probes.append((eid, t["face"]))
    people = sorted({e for e, _f in probes})
    print(f"  {len(probes)} labelled CCTV passes over {len(people)} people, "
          f"{len(days)} days\n")

    # One gallery per held-out person, reused across that person's probes and
    # every threshold. Rebuilding per probe would be 371 constructions.
    held = {}
    for e in people:
        keep = owner != e
        held[e] = Gallery(M[keep], owner[keep],
                          {k: v for k, v in names.items() if k != e},
                          thresholds=floor[keep])
    full = Gallery(M, owner, names, thresholds=floor)

    mg = settings.tracklet_face_margin
    print(f"  margin {mg}; per-row floors applied (augment_live_floor="
          f"{settings.augment_live_floor})\n")
    print(f"  {'thr':>6s} | {'IMPOSTOR (person removed)':^34s} | "
          f"{'GENUINE (person present)':^28s}")
    print(f"  {'':>6s} | {'named':>7s} {'FPIR':>8s} {'top score':>16s} | "
          f"{'correct':>8s} {'recall':>8s} {'wrong':>7s}")
    rows = []
    for thr in (0.15, 0.18, 0.21, 0.215, 0.24, 0.27, 0.30, 0.33, 0.36, 0.40):
        fa, fa_scores = 0, []
        ok = wrong = 0
        for e, f in probes:
            m = held[e].match(f, thr, mg)
            if m.employee_id is not None:
                fa += 1
                fa_scores.append(float(m.score))
            g = full.match(f, thr, mg)
            if g.employee_id == e:
                ok += 1
            elif g.employee_id is not None:
                wrong += 1
        fpir = fa / len(probes) * 100
        s = (f"med {np.median(fa_scores):.3f} max {max(fa_scores):.3f}"
             if fa_scores else "-")
        rows.append({"threshold": thr, "probes": len(probes), "false_accepts": fa,
                     "fpir_pct": fpir, "correct": ok, "wrong": wrong,
                     "recall_pct": ok / len(probes) * 100,
                     "fa_score_median": float(np.median(fa_scores)) if fa_scores else None,
                     "fa_score_max": float(max(fa_scores)) if fa_scores else None})
        mark = "  <- deployed" if abs(thr - settings.tracklet_face_threshold) < 1e-9 else ""
        print(f"  {thr:6.3f} | {fa:7d} {fpir:7.2f}% {s:>16s} | "
              f"{ok:8d} {ok / len(probes) * 100:7.1f}% {wrong:7d}{mark}")

    print(f"\n  FPIR = share of the {len(probes)} impostor probes given ANY "
          f"name. Each one would be a wrong attendance row.")
    if args.json:
        out = ROOT / args.json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"days": days, "probes": len(probes),
                                   "people": len(people), "rows": rows},
                                  indent=2, default=float))
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
