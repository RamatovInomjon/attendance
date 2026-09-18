#!/usr/bin/env python3
"""Can a missing CHECK_OUT be recovered from the passes we already stored?

THE QUESTION
------------
56 days in the production database have a check-in and no check-out. The face
path saw nothing later on 41 of them. Every pass, named or not, left a body
feature and usually a face template behind in `reid_pass` - so the question is
whether the departure is recoverable from those, and at what error rate.

This is NOT the rule `pseudo_gallery.py` refuses. That one is open-set: "here is
an unknown track among ~1100 a day, is it an employee?", where the base rate
(~45 recoverable employees) makes a body rule write more wrong rows than right.
This is constrained verification: "person X checked in at 09:12 and never
checked out; which of today's EXIT passes is X?", anchored on a pass the FACE
named earlier the SAME DAY, so clothing still holds.

THE HOLDOUT, AND WHY IT IS HONEST
---------------------------------
Measured on COMPLETE days - both ends face-confirmed - with the answer hidden:
candidates include the pass that really was the check-out, but no ranker may
look at `employee_id`, which is the label. That is a real holdout with real
labels, unlike the pseudo-grouping evaluation, which could only ever be scored
against a relabelling of its own output.

Read the ceiling before the accuracy. `set_recall` is how often the right pass
is even among the candidates; no ranker can beat it. The gap between it and a
ranker's top-1 is what better ranking could still win.

TWO WARNINGS ABOUT THE NUMBERS THIS PRINTS
------------------------------------------
*They are an optimistic bound.* A complete day is one where the exit WAS
captured. The days we want to fill are the ones where capture failed, and the
body path may well have failed there too - only 17 of the 56 have any linked
EXIT pass at all. Multiply accordingly.

*Accuracy here is not licence to write attendance.* A missing check-out is
visible: it is flagged incomplete and somebody investigates. A wrong one is
invisible - it writes a plausible `worked_seconds` that nothing reveals. Use
this to decide whether to SUGGEST a departure for one-click confirmation, not
to fill one in silently.

VECTOR SPACES ARE CHECKED, NOT ASSUMED
--------------------------------------
This database holds two body spaces (OSNet 512-d, ResNet101 2048-d) and two
face spaces. Cosines across two spaces succeed and mean nothing, which is why
those provenance columns exist; a candidate whose `model_name`/`face_model`
differs from the anchor's is skipped rather than scored.

    python bench/missing_checkout_eval.py --db <path to a database>
    python bench/missing_checkout_eval.py --db ... --tolerance 120 --verbose
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _dt(v):
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    return datetime.fromisoformat(str(v).replace("Z", "").split("+")[0])


def _vec(blob, dim):
    if not blob or not dim:
        return None
    a = np.frombuffer(blob, dtype=np.float32)
    if a.size != dim:
        return None
    n = np.linalg.norm(a)
    return a / n if n > 1e-9 else None


class Pass:
    __slots__ = ("id", "employee_id", "last_seen", "direction", "body", "body_model",
                 "face", "face_model", "pseudo_score", "pseudo_person_id",
                 # set by candidates_for(): is this pass comparable to the
                 # anchor at all? Two model spaces are present in this data and
                 # a cosine across them succeeds while meaning nothing.
                 "_same_body", "_same_face")

    def __init__(self, row):
        (self.id, self.employee_id, last_seen, self.direction, body, dim,
         self.body_model, face, face_dim, self.face_model, self.pseudo_score,
         self.pseudo_person_id) = row
        self.last_seen = _dt(last_seen)
        self.body = _vec(body, dim)
        self.face = _vec(face, face_dim)
        self._same_body = self._same_face = False


PASS_COLS = """id, employee_id, last_seen, direction, vector, dim, model_name,
               face_vector, face_dim, face_model, pseudo_score, pseudo_person_id"""


def load_days(db, complete: bool):
    """(employee, date, check_in, check_out) for days of the requested kind."""
    cond = ("d.check_out_time is not null" if complete
            else "d.check_out_time is null")
    return [(e, b, _dt(ci), _dt(co)) for e, b, ci, co in db.execute(
        f"""select d.employee_id, d.business_date, d.check_in_time, d.check_out_time
            from daily_attendance d
            where d.check_in_time is not null and {cond}""")]


def passes_by_day(db):
    out = defaultdict(list)
    for row in db.execute(f"select {PASS_COLS}, business_date from reid_pass"):
        out[row[-1]].append(Pass(row[:-1]))
    return out


def anchor_for(day_passes, employee_id, check_in):
    """The employee's own face-named pass that day: the clothing reference.

    Nearest to the check-in, because clothing and light drift over a day and
    the arrival is what the check-in actually was.
    """
    mine = [p for p in day_passes
            if p.employee_id == employee_id and (p.body is not None or p.face is not None)]
    if not mine:
        return None
    return min(mine, key=lambda p: abs((p.last_seen - check_in).total_seconds())
               if p.last_seen and check_in else 1e9)


def candidates_for(day_passes, anchor, check_in):
    """Every EXIT pass after the check-in, in the anchor's vector space(s).

    `employee_id` is deliberately NOT filtered on - it is the label. On a
    complete day the true pass carries the name, so excluding named passes
    would remove the answer and make the test meaningless.
    """
    out = []
    for p in day_passes:
        if p.direction != "EXIT" or p.last_seen is None or p.last_seen <= check_in:
            continue
        same_body = (p.body is not None and anchor.body is not None
                     and p.body_model == anchor.body_model
                     and p.body.shape == anchor.body.shape)
        same_face = (p.face is not None and anchor.face is not None
                     and p.face_model == anchor.face_model
                     and p.face.shape == anchor.face.shape)
        if same_body or same_face:
            p._same_body, p._same_face = same_body, same_face
            out.append(p)
    return out


def score(p, anchor):
    """(body_sim, face_sim) against the anchor, None where not comparable."""
    b = float(p.body @ anchor.body) if p._same_body else None
    f = float(p.face @ anchor.face) if p._same_face else None
    return b, f


RANKERS = {
    "latest":        lambda c: c["t"],
    "pseudo_score":  lambda c: c["pseudo"] or -1,
    "body":          lambda c: c["b"] if c["b"] is not None else -1,
    "face":          lambda c: c["f"] if c["f"] is not None else -1,
    "face_or_body":  lambda c: c["f"] if c["f"] is not None else (
                               c["b"] if c["b"] is not None else -1),
    "fused":         lambda c: (0.7 * (c["f"] if c["f"] is not None else 0)
                                + 0.3 * (c["b"] if c["b"] is not None else 0)),
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--tolerance", type=float, default=120,
                    help="seconds from the real check-out that counts as correct")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    db = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    by_day = passes_by_day(db)

    cases, skipped = [], defaultdict(int)
    for emp, bdate, ci, co in load_days(db, complete=True):
        dp = by_day.get(bdate, [])
        a = anchor_for(dp, emp, ci)
        if a is None:
            skipped["no anchor pass"] += 1
            continue
        cands = candidates_for(dp, a, ci)
        if not cands:
            skipped["no comparable EXIT candidate"] += 1
            continue
        rows, truth = [], []
        for p in cands:
            b, f = score(p, a)
            rows.append({"t": p.last_seen.timestamp(), "b": b, "f": f,
                         "pseudo": p.pseudo_score, "id": p.id,
                         # LABELS - scoring only, never visible to a ranker.
                         "is_them": p.employee_id == emp})
            truth.append(abs((p.last_seen - co).total_seconds()) <= args.tolerance)
        if not any(truth):
            skipped["answer not among candidates"] += 1
            continue
        cases.append((rows, truth))

    n = len(cases)
    print(f"database        {args.db}")
    print(f"tolerance       ±{args.tolerance:.0f}s from the recorded check-out")
    print(f"usable cases    {n} complete days with an anchor and a reachable answer")
    for k, v in sorted(skipped.items(), key=lambda kv: -kv[1]):
        print(f"   skipped: {k:<32} {v}")
    if not n:
        print("\nnothing to measure")
        return 1
    sizes = [len(r) for r, _ in cases]
    print(f"candidates/day  avg {np.mean(sizes):.1f}  median {np.median(sizes):.0f}  "
          f"max {max(sizes)}")
    print(f"\nset_recall is 100% by construction here: a day whose answer is not\n"
          f"among the candidates is excluded above and counted as a skip.\n")

    # RIGHT PERSON and RIGHT PASS are different failures with different costs.
    # Picking another pass of the SAME person (a lunch trip) puts a real
    # departure at the wrong time; picking a STRANGER attributes somebody
    # else's walk. Reporting one number for both hides which one is happening.
    print(f"{'ranker':<14} {'right pass':>11} {'right person':>13}"
          f"   {'@ margin>=0.05: person / coverage':>34}")
    print("-" * 78)
    for name, key in RANKERS.items():
        hits = who = cov = cov_who = 0
        for rows, truth in cases:
            order = sorted(range(len(rows)), key=lambda i: -key(rows[i]))
            best = order[0]
            hits += truth[best]
            who += rows[best]["is_them"]
            margin = (key(rows[best]) - key(rows[order[1]])) if len(order) > 1 else 1.0
            if name not in ("latest", "pseudo_score") and margin >= 0.05:
                cov += 1
                cov_who += rows[best]["is_them"]
        line = f"{name:<14} {100.0*hits/n:>10.1f}% {100.0*who/n:>12.1f}%"
        if name not in ("latest", "pseudo_score"):
            c = (f"{100.0*cov_who/cov:.1f}% on {100.0*cov/n:.0f}% of days"
                 if cov else "-")
            line += f"   {c:>34}"
        print(line)

    print("\nThe margin column is the operating point that matters: a suggestion\n"
          "is only worth showing when the best candidate clearly beats the next.\n"
          "Coverage below 100% is the price, and is the right trade when a wrong\n"
          "check-out is invisible and a missing one is not.")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
