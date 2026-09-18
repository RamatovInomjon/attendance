#!/usr/bin/env python3
"""Give a person with NO gallery entry a first reference, from their own passes.

WHY THIS IS NOT `scripts/augment_gallery.py`
--------------------------------------------
Augmentation extends an identity the gallery already holds. `augment.scan()`
judges a crop by `own - other`: how much better it matches its own person than
anybody else. For somebody with **zero** embeddings there is no "own" - the
comparison is against an empty set, `own` is -1.0, and every candidate they have
is refused as "resembles another person more than its own label". That is the
correct behaviour for augmentation and useless for the case here: three people
enrolled through the browser before captures were saved to disk, whose only
photograph was an embedding that a recognizer swap correctly discarded.

So this bootstraps the FIRST row, and it is deliberately a separate script with
a separate name, because the one check it cannot make is the one that matters
most.

WHAT STANDS IN FOR THAT CHECK
-----------------------------
Two things, and neither is `own - other`:

* **A human asserts the identity.** The capture was named by the recognizer
  that was deployed at the time, at a score the operator sets with
  `--min-score`. On the previous model (threshold 0.22, genuine median 0.335) a
  0.45 pass is far above anything an impostor produced. It is still an
  assertion, not a measurement, and the script prints the capture, its date, its
  camera and its score so a person can refuse it.

* **The floor still applies, unchanged.** `augment.calibrate()` gives the row
  the same flat corridor floor every live crop gets, and REFUSES it outright if
  any real face - an enrolment photograph, another candidate, or an
  unidentified corridor probe - already reaches it at or above that floor. So a
  bootstrapped row cannot fire for somebody else, whatever the human believed.
  That property is measured, and it is the one that makes a wrong assertion
  survivable.

A bootstrapped row is written exactly like an augmented one (`live:` source
tag, its own `threshold`), so `/gallery/review` lists it and an admin can remove
it with one click. It is a stopgap: it is dropped by the next recognizer swap
like any corridor crop. Re-enrol these people properly through *Ro'yxatdan
o'tkazish* - that saves photographs, which survive everything.

USAGE
-----
    python scripts/bootstrap_face.py --faceless --min-score 0.45
    python scripts/bootstrap_face.py --faceless --min-score 0.45 --apply
    python scripts/bootstrap_face.py --employee 55 --employee 56 --apply

Writes nothing without `--apply`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sqlalchemy import select

from app.config import settings


def faceless_employees() -> list[tuple[int, str]]:
    """Active people with no embedding at all - who this script is for."""
    from app.db.models import Employee, FaceEmbedding
    from app.db.session import session_scope
    with session_scope() as s:
        has_face = select(FaceEmbedding.employee_id).distinct()
        return [(int(i), n) for i, n in s.execute(
            select(Employee.id, Employee.full_name)
            .where(Employee.is_active.is_(True), Employee.id.not_in(has_face))
            .order_by(Employee.id)).all()]


def captures_for(employee_ids: set[int]) -> dict[int, list[dict]]:
    """Every saved capture naming one of these people, best score first."""
    out: dict[int, list[dict]] = {i: [] for i in employee_ids}
    root = settings.debug_dir
    if not root.is_dir():
        return out
    for meta_f in root.rglob("*.json"):
        try:
            m = json.loads(meta_f.read_text())
        except Exception:
            continue
        eid = m.get("employee_id")
        if eid not in out:
            continue
        if not meta_f.with_name(meta_f.stem + "_aligned.jpg").is_file():
            continue
        q = m.get("quality") or {}
        out[eid].append({
            "key": meta_f.stem, "employee_id": int(eid),
            "name": m.get("name", ""), "camera": m.get("camera", ""),
            "when": str(m.get("timestamp_local", ""))[:19],
            "score": float(m.get("score", 0.0)),
            "frames": int(m.get("embedded_frames", 0)),
            "aligner": float(q.get("aligner_score", 0.0)),
            "gate": q.get("gate"),
        })
    for rows in out.values():
        rows.sort(key=lambda r: -r["score"])
    return out


def build_candidates(picked: list[dict]):
    """Re-embed each chosen capture WITH THE CURRENT RECOGNIZER.

    The stored vectors cannot be reused: they are in the previous model's space,
    where a cosine against this gallery is meaningless and - the reason the
    provenance columns exist - still returns a plausible number.
    """
    import cv2
    from app.core.geometry import to_normalized_chw
    from app.core.recognizer import FaceRecognizer
    from app.services.augment import Candidate, crop_path

    rec = FaceRecognizer(settings.model_path(settings.recognizer_model),
                         batch_size=settings.embed_batch)
    cands = []
    for r in picked:
        p = crop_path(r["key"])
        img = cv2.imread(str(p)) if p else None
        if img is None:
            print(f"  ! {r['key']}: aligned crop unreadable, skipped")
            continue
        # debug_capture upscales the recognizer's 112x112 to 224 for viewing.
        if img.shape[:2] != (112, 112):
            img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
        v = rec.embed(to_normalized_chw(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[None])[0]
        c = Candidate(key=r["key"], employee_id=r["employee_id"], name=r["name"],
                      camera=r["camera"], when=r["when"], score=r["score"],
                      frames=r["frames"], aligner=r["aligner"])
        c.vec = v
        cands.append(c)
    return cands


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--employee", type=int, action="append", default=[],
                    help="employee id to bootstrap; repeatable")
    ap.add_argument("--faceless", action="store_true",
                    help="every active employee with no embedding at all")
    ap.add_argument("--min-score", type=float, default=0.45,
                    help="lowest capture score to trust (previous model's "
                         "scale; default 0.45)")
    ap.add_argument("--per-person", type=int, default=1,
                    help="how many crops to add each (default 1)")
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--min-aligner", type=float, default=0.95)
    ap.add_argument("--apply", action="store_true",
                    help="actually write; without it this is a dry run")
    args = ap.parse_args()

    from app.services import augment
    from app.services.enrollment import load_gallery

    targets = dict(faceless_employees()) if args.faceless else {}
    for i in args.employee:
        targets.setdefault(i, "")
    if not targets:
        print("nothing to do: no --employee given and nobody is faceless")
        return 0

    print(f"bootstrapping {len(targets)} person(s): "
          + ", ".join(f"{i}" + (f" ({n})" if n else "") for i, n in targets.items()))
    print(f"recognizer {settings.recognizer_model}, "
          f"floor {max(settings.threshold_for(settings.recognizer_model), settings.augment_live_floor):.3f}\n")

    found = captures_for(set(targets))
    picked: list[dict] = []
    for eid, rows in found.items():
        ok = [r for r in rows
              if r["score"] >= args.min_score
              and r["frames"] >= args.min_frames
              and r["aligner"] >= args.min_aligner
              and r["gate"] in (None, "PASS")]
        label = targets.get(eid) or (rows[0]["name"] if rows else str(eid))
        if not ok:
            best = f"{rows[0]['score']:.3f}" if rows else "none"
            print(f"  employee {eid} ({label}): NOTHING USABLE "
                  f"({len(rows)} capture(s), best score {best})")
            continue
        take = ok[:args.per_person]
        for r in take:
            print(f"  employee {eid} ({label}): {r['key']}")
            print(f"      score {r['score']:.3f} · {r['frames']} frames · "
                  f"aligner {r['aligner']:.2f} · {r['camera']} · {r['when']}")
        picked += take

    if not picked:
        print("\nno usable capture for anybody - lower --min-score, or re-enrol them")
        return 1

    print(f"\nre-embedding {len(picked)} crop(s) with the current recognizer...")
    cands = build_candidates(picked)
    if not cands:
        print("nothing could be re-embedded")
        return 1

    gal = load_gallery()
    augment.calibrate(cands, gal.M, gal.owner, gal.names)

    accepted = [c for c in cands if not c.rejected]
    print()
    for c in cands:
        head = f"  {c.name or c.employee_id} · {c.key}"
        if c.rejected:
            print(f"{head}\n      REFUSED: {c.rejected}")
        else:
            print(f"{head}\n      floor {c.threshold:.3f} · worst impostor "
                  f"{c.impostor:.3f}"
                  + (f" ({c.impostor_name})" if c.impostor_name else ""))

    if not accepted:
        print("\nevery candidate was refused by the floor - nothing to add")
        return 1
    if not args.apply:
        print(f"\ndry run - nothing written. Re-run with --apply to add "
              f"{len(accepted)} embedding(s).")
        return 0

    n = augment.add(accepted)
    print(f"\nadded {n} embedding(s). Restart the service to load them:")
    print("    ./scripts/restart.sh")
    print("These are corridor crops: the next recognizer swap drops them. "
          "Re-enrol these people through Ro'yxatdan o'tkazish when you can.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
