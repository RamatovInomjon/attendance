#!/usr/bin/env python3
"""Add live corridor faces to the gallery, so it stops being only ID photos.

    python scripts/augment_gallery.py --review          # pick candidates, no writes
    (or use the web UI: Galereya, admin only - same rules, same refusal)
    python scripts/augment_gallery.py --measure         # what would it do to FAR?
    python scripts/augment_gallery.py --apply
    python scripts/augment_gallery.py --revert          # remove every live entry
    python scripts/augment_gallery.py --recalibrate     # refresh existing floors

THE PROBLEM THIS SOLVES
-----------------------
The gallery is enrolment photography: frontal, lit, close. The corridor is not.
Measured on this install, the same people score ~0.85 against the gallery and
0.20-0.43 walking past, and 17 of 54 enrolled people were never recognised once
in a full day. The model is not the limit - the domain gap is. A face that was
recognised at 0.55 walking through the door is worth more, as a reference for
next time, than any studio photograph.

WHY THIS IS DANGEROUS, AND WHAT GUARDS IT
-----------------------------------------
Adding a MIS-recognised crop is not a small mistake. That face becomes a
permanent reference for the wrong person, so the same error gets easier next
time, and easier again - the gallery poisons itself, silently, and nothing in
the attendance data reveals it.

So candidates must clear all of:

  * a high score AND a real margin over the runner-up - the margin is what
    distinguishes "clearly this person" from "closest of several"
  * a pass that was decided by many agreeing frames, not a lucky one
  * the quality gate the pipeline itself applies (aligner score, pose, blur)
  * a similarity ceiling against what is already stored, so twenty frames of
    one walk do not become twenty near-identical vectors that outvote the rest

And then every surviving crop is given its OWN acceptance floor, just above the
highest similarity any other person's vector reaches against it. It cannot name
anybody below that floor, so it cannot manufacture a false accept against any
face we have measured - and because the floor is the lowest value with that
property, it still fires at the weakest similarity that is defensible.

This script is a thin front end. The rules live in `app/services/augment.py`,
which the web UI calls too, so the CLI and the UI cannot drift apart - the
previous version reimplemented all of it and had already drifted.

Everything added is tagged `live:` in `source_file`, so `--revert` removes
exactly these and leaves the enrolment untouched.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-score", type=float, default=None,
                    help="default: 1.5x the recognition threshold")
    ap.add_argument("--min-margin", type=float, default=0.10,
                    help="own-person similarity minus the best OTHER person, "
                         "recomputed here rather than read from the capture - "
                         "anything written before the margin fix stores 0.0")
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--min-aligner", type=float, default=0.95)
    ap.add_argument("--max-per-person", type=int, default=5)
    ap.add_argument("--max-similarity", type=float, default=0.92,
                    help="reject a candidate this close to one already kept")
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--recalibrate", action="store_true",
                    help="recompute the floors of the live rows already stored")
    ap.add_argument("--show-rejected", action="store_true")
    ap.add_argument("--out", default="data/gallery_candidates")
    args = ap.parse_args()

    from app.config import settings
    from app.core.onnx_env import preload_cuda_libs
    from app.services import augment

    thr = settings.threshold_for(settings.recognizer_model)

    if args.revert:
        existing = augment.added()
        if not args.apply:
            print(f"  {len(existing)} live embedding(s) would be removed. Add --apply.")
            return 0
        n = augment.remove([e["id"] for e in existing])
        print(f"  removed {n} live embedding(s); enrolment untouched")
        return 0

    if args.recalibrate:
        n = augment.recalibrate()
        print(f"  refreshed the floor on {n} live embedding(s)")
        for e in augment.added():
            flag = "  LOOKALIKE - review it" if e["reached"] else ""
            print(f"    {e['name'][:28]:28s} floor {e['threshold']:.3f}"
                  f"  reached at {e['reached']:.3f}{flag}" if e["reached"]
                  else f"    {e['name'][:28]:28s} floor {e['threshold']:.3f}")
        return 0

    preload_cuda_libs()
    cands = augment.scan(
        min_score=args.min_score, min_margin=args.min_margin,
        min_frames=args.min_frames, min_aligner=args.min_aligner,
        max_per_person=args.max_per_person, max_similarity=args.max_similarity,
        include_rejected=args.show_rejected)
    offered = [c for c in cands if not c.rejected]

    print(f"  candidates offered : {len(offered)}")
    print(f"    score >= {(args.min_score or thr * 1.5):.3f}  "
          f"margin >= {args.min_margin:.2f}  frames >= {args.min_frames}  "
          f"aligner >= {args.min_aligner:.2f}")
    if args.show_rejected:
        for c in cands:
            if c.rejected:
                print(f"    - {c.name[:24]:24s} {c.key[:28]:28s} {c.rejected}")
    if not offered:
        print("\n  Nothing qualifies. Either the thresholds are too strict for this\n"
              "  data, or margin is 0.0 throughout - captures written before the\n"
              "  margin fix carry no runner-up distance and cannot be judged.")
        return 1

    by_person = defaultdict(list)
    for c in offered:
        by_person[c.employee_id].append(c)
    print()
    for emp, group in sorted(by_person.items(), key=lambda kv: kv[1][0].name):
        worst = max(c.impostor for c in group)
        print(f"    {group[0].name[:28]:28s} +{len(group)}   "
              f"floor {group[0].threshold:.3f}   worst other face {worst:.3f}")

    if args.review:
        out = ROOT / args.out
        shutil.rmtree(out, ignore_errors=True)
        for c in offered:
            d = out / f"{c.employee_id:03d}_{c.name.replace(' ', '_')}"
            d.mkdir(parents=True, exist_ok=True)
            src = augment.crop_path(c.key)
            if src is not None:
                shutil.copy(src, d / f"s{c.score:.3f}_m{c.margin:.3f}_"
                                     f"t{c.threshold:.3f}_{c.camera}.jpg")
        print(f"\n  wrote {len(offered)} candidate(s) to {out}")
        print("  LOOK AT THEM. Any crop that is not the named person poisons that")
        print("  identity permanently, and the error compounds. Delete any that are")
        print("  wrong, then re-run with --measure.")
        return 0

    check = augment.check_impostors(offered)
    print(f"\n  recognition threshold            {check.threshold:.4f}")
    print(f"  gallery's own worst pair         {check.before:.4f}")
    if check.gallery_unsafe and check.pre_existing:
        print("    ^ ALREADY at or above the threshold, and NOT caused by this:")
        print(f"      {check.pre_existing[0]}")
        print(f"      {check.pre_existing[1]}")
        print("      Two enrolled people this close is a live false-accept risk.")
        print("      Only re-enrolling one of them fixes it; augmentation cannot,")
        print("      and no longer refuses on account of it.")
    print(f"  worst the selection can reach    {check.after:.4f}  "
          f"(after each crop's own floor)")
    for name, when, sim in check.unusable:
        print(f"    REJECTED {name} ({when}): already reached at {sim:.3f}")

    if not check.safe:
        print("\n  REFUSED: another face already reaches the crops listed above at")
        print("  or above the floor a corridor crop answers to. They are")
        print("  lookalikes, not references. Deselect them; the rest is fine.")
        return 1
    print("  SAFE: no crop can name anybody at a similarity another face reaches.")

    if args.measure or not args.apply:
        print("\n  Nothing written. Add --apply.")
        return 0

    n = augment.add(offered)
    print(f"\n  added {n} live embedding(s), tagged '{augment.TAG}', each with its floor")
    print("  Reload the gallery (POST /api/gallery/reload) or restart.")
    print("  Undo with: python scripts/augment_gallery.py --revert --apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
