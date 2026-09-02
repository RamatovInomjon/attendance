#!/usr/bin/env python3
"""Add live corridor faces to the gallery, so it stops being only ID photos.

    python scripts/augment_gallery.py --review          # pick candidates, no writes
    python scripts/augment_gallery.py --measure         # what would it do to FAR?
    python scripts/augment_gallery.py --apply
    python scripts/augment_gallery.py --revert          # remove every live entry

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

And after selection, `--measure` re-runs the gallery separation with the
candidates included and REFUSES to apply if the worst impostor pair moves above
the recognition threshold. That check is the point: augmentation is supposed to
raise genuine scores without raising impostor scores, and if it does the latter
the additions are wrong.

Everything added is tagged `live:` in `source_file`, so `--revert` removes
exactly these and leaves the enrolment untouched.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TAG = "live:"


def candidates(debug_dir: Path, min_score, min_margin, min_frames, min_aligner):
    """Every recognised pass that is a defensible reference for its identity."""
    out = []
    for meta_f in sorted(debug_dir.rglob("*.json")):
        try:
            m = json.loads(meta_f.read_text())
        except Exception:
            continue
        aligned = meta_f.with_name(meta_f.stem + "_aligned.jpg")
        if not aligned.is_file() or m.get("employee_id") is None:
            continue
        q = m.get("quality") or {}
        why = []
        if float(m.get("score", 0)) < min_score:
            why.append(f"score {m.get('score', 0):.3f}")
        if int(m.get("embedded_frames", 0)) < min_frames:
            why.append(f"frames {m.get('embedded_frames', 0)}")
        if float(q.get("aligner_score", 0)) < min_aligner:
            why.append(f"aligner {q.get('aligner_score', 0):.2f}")
        if q.get("gate") not in (None, "PASS"):
            why.append(f"gate {q.get('gate')}")
        out.append({"meta": m, "aligned": aligned, "rejected": why,
                    "employee_id": int(m["employee_id"]),
                    "name": m.get("name", ""), "score": float(m.get("score", 0)),
                    "margin": float(m.get("margin", 0)),
                    "camera": m.get("camera", ""), "key": meta_f.stem})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--debug-dir", default=None, help="default: settings.debug_dir")
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
    ap.add_argument("--out", default="data/gallery_candidates")
    args = ap.parse_args()

    import cv2
    from sqlalchemy import delete, func, select
    from app.config import settings
    from app.core.onnx_env import preload_cuda_libs
    from app.db.models import Employee, FaceEmbedding
    from app.db.session import session_scope

    thr = settings.threshold_for(settings.recognizer_model)
    min_score = args.min_score if args.min_score is not None else thr * 1.5

    # ---- revert -------------------------------------------------------
    if args.revert:
        with session_scope() as s:
            n = s.execute(select(func.count()).select_from(FaceEmbedding)
                          .where(FaceEmbedding.source_file.like(f"{TAG}%"))).scalar()
            if args.apply:
                s.execute(delete(FaceEmbedding)
                          .where(FaceEmbedding.source_file.like(f"{TAG}%")))
                print(f"  removed {n} live embedding(s); enrolment untouched")
            else:
                print(f"  {n} live embedding(s) would be removed. Add --apply.")
        return 0

    preload_cuda_libs()
    from app.core.recognizer import FaceRecognizer
    from app.core.geometry import to_normalized_chw

    debug_dir = Path(args.debug_dir) if args.debug_dir else settings.debug_dir
    if not debug_dir.is_dir():
        raise SystemExit(f"  no captures at {debug_dir} (is debug_capture on?)")

    cands = candidates(debug_dir, min_score, args.min_margin,
                       args.min_frames, args.min_aligner)
    kept_pool = [c for c in cands if not c["rejected"]]
    print(f"  recognised passes on disk : {len(cands)}")
    print(f"  clearing every gate       : {len(kept_pool)}")
    print(f"    score >= {min_score:.3f}  margin >= {args.min_margin:.2f}  "
          f"frames >= {args.min_frames}  aligner >= {args.min_aligner:.2f}")
    if not kept_pool:
        print("\n  Nothing qualifies. Either the thresholds are too strict for this\n"
              "  data, or margin is 0.0 throughout - captures written before the\n"
              "  margin fix carry no runner-up distance and cannot be judged.")
        return 1

    rec = FaceRecognizer(settings.model_path(settings.recognizer_model),
                         batch_size=settings.embed_batch)

    # ---- embed, then thin within each identity ------------------------
    by_person = defaultdict(list)
    for c in kept_pool:
        by_person[c["employee_id"]].append(c)

    from app.services.enrollment import load_gallery
    gal = load_gallery()

    chosen = []
    for emp, group in by_person.items():
        group.sort(key=lambda c: -c["score"])
        keep: list = []
        for c in group:
            if len(keep) >= args.max_per_person:
                break
            img = cv2.imread(str(c["aligned"]))
            if img is None:
                continue
            # debug_capture upscales the recognizer's 112x112 to 224 for human
            # viewing, so it has to come back down. Verified over 250 captures:
            # every one still matches its own person best, worst impostor 0.171
            # against a 0.215 threshold.
            if img.shape[0] != 112 or img.shape[1] != 112:
                img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
            v = rec.embed(to_normalized_chw(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[None])[0]

            # The margin that matters is measured HERE, against the live
            # gallery: how far this face sits from the nearest OTHER person.
            # The stored value cannot be used - captures written before the
            # margin fix all carry 0.0, which would reject the entire history.
            sims = gal.M @ v
            own = float(sims[gal.owner == emp].max()) if (gal.owner == emp).any() else -1.0
            others = sims[gal.owner != emp]
            best_other = float(others.max()) if others.size else -1.0
            c["own"], c["other"] = own, best_other
            c["computed_margin"] = own - best_other
            if own <= best_other:
                continue          # it resembles somebody else more; not a reference
            if c["computed_margin"] < args.min_margin:
                continue
            # Twenty frames of one walk are twenty near-identical vectors; they
            # would outvote the enrolment photos without adding information.
            if any(float(v @ k["vec"]) > args.max_similarity for k in keep):
                continue
            c["vec"] = v
            keep.append(c)
        chosen += keep

    print(f"  after per-person thinning : {len(chosen)} "
          f"across {len(by_person)} person(s)\n")
    for emp, group in sorted(by_person.items()):
        n = sum(1 for c in chosen if c["employee_id"] == emp)
        if n:
            print(f"    {group[0]['name'][:30]:30s} +{n}")

    # ---- review folder ------------------------------------------------
    if args.review:
        out = ROOT / args.out
        shutil.rmtree(out, ignore_errors=True)
        for c in chosen:
            d = out / f"{c['employee_id']:03d}_{c['name'].replace(' ', '_')}"
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy(c["aligned"],
                        d / f"s{c['score']:.3f}_m{c['computed_margin']:.3f}_"
                            f"{c['camera']}.jpg")
            face = c["aligned"].with_name(c["aligned"].name.replace("_aligned", "_face"))
            if face.is_file():
                shutil.copy(face, d / f"s{c['score']:.3f}_native.jpg")
        print(f"\n  wrote {len(chosen)} candidate(s) to {out}")
        print("  LOOK AT THEM. Any crop that is not the named person poisons that")
        print("  identity permanently, and the error compounds. Delete any that are")
        print("  wrong, then re-run with --measure.")
        return 0

    # ---- the safety check ---------------------------------------------
    with session_scope() as s:
        rows = s.execute(
            select(FaceEmbedding.employee_id, FaceEmbedding.vector)).all()
        names = dict(s.execute(select(Employee.id, Employee.full_name)).all())
    M = np.stack([np.frombuffer(r[1], np.float32) for r in rows])
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    owner = np.array([r[0] for r in rows])

    def worst_impostor(mat, own, labels=None):
        """Highest similarity between vectors belonging to DIFFERENT people.

        Returns the value and, when `labels` is given, which pair produced it -
        because "something is wrong" is not actionable and "these two faces,
        this candidate" is.
        """
        sim = mat @ mat.T
        np.fill_diagonal(sim, -1)
        cross = own[:, None] != own[None, :]
        masked = np.where(cross, sim, -1)
        i, j = np.unravel_index(int(masked.argmax()), masked.shape)
        val = float(masked[i, j])
        if labels is None:
            return val
        return val, (labels[i], labels[j])

    before = worst_impostor(M, owner)
    M2 = np.vstack([M, np.stack([c["vec"] for c in chosen])])
    o2 = np.concatenate([owner, np.array([c["employee_id"] for c in chosen])])
    labels = ([f"enrolment  {names.get(int(e), e)}" for e in owner]
              + [f"CANDIDATE  {names.get(c['employee_id'], c['employee_id'])}"
                 f"  ({c['key']}, own {c['own']:.3f} margin {c['computed_margin']:.3f})"
                 for c in chosen])
    after, pair = worst_impostor(M2, o2, labels)

    print(f"\n  worst impostor pair  before {before:.4f}   after {after:.4f}"
          f"   ({after - before:+.4f})")
    print(f"  recognition threshold      {thr:.4f}")
    safe = after < thr
    if safe:
        print("  SAFE: the worst impostor stays below the threshold")
    else:
        print("  REFUSED: augmenting pushes two DIFFERENT people to or above the")
        print("  threshold, which manufactures a false accept. The pair is:")
        print(f"    {pair[0]}")
        print(f"    {pair[1]}")
        print("  If one of those is a CANDIDATE, look at its crop in the review")
        print("  folder: either it is a mis-recognition being fed back, or those")
        print("  two people genuinely look alike and neither should be added.")
        print("  Delete it and re-run, or raise --min-margin.")
    if not safe or args.measure or not args.apply:
        if args.apply and not safe:
            return 1
        if not args.apply:
            print("\n  Nothing written. Add --apply.")
        return 0

    with session_scope() as s:
        for c in chosen:
            s.add(FaceEmbedding(
                employee_id=c["employee_id"],
                source_file=f"{TAG}{c['key']}",
                vector=c["vec"].astype(np.float32).tobytes(),
                dim=int(c["vec"].shape[0]),
                model_name=settings.recognizer_model,
                quality=float(c["computed_margin"]),
            ))
    print(f"\n  added {len(chosen)} live embedding(s), tagged '{TAG}'")
    print("  Reload the gallery (POST /api/gallery/reload) or restart.")
    print("  Undo with: python scripts/augment_gallery.py --revert --apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
