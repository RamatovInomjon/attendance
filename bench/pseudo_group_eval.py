#!/usr/bin/env python3
"""How many DISTINCT people were the unknown passes? Scored on production data.

    python bench/pseudo_group_eval.py --persons /media/.../data/persons --day 20260907
    python bench/pseudo_group_eval.py --persons ... --day 20260907 --sweep
    python bench/pseudo_group_eval.py --persons ... --day 20260907 --json out.json

WHAT THIS ANSWERS
-----------------
`unknown_sighting` counts PASSES. An operator looking at 1082 rows for one day
cannot tell one visitor seen five times from five visitors seen once, and the
building has no idea how many distinct people walked through it.
`app/services/pseudo_gallery.py` groups those passes into pseudo-identities.
This measures how well, on real production passes rather than on a prototype's.

THE GROUND TRUTH, AND ITS LIMIT
-------------------------------
Unregistered people carry no label - that is what makes them unregistered. So
truth is built the way the research built it: passes whose best face is large
enough to trust (inter-pupil distance >= `pseudo_face_ipd_min`) are linked to
each other at a STRICT face similarity (0.45, well above the 0.35 the grouping
itself uses) and the connected components of that graph are taken as "the same
person".

That truth is only defined on the passes that HAVE a good face - 23-28% of them
here. It says nothing about the rest, so pair recall below is recall over the
truth-covered subset, and `pseudo_people` counts every pass. Both numbers are
reported rather than blended, because blending them would hide which one the
grouping is actually good at.

Running order is chronological and online: each pass sees only what came
before it, exactly as the deployment does. No pass is ever grouped using a
later one.

WHAT IS PRODUCTION AND WHAT IS RECOMPUTED
-----------------------------------------
The stored crops and `_pass.json` are production's own. The BODY feature is
recomputed with the currently configured ReID model (the stored `reid_vector`
belongs to whichever model was deployed at the time, and this is here to
compare models). The FACE feature is computed by the production face path -
`HeadDetector` -> `FaceAligner` -> `FaceRecognizer` - run over the stored body
crops, which is the same chain the live pipeline runs over full frames.

That last point is a real limitation and is not hidden: a face inside a 320-px
body crop is smaller than the same face in the 4K frame, so IPD here is a lower
bound and the fraction of passes with a usable face is pessimistic. Use
`bench/tracklet_recheck_eval.py` on the recordings for the full-frame view.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- #
#  reading the corpus
# --------------------------------------------------------------------------- #
def tracklets(root: Path, day: str, kind: str) -> list[dict]:
    """One entry per stored person-pass, in time order."""
    base = root / kind / day
    if not base.is_dir():
        return []
    out = []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        crops = sorted(d.glob("*.jpg"))
        if not crops:
            continue
        meta = {}
        f = d / "_pass.json"
        if f.is_file():
            try:
                meta = json.loads(f.read_text())
            except Exception:
                meta = {}
        # `known/<day>/<Person_Name>/` is one folder per PERSON, holding every
        # crop of theirs that day; split it back into passes by camera+track,
        # which the filenames carry. `unknown/` is already one folder per pass.
        if kind == "known":
            by_pass: dict[tuple, list] = defaultdict(list)
            for c in crops:
                parts = c.stem.split("_")
                try:
                    cam, tid = parts[2], parts[3]
                except IndexError:
                    continue
                by_pass[(cam, tid)].append(c)
            for (cam, tid), files in by_pass.items():
                out.append({"dir": d, "name": d.name, "camera": cam,
                            "track": tid, "crops": sorted(files),
                            "employee_id": meta.get("employee_id"),
                            "stamp": sorted(files)[0].stem[:15]})
        else:
            out.append({"dir": d, "name": d.name,
                        "camera": meta.get("camera", d.name.split("_")[0]),
                        "track": str(meta.get("track_id", "")),
                        "crops": crops, "employee_id": None,
                        "stamp": crops[0].stem[:15]})
    out.sort(key=lambda t: t["stamp"])
    return out


# --------------------------------------------------------------------------- #
#  features
# --------------------------------------------------------------------------- #
class Features:
    """The production face and body chains, run over stored crops."""

    def __init__(self):
        from app.config import settings
        from app.core.onnx_env import preload_cuda_libs
        preload_cuda_libs()
        from app.core.aligner import FaceAligner
        from app.core.head_detector import CLS_HEAD, HeadDetector
        from app.core.recognizer import FaceRecognizer
        from app.core.reid import PersonReID

        self.s = settings
        self.CLS_HEAD = CLS_HEAD
        self.head = HeadDetector(settings.model_path(settings.head_model),
                                 size=settings.head_input,
                                 conf=min(settings.head_conf, settings.track_low_thresh))
        self.aligner = FaceAligner(settings.model_path(settings.aligner_model),
                                   crop_size=settings.align_crop_size,
                                   margin=settings.align_margin,
                                   mode=settings.align_mode)
        self.rec = FaceRecognizer(settings.model_path(settings.recognizer_model),
                                  batch_size=settings.embed_batch)
        self.reid = PersonReID(settings.model_path(settings.reid_model),
                               batch_size=settings.embed_batch)

    def body(self, imgs) -> np.ndarray:
        from app.core.reid import aggregate, sharpness_of
        embs = self.reid.embed(imgs)
        return aggregate(embs, None, [sharpness_of(i) for i in imgs])

    def face(self, imgs, gates: bool) -> tuple[np.ndarray | None, float, int]:
        """(template, best ipd, frames used) for one pass.

        `gates` applies the live quality gates, which is what
        `CameraWorker._second_chance` sees. Without them every alignable head
        contributes, which is what the research prototype measured - the two
        differ and the difference is the point of the `--no-gates` flag.
        """
        from app.core.quality import assess
        boxes, keep = [], []
        for i, im in enumerate(imgs):
            dets = [d for d in self.head.detect(im)
                    if d.cls == self.CLS_HEAD and d.score >= self.s.head_conf]
            if not dets:
                continue
            b = max(dets, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
            boxes.append(b.box.astype(np.float32))
            keep.append(i)
        if not boxes:
            return None, 0.0, 0

        # Align and gate first, then embed the survivors in ONE batch. The
        # recogniser is a 130 MB IR-101; calling it once per crop spends most
        # of its time on fixed per-call cost, and a pass has up to ten crops.
        aligned, weights, best_ipd = [], [], 0.0
        for i, box in zip(keep, boxes):
            faces = self.aligner.align(imgs[i], [box], is_bgr=True)
            if not faces:
                continue
            f = faces[0]
            q = assess(box, f.aligned, f.landmarks, f.score,
                       min_face_px=self.s.min_face_px,
                       min_laplacian_var=self.s.min_laplacian_var,
                       min_aligner_score=self.s.min_aligner_score,
                       max_yaw_deg=self.s.max_yaw_deg,
                       max_pitch_deg=self.s.max_pitch_deg)
            if gates and not q.ok:
                continue
            if not gates and f.score < 0.5:
                # Even ungated, a head the aligner did not believe held a face
                # regresses symmetric landmarks off a hairline. That is not a
                # weak face, it is a made-up one.
                continue
            aligned.append(f.aligned)
            weights.append(float(f.score) * float(np.sqrt(max(q.ipd, 1e-6))))
            best_ipd = max(best_ipd, float(q.ipd))
        if not aligned:
            return None, best_ipd, 0
        embs = self.rec.embed(np.stack(aligned))
        w = np.asarray(weights, np.float32)
        acc = (embs * w[:, None]).sum(axis=0)
        n = len(aligned)
        norm = float(np.linalg.norm(acc))
        if norm <= 1e-6:
            return None, best_ipd, n
        return (acc / norm).astype(np.float32), best_ipd, n


# --------------------------------------------------------------------------- #
#  truth and scoring
# --------------------------------------------------------------------------- #
def truth_clusters(F: np.ndarray, threshold: float) -> np.ndarray:
    """Connected components of "these two faces are the same person"."""
    n = len(F)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    S = F @ F.T
    for i in range(n):
        for j in np.where(S[i] >= threshold)[0]:
            if j > i:
                a, b = find(i), find(int(j))
                if a != b:
                    parent[a] = b
    lab, out = {}, []
    for i in range(n):
        r = find(i)
        out.append(lab.setdefault(r, len(lab)))
    return np.asarray(out)


def pair_scores(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float, int, int]:
    """Precision and recall over PAIRS, the standard clustering measure.

    Pairs, not cluster identity: a grouping that splits one person in two is
    wrong in proportion to how many of their pass-pairs it broke, which is what
    matters for counting.
    """
    same_p = pred[:, None] == pred[None, :]
    same_t = truth[:, None] == truth[None, :]
    iu = np.triu_indices(len(pred), k=1)
    p, t = same_p[iu], same_t[iu]
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    return prec, rec, tp, fp


def run_gallery(passes, mode: str, day: date, model: str) -> np.ndarray:
    """Group every pass online with the production gallery; return its labels."""
    from app.config import settings
    from app.db.models import ReidPass
    from app.db.session import session_scope
    from app.services.pseudo_gallery import PseudoGallery

    labels = np.full(len(passes), -1, np.int64)
    with session_scope() as s:
        g = PseudoGallery()
        base = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        for i, t in enumerate(passes):
            row = ReidPass(camera_id=1 if t["camera"] == "Entrance" else 2,
                           camera_name=t["camera"], track_id=i,
                           first_seen=base, last_seen=base, business_date=day,
                           direction="UNKNOWN", dim=len(t["body"]),
                           model_name=model)
            s.add(row)
            s.flush()
            p = g.place(s, row,
                        face=t["face"] if mode in ("face", "face_body") else None,
                        body=t["body"] if mode in ("body", "face_body") else None,
                        face_ipd=t["ipd"], body_model=model)
            labels[i] = p.id if p is not None else -(i + 1)
        stats = g.stats()
        s.rollback()
    return labels, stats


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--persons", required=True,
                    help="a data/persons tree copied from production")
    ap.add_argument("--day", required=True, help="YYYYMMDD")
    ap.add_argument("--limit", type=int, default=0, help="first N passes, 0 = all")
    ap.add_argument("--max-crops", type=int, default=10)
    ap.add_argument("--truth-threshold", type=float, default=0.45)
    ap.add_argument("--kind", default="unknown", choices=("unknown", "known"),
                    help="which corpus to group. `known` is the NON-CIRCULAR "
                         "check: those passes carry a real identity from the "
                         "face path, so truth does not come from the same "
                         "feature the grouping uses.")
    ap.add_argument("--no-gates", action="store_true",
                    help="build face templates from every alignable head, not "
                         "only the frames the live gates admit")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--wide-sweep", action="store_true",
                    help="a larger threshold grid; needs --cache to be sane")
    ap.add_argument("--cache", default=None,
                    help="npz of the extracted features. Written if absent, "
                         "read if present - extraction is ~20 min for a day "
                         "and a threshold sweep must not pay it twice.")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    import cv2
    from app.config import settings

    root = Path(args.persons)
    passes = tracklets(root, args.day, args.kind)
    if args.limit:
        passes = passes[: args.limit]
    if not passes:
        raise SystemExit(f"  no passes under {root / args.kind / args.day}")
    print(f"  {len(passes)} {args.kind} pass(es) from {root}/{args.kind}/{args.day}")
    print(f"  reid  {settings.reid_model}")
    print(f"  face  {settings.recognizer_model}  gates="
          f"{'off' if args.no_gates else 'on'}")
    print(f"  thresholds: face {settings.pseudo_face_threshold} "
          f"body {settings.pseudo_body_threshold} "
          f"ipd>={settings.pseudo_face_ipd_min}\n")

    cache = Path(args.cache) if args.cache else None
    if cache and cache.is_file():
        z = np.load(cache, allow_pickle=True)
        names_c = list(z["name"])
        by_name = {t["name"] + "|" + str(t["track"]): t for t in passes}
        kept = []
        for i, nm in enumerate(names_c):
            t = by_name.get(str(nm))
            if t is None:
                continue
            t["body"] = z["body"][i]
            f = z["face"][i]
            t["face"] = None if not np.any(f) else f
            t["ipd"] = float(z["ipd"][i])
            t["nface"] = int(z["nface"][i])
            kept.append(t)
        passes = kept
        print(f"  loaded {len(passes)} cached feature set(s) from {cache}\n")
        return _score(args, passes, settings)

    feats = Features()
    kept = []
    for n, t in enumerate(passes, 1):
        imgs = [cv2.imread(str(c)) for c in t["crops"][: args.max_crops]]
        imgs = [i for i in imgs if i is not None]
        if not imgs:
            continue
        t["body"] = feats.body(imgs)
        t["face"], t["ipd"], t["nface"] = feats.face(imgs, gates=not args.no_gates)
        kept.append(t)
        if n % 25 == 0 or n == len(passes):
            good = sum(1 for x in kept
                       if x["face"] is not None and x["ipd"] >= settings.pseudo_face_ipd_min)
            print(f"    {n}/{len(passes)} features  ({good} with a usable face)",
                  end="\r", flush=True)
    passes = kept
    print()

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        d = len(passes[0]["body"])
        fd = next((len(t["face"]) for t in passes if t["face"] is not None), d)
        np.savez_compressed(
            cache,
            name=np.array([t["name"] + "|" + str(t["track"]) for t in passes]),
            body=np.stack([t["body"] for t in passes]),
            face=np.stack([t["face"] if t["face"] is not None
                           else np.zeros(fd, np.float32) for t in passes]),
            ipd=np.array([t["ipd"] for t in passes], np.float32),
            nface=np.array([t["nface"] for t in passes], np.int32))
        print(f"  cached features -> {cache}")

    return _score(args, passes, settings)


def _score(args, passes, settings):
    usable = [i for i, t in enumerate(passes)
              if t["face"] is not None and t["ipd"] >= settings.pseudo_face_ipd_min]
    print(f"\n  {len(usable)}/{len(passes)} passes carry a usable face "
          f"({len(usable) / len(passes) * 100:.1f}%) - this is the ceiling on "
          f"how much can be regrouped at all")
    if len(usable) < 2:
        raise SystemExit("  too few usable faces to build a truth set")

    if args.kind == "known":
        # REAL identities. `known/<day>/<Person_Name>/` was written because the
        # face path named that pass at the time, so the folder is a label that
        # owes nothing to the feature being evaluated. Every pass counts here,
        # not only the ones with a usable face - which is why the count error
        # below is the honest one and the unknown-corpus number is not.
        usable = list(range(len(passes)))
        lab: dict = {}
        truth = np.array([lab.setdefault(t["name"], len(lab)) for t in passes])
        n_truth = len(lab)
        print(f"  truth = the face path's own labels: {n_truth} people "
              f"across {len(passes)} passes  (NON-CIRCULAR)")
    else:
        F = np.stack([passes[i]["face"] for i in usable])
        truth = truth_clusters(F, args.truth_threshold)
        n_truth = len(set(truth.tolist()))
        print(f"  truth (face sim >= {args.truth_threshold}, connected "
              f"components): {n_truth} distinct people among those "
              f"{len(usable)} passes")
        print(f"  NOTE: this truth uses the SAME face feature the grouping "
              f"uses, so a grouping threshold near {args.truth_threshold} "
              f"flatters itself. Cross-check with --kind known.")

    y, m, d = int(args.day[:4]), int(args.day[4:6]), int(args.day[6:])
    day = date(y, m, d)
    rows = []
    for mode in ("body", "face", "face_body"):
        labels, stats = run_gallery(passes, mode, day, settings.reid_model)
        prec, rec, tp, fp = pair_scores(labels[usable], truth)
        n_groups = len(set(labels.tolist()))
        err = (n_groups - n_truth) / n_truth * 100 if n_truth else float("nan")
        rows.append({"mode": mode, "precision": prec, "recall": rec,
                     "pairs_tp": tp, "pairs_fp": fp, "groups": n_groups,
                     "groups_on_usable": len(set(labels[usable].tolist())),
                     "truth_people": n_truth, **stats})
        print(f"\n  {mode:10s} pair precision {prec * 100:5.1f}%  "
              f"recall {rec * 100:5.1f}%   "
              f"pseudo-people {n_groups:4d} over all {len(passes)} passes")
        print(f"             {len(set(labels[usable].tolist())):4d} over the "
              f"{len(usable)} truth-covered passes, against {n_truth} true "
              f"people  ->  count error "
              f"{(len(set(labels[usable].tolist())) - n_truth) / n_truth * 100:+.0f}%")
        print(f"             linked by face {stats['by_face']}, "
              f"by body {stats['by_body']}, new {stats['created']}")

    if args.sweep:
        print(f"\n  {'face thr':>9s} {'body thr':>9s} {'prec':>7s} {'recall':>7s} "
              f"{'groups':>7s} {'count err':>10s}")
        of, ob = settings.pseudo_face_threshold, settings.pseudo_body_threshold
        faces = ((0.30, 0.33, 0.35, 0.38, 0.40, 0.43, 0.45, 0.48)
                 if args.wide_sweep else (0.30, 0.35, 0.40))
        bodies = ((0.70, 0.75, 0.80, 0.85, 0.90, 1.01)
                  if args.wide_sweep else (0.70, 0.75, 0.80))
        try:
            for ft in faces:
                for bt in bodies:
                    settings.pseudo_face_threshold = ft
                    settings.pseudo_body_threshold = bt
                    labels, _st = run_gallery(passes, "face_body", day,
                                              settings.reid_model)
                    prec, rec, _tp, _fp = pair_scores(labels[usable], truth)
                    g = len(set(labels[usable].tolist()))
                    rows.append({"mode": "face_body", "face_threshold": ft,
                                 "body_threshold": bt, "precision": prec,
                                 "recall": rec, "groups_on_usable": g,
                                 "truth_people": n_truth})
                    print(f"  {ft:9.2f} {bt:9.2f} {prec * 100:6.1f}% "
                          f"{rec * 100:6.1f}% {g:7d} "
                          f"{(g - n_truth) / n_truth * 100:+9.0f}%")
        finally:
            settings.pseudo_face_threshold, settings.pseudo_body_threshold = of, ob

    if args.json:
        out = ROOT / args.json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"day": args.day, "passes": len(passes), "usable_faces": len(usable),
             "truth_people": n_truth, "gates": not args.no_gates,
             "reid_model": settings.reid_model, "results": rows}, indent=2))
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
