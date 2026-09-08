#!/usr/bin/env python3
"""Derive the grouping and re-match thresholds from six days of production.

    python bench/group_calibrate.py --persons <tree> \
        --calibrate 20260902,20260903,20260904 \
        --validate  20260905,20260906,20260907

THE SPLIT IS STRATIFIED BY DAY OF WEEK, AND THAT WAS DECIDED FIRST
------------------------------------------------------------------
This corridor's weekend is a different population, not a quieter version of the
same one: the face path names 11.7-13.3% of passes on a weekday and 2.4-5.1% on
a Saturday or Sunday, because almost everybody through the door then is a
visitor. A split that put both weekend days on one side would be calibrating on
one population and validating on another, and any threshold that failed to
transfer would be telling you about the calendar rather than about the rule.

So each side gets two weekdays and one weekend day:

    calibrate  Wed 09-02, Fri 09-04, Sat 09-05
    validate   Thu 09-03, Sun 09-06, Mon 09-07

That was chosen from the day-of-week table and the naming rates BEFORE any
grouping result was computed, which is the only point at which choosing a split
is legitimate.

WHY A SPLIT, AND WHY IT IS NOT NEGOTIABLE
-----------------------------------------
A threshold picked on the data it is then reported against is not a
measurement, it is a description of that data. The deployed
`pseudo_face_threshold = 0.35` came from the research corpus and, swept on one
production day, "wanted" to be 0.40 - but that sweep chose AND reported on the
same 2178 passes, so the improvement it showed was partly the grid finding
noise.

So: the grid is searched on the CALIBRATION days only. The chosen point is then
run once on the VALIDATION days and that number is the one quoted. If the two
disagree, the honest answer is that the threshold does not transfer, and it is
reported as such rather than re-tuned until it does.

TWO TRUTHS, AND ONE OF THEM IS NOT CIRCULAR
-------------------------------------------
* `known/` passes carry the identity the FACE PATH assigned them at the time.
  That label owes nothing to the grouping rule, so precision and recall against
  it are honest. It is the smaller corpus (9-43 people a day) and it is biased
  toward people the face path can already see - but it cannot flatter itself.

* `unknown/` passes have no label. Truth is the research's construction:
  connected components of "face similarity >= 0.45" among passes with a usable
  face. This IS circular with respect to the face threshold - a grouping
  threshold approaching 0.45 will look better than it is - which is exactly why
  the known corpus is measured beside it and why the recommendation follows
  whichever of the two is more conservative.

WHAT IS MEASURED
----------------
1. GROUPING - can the passes of one unregistered person be collected? Reported
   as pair precision/recall and as the error in the DISTINCT-PERSON COUNT,
   which is the number an operator actually wants.
2. NAMING - the whole-pass face template against the enrolment gallery. On
   `known/` this gives real precision (does it reach the same person the live
   path did); on `unknown/` it gives the recovery count, which is the
   attendance gain and has no label.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from pseudo_group_eval import (Features, pair_scores, run_gallery,  # noqa: E402
                               tracklets, truth_clusters)
from app.services.reid_worker import slug  # noqa: E402


# --------------------------------------------------------------------------- #
#  features, cached per day
# --------------------------------------------------------------------------- #
def day_features(root: Path, day: str, kind: str, cache_dir: Path,
                 feats, max_crops: int) -> list[dict]:
    import cv2
    cache = cache_dir / f"feat_{kind}_{day}.npz"
    passes = tracklets(root, day, kind)
    if not passes:
        return []
    if cache.is_file():
        z = np.load(cache, allow_pickle=True)
        by = {t["name"] + "|" + str(t["track"]): t for t in passes}
        out = []
        for i, nm in enumerate(z["name"]):
            t = by.get(str(nm))
            if t is None:
                continue
            t["body"] = z["body"][i]
            f = z["face"][i]
            t["face"] = None if not np.any(f) else f
            t["ipd"] = float(z["ipd"][i])
            t["day"] = day
            out.append(t)
        return out

    kept = []
    for n, t in enumerate(passes, 1):
        imgs = [cv2.imread(str(c)) for c in t["crops"][:max_crops]]
        imgs = [i for i in imgs if i is not None]
        if not imgs:
            continue
        t["body"] = feats.body(imgs)
        t["face"], t["ipd"], _n = feats.face(imgs, gates=True)
        t["day"] = day
        kept.append(t)
        if n % 50 == 0 or n == len(passes):
            print(f"      {kind}/{day} {n}/{len(passes)}", end="\r", flush=True)
    print()
    cache.parent.mkdir(parents=True, exist_ok=True)
    fd = next((len(t["face"]) for t in kept if t["face"] is not None), 512)
    np.savez_compressed(
        cache,
        name=np.array([t["name"] + "|" + str(t["track"]) for t in kept]),
        body=np.stack([t["body"] for t in kept]),
        face=np.stack([t["face"] if t["face"] is not None
                       else np.zeros(fd, np.float32) for t in kept]),
        ipd=np.array([t["ipd"] for t in kept], np.float32))
    return kept


# --------------------------------------------------------------------------- #
#  one day, one setting
# --------------------------------------------------------------------------- #
def score_day(passes, kind, settings, face_thr, body_thr, truth_thr=0.45):
    """Group one day at one setting; return the metrics dict."""
    of, ob = settings.pseudo_face_threshold, settings.pseudo_body_threshold
    settings.pseudo_face_threshold, settings.pseudo_body_threshold = face_thr, body_thr
    try:
        y, m, d = int(passes[0]["day"][:4]), int(passes[0]["day"][4:6]), \
            int(passes[0]["day"][6:])
        labels, stats = run_gallery(passes, "face_body", date(y, m, d),
                                    settings.reid_model)
    finally:
        settings.pseudo_face_threshold, settings.pseudo_body_threshold = of, ob

    if kind == "known":
        idx = list(range(len(passes)))
        lab: dict = {}
        truth = np.array([lab.setdefault(t["name"], len(lab)) for t in passes])
    else:
        idx = [i for i, t in enumerate(passes)
               if t["face"] is not None and t["ipd"] >= settings.pseudo_face_ipd_min]
        if len(idx) < 2:
            return None
        truth = truth_clusters(np.stack([passes[i]["face"] for i in idx]), truth_thr)

    prec, rec, tp, fp = pair_scores(labels[idx], truth)
    n_truth = len(set(truth.tolist()))
    n_pred = len(set(labels[idx].tolist()))
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"day": passes[0]["day"], "kind": kind, "face": face_thr,
            "body": body_thr, "precision": prec, "recall": rec, "f1": f1,
            "truth_people": n_truth, "pred_people": n_pred,
            "count_err": (n_pred - n_truth) / max(n_truth, 1) * 100,
            "groups_all": len(set(labels.tolist())), "passes": len(passes),
            "covered": len(idx)}


def run_gallery_multiday(stream, model):
    """One gallery across several days, chronologically.

    Body templates expire at each business-date boundary on their own - the
    gallery drops them when `body_date` no longer matches - so this exercises
    exactly the rule the deployment runs, rather than a special cross-day mode.
    """
    from app.db.models import ReidPass
    from app.db.session import session_scope
    from app.services.pseudo_gallery import PseudoGallery
    from datetime import datetime, timezone

    labels = np.full(len(stream), -1, np.int64)
    with session_scope() as s:
        g = PseudoGallery()
        for i, t in enumerate(stream):
            d = t["day"]
            bday = date(int(d[:4]), int(d[4:6]), int(d[6:]))
            when = datetime(bday.year, bday.month, bday.day, tzinfo=timezone.utc)
            row = ReidPass(camera_id=1 if t["camera"] == "Entrance" else 2,
                           camera_name=t["camera"], track_id=i, first_seen=when,
                           last_seen=when, business_date=bday,
                           direction="UNKNOWN", dim=len(t["body"]),
                           model_name=model)
            s.add(row)
            s.flush()
            p = g.place(s, row, face=t["face"], body=t["body"],
                        face_ipd=t["ipd"], body_model=model)
            labels[i] = p.id if p is not None else -(i + 1)
        stats = g.stats()
        s.rollback()
    return labels, stats


def agg(rows):
    """Average over days, weighting each day equally - a busy day must not
    decide a threshold for the quiet ones."""
    rows = [r for r in rows if r]
    if not rows:
        return None
    return {k: float(np.mean([r[k] for r in rows]))
            for k in ("precision", "recall", "f1", "count_err")} | {
        "truth_people": sum(r["truth_people"] for r in rows),
        "pred_people": sum(r["pred_people"] for r in rows),
        "passes": sum(r["passes"] for r in rows),
        "covered": sum(r["covered"] for r in rows),
        "groups_all": sum(r["groups_all"] for r in rows)}


# --------------------------------------------------------------------------- #
#  naming: the whole-pass template against the enrolment gallery
# --------------------------------------------------------------------------- #
def day_people(db: Path) -> dict[str, set]:
    """Which employees production recorded on each day.

    An unknown pass that the template names is unlabelled, so it cannot be
    scored directly. It CAN be corroborated: if production saw that person
    elsewhere on the same day, the template found them in a pass the live vote
    missed - which is what a recovery looks like. If the person appears nowhere
    in that whole day, it has the shape of a false accept and is counted apart.
    """
    import sqlite3
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out: dict[str, set] = {}
    for d, e in con.execute(
            "select business_date, employee_id from recognition_event "
            "where employee_id is not null and voided_at is null"):
        out.setdefault(str(d).replace("-", ""), set()).add(int(e))
    con.close()
    return out


def naming(passes, kind, gallery, thr, margin, min_ipd, seen=None):
    """(agreement on labelled passes, names produced on unlabelled ones)."""
    agree = wrong = called = 0
    corrob = novel = 0
    for t in passes:
        if t["face"] is None or t["ipd"] < min_ipd:
            continue
        m = gallery.match(t["face"], thr, margin)
        if m.employee_id is None:
            continue
        called += 1
        if kind == "known":
            # `known/<Person_Name>/` is the face path's own answer. The folder
            # name went through `slug()`, so the gallery's name has to as well -
            # comparing "Firstname Lastname" with "Firstname_Lastname" never
            # matches, and a silent always-false makes precision read 0% on a
            # rule that is actually right 98% of the time.
            if slug(gallery.name(m.employee_id)) == t["name"]:
                agree += 1
            else:
                wrong += 1
        elif seen is not None:
            if m.employee_id in seen.get(t["day"], set()):
                corrob += 1
            else:
                novel += 1
    return {"called": called, "agree": agree, "wrong": wrong,
            "corroborated": corrob, "novel": novel}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--persons", required=True)
    ap.add_argument("--calibrate", required=True, help="comma-separated YYYYMMDD")
    ap.add_argument("--validate", required=True)
    ap.add_argument("--cache-dir", default="data/bench/feat_cache")
    ap.add_argument("--max-crops", type=int, default=10)
    ap.add_argument("--db", default=None,
                    help="production DB, to corroborate recovered names")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from app.config import settings
    from app.services.enrollment import load_gallery

    root = Path(args.persons)
    cal_days = [d.strip() for d in args.calibrate.split(",") if d.strip()]
    val_days = [d.strip() for d in args.validate.split(",") if d.strip()]
    cache = ROOT / args.cache_dir
    gallery = load_gallery()
    print(f"  gallery {len(gallery)} embeddings / {gallery.n_people} people")
    print(f"  reid    {settings.reid_model}")
    print(f"  calibrate on {cal_days}\n  validate  on {val_days}\n")

    feats = None
    data: dict = {}
    for day in cal_days + val_days:
        for kind in ("unknown", "known"):
            c = cache / f"feat_{kind}_{day}.npz"
            if not c.is_file() and feats is None:
                feats = Features()
            data[(kind, day)] = day_features(root, day, kind, cache, feats,
                                             args.max_crops)
            n = len(data[(kind, day)])
            good = sum(1 for t in data[(kind, day)]
                       if t["face"] is not None
                       and t["ipd"] >= settings.pseudo_face_ipd_min)
            print(f"    {kind:8s} {day}  {n:5d} passes, {good:4d} with a usable "
                  f"face ({good / max(n, 1) * 100:4.1f}%)")

    # ---- grid search, CALIBRATION ONLY, on the NON-CIRCULAR corpus -----
    #
    # THE GRID IS SEARCHED ON `known/`, NOT ON `unknown/`, AND THAT IS THE
    # WHOLE METHODOLOGICAL POINT. The unknown-corpus truth is connected
    # components of face similarity at 0.45, so a grouping rule that uses the
    # same feature is scored against a relabelling of its own output: at a
    # grouping threshold of 0.45 its "precision" is 100% by construction, and
    # every step toward 0.45 looks like an improvement. Swept that way, the
    # grid recommends raising the face threshold. Swept on real identities it
    # recommends LOWERING it. Both cannot be right, and only one of them is
    # measuring anything.
    #
    # `known/` labels come from the face path's decision at the time and owe
    # nothing to the grouping rule. Its bias is different and it is stated
    # rather than hidden: those people have good faces and pass several times a
    # day, so the recall attainable there is optimistic for a one-visit
    # visitor. It cannot, however, flatter itself.
    grid_face = (0.20, 0.24, 0.26, 0.28, 0.30, 0.32, 0.35, 0.40)
    grid_body = (0.60, 0.65, 0.70, 0.75, 0.80, 1.01)   # 1.01 = body disabled
    print(f"\n  {'=' * 78}\n  GRID on the calibration days, KNOWN corpus "
          f"(real identities - NOT circular)\n  {'=' * 78}")
    print(f"  {'face':>5s} {'body':>5s} {'prec':>7s} {'recall':>7s} {'F1':>7s} "
          f"{'count err':>10s}")
    results = []
    for ft in grid_face:
        for bt in grid_body:
            rows = [score_day(data[("known", d)], "known", settings, ft, bt)
                    for d in cal_days if data[("known", d)]]
            a = agg(rows)
            if not a:
                continue
            a |= {"face": ft, "body": bt, "split": "calibrate", "corpus": "known"}
            results.append(a)
            print(f"  {ft:5.2f} {bt:5.2f} {a['precision'] * 100:6.1f}% "
                  f"{a['recall'] * 100:6.1f}% {a['f1'] * 100:6.1f}% "
                  f"{a['count_err']:+9.0f}%")

    best_f1 = max(results, key=lambda r: r["f1"])
    # The count-optimal point is reported but NOT recommended on its own: a
    # rule can land on the right total by cancelling merges against splits.
    best_cnt = min(results, key=lambda r: abs(r["count_err"]))
    print(f"\n  best pair-F1        : face {best_f1['face']:.2f} body "
          f"{best_f1['body']:.2f}   P {best_f1['precision'] * 100:.1f} "
          f"R {best_f1['recall'] * 100:.1f} F1 {best_f1['f1'] * 100:.1f}%  "
          f"count err {best_f1['count_err']:+.0f}%")
    print(f"  best count accuracy : face {best_cnt['face']:.2f} body "
          f"{best_cnt['body']:.2f}   F1 {best_cnt['f1'] * 100:.1f}%  "
          f"count err {best_cnt['count_err']:+.0f}%")

    # ---- the circular corpus, for the record ----------------------------
    print(f"\n  {'=' * 78}\n  the same points on the UNKNOWN corpus "
          f"(face-cluster truth - CIRCULAR, do not select on it)\n  {'=' * 78}")
    print(f"  {'face':>5s} {'body':>5s} {'prec':>7s} {'recall':>7s} {'F1':>7s} "
          f"{'count err':>10s}")
    known_rows = []
    for ft, bt in sorted({(0.35, 0.75), (best_f1["face"], best_f1["body"]),
                          (best_cnt["face"], best_cnt["body"]),
                          (0.28, 0.75), (0.40, 0.80)}):
        rows = [score_day(data[("unknown", d)], "unknown", settings, ft, bt)
                for d in cal_days if data[("unknown", d)]]
        a = agg(rows)
        if not a:
            continue
        a |= {"face": ft, "body": bt, "split": "calibrate", "corpus": "unknown"}
        known_rows.append(a)
        print(f"  {ft:5.2f} {bt:5.2f} {a['precision'] * 100:6.1f}% "
              f"{a['recall'] * 100:6.1f}% {a['f1'] * 100:6.1f}% "
              f"{a['count_err']:+9.0f}%")

    # ---- VALIDATION, run once ------------------------------------------
    print(f"\n  {'=' * 74}\n  VALIDATION - held out, each setting run ONCE\n"
          f"  {'=' * 74}")
    chosen = [("deployed", 0.35, 0.75),
              ("best F1", best_f1["face"], best_f1["body"]),
              ("best count", best_cnt["face"], best_cnt["body"])]
    val = []
    for label, ft, bt in chosen:
        for kind in ("unknown", "known"):
            rows = [score_day(data[(kind, d)], kind, settings, ft, bt)
                    for d in val_days if data[(kind, d)]]
            a = agg(rows)
            if not a:
                continue
            a |= {"label": label, "face": ft, "body": bt, "corpus": kind,
                  "split": "validate"}
            val.append(a)
            print(f"  {label:11s} face {ft:.2f} body {bt:.2f}  {kind:8s}  "
                  f"prec {a['precision'] * 100:5.1f}%  rec {a['recall'] * 100:5.1f}%  "
                  f"F1 {a['f1'] * 100:5.1f}%  count {a['count_err']:+5.0f}%  "
                  f"({a['pred_people']} groups vs {a['truth_people']} people)")

    # ---- what grouping is WORTH, on the validation days -----------------
    tot_pass = sum(len(data[("unknown", d)]) for d in val_days)
    print(f"\n  {'=' * 74}\n  WHAT AN OPERATOR SEES, validation days\n  {'=' * 74}")
    for label, ft, bt in chosen:
        rows = [score_day(data[("unknown", d)], "unknown", settings, ft, bt)
                for d in val_days if data[("unknown", d)]]
        a = agg(rows)
        print(f"  {label:11s}: {tot_pass} unknown passes -> "
              f"{a['groups_all']} pseudo-people "
              f"({(1 - a['groups_all'] / tot_pass) * 100:.0f}% fewer rows to review)")

    # ---- CROSS-DAY: one gallery over the whole week ---------------------
    #
    # Everything above scores each day on its own, because that is how the
    # truth is built. Production does not restart at midnight: a face template
    # survives, so somebody seen on Tuesday and Thursday should be ONE
    # pseudo-person. This measures that, and it is the number that answers
    # "how many distinct people came through this week".
    print(f"\n  {'=' * 74}\n  CROSS-DAY: one gallery over all "
          f"{len(cal_days + val_days)} days, in order\n  {'=' * 74}")
    all_days = sorted(cal_days + val_days)
    stream = [t for d in all_days for t in data[("unknown", d)]]
    stream.sort(key=lambda t: (t["day"], t["stamp"]))
    cross = []
    for label, ft, bt in chosen:
        of, ob = settings.pseudo_face_threshold, settings.pseudo_body_threshold
        settings.pseudo_face_threshold, settings.pseudo_body_threshold = ft, bt
        try:
            labels, stats = run_gallery_multiday(stream, settings.reid_model)
        finally:
            settings.pseudo_face_threshold = of
            settings.pseudo_body_threshold = ob
        idx = [i for i, t in enumerate(stream)
               if t["face"] is not None and t["ipd"] >= settings.pseudo_face_ipd_min]
        truth = truth_clusters(np.stack([stream[i]["face"] for i in idx]), 0.45)
        prec, rec, _tp, _fp = pair_scores(labels[idx], truth)
        n_truth = len(set(truth.tolist()))
        n_pred = len(set(labels[idx].tolist()))
        per_day_sum = sum(len(set(run_gallery(
            data[("unknown", d)], "face_body",
            date(int(d[:4]), int(d[4:6]), int(d[6:])),
            settings.reid_model)[0].tolist())) for d in all_days) \
            if label == "deployed" else None
        cross.append({"label": label, "face": ft, "body": bt,
                      "passes": len(stream), "groups_all": len(set(labels.tolist())),
                      "precision": prec, "recall": rec,
                      "truth_people": n_truth, "pred_people": n_pred,
                      "count_err": (n_pred - n_truth) / max(n_truth, 1) * 100})
        print(f"  {label:11s} face {ft:.2f} body {bt:.2f}: "
              f"{len(stream)} passes -> {len(set(labels.tolist()))} pseudo-people; "
              f"on the {len(idx)} truth-covered passes {n_pred} groups vs "
              f"{n_truth} people ({(n_pred - n_truth) / max(n_truth, 1) * 100:+.0f}%), "
              f"pair prec {prec * 100:.1f}% rec {rec * 100:.1f}%")

    # ---- naming ---------------------------------------------------------
    print(f"\n  {'=' * 74}\n  NAMING: whole-pass template vs the enrolment "
          f"gallery\n  {'=' * 74}")
    seen = day_people(Path(args.db)) if args.db else None
    print(f"  {'thr':>5s} {'agree':>7s} {'wrong':>7s} {'precision':>10s} "
          f"{'named':>7s} {'corrob':>7s} {'novel':>7s} {'corrob%':>8s}")
    name_rows = []
    for thr in (0.21, 0.24, 0.27, 0.30, 0.33, 0.36, 0.40):
        k = {"called": 0, "agree": 0, "wrong": 0}
        u = {"called": 0, "corroborated": 0, "novel": 0}
        for d in cal_days + val_days:
            r = naming(data[("known", d)], "known", gallery, thr,
                       settings.tracklet_face_margin, settings.pseudo_face_ipd_min)
            for key in k:
                k[key] += r[key]
            r2 = naming(data[("unknown", d)], "unknown", gallery, thr,
                        settings.tracklet_face_margin,
                        settings.pseudo_face_ipd_min, seen)
            for key in u:
                u[key] += r2[key]
        p = k["agree"] / max(k["agree"] + k["wrong"], 1) * 100
        cor = u["corroborated"] / max(u["called"], 1) * 100
        name_rows.append({"threshold": thr, **k, "unknown_named": u["called"],
                          "corroborated": u["corroborated"], "novel": u["novel"],
                          "precision": p, "corroborated_pct": cor})
        print(f"  {thr:5.2f} {k['agree']:7d} {k['wrong']:7d} {p:9.1f}% "
              f"{u['called']:7d} {u['corroborated']:7d} {u['novel']:7d} "
              f"{cor:7.1f}%")

    if args.json:
        out = ROOT / args.json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "calibrate": cal_days, "validate": val_days,
            "grid": results, "known_crosscheck": known_rows,
            "validation": val, "naming": name_rows, "cross_day": cross,
            "best_f1": best_f1, "best_count": best_cnt}, indent=2, default=float))
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
