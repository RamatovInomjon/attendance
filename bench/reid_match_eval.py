#!/usr/bin/env python3
"""Measure cross-camera ReID matching on real passes, with face labels as truth.

    python bench/reid_match_eval.py --dir data/reid_passes
    python bench/reid_match_eval.py --dir data/reid_passes --sweep

Every pass in the corpus already carries what the FACE path decided. Passes it
named give ground truth for free: two passes with the same `employee_id` on
different cameras are a genuine pair, two with different ids are an impostor
pair. So the body-ReID matching rule can be scored without labelling anything by
hand.

This measures the RULE, not the model - `bench/eval_reid.py` already checked the
model against its published figure. What is at stake here is whether the
threshold and margin carried over from a 45-person research test set survive
contact with this corridor. The face threshold taught that lesson the expensive
way: the deployed 0.18 turned out to sit BELOW the worst impostor actually seen.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="data/reid_passes")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-crops", type=int, default=8,
                    help="crops per pass, matching body_crop_max_per_pass")
    ap.add_argument("--sweep", action="store_true",
                    help="report the operating curve, not just the setting")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import cv2
    from app.config import settings
    from app.core.reid import PersonReID, aggregate, sharpness_of

    root = ROOT / args.dir
    if not root.is_dir():
        raise SystemExit(f"  no corpus at {root}")
    reid = PersonReID(args.model or str(settings.model_path(settings.reid_model)))
    print(f"  model {reid.model_name}  {reid.height}x{reid.width}  embed={reid.dim}")

    passes = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        meta_f = d / "_pass.json"
        if not meta_f.is_file():
            continue
        meta = json.loads(meta_f.read_text())
        crops = sorted(d.glob("*.jpg"))[: args.max_crops]
        if len(crops) < settings.reid_min_crops:
            continue
        imgs = [cv2.imread(str(c)) for c in crops]
        imgs = [i for i in imgs if i is not None]
        if not imgs:
            continue
        embs = reid.embed(imgs)
        scores = [float(c.stem.rsplit("_s", 1)[-1]) if "_s" in c.stem else 1.0
                  for c in crops[: len(imgs)]]
        feat = aggregate(embs, scores, [sharpness_of(i) for i in imgs])
        passes.append({"camera": meta.get("camera", ""),
                       "employee_id": meta.get("employee_id"),
                       "folder": d.name, "feat": feat, "crops": len(imgs)})

    labelled = [p for p in passes if p["employee_id"] is not None]
    cams = sorted({p["camera"] for p in passes})
    print(f"  {len(passes)} passes ({len(labelled)} face-labelled), cameras {cams}")
    if len(cams) < 2:
        raise SystemExit("  need two cameras to score cross-camera matching")

    gal_cam = next((c for c in cams if c.lower() == "entrance"), cams[0])
    qry_cam = next((c for c in cams if c.lower() == "exit"), cams[1])
    gallery = [p for p in labelled if p["camera"] == gal_cam]
    queries = [p for p in labelled if p["camera"] == qry_cam]
    print(f"  protocol gallery={gal_cam} ({len(gallery)}) -> "
          f"query={qry_cam} ({len(queries)})\n")
    if not gallery or not queries:
        raise SystemExit("  no labelled passes on one of the cameras")

    G = np.stack([p["feat"] for p in gallery])
    gid = np.array([p["employee_id"] for p in gallery])
    genuine, impostor, rows = [], [], []
    for q in queries:
        sims = G @ q["feat"]
        order = np.argsort(-sims)
        top, second = float(sims[order[0]]), (
            float(sims[order[1]]) if len(order) > 1 else -1.0)
        correct = bool(gid[order[0]] == q["employee_id"])
        # A query whose person is not in the gallery at all is an impostor -
        # the open-set case, and the common one here.
        present = bool((gid == q["employee_id"]).any())
        (genuine if present else impostor).append(top)
        rows.append({"folder": q["folder"], "top": top, "margin": top - second,
                     "correct": correct, "present": present})

    def report(thr, margin):
        tp = sum(1 for r in rows if r["present"] and r["correct"]
                 and r["top"] >= thr and r["margin"] >= margin)
        fp = sum(1 for r in rows if r["top"] >= thr and r["margin"] >= margin
                 and not (r["present"] and r["correct"]))
        claimed = tp + fp
        return tp, fp, claimed

    g, i = np.array(genuine), np.array(impostor)
    for name, a in (("genuine ", g), ("impostor", i)):
        if len(a):
            print(f"  {name} n={len(a):3d}  min={a.min():.3f} "
                  f"median={np.median(a):.3f} p90={np.percentile(a, 90):.3f} "
                  f"max={a.max():.3f}")
    if len(i):
        print(f"\n  worst impostor {i.max():.3f}"
              f"   configured threshold {settings.reid_match_threshold:.3f}"
              + ("   <-- ACCEPTS IMPOSTORS"
                 if settings.reid_match_threshold <= i.max() else "   ok"))

    thr, mg = settings.reid_match_threshold, settings.reid_match_margin
    tp, fp, claimed = report(thr, mg)
    print(f"\n  at threshold {thr:.3f} margin {mg:.3f}: "
          f"{tp} correct, {fp} wrong, {claimed} claimed of {len(rows)} queries")

    if args.sweep:
        print(f"\n  {'thr':>6s} {'margin':>7s} {'correct':>8s} {'wrong':>6s} "
              f"{'precision':>10s}")
        for t in (0.55, 0.60, 0.65, 0.6769, 0.70, 0.75, 0.80):
            for m in (0.0, 0.05):
                tp, fp, cl = report(t, m)
                pr = tp / cl if cl else float("nan")
                print(f"  {t:6.3f} {m:7.2f} {tp:8d} {fp:6d} {pr:10.2f}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "model": reid.model_name, "passes": len(passes),
            "labelled": len(labelled), "gallery": len(gallery),
            "queries": len(queries),
            "genuine": {"n": len(g), "median": float(np.median(g)) if len(g) else None,
                        "min": float(g.min()) if len(g) else None},
            "impostor": {"n": len(i), "median": float(np.median(i)) if len(i) else None,
                         "max": float(i.max()) if len(i) else None},
            "threshold": thr, "margin": mg,
        }, indent=2))
        print(f"\n  written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
