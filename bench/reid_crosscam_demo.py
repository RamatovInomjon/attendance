#!/usr/bin/env python3
"""Run the cross-camera ReID rule over a pass corpus and show what it found.

    python bench/reid_crosscam_demo.py --dir data/reid_passes --out data/reid_pairs

Every pass is treated as UNKNOWN - the face labels are hidden from the matcher
and used only afterwards, to say whether each pair it claimed was right. That is
the honest test: in production this runs on people the face path could not name,
so it must not be handed the answer.

Writes one side-by-side JPEG per claimed pair, Entrance on the left and Exit on
the right, so a human can check the verdict rather than take the number on
trust.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="data/reid_passes")
    ap.add_argument("--out", default="data/reid_pairs")
    ap.add_argument("--keep", type=int, default=10, help="crops per pass")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--margin", type=float, default=None)
    args = ap.parse_args()

    import cv2
    from app.config import settings
    from app.core.reid import PersonReID, aggregate, sharpness_of
    from app.services.reid_worker import _spread

    thr = args.threshold if args.threshold is not None else settings.reid_match_threshold
    mg = args.margin if args.margin is not None else settings.reid_match_margin
    reid = PersonReID(str(settings.model_path(settings.reid_model)))
    print(f"  model {reid.model_name}   threshold {thr:.3f}  margin {mg:.3f}\n")

    root, out = ROOT / args.dir, ROOT / args.out
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    passes = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        mf = d / "_pass.json"
        if not mf.is_file():
            continue
        meta = json.loads(mf.read_text())
        files = sorted(d.glob("*.jpg"))
        if len(files) < settings.reid_min_crops:
            continue
        pick = [files[i] for i in _spread(list(range(len(files))), args.keep)]
        imgs, keep = [], []
        for f in pick:
            im = cv2.imread(str(f))
            if im is not None:
                imgs.append(im)
                keep.append(f)
        if len(imgs) < settings.reid_min_crops:
            continue
        embs = reid.embed(imgs)
        scores = [float(f.stem.rsplit("_s", 1)[-1]) if "_s" in f.stem else 1.0
                  for f in keep]
        passes.append({
            "folder": d.name, "camera": meta.get("camera", ""),
            "truth": meta.get("employee_id"), "name": meta.get("name"),
            "feat": aggregate(embs, scores, [sharpness_of(i) for i in imgs]),
            "files": keep,
        })

    gal = [p for p in passes if p["camera"] == "Entrance"]
    qry = [p for p in passes if p["camera"] == "Exit"]
    print(f"  {len(passes)} passes: Entrance {len(gal)} -> Exit {len(qry)}\n")
    G = np.stack([p["feat"] for p in gal])

    claims = []
    used = set()
    for q in qry:
        sims = G @ q["feat"]
        order = np.argsort(-sims)
        # One entrance may only be claimed once, as in the live matcher.
        cand = [i for i in order if gal[i]["folder"] not in used]
        if len(cand) < 2:
            continue
        top, second = float(sims[cand[0]]), float(sims[cand[1]])
        if top < thr or (top - second) < mg:
            continue
        g = gal[cand[0]]
        used.add(g["folder"])
        # Verdict comes from the FACE labels, which the matcher never saw.
        if q["truth"] is None or g["truth"] is None:
            verdict = "unverifiable"
        else:
            verdict = "CORRECT" if q["truth"] == g["truth"] else "WRONG"
        claims.append((top, top - second, g, q, verdict))

    claims.sort(key=lambda c: -c[0])
    ok = sum(1 for c in claims if c[4] == "CORRECT")
    bad = sum(1 for c in claims if c[4] == "WRONG")
    unk = sum(1 for c in claims if c[4] == "unverifiable")
    print(f"  {len(claims)} pairs claimed: {ok} correct, {bad} wrong, "
          f"{unk} unverifiable (neither side face-labelled)\n")

    print(f"  {'score':>6s} {'margin':>7s}  {'verdict':13s} who")
    for i, (top, m, g, q, verdict) in enumerate(claims, 1):
        who = (g["name"] or q["name"] or "?") if verdict != "WRONG" else \
              f"{g['name']} =/= {q['name']}"
        print(f"  {top:6.3f} {m:7.3f}  {verdict:13s} {who}")

        # Side by side: entrance left, exit right, so the pair can be judged.
        def strip(p, n=3):
            ims = [cv2.resize(cv2.imread(str(f)), (140, 340))
                   for f in p["files"][:n]]
            return cv2.hconcat(ims) if ims else None
        a, b = strip(g), strip(q)
        if a is None or b is None:
            continue
        gap = np.full((340, 12, 3), 255, np.uint8)
        canvas = cv2.hconcat([a, gap, b])
        label = f"{verdict}  s={top:.3f} m={m:.3f}   ENTRANCE | EXIT"
        bar = np.full((28, canvas.shape[1], 3), 30, np.uint8)
        cv2.putText(bar, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / f"{i:02d}_{verdict}_{top:.3f}.jpg"),
                    cv2.vconcat([bar, canvas]), [cv2.IMWRITE_JPEG_QUALITY, 92])

    (out / "_pairs.json").write_text(json.dumps([{
        "score": round(t, 4), "margin": round(m, 4), "verdict": v,
        "entrance": g["folder"], "exit": q["folder"],
        "entrance_name": g["name"], "exit_name": q["name"],
    } for t, m, g, q, v in claims], indent=2, ensure_ascii=False))
    print(f"\n  side-by-side images in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
