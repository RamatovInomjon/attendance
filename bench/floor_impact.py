#!/usr/bin/env python3
"""What the corridor-crop floor actually changed, pass by pass, with pictures.

    python bench/floor_impact.py --out data/floor_impact
    python bench/floor_impact.py --out /tmp/x --limit 200

`augment_live_floor` was chosen from a rate on a proxy population: faces the
pipeline had named nobody, which is not the same as faces that are nobody. This
script answers the narrower, checkable question instead - of the passes the
system ACTUALLY named, which ones does the floor now refuse? - and puts the
face of every one of them on a contact sheet, because whether a refusal was a
false accept removed or a genuine recognition lost is a judgement only a person
who knows these faces can make.

WHAT IT MEASURES, AND THE ONE APPROXIMATION
-------------------------------------------
Each `data/debug` capture is one frame of one pass: the highest-scoring frame
for the identity that pass committed. Identity is decided by consensus over the
whole pass, so this is not a full re-run. It is a sound test of REFUSAL,
though: if the best frame no longer clears the bar through any row, no weaker
frame of that pass does either.

Three galleries are scored:

  * WITHOUT floors - the gallery as it behaved before 04-09, every row at the
    global threshold. This is the state that produced the stored decisions.
  * WITH floors - as it behaves now: corridor crops at `augment_live_floor`.
  * ENROLMENT ONLY - the studio photographs alone, for reference.

Every image it writes is a face already stored under `data/debug`. Nothing
leaves the machine and no new biometric data is created.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _decide(per_row, owner, people, slot, floors, thr, margin, use_floors):
    """Per-person max then the two acceptance rules, mirroring Gallery."""
    s = per_row - floors + thr if use_floors else per_row
    out = np.full(len(people), -4.0, np.float32)
    np.maximum.at(out, slot, s)
    if len(people) == 1:
        return (int(people[0]), float(out[0])) if out[0] >= thr else (None, float(out[0]))
    idx = np.argpartition(out, -2)[-2:]
    if out[idx[0]] > out[idx[1]]:
        idx = idx[::-1]
    second, top = int(idx[0]), int(idx[1])
    if out[top] >= thr and (out[top] - out[second]) >= margin:
        return int(people[top]), float(out[top])
    return None, float(out[top])


def _sheet(items, path, cols=8, cell=132, pad=22):
    """A labelled contact sheet, so a person can judge a whole class at once."""
    import cv2
    if not items:
        return 0
    rows = (len(items) + cols - 1) // cols
    H, W = rows * (cell + pad), cols * cell
    canvas = np.full((H, W, 3), 245, np.uint8)
    for k, (img_path, label) in enumerate(items):
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        im = cv2.resize(im, (cell, cell), interpolation=cv2.INTER_AREA)
        r, c = divmod(k, cols)
        y, x = r * (cell + pad), c * cell
        canvas[y:y + cell, x:x + cell] = im
        for i, line in enumerate(label.split("\n")[:2]):
            cv2.putText(canvas, line[:22], (x + 2, y + cell + 10 + i * 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.29, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)
    return len(items)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Inside data/debug, beside the captures it is made from, so the sheet and
    # the faces on it live together. The underscore keeps it out of the way of
    # the per-person folders that `augment.scan` and `crop_path` walk.
    ap.add_argument("--out", default="data/debug/_floor_impact")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import cv2
    from app.config import settings
    from app.core.geometry import to_normalized_chw
    from app.core.onnx_env import preload_cuda_libs
    from app.services.augment import TAG, _gallery_rows

    thr = settings.threshold_for(settings.recognizer_model)
    margin = settings.second_best_margin
    floor_cfg = max(thr, float(settings.augment_live_floor))
    M, owner, ids, tags, names, floors = _gallery_rows()
    if not len(M):
        print("  the gallery is empty"); return 1
    live = np.array([t.startswith(TAG) for t in tags])
    people = np.unique(owner)
    slot = np.searchsorted(people, owner).astype(np.int64)
    print(f"  gallery    {len(M)} rows: {(~live).sum()} enrolment, {live.sum()} corridor")
    print(f"  threshold  {thr:.3f}   corridor floor {floor_cfg:.3f}   margin {margin:.3f}")

    preload_cuda_libs()
    from app.core.recognizer import FaceRecognizer
    rec = FaceRecognizer(settings.model_path(settings.recognizer_model),
                         batch_size=settings.embed_batch)

    files = sorted(glob.glob(str(settings.debug_dir / "**" / "*.json"), recursive=True))
    if args.limit:
        files = files[-args.limit:]
    caps, meta = [], []
    for jf in files:
        aligned = jf[:-5] + "_aligned.jpg"
        if not os.path.exists(aligned):
            continue
        try:
            m = json.loads(open(jf).read())
        except Exception:
            continue
        if m.get("employee_id") is None:
            continue
        im = cv2.imread(aligned)
        if im is None:
            continue
        if im.shape[:2] != (112, 112):
            im = cv2.resize(im, (112, 112), interpolation=cv2.INTER_AREA)
        caps.append(to_normalized_chw(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
        meta.append({"emp": int(m["employee_id"]), "name": m.get("name", ""),
                     "score": float(m.get("score", 0.0)), "cam": m.get("camera", ""),
                     "stem": Path(jf).stem, "face": jf[:-5] + "_face.jpg",
                     "aligned": aligned})
    if not caps:
        print("  no captures under", settings.debug_dir); return 1
    Q = rec.embed(np.stack(caps))
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12
    S = Q @ M.T
    print(f"  captures   {len(Q)} recognised passes\n")

    kept, lost, changed = [], [], []
    enrol_only_lost = 0
    for i, m in enumerate(meta):
        before, sb = _decide(S[i], owner, people, slot, floors, thr, margin, False)
        after, sa = _decide(S[i], owner, people, slot, floors, thr, margin, True)
        eo = np.where(~live, S[i], -2.0)
        m["before"], m["after"] = before, after
        m["s_before"], m["s_after"] = sb, sa
        m["enrol_best"] = float(eo[eo > -1.5].max()) if (~live).any() else -1.0
        if before is not None and after is None:
            lost.append(m)
            if m["enrol_best"] >= thr:
                enrol_only_lost += 1
        elif before is not None and after is not None and before != after:
            changed.append(m)
        elif after is not None:
            kept.append(m)

    named = sum(1 for m in meta if m["before"] is not None)
    print(f"  named before the floor : {named} of {len(meta)}")
    print(f"  still named after      : {len(kept)}  ({100*len(kept)/max(named,1):.1f}%)")
    print(f"  NO LONGER NAMED        : {len(lost)}  ({100*len(lost)/max(named,1):.1f}%)")
    print(f"  named as someone else  : {len(changed)}")
    print(f"\n  Of the {len(lost)} refused, {len(lost) - enrol_only_lost} rested on a")
    print(f"  corridor crop ALONE - no enrolment photo of that person reached")
    print(f"  {thr:.3f}. Those are the passes the floor is actually deciding.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lost.sort(key=lambda m: -m["s_before"])
    sheets = [
        ("refused_by_the_floor", lost,
         "every pass the corridor floor now refuses - judge each face"),
        ("identity_changed", changed, "the floor moved these to a different person"),
    ]
    for tag, items, why in sheets:
        rows = [(m["face"] if os.path.exists(m["face"]) else m["aligned"],
                 f"{m['name'][:20]}\n{m['s_before']:.3f}->{m['s_after']:.3f}")
                for m in items]
        n = _sheet(rows, out / f"{tag}.jpg")
        if n:
            print(f"\n  {out / (tag + '.jpg')}  ({n} faces) - {why}")

    with open(out / "refused.csv", "w") as f:
        f.write("stem,name,camera,score_before,score_after,best_enrolment_photo\n")
        for m in lost:
            f.write(f"{m['stem']},{m['name']},{m['cam']},{m['s_before']:.4f},"
                    f"{m['s_after']:.4f},{m['enrol_best']:.4f}\n")
    print(f"  {out / 'refused.csv'}  - the same list as data")
    print("\n  A refusal is not automatically a false accept removed, nor")
    print("  automatically a genuine recognition lost. Look at the faces.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
