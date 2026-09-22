#!/usr/bin/env python3
"""Replay the SAME faces through the previous and the current recognizer.

WHY THIS EXISTS
---------------
After the 2026-09-18 swap to the S3 IR-101, a weekday named ~26 people where
weekdays before it named ~35. That compares two SYSTEMS, not two models: the
swap also dropped the 58 corridor crops the previous gallery carried
(`enroll.py` cannot carry vectors across a recognizer), and those crops existed
precisely because webcam enrolment photographs miss people on CCTV. It also
compares two different sets of days.

So this holds the faces fixed and varies one thing at a time:

    OLD      previous model + previous gallery (enrolment + its 58 crops)
             at the threshold production actually ran, 0.22
    NEW      current model + current gallery at 0.28 - what runs now
    NEW+     current model + current gallery + the SAME 58 corridor crops,
             re-embedded with the current model and floored at 0.44

OLD vs NEW is the question you asked. NEW vs NEW+ isolates the corridor crops.
OLD vs NEW+ is then the model itself.

THE PROBES
----------
Every pass since the start of the export that kept an aligned face:
named passes from `data/debug/*/<stem>_aligned.jpg` (upscaled to 224 for
viewing, so resized back to 112), and unknown passes from
`media/snapshots/unknown_0_<cam>_<track>_<epoch>.jpg`, matched to their
`unknown_sighting` row by camera, track and time.

WHAT A SINGLE FRAME IS AND IS NOT
---------------------------------
Each probe is ONE aligned frame - the track's best - scored with the live
matcher (`Gallery.match` at the live `second_best_margin`). The live pipeline
votes over every frame of a pass and needs several to agree, so a one-frame
replay names MORE than production would, for every system alike. The absolute
rates are therefore not production's; the DIFFERENCES between systems, on
identical probes under an identical rule, are the result.

And more names is not better names. On unknown probes most faces are visitors,
so a system that names more of them may be admitting more strangers. Where two
independent systems NAME THE SAME PERSON for the same face, that agreement is
evidence; where only one does, it is a candidate for a human to look at, and
`--sheet` writes those out side by side with the claimed person's photograph.

    python bench/replay_models.py --export /media/.../gpu6_prod_20260922
    python bench/replay_models.py --export ... --since 2026-09-18 --sheet out/
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OLD_MODEL = "adaface_ir101_finetune_fp16.onnx"
NEW_MODEL = "ir101S3v2s_sr10final_fp16.onnx"
OLD_THR = 0.22          # production's override before the swap, not the table's 0.215
NEW_THR = 0.28
NEW_FLOOR = 0.44
SWAP_UTC = datetime(2026, 9, 18, 6, 38, tzinfo=timezone.utc)   # 11:38 Tashkent

_UNK = re.compile(r"unknown_0_(\d+)_(\d+)_(\d+)\.jpg$")
_BODY = re.compile(r"body_0_(\d+)_(\d+)_(\d+)\.jpg$")


def _load_img(path: Path):
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        return None
    if img.shape[:2] != (112, 112):
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _utc(s):
    """Aware UTC. Debug captures carry `+05:00`; database values are naive UTC.

    The offset must be PARSED, not stripped: stripping it reads 11:54 Tashkent
    as 11:54 UTC and files every morning pass around the swap on the wrong side.
    """
    d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)


def named_probes(export: Path, since: str):
    """Passes the live system named, with the name it gave them."""
    out = []
    for meta in (export / "data" / "debug").rglob("*.json"):
        try:
            m = json.loads(meta.read_text())
        except Exception:
            continue
        if m.get("employee_id") is None:
            continue
        img = meta.with_name(meta.stem + "_aligned.jpg")
        when = str(m.get("timestamp_local", ""))[:10]
        if not img.is_file() or when < since:
            continue
        ts = str(m.get("timestamp_utc") or m.get("timestamp_local") or "")
        out.append({"kind": "named", "path": img, "live_id": int(m["employee_id"]),
                    "date": when, "ts": ts, "live_score": float(m.get("score", 0))})
    return out


def unknown_probes(export: Path, db, since: str):
    """Unknown passes, each joined to its aligned face by camera, track, time."""
    snaps = export / "media" / "snapshots"
    faces = {}
    for p in snaps.glob("unknown_0_*.jpg"):
        mm = _UNK.search(p.name)
        if mm:
            cam, trk, ep = map(int, mm.groups())
            faces.setdefault((cam, trk), []).append((ep, p))
    out = []
    for sid, snap, bdate, last in db.execute(
            "select id, snapshot, business_date, last_seen from unknown_sighting "
            "where business_date >= ? and snapshot is not null", (since,)):
        mm = _BODY.search(snap or "")
        if not mm:
            continue
        cam, trk, ep = map(int, mm.groups())
        cands = faces.get((cam, trk), [])
        best = min(cands, key=lambda c: abs(c[0] - ep), default=None)
        if best is None or abs(best[0] - ep) > 3:
            continue
        out.append({"kind": "unknown", "path": best[1], "live_id": None,
                    "date": str(bdate), "ts": str(last), "sighting": sid})
    return out


def gallery_from(db_path: Path, model_key: str):
    """(vectors, owners, floors, names) for one model's rows of a gallery DB."""
    from app.config import recognizer_key
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    names = dict(db.execute("select id, full_name from employee"))
    V, O, F, SRC = [], [], [], []
    for eid, vec, dim, model, thr, src in db.execute(
            "select f.employee_id, f.vector, f.dim, f.model_name, f.threshold, "
            "f.source_file from face_embedding f join employee e on e.id=f.employee_id "
            "where e.is_active = 1"):
        if recognizer_key(model or "") != recognizer_key(model_key):
            continue
        a = np.frombuffer(vec, dtype=np.float32)
        if a.size != (dim or 512):
            continue
        V.append(a); O.append(eid); F.append(thr if thr else 0.0); SRC.append(src or "")
    db.close()
    return (np.stack(V) if V else np.zeros((0, 512), np.float32),
            np.array(O, np.int64), np.array(F, np.float32), names, SRC)


def embed_all(model_name: str, images: list):
    from app.config import settings
    from app.core.geometry import to_normalized_chw
    from app.core.recognizer import FaceRecognizer
    rec = FaceRecognizer(settings.model_path(model_name), batch_size=64)
    out = np.zeros((len(images), 512), np.float32)
    for i in range(0, len(images), 64):
        batch = np.stack([to_normalized_chw(im) for im in images[i:i + 64]])
        out[i:i + len(batch)] = rec.embed(batch)
    return out


def decide(gallery, E, thr):
    from app.config import settings
    ms = gallery.match_batch(E, thr, settings.second_best_margin)
    return [(m.employee_id, float(m.score)) for m in ms]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True, type=Path)
    ap.add_argument("--since", default="2026-09-02")
    ap.add_argument("--sheet", type=Path, help="write review sheets of disagreements here")
    ap.add_argument("--csv", type=Path, help="write every probe's three decisions here")
    args = ap.parse_args()

    from app.core.gallery import Gallery

    ex = args.export
    live_db = next((ex / "data").glob("ematsy_prod_*.db"))
    old_gal_db = ex / "data" / "gallery_pre_swap_20260918.db"
    db = sqlite3.connect(f"file:{live_db}?mode=ro", uri=True)

    probes = named_probes(ex, args.since) + unknown_probes(ex, db, args.since)
    imgs, keep = [], []
    for p in probes:
        im = _load_img(p["path"])
        if im is not None:
            imgs.append(im); keep.append(p)
    probes = keep
    print(f"probes: {sum(p['kind']=='named' for p in probes)} named, "
          f"{sum(p['kind']=='unknown' for p in probes)} unknown  (since {args.since})")

    # --- galleries -------------------------------------------------------
    Vo, Oo, Fo, names_o, Src_o = gallery_from(old_gal_db, OLD_MODEL)
    Vn, On, Fn, names, _ = gallery_from(live_db, NEW_MODEL)
    g_old = Gallery(Vo, Oo, names_o, thresholds=Fo)
    g_new = Gallery(Vn, On, names, thresholds=Fn)

    # The old gallery's corridor crops, re-embedded with the NEW model from the
    # captures they were made from - same faces, the other model's space.
    crop_imgs, crop_own = [], []
    for eid, src, thr in zip(Oo, Src_o, Fo):
        if not src.startswith("live:"):
            continue
        key = src[len("live:"):]
        hit = next((ex / "data" / "debug").rglob(f"{key}_aligned.jpg"), None)
        im = _load_img(hit) if hit else None
        if im is not None:
            crop_imgs.append(im); crop_own.append(int(eid))
    n_crops_old = int(sum(s.startswith("live:") for s in Src_o))
    print(f"old gallery {len(Vo)} rows ({n_crops_old} corridor crops); "
          f"new gallery {len(Vn)} rows; {len(crop_imgs)}/{n_crops_old} old crops "
          f"still have their capture on disk")

    print("embedding with the previous model...")
    E_old = embed_all(OLD_MODEL, imgs)
    print("embedding with the current model...")
    E_new = embed_all(NEW_MODEL, imgs)
    if crop_imgs:
        C_new = embed_all(NEW_MODEL, crop_imgs)
        g_plus = Gallery(np.vstack([Vn, C_new]),
                         np.concatenate([On, np.array(crop_own, np.int64)]),
                         names, thresholds=np.concatenate(
                             [Fn, np.full(len(C_new), NEW_FLOOR, np.float32)]))
    else:
        g_plus = g_new

    d_old = decide(g_old, E_old, OLD_THR)
    d_new = decide(g_new, E_new, NEW_THR)
    d_plus = decide(g_plus, E_new, NEW_THR)

    # --- report ----------------------------------------------------------
    def period(p):
        try:
            return "after swap" if _utc(p["ts"]) >= SWAP_UTC else "before swap"
        except Exception:
            return "after swap" if p["date"] >= "2026-09-18" else "before swap"

    rows = []
    for p, a, b, c in zip(probes, d_old, d_new, d_plus):
        rows.append({**p, "old": a[0], "old_s": a[1], "new": b[0], "new_s": b[1],
                     "plus": c[0], "plus_s": c[1], "period": period(p)})

    for kind in ("named", "unknown"):
        print(f"\n=== {kind.upper()} probes ===")
        for per in ("before swap", "after swap"):
            R = [r for r in rows if r["kind"] == kind and r["period"] == per]
            if not R:
                continue
            n = len(R)
            line = f"  {per:<12} n={n:<6}"
            for sysname in ("old", "new", "plus"):
                named_n = sum(r[sysname] is not None for r in R)
                line += f"  {sysname.upper():>4} names {100*named_n/n:5.1f}%"
            print(line)
            if kind == "named":
                for sysname in ("old", "new", "plus"):
                    agree = sum(r[sysname] == r["live_id"] for r in R)
                    other = sum(r[sysname] is not None and r[sysname] != r["live_id"] for r in R)
                    print(f"      {sysname.upper():>4}: same name as live {100*agree/n:5.1f}%"
                          f"   a DIFFERENT name {other}")

    U = [r for r in rows if r["kind"] == "unknown"]
    both = [r for r in U if r["old"] is not None and r["old"] == r["plus"]]
    only_old = [r for r in U if r["old"] is not None and r["plus"] is None]
    only_plus = [r for r in U if r["plus"] is not None and r["old"] is None]
    clash = [r for r in U if r["old"] is not None and r["plus"] is not None
             and r["old"] != r["plus"]]
    print(f"\n=== UNKNOWNS each system would name ({len(U)} probes) ===")
    print(f"  OLD and NEW+ agree on the same person : {len(both)}")
    print(f"  only OLD names                        : {len(only_old)}")
    print(f"  only NEW+ names                       : {len(only_plus)}")
    print(f"  both name, DIFFERENT people           : {len(clash)}")
    print("  top people among the agreed:",
          ", ".join(f"{names.get(k, k)} {v}" for k, v in
                    Counter(r["old"] for r in both).most_common(8)))

    if args.csv:
        import csv
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["kind", "period", "date", "path", "sighting", "live",
                        "old", "old_score", "new", "new_score", "plus", "plus_score"])
            for r in rows:
                w.writerow([r["kind"], r["period"], r["date"], r["path"],
                            r.get("sighting", ""), names.get(r["live_id"], ""),
                            names.get(r["old"], ""), f"{r['old_s']:.3f}",
                            names.get(r["new"], ""), f"{r['new_s']:.3f}",
                            names.get(r["plus"], ""), f"{r['plus_s']:.3f}"])
        print(f"\nwrote {args.csv}")

    if args.sheet:
        _sheets(args.sheet, {"agreed": both, "only_old": only_old,
                             "only_new_plus": only_plus, "clash": clash}, names)
    return 0


def _sheets(outdir: Path, groups: dict, names: dict):
    """One contact sheet per group: probe face | claimed person's photograph."""
    import cv2
    from app.config import settings
    outdir.mkdir(parents=True, exist_ok=True)
    photo_cache = {}

    def photo(eid):
        if eid in photo_cache:
            return photo_cache[eid]
        from app.services.augment import profile_image
        p = profile_image(eid)
        im = cv2.imread(str(p)) if p else None
        photo_cache[eid] = cv2.resize(im, (112, 112)) if im is not None else None
        return photo_cache[eid]

    for gname, R in groups.items():
        tiles = []
        for r in R[:120]:
            face = cv2.imread(str(r["path"]))
            if face is None:
                continue
            face = cv2.resize(face, (112, 112))
            eid = r["old"] if r["old"] is not None else r["plus"]
            ph = photo(eid)
            if ph is None:
                ph = np.zeros((112, 112, 3), np.uint8)
            tile = np.hstack([face, ph])
            tile = cv2.copyMakeBorder(tile, 0, 26, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            label = f"{names.get(eid, eid)}"[:22]
            cv2.putText(tile, label, (3, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
            cv2.putText(tile, f"o{r['old_s']:.2f} n{r['plus_s']:.2f}", (3, 136),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)
            tiles.append(tile)
        if not tiles:
            continue
        cols = 6
        while len(tiles) % cols:
            tiles.append(np.full_like(tiles[0], 255))
        grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        cv2.imwrite(str(outdir / f"{gname}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  sheet {outdir / (gname + '.jpg')}  ({min(len(R), 120)} of {len(R)})")


if __name__ == "__main__":
    raise SystemExit(main())
