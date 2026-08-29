#!/usr/bin/env python3
"""One command that answers 'did this change make the algorithm worse?'

Speed alone is a trap: every optimisation here could be made faster by
degrading the model, and a wall-clock number would applaud it. So this measures
both halves and writes them as JSON, and `--compare` diffs two runs and fails
loudly when accuracy moved in the wrong direction.

    python scripts/benchmark.py --out data/bench/before.json
    ...make a change...
    python scripts/benchmark.py --out data/bench/after.json
    python scripts/benchmark.py --compare data/bench/before.json data/bench/after.json

Accuracy comes from two independent places, because they fail differently:

* the enrolment gallery (d-prime, rank-1) catches a recognizer that has been
  degraded - quantised too far, wrong weights, wrong preprocessing;
* replayed corridor clips catch a pipeline that has been degraded - a detector
  that finds fewer heads, an aligner that produces worse crops, a tracker that
  fragments, direction that stops resolving. The gallery cannot see any of that.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Anything worse than this is a regression, not noise. d-prime and rank-1 come
# from a fixed gallery and are deterministic, so their tolerance is tight;
# per-frame timing varies with machine load, so it gets a wide band and is
# reported rather than enforced.
TOLERANCE = {
    "gallery.d_prime": -0.05,          # absolute drop allowed
    "gallery.rank1": -0.0,             # rank-1 must never fall
    "gallery.weakest_genuine": -0.005,
    "clips.recognized_tracks": -0,     # must not recognise fewer people
    # faces_embedded is the honest "are we getting fewer chances to
    # recognise" signal; track_observations is reported, not enforced.
    "clips.faces_embedded": -0.05,
    "clips.direction_resolved": -0.02,
}


def _gallery_metrics(limit_people: int | None = None) -> dict:
    """Genuine/impostor separation over the enrolment set."""
    import numpy as np, cv2, glob, os
    from app.core.onnx_env import preload_cuda_libs
    preload_cuda_libs()
    from app.core.detector import YoloFaceDetector
    from app.core.aligner import FaceAligner
    from app.core.recognizer import FaceRecognizer
    from app.config import settings

    gal_dir = ROOT / "face_id_users"
    if not gal_dir.is_dir():
        return {"skipped": "face_id_users/ not present"}

    det = YoloFaceDetector(str(settings.model_path(settings.detector_model)),
                           imgsz=1280, conf=0.35)
    ali = FaceAligner(settings.model_path(settings.aligner_model),
                      crop_size=settings.align_crop_size,
                      margin=settings.align_margin, mode=settings.align_mode)
    rec = FaceRecognizer(settings.model_path(settings.recognizer_model),
                         batch_size=settings.embed_batch)

    people = sorted(d for d in os.listdir(gal_dir) if (gal_dir / d).is_dir())
    if limit_people:
        people = people[:limit_people]
    embs, labels = [], []
    t0 = time.perf_counter()
    for pid, person in enumerate(people):
        for p in sorted(glob.glob(str(gal_dir / person / "*.png"))):
            bgr = cv2.imread(p)
            if bgr is None:
                continue
            d = det.detect(bgr)
            if not d:
                continue
            box = max(d, key=lambda x: (x.box[2] - x.box[0]) * (x.box[3] - x.box[1])).box
            faces = ali.align(bgr, [box], is_bgr=True)
            if not faces:
                continue
            e = rec.embed(np.stack([faces[0].aligned]),
                          keypoints=np.stack([faces[0].keypoints])
                          if rec.needs_keypoints else None)
            embs.append(e[0] / (np.linalg.norm(e[0]) + 1e-12))
            labels.append(pid)
    dt = time.perf_counter() - t0
    if len(embs) < 4:
        return {"skipped": f"only {len(embs)} embeddings"}

    E = np.stack(embs).astype(np.float32)
    y = np.array(labels)
    S = E @ E.T
    iu = np.triu_indices(len(E), k=1)
    same = y[iu[0]] == y[iu[1]]
    gen, imp = S[iu][same], S[iu][~same]
    d_prime = float(abs(gen.mean() - imp.mean()) /
                    np.sqrt(0.5 * (gen.var() + imp.var()) + 1e-12))

    # rank-1, leave-one-out
    Sx = S.copy()
    np.fill_diagonal(Sx, -2.0)
    rank1 = float((y[Sx.argmax(1)] == y).mean())

    return {
        "images": len(embs), "people": len(set(labels)),
        "d_prime": round(d_prime, 4),
        "rank1": round(rank1, 6),
        "genuine_mean": round(float(gen.mean()), 4),
        "genuine_min": round(float(gen.min()), 4),
        "weakest_genuine": round(float(gen.min()), 4),
        "impostor_mean": round(float(imp.mean()), 4),
        "impostor_max": round(float(imp.max()), 4),
        "embed_seconds": round(dt, 1),
    }


def _clip_metrics(max_clips: int = 8, max_frames: int = 400) -> dict:
    """Replay recorded corridor footage through the real pipeline."""
    import numpy as np, cv2, glob, collections
    from app.core.onnx_env import preload_cuda_libs
    preload_cuda_libs()
    from app.core.pipeline import CameraPipeline, Frame
    from app.services.enrollment import load_gallery

    clips = sorted(glob.glob(str(ROOT / "data/recordings/*/*.mp4")))[:max_clips]
    if not clips:
        return {"skipped": "no recordings"}

    # Real per-camera geometry. Without it DirectionConfig has no line, so
    # `configured` is False, the depth-only travel guard never applies, and
    # every verdict comes back "depth only (...)" - which is NOT how production
    # behaves and makes any direction comparison meaningless.
    from app.core.direction import config_from_camera
    from app.db.models import Camera
    from app.db.session import session_scope
    from sqlalchemy import select
    cfgs = {}
    try:
        with session_scope() as _s:
            for cam in _s.execute(select(Camera)).scalars():
                cfgs[cam.name] = config_from_camera(cam)
    except Exception:
        pass

    gallery = load_gallery()
    ts, tracks, track_obs, faces_embedded = [], [], 0, 0
    names, dirs = set(), collections.Counter()
    for clip in clips:
        # clips are data/recordings/<CameraName>/<file>.mp4
        cam_name = Path(clip).parent.name
        pipe = CameraPipeline("bench", gallery, direction_cfg=cfgs.get(cam_name))
        cap = cv2.VideoCapture(clip)
        t_base = 1_700_000_000.0     # fixed epoch: replay must not depend on "now"
        i = 0
        while i < max_frames:
            ok, im = cap.read()
            if not ok:
                break
            # Synthetic timestamps, NOT time.time(). Track ageing and the
            # direction trajectory are measured in seconds, so feeding
            # wall-clock makes replay depend on how fast the machine happens to
            # run: a slower build ages tracks sooner, prunes them differently,
            # and reports different recognitions. That produced a phantom
            # "regression" where an encrypted build appeared to lose a person.
            # 20 fps is what the cameras deliver.
            t0 = time.perf_counter()
            r = pipe.process(Frame(image=im, ts=t_base + i / 20.0, index=i))
            ts.append((time.perf_counter() - t0) * 1000)
            track_obs += len(r.tracks)      # live tracks this frame, NOT detections
            faces_embedded += int(r.timings.get("faces", 0) or 0)
            tracks.extend(r.completed)
            i += 1
        tracks.extend(pipe._prune(t_base + i / 20.0 + 1e6))
        cap.release()

    for t in tracks:
        dirs[t.direction] += 1
        if t.name and not t.name.startswith("_") and t.name not in ("", "…"):
            names.add(t.name)
    ts.sort()
    n = max(len(ts), 1)
    resolved = sum(v for k, v in dirs.items() if k != "UNKNOWN")
    return {
        "clips": len(clips), "frames": len(ts),
        "ms_median": round(ts[n // 2], 3),
        "ms_p90": round(ts[int(n * 0.9)], 3),
        "ms_p99": round(ts[int(n * 0.99)], 3),
        # live track-observations summed over frames. Fewer is not worse:
        # merging fragmented tracks legitimately reduces it.
        "track_observations": track_obs,
        "faces_embedded": faces_embedded,
        "tracks": len(tracks),
        "recognized_tracks": len(names),
        "recognized_names": sorted(names),
        "direction_resolved": round(resolved / max(len(tracks), 1), 4),
        "by_direction": dict(dirs),
    }


def run(args) -> dict:
    from app.config import settings
    return {
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "models": {
            "head": settings.head_model,
            "aligner": settings.aligner_model,
            "recognizer": settings.recognizer_model,
        },
        "settings": {
            "threshold": settings.recognition_threshold,
            "second_best_margin": settings.second_best_margin,
            "min_face_px": settings.min_face_px,
            "process_every_nth": settings.process_every_nth,
        },
        "gallery": _gallery_metrics(args.people),
        "clips": _clip_metrics(args.clips, args.frames),
    }


def _get(d: dict, dotted: str):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def compare(before: dict, after: dict) -> int:
    print(f"  before : {before['when']}  {before['models']}")
    print(f"  after  : {after['when']}  {after['models']}")
    print()
    worse = []
    for key, tol in TOLERANCE.items():
        b, a = _get(before, key), _get(after, key)
        if b is None or a is None:
            print(f"  {key:34s} (missing in one run)")
            continue
        delta = a - b
        rel = delta / b if (b and abs(tol) < 1 and "heads" in key or "direction" in key) else None
        limit = tol * abs(b) if rel is not None else tol
        bad = delta < limit
        mark = "WORSE" if bad else "ok"
        print(f"  {key:34s} {b:>10.4f} -> {a:>10.4f}  ({delta:+.4f})  {mark}")
        if bad:
            worse.append((key, b, a))

    for key in ("clips.ms_median", "clips.ms_p90"):
        b, a = _get(before, key), _get(after, key)
        if b and a:
            print(f"  {key:34s} {b:>10.3f} -> {a:>10.3f}  ({(a-b)/b*100:+.1f}%)  [speed, not enforced]")

    bn = set(_get(before, "clips.recognized_names") or [])
    an = set(_get(after, "clips.recognized_names") or [])
    if bn - an:
        print(f"\n  NO LONGER RECOGNISED: {sorted(bn - an)}")
        worse.append(("clips.recognized_names", len(bn), len(an)))
    if an - bn:
        print(f"  newly recognised: {sorted(an - bn)}")

    print()
    if worse:
        print(f"  REGRESSION in {len(worse)} metric(s) - do not ship")
        return 1
    print("  no accuracy regression")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--people", type=int, default=None,
                    help="limit gallery people (faster smoke run)")
    args = ap.parse_args()

    if args.compare:
        b = json.loads(args.compare[0].read_text())
        a = json.loads(args.compare[1].read_text())
        return compare(b, a)

    res = run(args)
    text = json.dumps(res, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"  written {args.out}")
    g, c = res["gallery"], res["clips"]
    if "skipped" not in g:
        print(f"  gallery: d'={g['d_prime']}  rank-1={g['rank1']:.4f}  "
              f"weakest genuine={g['weakest_genuine']}  ({g['images']} imgs)")
    if "skipped" not in c:
        print(f"  clips  : {c['ms_median']} ms median, {c['ms_p90']} p90 | "
              f"{c['recognized_tracks']} recognised | "
              f"direction resolved {c['direction_resolved']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
