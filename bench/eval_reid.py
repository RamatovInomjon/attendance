#!/usr/bin/env python3
"""Score the exported ReID ONNX model on the cross-camera test set.

    python bench/eval_reid.py
    python bench/eval_reid.py --model models/reid_..._fp16.onnx --splits 200

THIS IS THE CHECK THAT MATTERS. Every preprocessing mistake a ReID model can
suffer - BGR instead of RGB, a letterbox instead of a squash, a missed
normalisation - produces embeddings of exactly the right shape and norm that are
quietly wrong. Nothing downstream complains; matching just degrades.

So the exported model is scored on the SAME test set, with the SAME protocol and
the SAME metric code as the research project, and the number is compared against
what that project published. If it lands near `natijalar/qayta_baho.tsv`, the
preprocessing in `app/core/reid.py` is right. If it does not, it is wrong, and
no amount of matching logic downstream will rescue it.

Protocol (tools/reid/crosscam.py): gallery = Entrance, query = Exit; half the
two-camera identities are enrolled, the rest become impostors; repeated over N
independent random enrolment splits because 45 people is too few for one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PHD = Path("/home/inomjon/projectAI/phd/dissertatsiya2")
PUBLISHED = {   # natijalar/qayta_baho.tsv, the row for the model we exported
    "resnet101_ibn_256x128_s0_e2048_es@h100": dict(mAP=92.24, R1=93.49,
                                                   fnir10=31.10, thr1=0.6769),
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="default: settings.reid_model")
    ap.add_argument("--test-dir", default=str(PHD / "test_reid"))
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--expect", default="resnet101_ibn_256x128_s0_e2048_es@h100")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # The research project's metric and split code, used verbatim so the
    # comparison is like for like.
    sys.path.insert(0, str(PHD / "tools" / "reid"))
    try:
        from crosscam import cam_of, kamera_maydoni, split_open_set_crosscam
        from reid_eval import closed_set_metrics, open_set_curve
    except ImportError as e:
        raise SystemExit(f"  cannot import the research eval code from {PHD}: {e}")

    import cv2
    from app.config import settings
    from app.core.reid import PersonReID

    model = args.model or str(settings.model_path(settings.reid_model))
    reid = PersonReID(model)
    print(f"  model   {Path(model).name}")
    print(f"  input   {reid.height}x{reid.width}  embed={reid.dim}  "
          f"colour={reid.colour}  provider={reid.provider}")

    test_dir = Path(args.test_dir)
    if not test_dir.is_dir():
        raise SystemExit(f"  no test set at {test_dir}")

    # Same folder-per-identity walk and camera detection as CrossCamSet.
    IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    every = [f.name for d in test_dir.iterdir() if d.is_dir()
             for f in d.iterdir() if f.suffix.lower() in IMG]
    field = kamera_maydoni(every)
    paths, pids, cams = [], [], []
    for pid, d in enumerate(sorted(p for p in test_dir.iterdir() if p.is_dir())):
        kj = d / "kamera.json"
        mapping = json.loads(kj.read_text(encoding="utf-8")) if kj.exists() else None
        for f in sorted(x for x in d.iterdir() if x.suffix.lower() in IMG):
            paths.append(f)
            pids.append(pid)
            cams.append(cam_of(f.name, mapping, field))
    pids, cams = np.array(pids), np.array(cams)
    print(f"  test    {len(set(pids))} people, {len(paths)} images, "
          f"cameras {sorted(set(cams))}")

    feats = []
    B = 32
    for i in range(0, len(paths), B):
        crops = [cv2.imread(str(p)) for p in paths[i:i + B]]
        crops = [c for c in crops if c is not None]
        feats.append(reid.embed(crops))
    f = np.concatenate(feats).astype(np.float32)
    f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-12
    print(f"  features {f.shape}")

    # Entrance is the gallery, Exit the query - the operating direction.
    low = {c.lower(): c for c in set(cams)}
    gal = low.get("entrance")
    qry = low.get("exit")
    print(f"  protocol gallery={gal or 'auto'} -> query={qry or 'auto'}, "
          f"{args.splits} splits\n")

    got = {k: [] for k in ("mAP", "R1", "fnir1", "fnir10", "thr1")}
    for seed in range(args.splits):
        gf, gp, qf, qp, imp, _meta = split_open_set_crosscam(
            f, pids, cams, gallery_cam=gal, query_cam=qry, seed=seed)
        if len(qp) == 0 or len(imp) == 0:
            continue
        c = closed_set_metrics(qf, qp, gf, gp)
        o = open_set_curve(qf, qp, imp, gf, gp)
        got["mAP"].append(c["mAP"] * 100)
        got["R1"].append(c["Rank-1"] * 100)
        got["fnir1"].append(o["FNIR@FPIR=0.01"] * 100)
        got["fnir10"].append(o["FNIR@FPIR=0.1"] * 100)
        got["thr1"].append(o["threshold@FPIR=0.01"])
    summary = {k: (float(np.mean(v)), float(np.std(v))) for k, v in got.items()}

    ref = PUBLISHED.get(args.expect)
    print(f"  {'metric':10s} {'ONNX':>16s}   {'published':>10s}   {'delta':>8s}")
    ok = True
    for key, label in (("mAP", "mAP"), ("R1", "R1"), ("fnir10", "FNIR@10%"),
                       ("thr1", "thr@FPIR1%")):
        m, sd = summary[key]
        if ref and key in ref:
            d = m - ref[key]
            # Tolerance is the split-to-split spread; fp16 and a different
            # resize implementation cannot move the mean beyond that.
            lim = max(1.0, sd * 0.5) if key != "thr1" else 0.02
            flag = "ok" if abs(d) <= lim else "** DIFFERS"
            ok &= abs(d) <= lim
            print(f"  {label:10s} {m:9.2f} ±{sd:5.2f}   {ref[key]:10.2f}   "
                  f"{d:+8.2f}  {flag}")
        else:
            print(f"  {label:10s} {m:9.2f} ±{sd:5.2f}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"model": Path(model).name, "splits": args.splits,
             "test_dir": str(test_dir), **{k: v for k, v in summary.items()}},
            indent=2))
        print(f"\n  written {args.out}")
    if ref and not ok:
        print("\n  THE EXPORT DOES NOT REPRODUCE THE PUBLISHED RESULT.")
        print("  Suspect preprocessing before anything else: colour order,")
        print("  resize mode, normalisation. Do not build matching on this.")
        return 1
    print("\n  reproduces the published result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
