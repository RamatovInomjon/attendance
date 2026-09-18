#!/usr/bin/env python3
"""Carry the production operating point from one recognizer to another.

    python bench/calibrate_from_corpus.py \
        --faces /media/inomjon/T7/face_eval_20260915 --prod data/gpu6_eval_20260915 \
        --reference adaface_ir101_finetune_fp16.onnx --reference-threshold 0.22 \
        --candidate ir101S3v2s_sr10final_fp16.onnx

WHY A THRESHOLD CANNOT BE COPIED
--------------------------------
Each recognizer puts its impostor distribution on its own scale: 0.22 keeps the
deployed IR-101 below a pair false-accept rate of ~1e-4 on this corridor, and
the same number on the new IR-101 - whose scores run ~0.04 higher - would admit
several times as many wrong names, with nothing in the attendance data to show
it. So the operating point is carried over as a RATE, not a number: measure the
pair-FAR the reference model actually runs at in production, then find the
score at which the candidate produces that same rate, on the same faces,
against the same gallery.

The faces are `bench/extract_native_faces.py` output labelled the way
`bench/fair_ab.py` labels them - by pixels against production's own saved
crops, never by either recognizer - so neither model chooses its own test.

Two numbers come out, both starting points that `augment_gallery.py --measure`
and the review page refine once the new model has run for a few days:

  recognizer_thresholds[candidate]   the global threshold
  augment_live_floor                 the corridor-crop floor, kept at the same
                                     position between threshold and typical
                                     genuine score as it has today
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
sys.path.insert(0, str(ROOT / "bench"))

FARS = (1e-4, 1e-3, 1e-2)
PER_TRACK = 20


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", default="/media/inomjon/T7/face_eval_20260915")
    ap.add_argument("--prod", default="data/gpu6_eval_20260915")
    ap.add_argument("--reference", default="adaface_ir101_finetune_fp16.onnx")
    ap.add_argument("--reference-threshold", type=float, default=0.22,
                    help="what production runs the reference model at")
    ap.add_argument("--reference-floor", type=float, default=0.35,
                    help="augment_live_floor in force for the reference model")
    ap.add_argument("--candidate", default="ir101S3v2s_sr10final_fp16.onnx")
    ap.add_argument("--out", default="data/bench/calibrate_from_corpus.json")
    args = ap.parse_args()

    from app.config import settings
    from app.core.onnx_env import preload_cuda_libs
    preload_cuda_libs()
    from app.core.recognizer import FaceRecognizer
    import fair_ab as fa

    prod = (ROOT / args.prod).resolve()
    FC = fa.Faces(Path(args.faces))
    track_label, matched, _ = fa.label_tracks(FC, prod)
    F = FC.m
    gal, owners, people, _fail = fa.build_galleries(prod)
    ids = np.array(sorted(set(owners.tolist())))
    col = {int(p): j for j, p in enumerate(ids)}

    by_seg = defaultdict(list)
    for i, sk in enumerate(F["skey"]):
        by_seg[sk].append(i)
    gen, unk, lab = [], [], []
    for sk, idx in by_seg.items():
        if len(idx) < 3:
            continue
        idx = np.array(idx)
        if len(idx) > PER_TRACK:
            idx = idx[np.linspace(0, len(idx) - 1, PER_TRACK).astype(int)]
        kind, emp = track_label.get(sk, ("unlabelled", 0))
        if kind == "genuine" and emp in col:
            gen.extend(idx.tolist())
            lab.extend([col[emp]] * len(idx))
        elif kind != "voided":
            unk.extend(idx.tolist())          # unknown AND unlabelled passes
    gen, lab, unk = np.array(gen), np.array(lab), np.array(unk)
    print(f"  labelled genuine frames {len(gen):,} ({len(set(lab))} people), "
          f"unnamed corridor frames {len(unk):,}; matched {dict(matched)}")

    report = {"reference": args.reference, "candidate": args.candidate,
              "reference_threshold": args.reference_threshold,
              "genuine_frames": int(len(gen)), "unnamed_frames": int(len(unk)),
              "models": {}}
    stats = {}
    for name in (args.reference, args.candidate):
        rec = FaceRecognizer(settings.model_path(name), batch_size=16)
        G = rec.embed(fa.to_chw(gal["pipe"]))
        pg = FC.pixels("pipe", gen)
        E = np.concatenate([rec.embed(fa.to_chw(pg[c:c + 1024])) for c in range(0, len(pg), 1024)])
        pu = FC.pixels("pipe", unk)
        U = np.concatenate([rec.embed(fa.to_chw(pu[c:c + 1024])) for c in range(0, len(pu), 1024)])
        del rec, pg, pu
        P = fa.per_person(E @ G.T, owners, ids)
        g = P[np.arange(len(gen)), lab]
        mask = np.ones_like(P, bool)
        mask[np.arange(len(gen)), lab] = False
        pool = P[mask]                                   # every non-owner pair
        other = np.where(mask, P, -2).max(axis=1)        # best wrong name per frame
        ok = P.argmax(1) == lab
        PU = fa.per_person(U @ G.T, owners, ids)
        u_best = PU.max(axis=1)
        stats[name] = dict(g=g, pool=pool, other=other, ok=ok, u_best=u_best)
        print(f"  {name}: genuine median {np.median(g):.3f}  impostor pool max "
              f"{pool.max():.3f}  p99.99 {np.quantile(pool, .9999):.3f}")

    ref, cand = stats[args.reference], stats[args.candidate]

    # -- the reference model's operating point, as rates ---------------------
    t0 = args.reference_threshold
    pair_far = float(np.mean(ref["pool"] >= t0))
    probe_far = float(np.mean(ref["other"] >= t0))
    ref_tar = float(np.mean((ref["g"] >= t0) & ref["ok"]))
    ref_unk = float(np.mean(ref["u_best"] >= t0))
    print(f"\n  reference at {t0:.3f}: pair FAR {pair_far:.2e}, {probe_far:.4f} of "
          f"genuine frames name someone else, TAR {ref_tar:.3f}, "
          f"unnamed frames that would be named {ref_unk:.4f}")

    # -- the candidate at the SAME rate --------------------------------------
    def thr_at(pool, far):
        k = max(1, int(round(far * len(pool))))
        return float(np.partition(pool, len(pool) - k)[len(pool) - k])

    t_same = thr_at(cand["pool"], pair_far) if pair_far > 0 else float(cand["pool"].max()) + 0.01
    rows = []
    for label, t in [("same pair-FAR as production", t_same)] + \
                    [(f"pair-FAR {f:g}", thr_at(cand["pool"], f)) for f in FARS] + \
                    [("max impostor + 0.01", float(cand["pool"].max()) + 0.01)]:
        rows.append({"rule": label, "threshold": round(t, 3),
                     "tar": float(np.mean((cand["g"] >= t) & cand["ok"])),
                     "probe_far": float(np.mean(cand["other"] >= t)),
                     "unnamed_named": float(np.mean(cand["u_best"] >= t))})
    print(f"\n  candidate {args.candidate}")
    print(f"  {'rule':30s} {'thr':>6s} {'TAR':>6s} {'wrong-name/frame':>17s} {'unnamed named':>14s}")
    for r in rows:
        print(f"  {r['rule']:30s} {r['threshold']:6.3f} {r['tar']:6.3f} "
              f"{r['probe_far']:17.4f} {r['unnamed_named']:14.4f}")
    ref_rows = [{"rule": f"pair-FAR {f:g}", "threshold": round(thr_at(ref['pool'], f), 3)} for f in FARS]

    # -- the corridor-crop floor: same relative position ---------------------
    # 0.35 sits a fixed way up the gap between the threshold and where a
    # typical genuine corridor match lands. Keep that ratio; the absolute
    # value follows the candidate's scale.
    ref_gap = float(np.median(ref["g"])) - t0
    cand_gap = float(np.median(cand["g"])) - t_same
    floor = t_same + (args.reference_floor - t0) * (cand_gap / ref_gap if ref_gap > 0 else 1.0)
    print(f"\n  corridor-crop floor: reference {args.reference_floor:.3f} sits "
          f"{(args.reference_floor - t0) / ref_gap:.2f} of the threshold->median gap; "
          f"same position for the candidate = {floor:.3f}")

    report["models"][args.reference] = {"pair_far_at_threshold": pair_far,
                                        "probe_far": probe_far, "tar": ref_tar,
                                        "genuine_median": float(np.median(ref["g"])),
                                        "thresholds": ref_rows}
    report["models"][args.candidate] = {"genuine_median": float(np.median(cand["g"])),
                                        "thresholds": rows}
    report["recommended"] = {"recognizer_threshold": round(t_same, 3),
                             "augment_live_floor": round(floor, 3)}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\n  recommended  recognizer_thresholds[{args.candidate!r}] = {t_same:.3f}"
          f"   augment_live_floor = {floor:.3f}")
    print(f"  report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
