#!/usr/bin/env python3
"""A/B two recognizers on the faces production actually saw, labelled by people.

    python bench/compare_recognizers.py \
        --data data/gpu6_eval_20260915 \
        --models adaface_ir101_finetune_fp16.onnx ir50S3v2s_sr10final.onnx

WHY THESE FACES
---------------
`models/README.md` records the lesson from the ViT-KPRPE candidate: both models
hit rank-1 1.0000 on the enrolment gallery, so gallery d' measured headroom on a
saturated set and said nothing about the corridor. This scores the 112x112 crops
the live pipeline itself chose - the `why_` crop behind every recognition event
and the `unknown_` crop behind every unnamed pass - so detection, alignment and
quality gating are held fixed and ONLY the recognizer differs.

WHAT EACH SET CAN AND CANNOT SAY
--------------------------------
genuine    live events nobody voided. Chosen BY the current model (it accepted
           them), so the challenger can only lose ground here. It measures
           regression, not improvement.
voided     events an admin marked "not this person". Confirmed impostor pairs.
tracklet   passes the face path never named, named later by an admin through a
           pseudo-person. Confirmed genuine pairs the current model MISSED - the
           only set that can show a challenger recovering something.
unknown    every unnamed pass. Unlabelled: it holds both visitors and missed
           employees, so it is reported as a naming rate and a contact sheet to
           look at, never as FAR.

THRESHOLDS DO NOT TRANSFER
--------------------------
Each model gets its own operating point, derived the same way for both: the
highest clean impostor score seen (every genuine probe scored against every
enrolled person who is NOT its owner), so neither is flattered by a number
tuned for the other.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CROP_RE = re.compile(r"(evt|why|unknown)_(\d+)_(\d+)_(-?\d+)_(\d+)\.jpg")


def _epoch(snapshot: str | None) -> int:
    m = re.search(r"_(\d+)\.jpg$", snapshot or "")
    return int(m.group(1)) if m else 0


def load_crop(path: Path) -> np.ndarray:
    """A stored aligned crop back into the tensor the recognizer was fed.

    Inverse of `worker._save_snapshot`: RGB uint8 -> BGR on disk, so read BGR,
    return RGB CHW in [-1, 1]. JPEG q92 is lossy, but both models read the same
    bytes, so it costs them equally.
    """
    import cv2
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    if bgr.shape[:2] != (112, 112):
        bgr = cv2.resize(bgr, (112, 112), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.transpose(rgb / 127.5 - 1.0, (2, 0, 1))


def build_sets(data: Path):
    db = sqlite3.connect(f"file:{data / 'ematsy.db'}?mode=ro", uri=True)
    q = lambda s, *a: list(db.execute(s, a))

    crops = defaultdict(list)
    for f in (data / "snapshots").iterdir():
        m = CROP_RE.match(f.name)
        if m:
            crops[(m[1], int(m[2]), int(m[3]), int(m[4]))].append((int(m[5]), f))

    def find(kind, emp, cam, track, ep):
        best = min(crops.get((kind, emp, cam, track), []),
                   key=lambda x: abs(x[0] - ep), default=None)
        return best[1] if best and abs(best[0] - ep) <= 3 else None

    people = {r[0]: (r[1], r[2]) for r in
              q("select id, full_name, folder from employee where is_active = 1")}

    events = []
    for eid, emp, cam, track, snap, voided, source, score, tr in q(
            "select id, employee_id, camera_id, track_id, snapshot, "
            "voided_at is not null, source, score, transition "
            "from recognition_event"):
        # Two crops per event, and the names mislead: `evt_` is the frame that
        # SCORED highest for the committed person (the current model's own
        # argmax), `why_` is the CLEAREST frame, picked by quality and blind to
        # any model's score. See pipeline.CompletedTrack.crop / score_crop.
        ep = _epoch(snap)
        evt, why = find("evt", emp, cam, track, ep), find("why", emp, cam, track, ep)
        if evt is None or why is None:
            continue
        kind = "voided" if voided else ("tracklet" if source == "tracklet" else "genuine")
        events.append({"id": eid, "emp": emp, "kind": kind, "evt": evt, "why": why,
                       "prod_score": score, "transition": tr})

    unknowns = []
    for uid, cam, track, snap, best, near in q(
            "select id, camera_id, track_id, snapshot, best_score, "
            "nearest_employee_id from unknown_sighting"):
        p = find("unknown", 0, cam, track, _epoch(snap))
        if p is not None:
            unknowns.append({"id": uid, "path": p, "prod_score": best, "near": near})

    enrol = q("select employee_id, source_file from face_embedding "
              "where source_file not like 'live:%'")
    return people, events, unknowns, enrol


class Model:
    def __init__(self, name: str):
        from app.config import settings
        from app.core.recognizer import FaceRecognizer
        self.name = name
        self.rec = FaceRecognizer(settings.model_path(name), batch_size=64)

    def embed(self, tensors: np.ndarray) -> np.ndarray:
        return self.rec.embed(tensors)


def enrol_gallery(models, people, enrol):
    """Enrolment photographs through the real enrolment path, once per model.

    Detection and alignment are shared, so the gallery differs between models
    only in the recognizer, exactly as the probes do.
    """
    import cv2
    from app.config import settings
    from app.services.enrollment import Enroller

    first = Enroller(recognizer=models[0].rec)
    gallery_dir = settings.root / "face_id_users"
    tensors, owners, skipped = [], [], []
    for emp, src in enrol:
        if emp not in people or not people[emp][1]:
            skipped.append((emp, src, "no enrolment folder"))
            continue
        path = gallery_dir / people[emp][1] / Path(src).name
        bgr = cv2.imread(str(path))
        if bgr is None:
            skipped.append((emp, src, "image missing"))
            continue
        dets = first.detector.detect(bgr)
        if not dets:
            skipped.append((emp, src, "no face"))
            continue
        d = max(dets, key=lambda x: (x.box[2] - x.box[0]) * (x.box[3] - x.box[1]))
        faces = first.aligner.align(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), [d.box])
        if not faces:
            skipped.append((emp, src, "no alignment"))
            continue
        tensors.append(faces[0].aligned)
        owners.append(emp)
    T = np.stack(tensors).astype(np.float32)
    return T, np.array(owners, np.int64), skipped


def per_person(sims: np.ndarray, owners: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """(probes, rows) -> (probes, people): best row per person, as Gallery.match."""
    out = np.full((sims.shape[0], len(ids)), -2.0, np.float32)
    for j, pid in enumerate(ids):
        out[:, j] = sims[:, owners == pid].max(axis=1)
    return out


def dprime(g, i):
    return float((np.mean(g) - np.mean(i)) / np.sqrt(0.5 * (np.var(g) + np.var(i))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/gpu6_eval_20260915")
    ap.add_argument("--models", nargs="+",
                    default=["adaface_ir101_finetune_fp16.onnx", "ir50S3v2s_sr10final.onnx"])
    ap.add_argument("--reference-threshold", type=float, default=0.22,
                    help="the live threshold of the FIRST model; its pair FAR on "
                         "the clearest-frame set becomes the operating point "
                         "every model is compared at")
    ap.add_argument("--exclude", default="",
                    help="comma-separated event ids to drop from every set "
                         "(label errors found by looking, applied to all models)")
    ap.add_argument("--out", default="data/bench/compare_recognizers")
    args = ap.parse_args()

    from app.core.onnx_env import preload_cuda_libs
    preload_cuda_libs()

    data = (ROOT / args.data).resolve()
    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    people, events, unknowns, enrol = build_sets(data)
    drop = {int(x) for x in args.exclude.split(",") if x.strip().isdigit()}
    events = [e for e in events if e["id"] not in drop]
    models = [Model(m) for m in args.models]

    GT, owners, skipped = enrol_gallery(models, people, enrol)
    enrolled = np.array(sorted(set(owners.tolist())), np.int64)
    col = {int(p): j for j, p in enumerate(enrolled)}
    events = [e for e in events if e["emp"] in col]
    kinds = {k: [i for i, e in enumerate(events) if e["kind"] == k]
             for k in ("genuine", "voided", "tracklet")}
    print(f"  data      {data}")
    print("  events    " + ", ".join(f"{k} {len(v)}" for k, v in kinds.items())
          + (f"  ({len(drop)} excluded by hand)" if drop else ""))
    print(f"  unknowns  {len(unknowns)}")
    print(f"  gallery   {len(GT)} photographs / {len(enrolled)} people "
          f"({len(skipped)} skipped: {sorted({x[2] for x in skipped})})\n")

    why_t = np.stack([load_crop(e["why"]) for e in events])
    evt_t = np.stack([load_crop(e["evt"]) for e in events])
    un_t = np.stack([load_crop(u["path"]) for u in unknowns])
    lab_all = np.array([col[e["emp"]] for e in events], np.int64)

    report = {"data": str(data), "gallery_rows": int(len(GT)),
              "gallery_people": int(len(enrolled)), "excluded": sorted(drop),
              "models": {}}
    target_far = None
    keep = {}

    for mi, m in enumerate(models):
        t0 = time.time()
        G = m.embed(GT)
        PW = per_person(m.embed(why_t) @ G.T, owners, enrolled)
        PV = per_person(m.embed(evt_t) @ G.T, owners, enrolled)
        PU = per_person(m.embed(un_t) @ G.T, owners, enrolled)
        took = time.time() - t0

        S = G @ G.T
        same = owners[:, None] == owners[None, :]
        np.fill_diagonal(same, False)
        other = owners[:, None] != owners[None, :]
        gal = {"dprime": dprime(S[same], S[other]),
               "worst_genuine": float(S[same].min()),
               "best_impostor": float(S[other].max())}

        gi = np.array(kinds["genuine"], np.int64)
        gl = lab_all[gi]

        def scores(P):
            g = P[gi, gl]
            mask = np.ones_like(P[gi], bool)
            mask[np.arange(len(gi)), gl] = False
            pairs = P[gi][mask]                       # every non-owner pair
            other_max = np.where(mask, P[gi], -2).max(axis=1)
            return g, pairs, other_max

        gw, pw, ow = scores(PW)        # clearest frame: the fair set
        gv, pv, ov = scores(PV)        # top-scoring frame: current model's argmax

        # One operating point for everybody: the pair FAR the reference model
        # actually runs at in production, measured on the fair set.
        if mi == 0:
            target_far = float(np.mean(pw >= args.reference_threshold))
        thr = float(np.quantile(pw, 1.0 - target_far)) if target_far > 0 else float(pw.max()) + 1e-3

        def ident(P, idx, t):
            lab = lab_all[idx]
            return (P[idx, lab] >= t) & (P[idx].argmax(axis=1) == lab)

        r = {
            "embed_seconds": round(took, 1),
            "gallery": gal,
            "clearest_frame": {
                "probes": int(len(gi)),
                "rank1": float(np.mean(PW[gi].argmax(axis=1) == gl)),
                "genuine_median": float(np.median(gw)),
                "genuine_p25": float(np.quantile(gw, .25)),
                "impostor_pair_p9999": float(np.quantile(pw, .9999)),
                "impostor_pair_p999": float(np.quantile(pw, .999)),
                "dprime": dprime(gw, pw),
                **{f"tar_at_pair_far_{f:g}": float(np.mean(ident(PW, gi, float(np.quantile(pw, 1 - f)))))
                   for f in (1e-4, 1e-3, 1e-2)},
            },
            "top_scoring_frame": {
                "rank1": float(np.mean(PV[gi].argmax(axis=1) == gl)),
                "genuine_median": float(np.median(gv)),
                "dprime": dprime(gv, pv),
            },
            "operating_point": {
                "pair_far": target_far,
                "threshold": thr,
                "per_probe_false_name_rate": float(np.mean(ow >= thr)),
                "identified_clearest": float(np.mean(ident(PW, gi, thr))),
                "identified_top_scoring": float(np.mean(ident(PV, gi, thr))),
                "identified_best_of_two": float(np.mean(ident(PW, gi, thr) | ident(PV, gi, thr))),
            },
        }
        vi = np.array(kinds["voided"], np.int64)
        v_best = np.maximum(PW[vi, lab_all[vi]], PV[vi, lab_all[vi]])
        r["voided"] = {"total": int(len(vi)),
                       "still_named": int(np.sum(ident(PW, vi, thr) | ident(PV, vi, thr))),
                       "score_vs_wrong_name": [round(float(x), 3) for x in v_best]}
        ti = np.array(kinds["tracklet"], np.int64)
        r["tracklet"] = {"total": int(len(ti)),
                         "named": int(np.sum(ident(PW, ti, thr) | ident(PV, ti, thr))),
                         "score": [round(float(x), 3) for x in
                                   np.maximum(PW[ti, lab_all[ti]], PV[ti, lab_all[ti]])]}
        u_best, u_who = PU.max(axis=1), enrolled[PU.argmax(axis=1)]
        r["unknown"] = {"total": int(len(unknowns)), "named": int(np.sum(u_best >= thr)),
                        "rate": float(np.mean(u_best >= thr))}
        report["models"][m.name] = r
        keep[m.name] = {"thr": thr, "u_best": u_best, "u_who": u_who,
                        "PW": PW, "ow": ow, "gi": gi}

        c, o = r["clearest_frame"], r["operating_point"]
        print(f"  == {m.name}   ({took:.0f}s)")
        print(f"     enrolment set     d' {gal['dprime']:.2f}   (saturated - for the record only)")
        print(f"     clearest frame    rank-1 {c['rank1']:.4f}  d' {c['dprime']:.2f}  genuine median {c['genuine_median']:.3f}")
        print(f"                       TAR @pairFAR 1e-4 {c['tar_at_pair_far_0.0001']:.3f}   1e-3 {c['tar_at_pair_far_0.001']:.3f}   1e-2 {c['tar_at_pair_far_0.01']:.3f}")
        print(f"     top-scoring frame rank-1 {r['top_scoring_frame']['rank1']:.4f}  d' {r['top_scoring_frame']['dprime']:.2f}")
        print(f"     at pair FAR {target_far:.2e}  threshold {thr:.3f}")
        print(f"        identified   clearest {o['identified_clearest']:.3f}   top-scoring {o['identified_top_scoring']:.3f}   either {o['identified_best_of_two']:.3f}")
        print(f"        false names  {o['per_probe_false_name_rate']:.4f} of genuine probes name someone else above threshold")
        print(f"        voided       {r['voided']['still_named']}/{r['voided']['total']} still named")
        print(f"        tracklet     {r['tracklet']['named']}/{r['tracklet']['total']} recovered (passes the face path missed)")
        print(f"        unknowns     {r['unknown']['named']}/{r['unknown']['total']} named ({100 * r['unknown']['rate']:.2f}%)\n")

    if len(models) == 2:
        a, b = (keep[m.name] for m in models)
        an, bn = a["u_best"] >= a["thr"], b["u_best"] >= b["thr"]
        both = np.where(an & bn)[0]
        report["unknown_overlap"] = {
            "both": int(len(both)),
            "both_same_person": int(np.sum(a["u_who"][both] == b["u_who"][both])),
            "only_first": int(np.sum(an & ~bn)), "only_second": int(np.sum(bn & ~an))}
        u = report["unknown_overlap"]
        print(f"  unknowns named by both {u['both']} (same person {u['both_same_person']}), "
              f"only {models[0].name} {u['only_first']}, only {models[1].name} {u['only_second']}")
        report["contact_sheets"] = {
            tag: contact_sheet(out / f"unknown_{tag}.jpg", np.where(mask)[0], k["u_who"],
                               k["u_best"], unknowns, owners, GT, people)
            for tag, mask, k in (("only_" + Path(models[1].name).stem, bn & ~an, b),
                                 ("only_" + Path(models[0].name).stem, an & ~bn, a))}
        # The strongest "impostors" in the genuine set, for both models: these
        # are the probes to look at before trusting any FAR - a tracker swap or
        # a wrong committed name looks exactly like a model error.
        worst = np.argsort(-np.maximum(a["ow"], b["ow"]))[:40]
        report["impostor_tail_events"] = [int(events[a["gi"][i]]["id"]) for i in worst]
        tail_sheet(out / "impostor_tail.jpg", worst, a, b, events, lab_all, owners,
                   enrolled, GT, people)

    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"\n  report -> {out / 'report.json'}")
    return 0


def tail_sheet(path, worst, a, b, events, lab_all, owners, enrolled, GT, people):
    """probe | labelled person | the other person it matched, per tail probe."""
    import cv2
    from app.core.geometry import aligned_to_uint8

    def ref(emp):
        row = int(np.where(owners == emp)[0][0])
        return cv2.cvtColor(aligned_to_uint8(GT[row]), cv2.COLOR_RGB2BGR)

    tiles = []
    for i in worst:
        e = events[a["gi"][i]]
        P = a["PW"][a["gi"][i]].copy()
        P[lab_all[a["gi"][i]]] = -2
        other_emp = int(enrolled[P.argmax()])
        trio = np.concatenate([cv2.imread(str(e["why"])), ref(e["emp"]), ref(other_emp)], axis=1)
        canvas = np.full((150, 336, 3), 255, np.uint8)
        canvas[:112] = trio
        cv2.putText(canvas, f"ev{e['id']} {people[e['emp']][0][:16]}", (2, 126),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"vs {people[other_emp][0][:16]}  A{a['ow'][i]:.2f} B{b['ow'][i]:.2f}",
                    (2, 143), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 160), 1, cv2.LINE_AA)
        tiles.append(canvas)
    cols = 5
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.concatenate(tiles[r:r + cols], axis=1) for r in range(0, len(tiles), cols)]
    cv2.imwrite(str(path), np.concatenate(rows, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 90])


def contact_sheet(path: Path, idx, who, best, unknowns, owners, GT, people, limit=120):
    """Probe crop beside the enrolment photo it was matched to, highest first."""
    import cv2
    from app.core.geometry import aligned_to_uint8
    order = sorted(idx.tolist(), key=lambda i: -best[i])[:limit]
    if not order:
        return None
    tiles = []
    for i in order:
        probe = cv2.imread(str(unknowns[i]["path"]))
        ref_row = int(np.where(owners == who[i])[0][0])
        ref = cv2.cvtColor(aligned_to_uint8(GT[ref_row]), cv2.COLOR_RGB2BGR)
        pair = np.concatenate([probe, ref], axis=1)
        pair = cv2.resize(pair, (224, 112))
        canvas = np.full((140, 224, 3), 255, np.uint8)
        canvas[:112] = pair
        name = people.get(int(who[i]), ("?", ""))[0][:22]
        cv2.putText(canvas, f"{best[i]:.3f} {name}", (3, 131),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"u{unknowns[i]['id']}", (180, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(canvas)
    cols = 8
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.concatenate(tiles[r:r + cols], axis=1) for r in range(0, len(tiles), cols)]
    cv2.imwrite(str(path), np.concatenate(rows, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
