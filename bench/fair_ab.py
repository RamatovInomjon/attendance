#!/usr/bin/env python3
"""A fair A/B of two recognizers on native-4K corridor faces.

    python bench/fair_ab.py --faces /media/inomjon/T7/face_eval_20260915 \
        --prod data/gpu6_eval_20260915

Consumes `bench/extract_native_faces.py` output. Written after
`compare_recognizers.py` was challenged as unfair, and it answers the three
ways that test could have favoured the deployed model:

ALIGNMENT   The challenger was trained on another aligner. Every face here
            carries three alignments of the SAME native crop, and each model is
            scored under each, against a gallery aligned the same way.

LABELS      Which track is which person is decided WITHOUT a recognizer: a
            replayed face is matched to the crop production saved by normalized
            cross-correlation of pixels. Same frame, same aligner, same bitstream
            gives NCC ~1.0; nothing else comes close.

SELECTION   Those labels still exist only for passes the deployed model named.
            So a label-free verification set is reported too: two faces from
            the same track are the same person, two faces from different tracks
            in the SAME FRAME are different people. It includes every unnamed
            pass and depends on no model and no production decision at all.
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

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ALIGN003 = Path("/home/inomjon/projectAI/1/data_aug/vb30k")

MODELS = {"IR101": "adaface_ir101_finetune_fp16.onnx",
          "IR50": "ir50S3v2s_sr10final.onnx",
          "IR101new": "ir101S3v2s_sr10final.onnx"}
CONDITIONS = [                       # (label, model, alignment)
    ("C1 IR101 pipe", "IR101", "pipe"),
    ("C2 IR50 pipe", "IR50", "pipe"),
    ("C3 IR50 d50", "IR50", "d50"),
    ("C4 IR50 r003", "IR50", "r003"),
    ("C5 IR101 d50", "IR101", "d50"),
    ("C6 IR101new pipe", "IR101new", "pipe"),
    ("C7 IR101new d50", "IR101new", "d50"),
    ("C8 IR101new r003", "IR101new", "r003"),
]
# Which conditions take turns choosing the hardest impostor samples, so no one
# model picks the set: the deployed model and each challenger, drop-in and at
# its own training alignment.
SAMPLE_PICKERS = ("C1 IR101 pipe", "C2 IR50 pipe", "C4 IR50 r003",
                  "C6 IR101new pipe", "C8 IR101new r003")
NCC_MIN, NCC_MARGIN = 0.93, 0.12
PER_TRACK = 20
MAX_IMPOSTOR_PAIRS = 60_000


def to_chw(u8_rgb: np.ndarray) -> np.ndarray:
    return (np.asarray(u8_rgb, np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2)


def ncc(a, B):
    a = a.astype(np.float32).ravel()
    a = (a - a.mean()) / (a.std() + 1e-6)
    B = B.reshape(len(B), -1).astype(np.float32)
    B = (B - B.mean(1, keepdims=True)) / (B.std(1, keepdims=True) + 1e-6)
    return B @ a / a.size


class Faces:
    """Metadata for every extracted face; pixels loaded per file, on demand.

    A busy 5-minute segment holds thousands of faces in three alignments, and
    holding every segment's pixels at once does not fit in this machine's RAM.
    """
    META = ("t", "frame", "track", "face_px", "r003_ok", "box")

    def __init__(self, faces_dir: Path):
        self.files = [f for f in sorted(faces_dir.glob("*.npz"))]
        parts = defaultdict(list)
        for fi, f in enumerate(self.files):
            with np.load(f) as z:
                n = len(z["t"])
                if not n:
                    continue
                for k in self.META:
                    parts[k].append(z[k])
                parts["cam"].append(np.array([str(z["camera"])] * n))
                parts["file"].append(np.full(n, fi))
                parts["row"].append(np.arange(n))
                parts["seg"].append(np.array([f.stem] * n))
        self.m = {k: np.concatenate(v) for k, v in parts.items()}
        # Faces the full 003 aligner could not place were stored as BLACK
        # crops. Two black crops embed identically, so they land as impostor
        # pairs at cosine 1.0 and destroy the 1e-3 operating point of any
        # condition that reads them - and they drag that condition's genuine
        # scores to zero as well. Drop them everywhere, so all conditions are
        # judged on exactly the same faces (2.06% of a full day).
        keep = self.m["r003_ok"].astype(bool)
        self.dropped_r003 = int((~keep).sum())
        self.m = {k: v[keep] for k, v in self.m.items()}
        self.m["row"] = self.m["row"]        # row stays the row WITHIN its npz
        self.m["tkey"] = np.array([f"{s}#{t}" for s, t in zip(self.m["seg"], self.m["track"])])
        self.m["skey"] = self._segments()
        self._cache = (None, None)

    def _segments(self, min_iou: float = 0.3) -> np.ndarray:
        """Split each tracker id wherever the head box jumps.

        The replay tracker follows PERSON boxes, and two people walking
        together swap heads inside them: Exit_20260907_114536#39 is five
        frames of Xamdamov Rustam and then 145 of a woman in glasses. One
        person's head overlaps itself frame to frame at IoU 0.7-0.97; every
        switch examined sat below 0.3. A label then covers only the segment
        its matched frame is in. Pure geometry - no recognizer involved.
        """
        m = self.m
        out = np.empty(len(m["t"]), object)
        for tk in np.unique(m["tkey"]):
            idx = np.where(m["tkey"] == tk)[0]
            idx = idx[np.argsort(m["frame"][idx])]
            seg, prev = 0, None
            for i in idx:
                if prev is not None:
                    a, b = m["box"][prev], m["box"][i]
                    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
                    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
                    inter = ix * iy
                    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
                    if inter / (u + 1e-6) < min_iou:
                        seg += 1
                out[i] = f"{tk}/{seg}"
                prev = i
        return out.astype(str)

    def pixels(self, key: str, idx: np.ndarray) -> np.ndarray:
        out = np.empty((len(idx), 112, 112, 3), np.uint8)
        order = np.argsort(self.m["file"][idx], kind="stable")
        for fi in np.unique(self.m["file"][idx]):
            sel = order[self.m["file"][idx][order] == fi]
            cached_key, arr = self._cache
            if cached_key != (fi, key):
                with np.load(self.files[fi]) as z:
                    arr = z[key]
                self._cache = ((fi, key), arr)
            out[sel] = arr[self.m["row"][idx[sel]]]
        return out


def label_tracks(FC, prod: Path):
    """production pass -> replay track, by pixels alone."""
    db = sqlite3.connect(f"file:{prod / 'ematsy.db'}?mode=ro", uri=True)
    cam_id = {n: i for i, n in db.execute("select id, name from camera")}
    snaps = defaultdict(list)
    for f in (prod / "snapshots").iterdir():
        m = re.match(r"(evt|why|unknown)_(\d+)_(\d+)_(-?\d+)_(\d+)\.jpg$", f.name)
        if m:
            snaps[(m[1], int(m[2]), int(m[3]), int(m[4]))].append((int(m[5]), f))

    def ep(s):
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()

    def crop_files(kind, emp, cam, track, snap):
        e = int(re.search(r"_(\d+)\.jpg$", snap).group(1)) if snap else 0
        return [f for x, f in snaps.get((kind, emp, cam, track), []) if abs(x - e) <= 3]

    passes = []
    for eid, emp, cam, track, ts, snap, voided, source in db.execute(
            "select id, employee_id, camera_id, track_id, ts, snapshot, "
            "voided_at is not null, source from recognition_event "
            "where business_date = '2026-09-07'"):
        span = db.execute(
            "select first_seen, last_seen from reid_pass where camera_id=? and track_id=? "
            "and business_date='2026-09-07' "
            "order by abs(julianday(last_seen) - julianday(?)) limit 1",
            (cam, track, ts)).fetchone()
        s0, s1 = (ep(span[0]), ep(span[1])) if span else (ep(ts) - 60, ep(ts))
        kind = "voided" if voided else ("tracklet" if source == "tracklet" else "genuine")
        # evt_ ONLY. It is the frame that produced the committed score, so it
        # shows the named person by definition. why_ is merely the clearest
        # frame of the pass, and when production's own tracker swapped heads it
        # is somebody else: event 988, committed as Xamdamov Rustam, saved a
        # why_ crop of the woman beside him, and labelling from it charged a
        # model that recognised her with a wrong name.
        files = crop_files("evt", emp, cam, track, snap)
        passes.append(("event", eid, emp, kind, cam, s0, s1, files))
    for uid, cam, track, fs, ls, snap in db.execute(
            "select id, camera_id, track_id, first_seen, last_seen, snapshot "
            "from unknown_sighting where business_date = '2026-09-07'"):
        passes.append(("unknown", uid, 0, "unknown", cam, ep(fs), ep(ls),
                       crop_files("unknown", 0, cam, track, snap)))

    names = {v: k for k, v in cam_id.items()}
    labels = defaultdict(set)
    matched = Counter()
    M = FC.m
    passes.sort(key=lambda p: (p[4], p[5]))          # camera, time: keeps the file cache warm
    for src, pid, emp, kind, cam, s0, s1, files in passes:
        cand = np.where((M["cam"] == names[cam]) & (M["t"] >= s0 - 4) & (M["t"] <= s1 + 4))[0]
        if not len(cand) or not files:
            continue
        pix = FC.pixels("pipe", cand)
        best = None
        for f in files:
            probe = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            v = ncc(probe, pix)
            i = int(np.argmax(v))
            others = v[M["tkey"][cand] != M["tkey"][cand[i]]]
            margin = float(v[i] - (others.max() if len(others) else -1))
            if best is None or v[i] > best[0]:
                best = (float(v[i]), cand[i], margin)
        matched["candidates"] += 1
        if best[0] >= NCC_MIN and best[2] >= NCC_MARGIN:
            labels[M["skey"][best[1]]].add((kind, emp, src, pid))
            matched[kind] += 1

    track_label, dropped = {}, 0
    for tk, labs in labels.items():
        ids = {(k if k != "genuine" else "genuine", e) for k, e, _s, _p in labs}
        if len(ids) > 1:
            dropped += 1                     # one replay track, two production identities
            continue
        (k, e), = ids
        track_label[tk] = (k, e)
    return track_label, matched, dropped


def build_galleries(prod: Path):
    """Enrolment photographs, aligned each of the three ways from one crop."""
    from app.config import settings
    from app.core.aligner import FaceAligner
    from app.core.detector import build_detector
    from app.core.onnx_env import best_providers
    sys.path.insert(0, str(ALIGN003))
    from gpu_align003 import Pipeline003
    from gpu_align import align112

    db = sqlite3.connect(f"file:{prod / 'ematsy.db'}?mode=ro", uri=True)
    people = {i: (n, f) for i, n, f in db.execute(
        "select id, full_name, folder from employee where is_active = 1")}
    enrol = list(db.execute("select employee_id, source_file from face_embedding "
                            "where source_file not like 'live:%'"))
    det = build_detector(settings.detector_kind, settings.model_path(settings.detector_model),
                         imgsz=settings.detect_width, conf=settings.detect_conf)
    ali = FaceAligner(settings.model_path(settings.aligner_model),
                      crop_size=settings.align_crop_size,
                      margin=settings.align_margin, mode=settings.align_mode)
    p003 = Pipeline003(str(ALIGN003 / "cfg003"), best_providers())
    out = {"pipe": [], "d50": [], "r003": []}
    owners, r003_fail = [], 0
    for emp, src in enrol:
        if emp not in people or not people[emp][1]:
            continue
        bgr = cv2.imread(str(settings.root / "face_id_users" / people[emp][1] / Path(src).name))
        if bgr is None:
            continue
        dets = det.detect(bgr)
        if not dets:
            continue
        box = np.asarray(max(dets, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1])).box,
                         np.float32)
        faces = ali.align(bgr, [box], is_bgr=True)
        if not faces:
            continue
        native, _ = ali._cut(bgr, box)
        pts, _s = p003.dfa_landmarks(native)
        d50 = cv2.cvtColor(align112(native, pts), cv2.COLOR_BGR2RGB)
        r003, _a, _b = p003.align(native)
        if r003 is None:
            r003_fail += 1
            r003 = d50
        else:
            r003 = cv2.cvtColor(r003, cv2.COLOR_BGR2RGB)
        from app.core.geometry import aligned_to_uint8
        out["pipe"].append(aligned_to_uint8(faces[0].aligned))
        out["d50"].append(d50)
        out["r003"].append(r003)
        owners.append(emp)
    return {k: np.stack(v) for k, v in out.items()}, np.array(owners), people, r003_fail


def per_person(S, owners, ids):
    out = np.full((S.shape[0], len(ids)), -2.0, np.float32)
    for j, p in enumerate(ids):
        out[:, j] = S[:, owners == p].max(axis=1)
    return out


def bootstrap_tar(tracks, faces_of, g, ok, nonowner, fars, reps, rng):
    """Paired bootstrap over TRACKS: frames of one pass are not independent."""
    T = len(tracks)
    out = np.zeros((reps, len(fars)))
    for r in range(reps):
        pick = rng.integers(0, T, T)
        idx = np.concatenate([faces_of[tracks[k]] for k in pick])
        pool = nonowner[idx].ravel()
        for j, f in enumerate(fars):
            k = max(1, int(round(f * len(pool))))
            t = np.partition(pool, len(pool) - k)[len(pool) - k]
            out[r, j] = np.mean((g[idx] >= t) & ok[idx])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", default="/media/inomjon/T7/face_eval_20260915")
    ap.add_argument("--prod", default="data/gpu6_eval_20260915")
    ap.add_argument("--out", default="data/bench/fair_ab")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--reps", type=int, default=1000)
    args = ap.parse_args()

    from app.config import settings
    from app.core.onnx_env import preload_cuda_libs
    preload_cuda_libs()
    from app.core.recognizer import FaceRecognizer

    prod = (ROOT / args.prod).resolve()
    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    FC = Faces(Path(args.faces))
    track_label, matched, dropped = label_tracks(FC, prod)
    F = FC.m
    N = len(F["t"])
    print(f"  dropped {FC.dropped_r003:,} faces the 003 aligner could not place (black crops)")
    print(f"  faces {N:,} in {len(set(F['tkey'])):,} tracks / {len(set(F['skey'])):,} head-continuous segments; matched "
          f"by pixels: {dict(matched)}; {dropped} tracks dropped for two identities")

    gal, owners, people, r003_fail = build_galleries(prod)
    ids = np.array(sorted(set(owners.tolist())))
    col = {int(p): j for j, p in enumerate(ids)}
    print(f"  gallery {len(owners)} photographs / {len(ids)} people "
          f"(full 003 aligner failed on {r003_fail})")

    by_track_g = defaultdict(list)
    for i, sk in enumerate(F["skey"]):
        by_track_g[sk].append(i)
    by_track_g = {k: np.array(v) for k, v in by_track_g.items() if len(v) >= 3}

    # Association glitches, removed WITHOUT looking at any face: a frame whose
    # head centre sits far from the median of its neighbours in the same track
    # is another person's head picked up by the person box.
    glitch = np.zeros(N, bool)
    cx = (F["box"][:, 0] + F["box"][:, 2]) / 2
    cy = (F["box"][:, 1] + F["box"][:, 3]) / 2
    size = np.maximum(F["box"][:, 2] - F["box"][:, 0], F["box"][:, 3] - F["box"][:, 1])
    for tk, idx in by_track_g.items():
        idx = idx[np.argsort(F["frame"][idx])]
        for k, i in enumerate(idx):
            nb = idx[max(0, k - 3):k + 4]
            nb = nb[nb != i]
            if len(nb) and np.hypot(cx[i] - np.median(cx[nb]), cy[i] - np.median(cy[nb])) > 0.75 * size[i]:
                glitch[i] = True
        by_track_g[tk] = idx[~glitch[idx]]
    by_track_g = {k: v for k, v in by_track_g.items() if len(v) >= 3}
    print(f"  association glitches removed: {int(glitch.sum())} frames")

    # ---- decide WHICH faces get embedded, before touching any pixels -------
    # A full day is ~290k faces; embedding all of them under every condition
    # does not fit in this machine's RAM. Gallery metrics use at most
    # PER_TRACK evenly spaced frames per pass (a loiterer must not outvote
    # everyone), and the label-free pairs are drawn from the metadata FIRST so
    # the frames they need are embedded even when the cap would have dropped
    # them - capping before pairing silently empties the impostor set, because
    # two passes rarely survive the cap in the same frame.
    capped = []
    for tk, idx in by_track_g.items():
        if len(idx) > PER_TRACK:
            idx = idx[np.linspace(0, len(idx) - 1, PER_TRACK).astype(int)]
        capped.extend(idx.tolist())
    capped = np.array(sorted(capped))

    gp_g, gp_cluster = [], []
    for ti, (tk, idx) in enumerate(by_track_g.items()):
        for _ in range(min(20, len(idx))):
            a, b = rng.choice(idx, 2, replace=False)
            if abs(F["t"][a] - F["t"][b]) >= 1.0:
                gp_g.append((a, b))
                gp_cluster.append(ti)
    tindex = {sk: i for i, sk in enumerate(by_track_g)}
    by_frame = defaultdict(list)
    for tk, idx in by_track_g.items():
        for i in idx:
            by_frame[(F["seg"][i], int(F["frame"][i]))].append(int(i))

    def iou(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-6)

    ip_g, ip_cluster = [], []
    for idx in by_frame.values():
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                i, j = idx[x], idx[y]
                if F["tkey"][i] != F["tkey"][j] and iou(F["box"][i], F["box"][j]) < 0.05:
                    ip_g.append((i, j))
                    ip_cluster.append(tuple(sorted((tindex[F["skey"][i]], tindex[F["skey"][j]]))))
    if len(ip_g) > MAX_IMPOSTOR_PAIRS:
        pick = rng.choice(len(ip_g), MAX_IMPOSTOR_PAIRS, replace=False)
        ip_g = [ip_g[k] for k in pick]
        ip_cluster = [ip_cluster[k] for k in pick]

    need = set(capped.tolist())
    for a, b in gp_g:
        need.add(int(a)); need.add(int(b))
    for a, b in ip_g:
        need.add(int(a)); need.add(int(b))
    sel = np.array(sorted(need))
    loc = {int(g): k for k, g in enumerate(sel)}
    print(f"  embedding {len(sel):,} of {N:,} faces "
          f"({len(capped):,} capped + label-free pair endpoints)")

    # everything below indexes the embedded subset
    F = {k: v[sel] for k, v in F.items()}
    by_track = {tk: np.array([loc[int(i)] for i in idx if int(i) in loc])
                for tk, idx in by_track_g.items()}
    by_track = {k: v for k, v in by_track.items() if len(v) >= 3}
    kind_of = np.array([track_label.get(sk, ("unlabelled", 0))[0] if sk in by_track else "dropped"
                        for sk in F["skey"]])
    emp_of = np.array([track_label.get(sk, ("", 0))[1] for sk in F["skey"]])
    capped_l = np.array([loc[int(i)] for i in capped])
    gen = capped_l[(kind_of[capped_l] == "genuine") & np.isin(emp_of[capped_l], ids)]
    gtracks = sorted(set(F["skey"][gen]))
    unk_tracks = sorted(tk for tk, (k, _e) in track_label.items()
                        if k == "unknown" and tk in by_track)
    gp = np.array([[loc[int(a)], loc[int(b)]] for a, b in gp_g])
    ip = np.array([[loc[int(a)], loc[int(b)]] for a, b in ip_g])
    gp_cluster = np.array(gp_cluster)
    clusters = sorted(set(ip_cluster))
    cmap = {c: k for k, c in enumerate(clusters)}
    ip_cluster = np.array([cmap[c] for c in ip_cluster])
    print(f"  labelled genuine: {len(gen):,} frames in {len(gtracks)} passes, "
          f"{len(set(emp_of[gen]))} people; unknown passes {len(unk_tracks)}")
    print(f"  label-free: {len(gp):,} same-pass pairs from {len(set(gp_cluster))} passes; "
          f"{len(ip):,} same-frame pairs from {len(clusters)} pairs of passes\n")

    # ---- embed, one video file at a time ----------------------------------
    # File-major: each npz array is decompressed once and every model that
    # needs that alignment reads it, instead of one full pass per condition.
    # Batch 16, not 64: all three models stay resident so each file's pixels
    # are read once, and three CUDA arenas sized for batch 64 do not fit in an
    # 8 GB card (the IR-101s alone ask for a 196 MB convolution buffer).
    recs = {m: FaceRecognizer(settings.model_path(f), batch_size=16) for m, f in MODELS.items()}
    needed = defaultdict(list)                     # alignment -> models
    for _name, m, a in CONDITIONS:
        needed[a].append(m)
    emb = {(m, a): np.zeros((len(sel), 512), np.float32)
           for a, ms in needed.items() for m in ms}
    for fi in range(len(FC.files)):
        rows = np.where(F["file"] == fi)[0]
        if not len(rows):
            continue
        for a, ms in needed.items():
            pix = FC.pixels(a, sel[rows])
            for m in ms:
                for c in range(0, len(rows), 1024):
                    emb[(m, a)][rows[c:c + 1024]] = recs[m].embed(to_chw(pix[c:c + 1024]))
            del pix
        FC._cache = (None, None)
    gal_emb = {(m, a): recs[m].embed(to_chw(gal[a])) for a, ms in needed.items() for m in ms}
    emb = {(m, a): (emb[(m, a)], gal_emb[(m, a)]) for (m, a) in emb}
    del recs

    vote_min = int(settings.vote_min_recognitions)
    vote_cons = float(settings.vote_consensus)
    report = {"faces": N, "matched": dict(matched), "gallery_people": int(len(ids)),
              "labelled_passes": len(gtracks), "unknown_passes": len(unk_tracks),
              "conditions": {}, "differences": {}, "samples": []}
    boot, table = {}, {}
    FARS = (1e-4, 1e-3, 1e-2)
    lab = np.array([col[int(e)] for e in emp_of[gen]])
    gpos = {int(f): k for k, f in enumerate(gen)}
    faces_of_gen = {tk: np.array([gpos[int(i)] for i in by_track[tk] if int(i) in gpos])
                    for tk in gtracks}

    for name, m, a in CONDITIONS:
        E, G = emb[(m, a)]
        P = per_person(E @ G.T, owners, ids)
        Pg = P[gen]
        g = Pg[np.arange(len(gen)), lab]
        ok = Pg.argmax(1) == lab
        NO = Pg.copy()
        NO[np.arange(len(gen)), lab] = np.nan
        NO = NO[~np.isnan(NO)].reshape(len(gen), len(ids) - 1)
        pool = NO.ravel()
        thr = {f: float(np.quantile(pool, 1 - f)) for f in FARS}
        r = {"rank1": float(ok.mean()),
             **{f"tar@{f:g}": float(np.mean((g >= t) & ok)) for f, t in thr.items()},
             **{f"thr@{f:g}": t for f, t in thr.items()}}
        B = bootstrap_tar(gtracks, faces_of_gen, g, ok, NO, FARS, args.reps, np.random.default_rng(args.seed))
        boot[name] = B
        for j, f in enumerate(FARS):
            r[f"tar@{f:g}_ci95"] = [float(np.quantile(B[:, j], .025)), float(np.quantile(B[:, j], .975))]

        # Pass level, production's consensus rule over every frame of the pass.
        t = thr[1e-3]
        def name_pass(tk):
            idx = by_track[tk]
            best = P[idx].argmax(1)
            above = best[P[idx, best] >= t]
            if len(above) < vote_min:
                return None
            who, n = Counter(above.tolist()).most_common(1)[0]
            return int(ids[who]) if n >= vote_min and n / len(above) >= vote_cons else None
        truth = {tk: int(emp_of[by_track[tk][0]]) for tk in gtracks}
        named = {tk: name_pass(tk) for tk in gtracks}
        r["passes_right"] = sum(named[tk] == truth[tk] for tk in gtracks)
        r["passes_wrong"] = sum(named[tk] not in (None, truth[tk]) for tk in gtracks)
        r["passes_missed"] = sum(named[tk] is None for tk in gtracks)
        r["passes_wrong_detail"] = [
            {"segment": tk, "truth": people[truth[tk]][0], "named": people[named[tk]][0],
             "frames": int(len(by_track[tk]))}
            for tk in gtracks if named[tk] not in (None, truth[tk])]
        r["unknown_named"] = {tk: name_pass(tk) for tk in unk_tracks}
        r["unknown_named_count"] = sum(v is not None for v in r["unknown_named"].values())

        gs = np.sum(E[gp[:, 0]] * E[gp[:, 1]], 1)
        iss = np.sum(E[ip[:, 0]] * E[ip[:, 1]], 1)
        for f in (1e-3, 1e-2):
            tt = float(np.quantile(iss, 1 - f))
            r[f"verify_tar@{f:g}"] = float(np.mean(gs >= tt))
        # cluster bootstrap for the label-free numbers
        vb = np.zeros((args.reps, 2))
        rr = np.random.default_rng(args.seed + 1)
        gtr = np.unique(gp_cluster)
        g_by = {c: np.where(gp_cluster == c)[0] for c in gtr}
        i_by = [np.where(ip_cluster == c)[0] for c in range(len(clusters))]
        for k in range(args.reps):
            gi = np.concatenate([g_by[c] for c in rr.choice(gtr, len(gtr))])
            ii = np.concatenate([i_by[c] for c in rr.integers(0, len(clusters), len(clusters))])
            for j, f in enumerate((1e-3, 1e-2)):
                vb[k, j] = np.mean(gs[gi] >= np.quantile(iss[ii], 1 - f))
        boot[name + " verify"] = vb
        r["verify_tar@0.001_ci95"] = [float(np.quantile(vb[:, 0], .025)), float(np.quantile(vb[:, 0], .975))]
        r["verify_tar@0.01_ci95"] = [float(np.quantile(vb[:, 1], .025)), float(np.quantile(vb[:, 1], .975))]
        table[name] = (P, thr)
        report["conditions"][name] = {k: v for k, v in r.items() if k != "unknown_named"}
        report["conditions"][name]["unknown_named"] = {k: v for k, v in r["unknown_named"].items() if v}
        ci = lambda key: "[%.3f, %.3f]" % tuple(r[key])
        print(f"  {name:14s} rank-1 {r['rank1']:.3f}   TAR @1e-4 {r['tar@0.0001']:.3f} {ci('tar@0.0001_ci95')}"
              f"   @1e-3 {r['tar@0.001']:.3f} {ci('tar@0.001_ci95')}   @1e-2 {r['tar@0.01']:.3f} {ci('tar@0.01_ci95')}")
        print(f"  {'':14s} passes right {r['passes_right']}/{len(gtracks)} wrong {r['passes_wrong']} "
              f"missed {r['passes_missed']}   unknown passes named {r['unknown_named_count']}/{len(unk_tracks)}"
              f"   label-free TAR @1e-3 {r['verify_tar@0.001']:.3f} {ci('verify_tar@0.001_ci95')}"
              f"  @1e-2 {r['verify_tar@0.01']:.3f} {ci('verify_tar@0.01_ci95')}")

    print("\n  paired differences (bootstrap over passes; CI excludes 0 = a real difference)")
    for x, y in (("C1 IR101 pipe", "C2 IR50 pipe"), ("C1 IR101 pipe", "C4 IR50 r003"),
                 ("C1 IR101 pipe", "C6 IR101new pipe"), ("C1 IR101 pipe", "C8 IR101new r003"),
                 ("C6 IR101new pipe", "C2 IR50 pipe"), ("C8 IR101new r003", "C4 IR50 r003"),
                 ("C7 IR101new d50", "C5 IR101 d50"), ("C5 IR101 d50", "C3 IR50 d50"),
                 ("C6 IR101new pipe", "C8 IR101new r003")):
        d = boot[x] - boot[y]
        dv = boot[x + " verify"] - boot[y + " verify"]
        cells = [f"@{f:g} {np.mean(d[:, j]):+.3f} [{np.quantile(d[:, j], .025):+.3f}, {np.quantile(d[:, j], .975):+.3f}]"
                 for j, f in enumerate(FARS)]
        cells.append(f"label-free@1e-2 {np.mean(dv[:, 1]):+.3f} [{np.quantile(dv[:, 1], .025):+.3f}, {np.quantile(dv[:, 1], .975):+.3f}]")
        report["differences"][f"{x} - {y}"] = cells
        print(f"  {x} minus {y}")
        print("     " + "   ".join(cells))

    samples(out, FC, sel, gal, owners, ids, col, gen, emp_of, table, people, rng, report)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"\n  report -> {out / 'report.json'}")
    return 0


def samples(out, FC, sel, gal, owners, ids, col, gen, emp_of, table, people, rng, report):
    """10 genuine + 10 impostor pairs, every condition's verdict on each."""
    F = FC.m
    lab = np.array([col[int(e)] for e in emp_of[gen]])

    # GENUINE: 10 different people, one random frame each. No selection on any
    # model's score, in either direction.
    gsel, seen = [], set()
    for i in rng.permutation(len(gen)):
        e = int(emp_of[gen[i]])
        if e not in seen:
            gsel.append((int(i), int(lab[i])))
            seen.add(e)
        if len(gsel) == 10:
            break

    # IMPOSTOR: a random impostor pair scores near zero under every model and
    # shows nothing, so these are the HARDEST: each probe's strongest non-owner,
    # ranked by how far past its own condition's threshold it reaches, taken in
    # turn from C1, C2, C3, C4 so no single model chooses the set.
    order = list(SAMPLE_PICKERS)
    ranked = {}
    for name in order:
        P, thr = table[name]
        Q = P[gen].copy()
        Q[np.arange(len(gen)), lab] = -2
        j = Q.argmax(1)
        excess = Q[np.arange(len(gen)), j] - thr[1e-3]
        ranked[name] = [(int(i), int(j[i])) for i in np.argsort(-excess)]
    isel, used, ptr = [], set(), {n: 0 for n in order}
    while len(isel) < 10:
        for name in order:
            while ptr[name] < len(ranked[name]):
                i, j = ranked[name][ptr[name]]
                ptr[name] += 1
                tk = F["skey"][gen[i]]
                if tk not in used:
                    isel.append((i, j))
                    used.add(tk)
                    break
            if len(isel) == 10:
                break

    ref_row = {int(p): int(np.where(owners == p)[0][0]) for p in ids}
    W = 900

    def tile(group, i, j):
        fi = int(gen[i])
        claimed, truth = int(ids[j]), int(emp_of[fi])
        # `fi` indexes the embedded subset; pixels live at the global row.
        pipe = FC.pixels("pipe", np.array([sel[fi]]))[0]
        r003 = FC.pixels("r003", np.array([sel[fi]]))[0]
        ref_p = gal["pipe"][ref_row[claimed]]
        strip = np.concatenate([pipe, r003, ref_p], 1)
        strip = cv2.resize(cv2.cvtColor(strip, cv2.COLOR_RGB2BGR), (504, 168), interpolation=cv2.INTER_CUBIC)
        c = np.full((168 + 46 + len(CONDITIONS) * 30 + 12, W, 3), 255, np.uint8)
        c[:168, :504] = strip
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(c, "pipeline", (40, 186), font, 0.5, (110, 110, 110), 1, cv2.LINE_AA)
        cv2.putText(c, "IR50 training", (178, 186), font, 0.5, (110, 110, 110), 1, cv2.LINE_AA)
        cv2.putText(c, "enrolment", (362, 186), font, 0.5, (110, 110, 110), 1, cv2.LINE_AA)
        who = people[truth][0][:26]
        what = people[claimed][0][:26]
        cv2.putText(c, group, (520, 26), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(c, f"face:  {who}", (520, 60), font, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(c, f"photo: {what}", (520, 88), font, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(c, f"{F['seg'][fi]}  frame {int(F['frame'][fi])}", (520, 116), font, 0.45, (110, 110, 110), 1, cv2.LINE_AA)
        cv2.putText(c, f"head {int(F['face_px'][fi])} px native", (520, 140), font, 0.45, (110, 110, 110), 1, cv2.LINE_AA)
        y = 214
        verdicts = {}
        for name, _m, _a in CONDITIONS:
            P, thr = table[name]
            s, t = float(P[fi, j]), thr[1e-3]
            accept = s >= t and int(P[fi].argmax()) == j
            right = accept if group == "GENUINE" else not accept
            verdicts[name] = {"score": round(s, 3), "thr": round(t, 3), "accept": bool(accept)}
            colour = (0, 120, 0) if right else (0, 0, 210)
            txt = (f"{name:14s}  score {s:+.3f}   threshold {t:.3f}   "
                   f"{'ACCEPT' if accept else 'reject'}   {'correct' if right else 'WRONG'}")
            cv2.putText(c, txt, (10, y), font, 0.58, colour, 1, cv2.LINE_AA)
            y += 30
        report["samples"].append({"group": group, "segment": str(F["seg"][fi]), "frame": int(F["frame"][fi]),
                                  "track": str(F["skey"][fi]), "face": who, "photo": what, **{"verdicts": verdicts}})
        return c

    for group, items, fname in (("GENUINE", gsel, "samples_genuine.jpg"),
                                ("IMPOSTOR", isel, "samples_impostor.jpg")):
        tiles = [tile(group, i, j) for i, j in items]
        sep = np.full((6, W, 3), 200, np.uint8)
        body = np.concatenate(sum(([t, sep] for t in tiles), [])[:-1], 0)
        cv2.imwrite(str(out / fname), body, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  {group.lower()} samples -> {out / fname}")


if __name__ == "__main__":
    raise SystemExit(main())
