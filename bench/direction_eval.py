#!/usr/bin/env python3
"""Re-decide dumped trajectories with the current and the legacy direction
algorithm, and score both against hand labels.

    python bench/direction_eval.py --dumps data/bench/traj_4k
    python bench/direction_eval.py --dumps data/bench/traj_4k --labels bench/direction_labels.json

The dumps come from bench/dump_trajectories.py. Labels map
"<camera>/<clip>/<track_id>" to ENTER, EXIT, STAY (walked and came back) or
NONE (nothing determinable / not a real person) - anything else is skipped.

"Legacy" is the rolling-window rule that ran in production until 2026-09-06:
a deque of the last 90 points, "the last sign change" for the tripwire, the
first third against the last third of that window for depth, a latch on the
last non-UNKNOWN verdict, and a 5 s staleness cut-off measured at the track's
last sighting. It is reproduced here so the two can be compared on identical
trajectories; note that the dumped synthesised heads already use the corrected
size, so the legacy depth signal is if anything better here than it was live.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.direction import Direction, DirectionConfig, Trajectory  # noqa: E402

LEGACY_WINDOW = 90
LEGACY_MAX_AGE_S = 5.0


def _side(line, x, y):
    (ax, ay), (bx, by) = line
    return (bx - ax) * (y - ay) - (by - ay) * (x - ax)


def _legacy_window_verdict(win, cfg: DirectionConfig):
    pts = list(win)
    n = len(pts)
    if n < cfg.min_points:
        return Direction.UNKNOWN, f"only {n} points"
    (t0, x0, y0, a0), (t1, x1, y1, a1) = pts[0][:4], pts[-1][:4]
    travel = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    if travel < cfg.min_travel:
        return Direction.UNKNOWN, f"stationary ({travel:.3f} < {cfg.min_travel})"
    # crossing: last clean sign change
    cross = Direction.UNKNOWN
    if cfg.configured:
        prev = 0
        for p in pts:
            s = _side(cfg.line, p[1], p[2])
            sign = 1 if s > 1e-4 else (-1 if s < -1e-4 else 0)
            if sign == 0:
                continue
            if prev and sign != prev:
                cross = Direction.ENTER if sign == cfg.inside_side else Direction.EXIT
            prev = sign
    # depth: first third vs last third means
    k = max(2, n // 3)
    first = sum(p[3] for p in pts[:k]) / k
    last = sum(p[3] for p in pts[-k:]) / k
    dep = Direction.UNKNOWN
    if first > 0 and last > 0:
        ratio = last / first
        growing = None
        if ratio >= cfg.min_area_ratio:
            growing = True
        elif ratio <= 1.0 / cfg.min_area_ratio:
            growing = False
        if growing is not None:
            inward = growing if cfg.depth_grows_inward else (not growing)
            dep = Direction.ENTER if inward else Direction.EXIT
    if cross is not Direction.UNKNOWN and dep is not Direction.UNKNOWN:
        if cross == dep:
            return cross, f"line+depth agree ({cross.value})"
        return cross, f"line={cross.value} (depth said {dep.value})"
    if cross is not Direction.UNKNOWN:
        return cross, f"line only ({cross.value})"
    if dep is not Direction.UNKNOWN:
        if not cfg.configured:
            return dep, f"depth only ({dep.value})"
        if travel >= cfg.depth_only_travel:
            return dep, f"depth only ({dep.value}, travel {travel:.2f})"
        return Direction.UNKNOWN, (f"no crossing, travel {travel:.2f} < "
                                   f"{cfg.depth_only_travel} (depth said {dep.value})")
    return Direction.UNKNOWN, "no usable signal"


def legacy_verdict(points, cfg: DirectionConfig):
    """The pipeline's per-frame latch plus the staleness cut at prune time."""
    win: deque = deque(maxlen=LEGACY_WINDOW)
    latched, latched_at, reason = Direction.UNKNOWN, 0.0, ""
    for p in points:
        win.append(p)
        d, why = _legacy_window_verdict(win, cfg)
        reason = why
        if d is not Direction.UNKNOWN:
            latched, latched_at = d, p[0]
    if not points:
        return Direction.UNKNOWN, "no points"
    last_seen = points[-1][0]
    if latched is not Direction.UNKNOWN and latched_at > 0 \
            and (last_seen - latched_at) > LEGACY_MAX_AGE_S:
        return Direction.UNKNOWN, f"stale({last_seen - latched_at:.0f}s, was {latched.value})"
    return latched, reason


def current_verdict(points, cfg: DirectionConfig, frame_w: int, frame_h: int):
    tr = Trajectory(capacity=max(8, len(points)))
    for p in points:
        t, cx, cy, area = p[:4]
        synth = bool(p[4]) if len(p) > 4 else False
        edge = bool(p[5]) if len(p) > 5 else False
        side = (max(area, 1e-9) * frame_w * frame_h) ** 0.5
        x, y = cx * frame_w, cy * frame_h
        box = [x - side / 2, y - side / 2, x + side / 2, y + side / 2]
        tr.add(t, box, frame_w, frame_h, synth=synth, edge=edge)
    return tr.direction(cfg)


def _norm(v: str) -> str:
    v = (v or "").upper()
    return v if v in ("ENTER", "EXIT") else "NONE"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default="data/bench/traj_4k")
    ap.add_argument("--labels", default="bench/direction_labels.json")
    ap.add_argument("--min-duration", type=float, default=1.0)
    ap.add_argument("--all", action="store_true", help="print every track, not only labelled ones")
    ap.add_argument("--json", default=None, help="write per-track verdicts here")
    args = ap.parse_args()

    labels = {}
    lp = ROOT / args.labels
    if lp.exists():
        labels = {k: v for k, v in json.loads(lp.read_text()).items() if not k.startswith("_")}

    rows = []
    for jf in sorted(glob.glob(str(ROOT / args.dumps / "*" / "*.json"))):
        d = json.loads(Path(jf).read_text())
        cfg = DirectionConfig(
            line=tuple(tuple(p) for p in d["line"]) if d.get("line") else None,
            inside_side=d.get("inside_side", 1),
            depth_grows_inward=bool(d.get("depth_grows_inward", True)))
        for tr in d["tracks"]:
            if tr["duration_s"] < args.min_duration or not tr["points"]:
                continue
            key = f"{tr['camera']}/{tr['clip']}/{tr['track_id']}"
            leg, leg_why = legacy_verdict(tr["points"], cfg)
            cur, cur_why = current_verdict(tr["points"], cfg, tr["frame_w"], tr["frame_h"])
            rows.append({
                "key": key, "camera": tr["camera"], "clip": tr["clip"],
                "track_id": tr["track_id"], "duration_s": round(tr["duration_s"], 1),
                "name": tr.get("name") or "", "embedded": tr.get("embedded", 0),
                "label": labels.get(key), "legacy": leg.value, "legacy_reason": leg_why,
                "current": cur.value, "current_reason": cur_why,
                "pipeline": tr.get("direction"), "pipeline_reason": tr.get("direction_reason"),
            })

    print(f"  {len(rows)} track(s) >= {args.min_duration}s from {args.dumps}")
    tally = {"legacy": collections.Counter(), "current": collections.Counter()}
    for r in rows:
        tally["legacy"][r["legacy"]] += 1
        tally["current"][r["current"]] += 1
    for k, c in tally.items():
        print(f"  {k:8s} ENTER={c['ENTER']:3d} EXIT={c['EXIT']:3d} UNKNOWN={c['UNKNOWN']:3d}")

    labelled = [r for r in rows if r["label"] is not None
                and r["label"].upper() in ("ENTER", "EXIT", "NONE")]
    if labelled:
        print(f"\n  {len(labelled)} labelled track(s)")
        for algo in ("legacy", "current"):
            right = wrong = refused = 0
            for r in labelled:
                truth = _norm(r["label"])
                got = r[algo]
                if truth == "NONE":
                    if got == "UNKNOWN":
                        right += 1
                    else:
                        wrong += 1
                elif got == truth:
                    right += 1
                elif got == "UNKNOWN":
                    refused += 1
                else:
                    wrong += 1
            print(f"  {algo:8s} correct={right:3d} wrong={wrong:3d} refused={refused:3d}  "
                  f"accuracy={right / len(labelled):.2f} wrong-rate={wrong / len(labelled):.2f}")
        print()
        print(f"  {'track':46s} {'dur':>5s} {'label':6s} {'legacy':8s} {'current':8s} reason (current)")
        for r in labelled:
            flag = "" if _norm(r["label"]) in (r["current"], "NONE" if r["current"] == "UNKNOWN" else "") \
                else "  <-- current"
            print(f"  {r['key'][:46]:46s} {r['duration_s']:5.1f} {r['label'][:6]:6s} "
                  f"{r['legacy']:8s} {r['current']:8s} {r['current_reason'][:44]}{flag}")
    if args.all or not labelled:
        print()
        print(f"  {'track':46s} {'dur':>5s} {'name':14s} {'legacy':8s} {'current':8s} reason (current)")
        for r in rows:
            mark = "" if r["legacy"] == r["current"] else "  *"
            print(f"  {r['key'][:46]:46s} {r['duration_s']:5.1f} {r['name'][:14]:14s} "
                  f"{r['legacy']:8s} {r['current']:8s} {r['current_reason'][:44]}{mark}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
        print(f"\n  written {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
