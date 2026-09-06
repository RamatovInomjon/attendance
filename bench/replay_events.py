#!/usr/bin/env python3
"""Replay a production database's recognised passes through the current
arbiter and attendance state machine, and compare with what was recorded.

    python bench/replay_events.py --db data/gpu6_export_20260904/ematsy.db
    python bench/replay_events.py --db ... --json data/bench/replay_events.json

Every stored `recognition_event` is one completed, identified pass with the
direction the pipeline gave it at the time. Those verdicts are taken as they
are - this does not re-run the trajectory logic, bench/direction_eval.py does
that on footage - but the DECISIONS made from them are re-derived:

* cross-camera grouping (`PassArbiter`), including the hold on a person who
  is still in view, reconstructed from `reid_pass` track spans;
* the group's direction (`resolve_direction`);
* the daily state machine (`AttendanceService._apply`, the production code).

The report counts what an attendance sheet suffers from: one walk written as
two opposite decisions, a check-out for somebody who came straight back, a
day whose only sighting moved nothing, and days with impossible times. The
same counts are computed from the stored transitions, so the two columns are
the old and the new policy on identical evidence.

The load-bearing assumption is stated once: a pass that was DUPLICATE_VIEW or
NO_DIRECTION in the record is still a pass, so the replay sees exactly the
sightings the live system saw and nothing it did not.
"""
from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings                                  # noqa: E402
from app.db.models import CameraRole, PresenceStatus              # noqa: E402
from app.services.arbiter import PassArbiter, PendingPass         # noqa: E402
from app.services.attendance import AttendanceService, business_date  # noqa: E402

UTC = timezone.utc


def _ts(v: str) -> datetime:
    s = str(v).replace("T", " ")
    try:
        t = datetime.fromisoformat(s)
    except ValueError:
        t = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def load(db_path: str) -> dict:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cams = {r["id"]: dict(r) for r in con.execute("select * from camera")}
    events = [dict(r) for r in con.execute(
        "select id, employee_id, camera_id, role, ts, business_date, score, margin, "
        "track_id, votes, snapshot, transition, direction, direction_reason "
        "from recognition_event where voided_at is null and employee_id is not null "
        "order by ts, id")]
    for e in events:
        e["ts"] = _ts(e["ts"])
        emb = 0
        for part in str(e.get("votes") or "").split("/"):
            if part.startswith("emb"):
                try:
                    emb = int(part[3:])
                except ValueError:
                    pass
        e["embedded"] = emb
    spans = []
    try:
        for r in con.execute("select camera_id, employee_id, first_seen, last_seen from reid_pass "
                             "where employee_id is not null"):
            spans.append((r["camera_id"], r["employee_id"], _ts(r["first_seen"]), _ts(r["last_seen"])))
    except sqlite3.OperationalError:
        pass
    daily = [dict(r) for r in con.execute("select * from daily_attendance")]
    con.close()
    return {"cameras": cams, "events": events, "spans": spans, "daily": daily}


# --- metrics --------------------------------------------------------------

def _pairs(applied: list[dict]) -> list[tuple]:
    """Consecutive applied (non-duplicate) events per person."""
    by_emp = collections.defaultdict(list)
    for e in applied:
        by_emp[e["employee_id"]].append(e)
    out = []
    for emp, rows in by_emp.items():
        rows.sort(key=lambda e: e["ts"])
        for a, b in zip(rows, rows[1:]):
            out.append((emp, a, b, (b["ts"] - a["ts"]).total_seconds()))
    return out


def metrics(events: list[dict], daily_rows: list[dict], window_s: float = 20.0) -> dict:
    """Counts from a list of decided events (each with transition/direction)."""
    m = collections.Counter()
    m["passes"] = len(events)
    for e in events:
        m[f"t:{e['transition']}"] += 1
    m["no_direction"] = sum(1 for e in events if e["transition"] == "NO_DIRECTION")
    m["no_direction_stale"] = sum(1 for e in events if e["transition"] == "NO_DIRECTION"
                                  and str(e["direction_reason"]).startswith("stale"))
    # A fused group whose views disagreed: the loser carries an opposite verdict.
    losers = [e for e in events if e["transition"] == "DUPLICATE_VIEW"]
    winners = [e for e in events if e["transition"] not in ("DUPLICATE_VIEW",)]
    contra = 0
    for l in losers:
        for w in winners:
            if w["employee_id"] != l["employee_id"]:
                continue
            if abs((w["ts"] - l["ts"]).total_seconds()) > window_s:
                continue
            wd = (w.get("group_direction") or w["direction"])
            if wd in ("ENTER", "EXIT") and l["direction"] in ("ENTER", "EXIT") and wd != l["direction"]:
                contra += 1
                break
    m["fused_contradictions"] = contra
    applied = [e for e in events if e["transition"] in ("CHECK_IN", "CHECK_OUT", "RE_SIGHTING")]
    flips = collections.Counter()
    for emp, a, b, dt in _pairs(applied):
        da = a.get("group_direction") or a["direction"]
        db = b.get("group_direction") or b["direction"]
        if da in ("ENTER", "EXIT") and db in ("ENTER", "EXIT") and da != db:
            if dt <= 60:
                flips["<=60s"] += 1
            elif dt <= 120:
                flips["60-120s"] += 1
    m["opposite_within_60s"] = flips["<=60s"]
    m["opposite_60_120s"] = flips["60-120s"]
    # The U-turn signature: a check-out immediately followed by that person
    # being seen walking IN (any transition), or a check-in followed by them
    # walking OUT. Somebody who really left does not reappear walking in
    # within two minutes; somebody who walked to the door and back does.
    by_emp = collections.defaultdict(list)
    for e in events:
        if e["transition"] not in ("DUPLICATE_VIEW", "DEBOUNCED"):
            by_emp[e["employee_id"]].append(e)
    u_out = u_in = 0
    for rows_ in by_emp.values():
        rows_.sort(key=lambda e: e["ts"])
        for i, a in enumerate(rows_):
            if a["transition"] not in ("CHECK_IN", "CHECK_OUT"):
                continue
            da = a.get("group_direction") or a["direction"]
            for b in rows_[i + 1:]:
                dt = (b["ts"] - a["ts"]).total_seconds()
                if dt > 120:
                    break
                db = b.get("group_direction") or b["direction"]
                if a["transition"] == "CHECK_OUT" and db == "ENTER":
                    u_out += 1
                    break
                if a["transition"] == "CHECK_IN" and db == "EXIT":
                    u_in += 1
                    break
    m["checkout_then_seen_entering_120s"] = u_out
    m["checkin_then_seen_leaving_120s"] = u_in
    # The transition the state machine gave an applied pass says whether the
    # verdict was consistent with the state: an EXIT while already out, or an
    # ENTER while already in, is a re-sighting - a missed or wrong movement.
    m["re_sightings"] = sum(1 for e in applied if e["transition"] == "RE_SIGHTING")
    # daily rows
    m["days"] = len(daily_rows)
    for r in daily_rows:
        ci, co = r.get("check_in_time"), r.get("check_out_time")
        st = r.get("status")
        m[f"d:{st}"] += 1
        if ci and co:
            ci_t = ci if isinstance(ci, datetime) else _ts(ci)
            co_t = co if isinstance(co, datetime) else _ts(co)
            if co_t < ci_t:
                m["out_before_in"] += 1
            if (co_t - ci_t).total_seconds() < 600:
                m["days_under_10min"] += 1
    return dict(m)


# --- replay ---------------------------------------------------------------

def replay(data: dict, window_s: float | None = None, live_hold: bool = True) -> tuple[list[dict], list[dict]]:
    """Re-decide every pass. Returns (events with new transitions, daily rows)."""
    cams = data["cameras"]
    events = sorted(data["events"], key=lambda e: (e["ts"], e["id"]))
    if not events:
        return [], []
    spans = data["spans"] if live_hold else []
    arb = PassArbiter(window_s=window_s)
    svc = AttendanceService()
    daily: dict[tuple, SimpleNamespace] = {}
    out: list[dict] = []
    last_by_cam: dict[tuple, tuple[datetime, str]] = {}

    def row_for(emp, bdate):
        key = (emp, bdate)
        if key not in daily:
            daily[key] = SimpleNamespace(
                employee_id=emp, business_date=bdate, presence=PresenceStatus.OUTSIDE,
                entered_at=None, check_in_time=None, check_out_time=None,
                worked_seconds=0, status="PRESENT", check_in_snapshot=None,
                check_out_snapshot=None, event_count=0)
        return daily[key]

    def apply(p: PendingPass, winner: bool, direction: str, reason: str):
        e = p.track.event
        cam = cams.get(p.camera_id, {})
        role = CameraRole(cam.get("role", "BOTH")) if cam else CameraRole.BOTH
        configured = all(cam.get(k) is not None for k in ("line_x1", "line_y1", "line_x2", "line_y2"))
        # the same-camera, same-direction cooldown, as AttendanceService._debounced
        key = (p.employee_id, p.camera_id)
        last = last_by_cam.get(key)
        if last is not None and (p.ts - last[0]) < svc.cooldown and \
                not (direction and last[1] and direction != last[1]):
            rec = dict(e, transition="DEBOUNCED", group_direction=direction, group_reason=reason)
            out.append(rec)
            return
        last_by_cam[key] = (p.ts, direction)
        bdate = business_date(p.ts)
        row = row_for(p.employee_id, bdate)
        row.event_count += 1
        if not winner:
            out.append(dict(e, transition="DUPLICATE_VIEW", group_direction=direction, group_reason=reason))
            return
        eff = svc._effective_role(role, direction)
        if eff is None and not configured:
            eff = role
        tr = svc._apply(row, eff, p.ts, e.get("snapshot"))
        out.append(dict(e, transition=tr, group_direction=direction, group_reason=reason))

    t0 = events[0]["ts"]
    base = t0.timestamp()
    i = 0
    now_dt = t0
    end = events[-1]["ts"] + timedelta(seconds=120)
    step = timedelta(seconds=1)
    while now_dt <= end:
        now = now_dt.timestamp() - base
        while i < len(events) and events[i]["ts"] <= now_dt:
            e = events[i]
            i += 1
            track = SimpleNamespace(
                embedded_frames=e["embedded"], best_score=float(e["score"] or 0.0),
                direction=e["direction"] or "UNKNOWN",
                direction_reason=e["direction_reason"] or "", event=e)
            arb.submit(PendingPass(
                employee_id=e["employee_id"], camera_id=e["camera_id"], role=None,
                ts=e["ts"], monotonic=e["ts"].timestamp() - base, track=track,
                snapshot=e.get("snapshot"), camera_name=cams.get(e["camera_id"], {}).get("name", "")))
        if spans:
            live_by_cam = collections.defaultdict(set)
            for cam_id, emp, a, b in spans:
                if a <= now_dt <= b:
                    live_by_cam[cam_id].add(emp)
            for cam_id in cams:
                arb.note_live(cam_id, live_by_cam.get(cam_id, set()), now)
        for g in arb.due(now):
            apply(g.winner, True, g.direction, g.direction_reason)
            for o in g.others:
                apply(o, False, o.track.direction, o.track.direction_reason)
        now_dt += step
    for g in arb.drain():
        apply(g.winner, True, g.direction, g.direction_reason)
        for o in g.others:
            apply(o, False, o.track.direction, o.track.direction_reason)

    # "Today" is the last day the export saw, not the wall clock: the export
    # was taken mid-afternoon with people still inside, and closing those
    # rows would invent NO_CHECKOUT flags the live system never wrote.
    today = business_date(events[-1]["ts"])
    rows = []
    for (emp, bdate), r in daily.items():
        if r.presence == PresenceStatus.INSIDE and bdate < today:
            svc._close_row(r)
        rows.append({"employee_id": emp, "business_date": str(bdate),
                     "check_in_time": r.check_in_time, "check_out_time": r.check_out_time,
                     "worked_seconds": r.worked_seconds, "status": r.status,
                     "presence": r.presence.value})
    out.sort(key=lambda e: (e["ts"], e["id"]))
    return out, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/gpu6_export_20260904/ematsy.db")
    ap.add_argument("--window", type=float, default=None)
    ap.add_argument("--no-live-hold", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--show-changes", action="store_true")
    ap.add_argument("--trace", default=None,
                    help="EMP[:YYYY-MM-DD]: print every pass of one person, recorded vs replayed")
    args = ap.parse_args()

    data = load(args.db)
    print(f"  {len(data['events'])} recognised passes, {len(data['spans'])} track spans, "
          f"{len(data['daily'])} daily rows from {args.db}")
    recorded = metrics(data["events"], data["daily"])
    new_events, new_daily = replay(data, window_s=args.window, live_hold=not args.no_live_hold)
    replayed = metrics(new_events, new_daily)

    keys = ["passes", "t:CHECK_IN", "t:CHECK_OUT", "t:RE_SIGHTING", "t:NO_DIRECTION",
            "t:DUPLICATE_VIEW", "t:DEBOUNCED", "fused_contradictions",
            "checkout_then_seen_entering_120s", "checkin_then_seen_leaving_120s",
            "opposite_within_60s", "opposite_60_120s", "days", "d:PRESENT", "d:NO_CHECKIN",
            "d:NO_CHECKOUT", "out_before_in", "days_under_10min"]
    print(f"\n  {'metric':26s} {'recorded':>9s} {'replayed':>9s}")
    for k in keys:
        print(f"  {k:26s} {recorded.get(k, 0):9d} {replayed.get(k, 0):9d}")

    if args.trace:
        emp_s, _, day = args.trace.partition(":")
        emp = int(emp_s)
        old_by_id = {e["id"]: e for e in data["events"]}
        print(f"\n  trace emp{emp} {day or ''}: recorded | replayed")
        for e in new_events:
            if e["employee_id"] != emp:
                continue
            if day and str(e["business_date"]) != day:
                continue
            o = old_by_id[e["id"]]
            mark = "" if o["transition"] == e["transition"] else "   <--"
            print(f"   #{e['id']:4d} cam{e['camera_id']} {e['ts'].astimezone(settings.tz):%H:%M:%S} "
                  f"{o['direction']:7s} {o['transition']:14s} | {e.get('group_direction'):7s} "
                  f"{e['transition']:14s} {str(e.get('group_reason'))[:52]}{mark}")
    if args.show_changes:
        old_by_id = {e["id"]: e for e in data["events"]}
        print(f"\n  passes whose decision changed:")
        for e in new_events:
            o = old_by_id[e["id"]]
            if o["transition"] != e["transition"] or (e.get("group_direction") != o["direction"]
                                                       and e["transition"] not in ("DUPLICATE_VIEW", "DEBOUNCED")):
                print(f"   #{e['id']:4d} emp{e['employee_id']:3d} cam{e['camera_id']} "
                      f"{e['ts'].astimezone(settings.tz):%m-%d %H:%M:%S} "
                      f"{o['direction']:7s} {o['transition']:14s} -> {e.get('group_direction'):7s} "
                      f"{e['transition']:14s} {str(e.get('group_reason'))[:60]}")
    if args.show_changes:
        def _t(v):
            if v is None:
                return "   -   "
            t = v if isinstance(v, datetime) else _ts(v)
            return t.astimezone(settings.tz).strftime("%H:%M:%S")
        old_days = {(r["employee_id"], str(r["business_date"])): r for r in data["daily"]}
        new_days = {(r["employee_id"], str(r["business_date"])): r for r in new_daily}
        print(f"\n  person-days whose times changed (recorded -> replayed):")
        for key in sorted(set(old_days) | set(new_days)):
            o, n = old_days.get(key), new_days.get(key)
            if o is None or n is None:
                print(f"   emp{key[0]:3d} {key[1]}  only in {'recorded' if n is None else 'replayed'}")
                continue
            same = (_t(o["check_in_time"]) == _t(n["check_in_time"])
                    and _t(o["check_out_time"]) == _t(n["check_out_time"])
                    and int(o["worked_seconds"] or 0) == int(n["worked_seconds"] or 0))
            if same:
                continue
            print(f"   emp{key[0]:3d} {key[1]}  in {_t(o['check_in_time'])}->{_t(n['check_in_time'])}  "
                  f"out {_t(o['check_out_time'])}->{_t(n['check_out_time'])}  "
                  f"worked {int(o['worked_seconds'] or 0) / 3600:5.2f}h->{int(n['worked_seconds'] or 0) / 3600:5.2f}h  "
                  f"{o['status']}->{n['status']}")
    if args.json:
        Path(args.json).write_text(json.dumps({"recorded": recorded, "replayed": replayed}, indent=1))
        print(f"\n  written {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
