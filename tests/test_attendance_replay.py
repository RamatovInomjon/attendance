"""Replay three production days through the decision logic and hold the line.

The fixture is an anonymised slice of the gpu6 export of 2026-09-02..04: every
recognised pass with the direction the pipeline gave it at the time, the
camera geometry, the ReID track spans (who was still in view when), and the
daily rows the OLD logic wrote. `bench/replay_events.py` re-decides those
passes with the current arbiter and state machine; this test pins the cases
that were inspected frame by frame and the aggregate counts, so a change to
the arbiter or the state machine that quietly brings a wrong check-out back
fails here rather than on the attendance sheet.

None of this re-runs the trajectory logic - the recorded verdicts are taken
as given. The direction algorithm itself is covered by
tests/test_direction_scenarios.py and bench/direction_eval.py.
"""
from __future__ import annotations

import json
from datetime import timezone
from pathlib import Path

import pytest

from bench.replay_events import _ts, metrics, replay

FIXTURE = Path(__file__).parent / "fixtures" / "attendance_replay_20260902_04.json"


@pytest.fixture(scope="module")
def replayed():
    raw = json.loads(FIXTURE.read_text())
    cams = {int(k): v for k, v in raw["cameras"].items()}
    events = []
    for e in raw["events"]:
        e = dict(e)
        e["ts"] = _ts(e["ts"])
        emb = 0
        for part in str(e.get("votes") or "").split("/"):
            if part.startswith("emb"):
                emb = int(part[3:])
        e["embedded"] = emb
        e["snapshot"] = None
        events.append(e)
    spans = [(c, emp, _ts(a), _ts(b)) for c, emp, a, b in raw["spans"]]
    data = {"cameras": cams, "events": events, "spans": spans, "daily": raw["daily"]}
    new_events, new_daily = replay(data)
    return {
        "data": data,
        "recorded": metrics(events, raw["daily"]),
        "replayed": metrics(new_events, new_daily),
        "events": {e["id"]: e for e in new_events},
        "daily": {(r["employee_id"], r["business_date"]): r for r in new_daily},
        "old_daily": {(r["employee_id"], r["business_date"]): r for r in raw["daily"]},
    }


def _local(dt):
    from app.config import settings
    return dt.astimezone(settings.tz).strftime("%H:%M:%S")


def test_the_fixture_is_the_whole_export():
    raw = json.loads(FIXTURE.read_text())
    assert len(raw["events"]) == 579
    assert len(raw["spans"]) == 535
    assert not any("name" in e for e in raw["events"]), "the fixture must stay anonymous"


def test_a_u_turn_at_the_door_no_longer_checks_anybody_out(replayed):
    """Employee 30, 2026-09-03 11:01: walked to the Exit camera and straight
    back. The Exit camera's EXIT (34 frames) beat the Entrance camera's
    ENTER (11 frames) on face strength, and the day's check-out became 11:01
    instead of the real 11:50 - 49 minutes lost. The later view decides."""
    ev = replayed["events"]
    assert ev[263]["group_direction"] == "ENTER"
    assert ev[263]["transition"] == "RE_SIGHTING"
    assert "[over Exit:EXIT" in ev[263]["group_reason"]
    assert ev[283]["transition"] == "CHECK_OUT"          # the real departure
    row = replayed["daily"][(30, "2026-09-03")]
    assert _local(row["check_out_time"]) == "11:50:53"
    assert row["worked_seconds"] >= 6100
    assert replayed["old_daily"][(30, "2026-09-03")]["worked_seconds"] < 3300


def test_the_second_inspected_u_turn(replayed):
    """Employee 51, 2026-09-04 10:47: the same pattern, 41 minutes lost."""
    ev = replayed["events"]
    assert ev[473]["group_direction"] == "ENTER" and ev[473]["transition"] == "RE_SIGHTING"
    assert ev[496]["transition"] == "CHECK_OUT"
    new = replayed["daily"][(51, "2026-09-04")]["worked_seconds"]
    old = replayed["old_daily"][(51, "2026-09-04")]["worked_seconds"]
    assert new >= old + 2000


def test_a_u_turn_seen_whole_by_the_entrance_camera(replayed):
    """Employee 21, 2026-09-04 15:39: seen walking toward the door on the
    Exit camera, then turning and walking back on the Entrance camera; the
    body crops show him passing under the Entrance camera. Recorded as a
    check-out; he never left."""
    ev = replayed["events"]
    assert ev[604]["group_direction"] == "ENTER"
    assert ev[604]["transition"] != "CHECK_OUT"


def test_a_contradiction_is_written_into_the_reason(replayed):
    """Every overruled view is named on the event row, so an operator can see
    what the other camera said."""
    contradicted = [e for e in replayed["events"].values() if "[over " in str(e.get("group_reason"))]
    assert len(contradicted) >= 5
    for e in contradicted:
        assert e["group_direction"] in ("ENTER", "EXIT")


def test_aggregate_counts_do_not_regress(replayed):
    rec, rep = replayed["recorded"], replayed["replayed"]
    assert rep["passes"] == rec["passes"]
    # Same evidence, so the same number of walks was seen; what changes is
    # which of them moved state and which way.
    assert rep["fused_contradictions"] <= rec["fused_contradictions"]
    assert rep["t:NO_DIRECTION"] <= rec["t:NO_DIRECTION"]
    assert rep["d:NO_CHECKOUT"] <= rec["d:NO_CHECKOUT"]
    assert rep["out_before_in"] <= 1        # employee 23: the second entry was debounced and never recorded
    assert rep["days_under_10min"] <= rec["days_under_10min"] + 1
