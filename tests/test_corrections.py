"""Voiding a wrong recognition, and labelling an unknown face.

The two things that carry weight here:

* **Attendance is DERIVED state.** `worked_seconds` accumulates, `presence` is
  a latch, `check_in_time` keeps the earliest, and a transition is a function
  of the state at the time. So a correction cannot subtract one event's effect;
  the day has to be replayed. The test that matters is the one where voiding
  the check-in must PROMOTE a later re-sighting into the check-in.
* **A correction must not become a way to author attendance.** Resolving an
  unknown records who it was and offers the face to the gallery. It writes no
  event, because a dropdown that can invent a check-in is a dropdown that can
  be wrong in a way nobody will ever see.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from sqlalchemy import select

from app.db.models import (
    CameraRole, DailyAttendance, Employee, FaceEmbedding, PresenceStatus,
    RecognitionEvent, UnknownSighting,
)
from app.db.session import session_scope
from app.services import corrections
from app.services.attendance import AttendanceService, business_date

ENTER, EXIT = "ENTER", "EXIT"
CAM_IN, CAM_OUT = 1, 2


@pytest.fixture(autouse=True)
def remove_test_rows():
    """One in-memory database is shared by the whole suite. Dependent rows are
    deleted EXPLICITLY - the test engine has no `PRAGMA foreign_keys`, so
    ON DELETE CASCADE never fires and SQLite reuses the freed employee id."""
    from sqlalchemy import delete
    with session_scope() as s:
        before = set(s.execute(select(Employee.id)).scalars())
        sightings = set(s.execute(select(UnknownSighting.id)).scalars())
    yield
    with session_scope() as s:
        made = [i for i in s.execute(select(Employee.id)).scalars() if i not in before]
        new_u = [i for i in s.execute(select(UnknownSighting.id)).scalars()
                 if i not in sightings]
        if new_u:
            s.execute(delete(UnknownSighting).where(UnknownSighting.id.in_(new_u)))
        if made:
            for model in (RecognitionEvent, DailyAttendance, FaceEmbedding):
                s.execute(delete(model).where(model.employee_id.in_(made)))
            s.execute(delete(Employee).where(Employee.id.in_(made)))


def _emp(name="Correction Target"):
    with session_scope() as s:
        e = Employee(full_name=name, is_active=True)
        s.add(e); s.flush()
        return e.id


def _at(h, m, sec=0):
    return datetime(2026, 8, 26, h, m, sec, tzinfo=timezone.utc)


def _pass(emp, cam, role, ts, direction, snapshot=None, score=0.5):
    with session_scope() as s:
        return AttendanceService(cooldown_s=0).record(
            s, employee_id=emp, camera_id=cam, role=role, ts=ts, score=score,
            direction=direction, snapshot=snapshot).transition


def _daily(emp, day=None):
    with session_scope() as s:
        return s.execute(select(DailyAttendance).where(
            DailyAttendance.employee_id == emp,
            DailyAttendance.business_date == (day or business_date(_at(9, 0))))
        ).scalar_one_or_none()


def _events(emp):
    with session_scope() as s:
        return s.execute(select(RecognitionEvent.id, RecognitionEvent.transition,
                                RecognitionEvent.voided_at)
                         .where(RecognitionEvent.employee_id == emp)
                         .order_by(RecognitionEvent.ts)).all()


# --- the case the whole design turns on -----------------------------------

def test_voiding_a_check_in_promotes_the_next_sighting_into_one():
    """A transition is a function of the state at the time, so it cannot be
    replayed as stored. Void the 09:00 arrival and the 09:30 re-sighting is no
    longer a re-sighting - it is when this person arrived."""
    emp = _emp()
    assert _pass(emp, CAM_IN, CameraRole.IN, _at(9, 0), ENTER) == "CHECK_IN"
    assert _pass(emp, CAM_IN, CameraRole.IN, _at(9, 30), ENTER) == "RE_SIGHTING"
    assert _daily(emp).check_in_time == _at(9, 0)

    first = _events(emp)[0][0]
    out = corrections.void_event(first, by="admin", reason="not him")
    assert out["ok"] and out["replayed"] == 1

    d = _daily(emp)
    assert d.check_in_time == _at(9, 30), "the survivor becomes the check-in"
    assert d.presence == PresenceStatus.INSIDE
    assert [t for _i, t, _v in _events(emp)] == ["CHECK_IN", "CHECK_IN"], \
        "the voided row keeps its old transition; the survivor is rewritten"


def test_worked_seconds_is_recomputed_not_patched():
    """Two intervals, and the first one voided. Subtracting its duration would
    be wrong the moment the check-in moves; only a replay gets this right."""
    emp = _emp()
    _pass(emp, CAM_IN, CameraRole.IN, _at(9, 0), ENTER)
    _pass(emp, CAM_OUT, CameraRole.OUT, _at(10, 0), EXIT)      # 1h
    _pass(emp, CAM_IN, CameraRole.IN, _at(11, 0), ENTER)
    _pass(emp, CAM_OUT, CameraRole.OUT, _at(13, 0), EXIT)      # 2h
    assert _daily(emp).worked_seconds == 3 * 3600

    first_out = [e for e in _events(emp) if e[1] == "CHECK_OUT"][0][0]
    corrections.void_event(first_out, by="admin")
    d = _daily(emp)
    # 09:00 in, 10:00 out is gone, so the 11:00 ENTER is a re-sighting inside an
    # interval that opened at 09:00 and closed at 13:00: four hours, not three.
    assert d.worked_seconds == 4 * 3600
    assert d.check_out_time == _at(13, 0)


def test_the_voided_event_is_kept_and_marked():
    """Voided, never deleted. A row that is gone can neither be audited nor
    learned from, and a confirmed false accept is the most valuable label this
    system produces."""
    emp = _emp()
    _pass(emp, CAM_IN, CameraRole.IN, _at(9, 0), ENTER)
    eid = _events(emp)[0][0]
    corrections.void_event(eid, by="inomjon", reason="that is Ilxom")
    with session_scope() as s:
        e = s.get(RecognitionEvent, eid)
        assert e is not None
        assert e.voided_at is not None
        assert e.voided_by == "inomjon"
        assert e.void_reason == "that is Ilxom"
        assert e.score == 0.5, "the evidence itself is never rewritten"


def test_a_duplicate_view_stays_a_duplicate_view_on_replay():
    """The losing half of a cross-camera pair recorded evidence and moved no
    state. Replaying it as a transition would invent exactly the second
    check-in the arbiter exists to prevent."""
    emp = _emp()
    _pass(emp, CAM_IN, CameraRole.IN, _at(9, 0), ENTER)
    with session_scope() as s:
        AttendanceService(cooldown_s=0).record(
            s, employee_id=emp, camera_id=CAM_OUT, role=CameraRole.OUT,
            ts=_at(9, 0, 5), score=0.4, direction=ENTER, apply_state=False)
    _pass(emp, CAM_IN, CameraRole.IN, _at(9, 30), ENTER)

    with session_scope() as s:
        AttendanceService().rebuild(s, emp, business_date(_at(9, 0)))
    kinds = [t for _i, t, _v in _events(emp)]
    assert kinds.count("DUPLICATE_VIEW") == 1
    assert _daily(emp).check_in_time == _at(9, 0)
    assert _daily(emp).presence == PresenceStatus.INSIDE


def test_rebuilding_a_finished_day_re_flags_the_missing_check_out():
    """The reset clears NO_CHECKOUT, which the end-of-day sweep had written for
    a day that is over. Re-deriving it is the only way a rebuild does not
    silently un-flag a day nobody closed."""
    emp = _emp()
    old = datetime.now(timezone.utc) - timedelta(days=3)
    with session_scope() as s:
        AttendanceService(cooldown_s=0).record(
            s, employee_id=emp, camera_id=CAM_IN, role=CameraRole.IN,
            ts=old, score=0.5, direction=ENTER)
    day = business_date(old)
    with session_scope() as s:
        AttendanceService().close_open_intervals(s, day)
    assert _daily(emp, day).status == "NO_CHECKOUT"
    with session_scope() as s:
        AttendanceService().rebuild(s, emp, day)
    assert _daily(emp, day).status == "NO_CHECKOUT"


def test_un_voiding_puts_the_day_back():
    emp = _emp()
    _pass(emp, CAM_IN, CameraRole.IN, _at(9, 0), ENTER)
    _pass(emp, CAM_IN, CameraRole.IN, _at(9, 30), ENTER)
    eid = _events(emp)[0][0]
    corrections.void_event(eid, by="admin")
    assert _daily(emp).check_in_time == _at(9, 30)
    assert corrections.unvoid_event(eid, by="admin")["ok"]
    assert _daily(emp).check_in_time == _at(9, 0)


def test_voiding_refuses_what_it_cannot_do():
    emp = _emp()
    _pass(emp, CAM_IN, CameraRole.IN, _at(9, 0), ENTER)
    eid = _events(emp)[0][0]
    assert corrections.void_event(999999, by="a")["error"] == "no such event"
    assert corrections.void_event(eid, by="a")["ok"]
    assert corrections.void_event(eid, by="a")["error"] == "already voided"


# --- resolving an unknown face --------------------------------------------

def _sighting(vector=True):
    v = np.random.default_rng(1).standard_normal(512).astype(np.float32)
    v /= np.linalg.norm(v)
    with session_scope() as s:
        u = UnknownSighting(camera_id=CAM_IN, track_id=7, business_date=business_date(_at(9, 0)),
                            frames=12, best_score=0.19,
                            vector=v.tobytes() if vector else None)
        s.add(u); s.flush()
        return u.id


def test_marking_a_visitor_is_the_label_the_thresholds_never_had():
    """A confirmed non-employee is an impostor probe. Every threshold in this
    system was set against 'the pipeline named nobody', which is not the same
    thing - some of those are enrolled people it simply missed."""
    sid = _sighting()
    out = corrections.resolve_sighting(sid, kind="visitor", employee_id=None, by="admin")
    assert out["ok"] and out["kind"] == "visitor"
    assert not out["offerable"], "a visitor's face is never offered to the gallery"
    probes = corrections.labelled_probes()
    assert len(probes["visitors"]) == 1
    assert np.allclose(np.linalg.norm(probes["visitors"], axis=1), 1.0)


def test_naming_an_employee_offers_the_face_but_writes_no_attendance():
    """The deliberate limit: a correction records what happened, it does not
    author a check-in. A dropdown that can invent attendance is a dropdown that
    can be wrong invisibly."""
    emp = _emp("Missed Person")
    sid = _sighting()
    out = corrections.resolve_sighting(sid, kind="employee", employee_id=emp, by="admin")
    assert out["ok"] and out["offerable"] and out["name"] == "Missed Person"
    assert _daily(emp) is None, "no attendance row may appear from a label"
    assert _events(emp) == []
    probes = corrections.labelled_probes()
    assert len(probes["missed"]) == 1 and probes["missed_employee_id"][0] == emp


def test_resolving_validates_its_input():
    sid = _sighting()
    assert not corrections.resolve_sighting(sid, kind="", employee_id=None, by="a")["ok"]
    assert not corrections.resolve_sighting(sid, kind="employee", employee_id=None, by="a")["ok"]
    assert not corrections.resolve_sighting(sid, kind="employee", employee_id=999999, by="a")["ok"]
    assert not corrections.resolve_sighting(999999, kind="visitor", employee_id=None, by="a")["ok"]
    assert corrections.resolve_sighting(sid, kind="unsure", employee_id=None, by="a")["ok"]


# --- the enrolment set's own worst pairs -----------------------------------

def test_worst_pairs_judges_a_row_by_the_floor_it_actually_answers_to():
    """The bug this replaced, reproduced.

    The warning used to rank pairs by raw similarity and label every row
    "enrolment". On gpu6 it therefore reported two ENROLMENT photographs of
    Mahmudjon Azimov and Oybek O'ljaboyev at 0.226 against a 0.220 threshold -
    when both were CORRIDOR CROPS answering to a 0.35 floor, and neither could
    name anybody at 0.226. Wrong about what it was showing, and wrong that
    there was anything to fix.

    So: a crop under its own floor must not appear, a crop over it must, and
    each side must say which kind it is.
    """
    from app.config import settings
    from app.services import augment
    rng = np.random.default_rng(101)
    thr = settings.threshold_for(settings.recognizer_model)
    floor = max(thr, settings.augment_live_floor)

    def blend(base, other, target):
        v = base * target + other * float(np.sqrt(max(0.0, 1 - target ** 2)))
        return (v / np.linalg.norm(v)).astype(np.float32)

    def unit():
        v = rng.standard_normal(512).astype(np.float32)
        return v / np.linalg.norm(v)

    a, far = unit(), unit()
    # Three DIFFERENT directions away from `a`, so each blend is close to `a`
    # and not to the others - otherwise the fixture accidentally makes two rows
    # near-identical and the test measures that instead of the floor.
    twin = blend(a, unit(), (thr + floor) / 2)    # over the threshold, UNDER the floor
    quiet = blend(a, unit(), (thr + floor) / 2)   # same, as a corridor crop
    hot = blend(a, unit(), min(0.95, floor + 0.4))  # over the floor as well

    made = []
    with session_scope() as s:
        for nm, v in (("Pair One", a), ("Pair Two", twin), ("Unrelated", far)):
            e = Employee(full_name=nm, is_active=True); s.add(e); s.flush()
            s.add(FaceEmbedding(employee_id=e.id, source_file="image_01.png",
                                vector=v.tobytes(), dim=512, model_name="m"))
            made.append(e.id)
        # Covered by its floor: must NOT be reported.
        s.add(FaceEmbedding(employee_id=made[2], source_file="live:quiet",
                            vector=quiet.tobytes(), dim=512, model_name="m",
                            threshold=floor))
        # Not covered: must be.
        s.add(FaceEmbedding(employee_id=made[2], source_file="live:loud",
                            vector=hot.tobytes(), dim=512, model_name="m",
                            threshold=floor))

    pairs = augment.worst_pairs(limit=20)
    files = {p["a"]["file"] for p in pairs} | {p["b"]["file"] for p in pairs}
    assert "live:quiet" not in files, "a crop under its own floor cannot false-accept"
    assert "live:loud" in files, "a crop over its own floor still can"
    names = {frozenset((p["a"]["name"], p["b"]["name"])) for p in pairs}
    assert frozenset(("Pair One", "Pair Two")) in names, \
        "two enrolment photos over the threshold are the real case"
    for p in pairs:
        assert p["similarity"] >= thr and p["effective"] >= thr
        assert p["a"]["employee_id"] != p["b"]["employee_id"]
        for side in (p["a"], p["b"]):
            assert side["live"] == side["file"].startswith("live:"), \
                "each side must say what it actually is"


def test_removing_an_enrolment_photo_refuses_to_un_enrol_anybody():
    """The guard `remove()` does not need, because `remove()` cannot reach an
    enrolment row at all. This can, so a person whose last photograph would go
    is refused - otherwise they stop being recognisable and nothing says why."""
    from app.services import augment
    rng = np.random.default_rng(102)
    v1 = rng.standard_normal(512).astype(np.float32); v1 /= np.linalg.norm(v1)
    v2 = rng.standard_normal(512).astype(np.float32); v2 /= np.linalg.norm(v2)

    with session_scope() as s:
        two = Employee(full_name="Has Two", is_active=True)
        one = Employee(full_name="Has One", is_active=True)
        s.add_all([two, one]); s.flush()
        rows = [FaceEmbedding(employee_id=two.id, source_file="image_01.png",
                              vector=v1.tobytes(), dim=512, model_name="m"),
                FaceEmbedding(employee_id=two.id, source_file="image_02.png",
                              vector=v2.tobytes(), dim=512, model_name="m"),
                FaceEmbedding(employee_id=one.id, source_file="image_01.png",
                              vector=v1.tobytes(), dim=512, model_name="m"),
                FaceEmbedding(employee_id=one.id, source_file="live:abc",
                              vector=v2.tobytes(), dim=512, model_name="m",
                              threshold=0.35)]
        s.add_all(rows); s.flush()
        two_a, only, corridor = rows[0].id, rows[2].id, rows[3].id

    out = augment.remove_enrolment([two_a, only, corridor])
    assert out["removed"] == 1, "only the person with a spare photo loses one"
    assert any("last photograph" in r for r in out["refused"])
    assert any("corridor-crop control" in r for r in out["refused"])
    with session_scope() as s:
        left = s.execute(select(FaceEmbedding.source_file)
                         .where(FaceEmbedding.employee_id.in_(
                             select(Employee.id).where(
                                 Employee.full_name.in_(["Has Two", "Has One"]))))
                         ).scalars().all()
    assert sorted(left) == ["image_01.png", "image_02.png", "live:abc"]


# --- the routes ------------------------------------------------------------

from fastapi.testclient import TestClient          # noqa: E402
from app.api.main import app                       # noqa: E402
from app.core.security import COOKIE_NAME, sign_session   # noqa: E402

client = TestClient(app, follow_redirects=False)


def _as(admin: bool) -> dict:
    tok = sign_session(user_id=1 if admin else 2,
                       username="inomjon" if admin else "operator", is_admin=admin)
    return {"Cookie": f"{COOKIE_NAME}={tok}"}


CORRECTION_ROUTES = [
    ("post", "/attendance/event/1/void"),
    ("post", "/attendance/event/1/unvoid"),
    ("post", "/attendance/unknown/1/resolve"),
    ("post", "/gallery/enrolment/remove"),
    ("get", "/gallery/enrolment/1"),
]


@pytest.mark.parametrize("method,path", CORRECTION_ROUTES)
def test_corrections_are_admin_only(method, path):
    """Rewriting attendance and deleting gallery rows is not an operator's to
    do. Anonymous gets a redirect to the login form; an operator gets 403."""
    assert getattr(client, method)(path).status_code in (303, 401, 403)
    assert getattr(client, method)(path, headers=_as(admin=False)).status_code \
        in (303, 403)


@pytest.mark.parametrize("evil", [
    "https://evil.example/steal",
    "//evil.example/steal",
    "http://evil.example",
])
def test_the_return_target_cannot_leave_the_site(evil):
    """`next` is caller input on an authenticated POST. An absolute or
    protocol-relative target would turn a correction button into an open
    redirect out of a signed-in session."""
    emp = _emp()
    _pass(emp, CAM_IN, CameraRole.IN, _at(9, 0), ENTER)
    eid = _events(emp)[0][0]
    r = client.post(f"/attendance/event/{eid}/void", headers=_as(admin=True),
                    data={"next": evil})
    assert r.status_code == 303
    assert not r.headers["location"].startswith(("http://", "https://", "//"))


def test_an_admin_can_void_through_the_route():
    emp = _emp()
    _pass(emp, CAM_IN, CameraRole.IN, _at(9, 0), ENTER)
    eid = _events(emp)[0][0]
    r = client.post(f"/attendance/event/{eid}/void", headers=_as(admin=True),
                    data={"next": "/", "reason": "wrong person"})
    assert r.status_code == 303
    with session_scope() as s:
        assert s.get(RecognitionEvent, eid).voided_by == "inomjon"


# --- finding the capture behind an event -----------------------------------

def test_the_capture_behind_an_event_is_found_by_time_and_score():
    """The bug that made voiding quietly useless.

    An event's `snapshot` names a BODY crop - `body_31_2_1974_1788437454.jpg`,
    keyed by employee/camera/track/epoch - while the debug capture is named for
    a human: `20260903_130030_0.253_Exit_001`, LOCAL time, score to three
    places. Deriving one from the other produced a key matching no gallery row,
    so voiding a wrong recognition failed to remove the crop that caused it -
    the single most important thing voiding does.
    """
    import json
    from app.config import settings

    ts = datetime(2026, 9, 3, 8, 0, 30, tzinfo=timezone.utc)
    local = ts.astimezone(settings.tz)
    stem = f"{local:%Y%m%d_%H%M%S}_0.253_Exit_001"
    folder = settings.debug_dir / "Void Test Person"
    folder.mkdir(parents=True, exist_ok=True)
    try:
        (folder / f"{stem}.json").write_text(json.dumps({"score": 0.253}))
        (folder / f"{stem}_face.jpg").write_bytes(b"\xff\xd8\xff")
        assert corrections.capture_stem(ts, 0.253, "Exit") == stem
        assert "face" in corrections.evidence(stem)
        # A pass whose capture has been swept, or was never written, is not an
        # error - debug_capture can be off. It just has no evidence to show.
        assert corrections.capture_stem(ts, 0.999, "Exit") == ""
        assert corrections.evidence("") == {}
    finally:
        for f in folder.glob(f"{stem}*"):
            f.unlink()
        folder.rmdir()


def test_voiding_removes_the_crop_the_event_actually_produced():
    """End to end on the real naming, not on a hand-written key."""
    import json
    from app.config import settings

    emp = _emp("Void Crop Person")
    ts = _at(9, 0)
    _pass(emp, CAM_IN, CameraRole.IN, ts, ENTER, score=0.512)
    local = ts.astimezone(settings.tz)
    stem = f"{local:%Y%m%d_%H%M%S}_0.512_Entrance_001"
    folder = settings.debug_dir / "Void Crop Person"
    folder.mkdir(parents=True, exist_ok=True)
    v = np.random.default_rng(2).standard_normal(512).astype(np.float32)
    v /= np.linalg.norm(v)
    try:
        (folder / f"{stem}.json").write_text(json.dumps({"score": 0.512}))
        with session_scope() as s:
            # The camera has to exist for the name to resolve.
            from app.db.models import Camera
            if s.get(Camera, CAM_IN) is None:
                s.add(Camera(id=CAM_IN, name="Entrance", role=CameraRole.IN,
                             rtsp_url="rtsp://x", enabled=False))
            s.add(FaceEmbedding(employee_id=emp, source_file=f"live:{stem}",
                                vector=v.tobytes(), dim=512, model_name="m",
                                threshold=0.35))
            s.add(FaceEmbedding(employee_id=emp, source_file="image_01.png",
                                vector=v.tobytes(), dim=512, model_name="m"))
        out = corrections.void_event(_events(emp)[0][0], by="admin", reason="wrong")
        assert out["capture"] == stem
        assert out["gallery_rows_removed"] == 1
        with session_scope() as s:
            left = s.execute(select(FaceEmbedding.source_file)
                             .where(FaceEmbedding.employee_id == emp)).scalars().all()
        assert left == ["image_01.png"], "the enrolment photo must be untouched"
    finally:
        for f in folder.glob(f"{stem}*"):
            f.unlink()
        if folder.is_dir():
            folder.rmdir()


def test_the_day_view_and_its_evidence_are_admin_gated():
    emp = _emp()
    _pass(emp, CAM_IN, CameraRole.IN, _at(9, 0), ENTER)
    day = business_date(_at(9, 0))
    eid = _events(emp)[0][0]
    for path in (f"/attendance/day/{emp}/{day}",
                 f"/attendance/event/{eid}/evidence/face"):
        assert client.get(path).status_code in (303, 401, 403)
    # The day view itself is readable by an operator - seeing the record is not
    # the same as rewriting it - but the void control and the face crops are not.
    r = client.get(f"/attendance/day/{emp}/{day}", headers=_as(admin=False))
    assert r.status_code == 200
    assert "Bu u emas" not in r.text
    assert client.get(f"/attendance/event/{eid}/evidence/face",
                      headers=_as(admin=False)).status_code == 403
    r = client.get(f"/attendance/day/{emp}/{day}", headers=_as(admin=True))
    assert r.status_code == 200 and "Bu u emas" in r.text


def test_the_evidence_route_rejects_an_unknown_kind():
    emp = _emp()
    _pass(emp, CAM_IN, CameraRole.IN, _at(9, 0), ENTER)
    eid = _events(emp)[0][0]
    assert client.get(f"/attendance/event/{eid}/evidence/../../etc/passwd",
                      headers=_as(admin=True)).status_code in (400, 404)
    assert client.get(f"/attendance/event/{eid}/evidence/passwd",
                      headers=_as(admin=True)).status_code == 400
