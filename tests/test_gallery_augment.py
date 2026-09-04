"""Adding corridor faces to the gallery, from the web UI.

Two things carry the weight here, and both are about damage a mistake would do
rather than about the feature working:

* **A mis-added crop is permanent and compounding.** It becomes a reference for
  the wrong person, so the same error gets easier next time, and nothing in the
  attendance data reveals it. Hence an admin gate AND a machine check that
  refuses a selection which makes two different people match.
* **The crops live outside the public media mount on purpose.** They are face
  images of identified people. Serving them by a caller-supplied key is a path
  traversal waiting to happen.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.core.security import COOKIE_NAME, sign_session
from app.services import augment

client = TestClient(app, follow_redirects=False)


# Explicit header rather than TestClient's `cookies=`, which MERGES with any
# cookie the client already holds - see tests/test_auth.py.
def _as(admin: bool) -> dict:
    tok = sign_session(user_id=1 if admin else 2,
                       username="inomjon" if admin else "operator",
                       is_admin=admin)
    return {"Cookie": f"{COOKIE_NAME}={tok}"}


# --- serving crops must not become a file-read primitive -------------------

@pytest.mark.parametrize("key", [
    "../../../etc/passwd",
    "..%2f..%2fetc%2fpasswd",
    "/etc/passwd",
    "....//....//etc/passwd",
    "a/../../../../root/.ssh/id_rsa",
])
def test_a_traversing_key_resolves_to_nothing(key):
    """`data/debug` sits outside the public mount deliberately - it holds face
    images. crop_path() resolves and checks containment, so a key that escapes
    the directory returns None rather than a file."""
    assert augment.crop_path(key) is None


def test_an_unknown_key_is_not_an_error():
    assert augment.crop_path("20990101_000000_0.999_Nowhere_001") is None


# --- every route is admin-only ---------------------------------------------

GALLERY_ROUTES = [
    ("get", "/gallery/review"),
    ("get", "/gallery/crop/anything"),
    ("post", "/gallery/augment"),
    ("post", "/gallery/augment/remove"),
]


@pytest.mark.parametrize("method,path", GALLERY_ROUTES)
def test_the_gallery_routes_refuse_an_operator(method, path):
    """A non-admin operator can sign in and use the rest of the app. Changing
    who the gallery thinks people are is not theirs to do."""
    r = getattr(client, method)(path, headers=_as(admin=False))
    assert r.status_code in (403, 303), (path, r.status_code)


@pytest.mark.parametrize("method,path", GALLERY_ROUTES)
def test_the_gallery_routes_refuse_an_anonymous_caller(method, path):
    r = getattr(client, method)(path)
    assert r.status_code in (303, 401, 403), (path, r.status_code)


@pytest.mark.parametrize("method,path", GALLERY_ROUTES)
def test_an_admin_is_not_refused(method, path):
    """The mirror of the two above: the gate must not lock everybody out."""
    r = getattr(client, method)(path, headers=_as(admin=True))
    assert r.status_code != 403, (path, r.status_code)


# --- removal must never touch the enrolment --------------------------------

def test_removal_only_deletes_corridor_additions():
    """The remove button is exposed in a UI, so its blast radius matters: it
    filters on the `live:` tag, and an enrolment row's id must be inert."""
    from sqlalchemy import select
    from app.db.models import Employee, FaceEmbedding
    from app.db.session import session_scope

    v = np.random.default_rng(0).standard_normal(512).astype(np.float32)
    with session_scope() as s:
        e = Employee(full_name="Augment Target", is_active=True)
        s.add(e); s.flush()
        enrol = FaceEmbedding(employee_id=e.id, source_file="face_id_users/x.png",
                              vector=v.tobytes(), dim=512, model_name="m", quality=0.9)
        live = FaceEmbedding(employee_id=e.id, source_file=f"{augment.TAG}cap1",
                             vector=v.tobytes(), dim=512, model_name="m", quality=0.4)
        s.add_all([enrol, live]); s.flush()
        enrol_id, live_id, emp_id = enrol.id, live.id, e.id

    assert augment.remove([enrol_id]) == 0, "an enrolment row must be untouchable"
    assert augment.remove([live_id]) == 1

    with session_scope() as s:
        left = s.execute(select(FaceEmbedding.id).where(
            FaceEmbedding.employee_id == emp_id)).scalars().all()
        assert left == [enrol_id]
        s.execute(FaceEmbedding.__table__.delete().where(
            FaceEmbedding.employee_id == emp_id))
        s.execute(Employee.__table__.delete().where(Employee.id == emp_id))


def test_removing_nothing_is_not_an_error():
    assert augment.remove([]) == 0


# --- the check that decides whether an addition is allowed -----------------

def _cand(emp, vec, name="X"):
    c = augment.Candidate(key=f"k{emp}", employee_id=emp, name=name,
                          camera="Entrance", when="", score=0.5)
    c.vec = vec / np.linalg.norm(vec)
    return c


def _people(names_and_vectors):
    """Enrol some employees with one embedding each; returns their ids."""
    from app.db.models import Employee, FaceEmbedding
    from app.db.session import session_scope
    made = []
    with session_scope() as s:
        for nm, v in names_and_vectors:
            e = Employee(full_name=nm, is_active=True); s.add(e); s.flush()
            s.add(FaceEmbedding(employee_id=e.id, source_file="enrol",
                                vector=np.asarray(v, np.float32).tobytes(),
                                dim=512, model_name="m"))
            made.append(e.id)
    return made


def _wipe(ids):
    from app.db.models import Employee, FaceEmbedding
    from app.db.session import session_scope
    with session_scope() as s:
        s.execute(FaceEmbedding.__table__.delete().where(
            FaceEmbedding.employee_id.in_(ids)))
        s.execute(Employee.__table__.delete().where(Employee.id.in_(ids)))


def _unit(rng):
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_a_candidate_that_looks_like_someone_else_is_refused():
    """The failure this feature must not cause: adding a crop that would let
    somebody ELSE be named as the person it is labelled with.

    It is no longer refused for moving a gallery-wide statistic. It is refused
    because B already reaches it at 0.99, far above the floor a corridor crop
    answers to - so it is a lookalike, not a reference - and the message names
    B rather than an unrelated pair of enrolment photographs.
    """
    rng = np.random.default_rng(7)
    a, b = _unit(rng), _unit(rng)
    made = _people([("Person A", a), ("Person B", b)])
    try:
        bad = _cand(made[0], b * 0.99 + a * 0.01, "Person A")
        chk = augment.check_impostors([bad])
        assert not chk.safe
        assert [u[0] for u in chk.unusable] == ["Person A"]
        assert bad.impostor_name == "Person B"
        assert bad.impostor > 0.9

        good = _cand(made[0], a * 0.98 + _unit(rng) * 0.02, "Person A")
        chk = augment.check_impostors([good])
        assert chk.safe and not chk.unusable
    finally:
        _wipe(made)


def test_a_pre_existing_enrolment_pair_does_not_block_an_addition():
    """The bug this replaced, reproduced.

    Two ENROLMENT photographs sitting above the threshold is a real problem -
    Narmatov/Qo'shmatov reached 0.208 against a 0.190 threshold on the live
    server - but it is not caused by any crop and no selection can cure it. The
    old gate measured the gallery's worst pair with the selection included, so
    that pair shut augmentation permanently while the message blamed whichever
    crop had been chosen. It must be REPORTED and must not refuse.
    """
    from app.config import settings
    rng = np.random.default_rng(11)
    thr = settings.threshold_for(settings.recognizer_model)

    a, noise, c = _unit(rng), _unit(rng), _unit(rng)
    # Two enrolment vectors deliberately far above the threshold from each other.
    look_alike = a * 0.9 + noise * 0.1
    look_alike /= np.linalg.norm(look_alike)
    made = _people([("Twin One", a), ("Twin Two", look_alike), ("Unrelated", c)])
    try:
        baseline = augment.check_impostors([])
        assert baseline.before > thr, "the fixture must reproduce the bad pair"
        assert baseline.gallery_unsafe
        assert baseline.pre_existing is not None

        # A crop for the unrelated third person, nowhere near either twin.
        crop = _cand(made[2], c * 0.97 + _unit(rng) * 0.03, "Unrelated")
        chk = augment.check_impostors([crop])
        assert chk.safe, "a pre-existing enrolment pair must not refuse this"
        assert chk.gallery_unsafe, "...but it must still be reported"
        assert chk.before == pytest.approx(baseline.before)
    finally:
        _wipe(made)


def _probes(vectors, nearest=None):
    """The corridor-probe tuple `calibrate` takes, without touching the DB."""
    P = np.stack([np.asarray(v, np.float32) for v in vectors])
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
    n = np.array(nearest if nearest is not None else [-1] * len(P), np.int64)
    return P, n


def test_every_corridor_crop_gets_the_same_flat_floor():
    """The floor is a policy, not a per-crop measurement.

    A per-row floor from each crop's own worst impostor measured WORSE on both
    axes than one flat number - a maximum over a few hundred probes is a noisy
    extreme-value estimate. See the table in app/config.py.
    """
    from app.config import settings
    rng = np.random.default_rng(3)
    a, b = _unit(rng), _unit(rng)
    made = _people([("Person A", a), ("Person B", b)])
    try:
        near = _cand(made[0], a * 0.8 + b * 0.2, "Person A")
        far = _cand(made[0], a, "Person A"); far.key = "kfar"
        augment.calibrate([near, far], np.stack([a, b]), np.array(made),
                          {made[0]: "Person A", made[1]: "Person B"},
                          probes=_probes([_unit(rng)]))
        want = max(settings.threshold_for(settings.recognizer_model),
                   settings.augment_live_floor)
        assert near.threshold == pytest.approx(want)
        assert far.threshold == pytest.approx(want)
        assert near.impostor > far.impostor, "the measurement still differs"
    finally:
        _wipe(made)


def test_a_floor_never_drops_below_the_global_threshold():
    """If the operating threshold is raised above the live floor, the floor
    follows it up. A corridor crop must never be the easiest row to match."""
    from app.config import settings
    rng = np.random.default_rng(5)
    a, b = _unit(rng), _unit(rng)
    made = _people([("Person A", a), ("Person B", b)])
    try:
        cand = _cand(made[0], a, "Person A")
        augment.calibrate([cand], np.stack([a, b]), np.array(made), {},
                          probes=_probes([_unit(rng)]))
        assert cand.threshold >= settings.threshold_for(settings.recognizer_model)
        assert cand.threshold >= settings.augment_live_floor
    finally:
        _wipe(made)


def test_a_crop_a_real_corridor_face_already_reaches_is_refused():
    """The measurement's job is now REFUSAL, and the probe set is what makes it
    work. A crop that an anonymous corridor face reaches above the floor is a
    lookalike: keeping it would let that face be given this person's name."""
    rng = np.random.default_rng(31)
    a, b = _unit(rng), _unit(rng)
    made = _people([("Person A", a), ("Person B", b)])
    try:
        crop = a * 0.7 + _unit(rng) * 0.3
        stranger = crop * 0.9 + _unit(rng) * 0.1      # ~0.99 against the crop
        c = _cand(made[0], crop, "Person A")
        augment.calibrate([c], np.stack([a, b]), np.array(made), {},
                          probes=_probes([stranger]))
        assert c.rejected, "a face reaching it above the floor must refuse it"
        assert "corridor face" in c.impostor_name
        assert "reaches this crop" in c.rejected
    finally:
        _wipe(made)


def test_a_probe_the_pipeline_already_attributed_to_this_person_is_ignored():
    """The probe set is faces the system named NOBODY - which includes enrolled
    people it simply missed. Counting those as impostors would refuse a crop
    for resembling its own subject, which is the exact case it exists to fix."""
    rng = np.random.default_rng(37)
    a, b = _unit(rng), _unit(rng)
    made = _people([("Person A", a), ("Person B", b)])
    try:
        crop = a * 0.7 + _unit(rng) * 0.3
        himself = crop * 0.9 + _unit(rng) * 0.1
        blamed = _cand(made[0], crop, "Person A")
        augment.calibrate([blamed], np.stack([a, b]), np.array(made), {},
                          probes=_probes([himself], nearest=[-1]))
        assert blamed.rejected, "an unattributed probe must still count"

        spared = _cand(made[0], crop, "Person A")
        augment.calibrate([spared], np.stack([a, b]), np.array(made), {},
                          probes=_probes([himself], nearest=[made[0]]))
        assert not spared.rejected, "a probe the system read as THIS person must not"
    finally:
        _wipe(made)


def test_a_sibling_candidate_too_close_is_refused_not_floored():
    """Two crops of different people that resemble each other are lookalikes.
    The old code gave them each a private higher floor, which hid the problem
    inside a number; now one of them is refused and the collision is named."""
    rng = np.random.default_rng(13)
    a, b = _unit(rng), _unit(rng)
    made = _people([("Person A", a), ("Person B", b)])
    try:
        shared = _unit(rng)
        ca = _cand(made[0], a * 0.15 + shared * 0.85, "Person A")
        cb = _cand(made[1], b * 0.15 + shared * 0.85, "Person B")
        cb.key = "kB"
        augment.calibrate([ca, cb], np.stack([a, b]), np.array(made), {},
                          probes=_probes([_unit(rng)]))
        assert ca.rejected and cb.rejected
        assert ca.impostor_name == "Person B"
    finally:
        _wipe(made)


def test_recalibrate_puts_every_stored_crop_on_the_current_floor():
    """The floor moves when `recognition_threshold_override` does, and stored
    rows keep whatever number was written when they were added. A row left on
    an old, lower floor is a row that stopped being protected."""
    from sqlalchemy import select, update
    from app.config import settings
    from app.db.models import FaceEmbedding
    from app.db.session import session_scope
    rng = np.random.default_rng(17)
    a, b = _unit(rng), _unit(rng)
    made = _people([("Person A", a), ("Person B", b)])
    try:
        ca = _cand(made[0], a * 0.8 + _unit(rng) * 0.2, "Person A")
        augment.calibrate([ca], np.stack([a, b]), np.array(made), {},
                          probes=_probes([_unit(rng)]))
        augment.add([ca])

        def floor_of():
            with session_scope() as s:
                return s.execute(select(FaceEmbedding.threshold).where(
                    FaceEmbedding.employee_id == made[0],
                    FaceEmbedding.source_file.like(f"{augment.TAG}%"))).scalar()

        want = max(settings.threshold_for(settings.recognizer_model),
                   settings.augment_live_floor)
        assert floor_of() == pytest.approx(want)

        # Simulate a row written under an older, lower policy.
        with session_scope() as s:
            s.execute(update(FaceEmbedding)
                      .where(FaceEmbedding.source_file.like(f"{augment.TAG}%"))
                      .values(threshold=0.19))
        assert floor_of() == pytest.approx(0.19)
        assert augment.recalibrate() >= 1
        assert floor_of() == pytest.approx(want), \
            "recalibrate must lift a stale floor back to the policy"
    finally:
        augment.remove([e["id"] for e in augment.added()])
        _wipe(made)


def test_an_empty_selection_reports_the_gallery_and_adds_nothing():
    chk = augment.check_impostors([])
    assert chk.after == -1.0 and chk.pair is None
    assert not chk.unusable
