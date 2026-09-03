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
    because the only floor that would make it safe - above its 0.99 similarity
    to Person B - is one it could never reach, so the crop is useless as well
    as dangerous, and the message names B rather than an unrelated pair.
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
        assert bad.threshold > 0.9

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


def test_the_floor_sits_just_above_the_worst_impostor():
    """Lowest FRR consistent with FAR 0: the floor is the smallest value that
    keeps every measured impostor out, and not one point higher."""
    from app.config import settings
    rng = np.random.default_rng(3)
    a, b = _unit(rng), _unit(rng)
    made = _people([("Person A", a), ("Person B", b)])
    try:
        # Deliberately blended so the similarity to B is a known, middling value.
        v = a * 0.8 + b * 0.2
        cand = _cand(made[0], v, "Person A")
        augment.calibrate([cand], np.stack([a, b]),
                          np.array(made), {made[0]: "Person A", made[1]: "Person B"})
        assert cand.impostor_name == "Person B"
        assert cand.threshold == pytest.approx(
            cand.impostor + settings.augment_threshold_margin)
        assert cand.threshold > cand.impostor, "the impostor must not get through"
    finally:
        _wipe(made)


def test_a_floor_never_drops_below_the_global_threshold():
    """A crop with no lookalike anywhere still answers to the global threshold.
    Being unlike everybody is not a licence to match at 0.05."""
    from app.config import settings
    rng = np.random.default_rng(5)
    a, b = _unit(rng), _unit(rng)
    made = _people([("Person A", a), ("Person B", b)])
    try:
        cand = _cand(made[0], a, "Person A")
        augment.calibrate([cand], np.stack([a, b]), np.array(made), {})
        assert cand.threshold >= settings.threshold_for(settings.recognizer_model)
    finally:
        _wipe(made)


def test_one_crop_raises_another_crop_s_floor():
    """Crops calibrate against EACH OTHER, not only against the studio photos.

    Two corridor faces of different people are the most realistic impostor pair
    available - the whole premise of the feature is that a corridor face and a
    studio face of the same person are 0.2 apart - so a floor measured against
    enrolment alone would be far too low.
    """
    rng = np.random.default_rng(13)
    a, b = _unit(rng), _unit(rng)
    made = _people([("Person A", a), ("Person B", b)])
    try:
        shared = _unit(rng)          # both crops drift toward the same point
        ca = _cand(made[0], a * 0.5 + shared * 0.5, "Person A")
        cb = _cand(made[1], b * 0.5 + shared * 0.5, "Person B")
        M, owner = np.stack([a, b]), np.array(made)

        alone = _cand(made[0], a * 0.5 + shared * 0.5, "Person A")
        augment.calibrate([alone], M, owner, {})
        augment.calibrate([ca, cb], M, owner, {})

        assert ca.threshold > alone.threshold
        assert ca.impostor_name == "Person B"
    finally:
        _wipe(made)


def test_adding_stores_the_floor_and_a_later_crop_refreshes_it():
    """A floor is a claim about the rest of the gallery, so it expires when the
    gallery changes. Adding a second crop must raise the first one's floor -
    otherwise the row added yesterday has an impostor it was never measured
    against, and the guarantee quietly stops being true."""
    from sqlalchemy import select
    from app.db.models import FaceEmbedding
    from app.db.session import session_scope
    rng = np.random.default_rng(17)
    a, b = _unit(rng), _unit(rng)
    made = _people([("Person A", a), ("Person B", b)])
    try:
        shared = _unit(rng)
        ca = _cand(made[0], a * 0.5 + shared * 0.5, "Person A")
        augment.calibrate([ca], np.stack([a, b]), np.array(made), {})
        augment.add([ca])

        def floor_of(emp):
            with session_scope() as s:
                return s.execute(
                    select(FaceEmbedding.threshold).where(
                        FaceEmbedding.employee_id == emp,
                        FaceEmbedding.source_file.like(f"{augment.TAG}%"))
                ).scalar()

        first = floor_of(made[0])
        assert first is not None and first > 0

        cb = _cand(made[1], b * 0.5 + shared * 0.5, "Person B")
        cb.key = "kB"
        augment.check_impostors([cb])        # calibrates against the stored crop
        augment.add([cb])

        assert floor_of(made[0]) > first, "the earlier crop must be re-measured"

        augment.remove([e["id"] for e in augment.added()
                        if e["employee_id"] == made[1]])
        assert floor_of(made[0]) == pytest.approx(first), \
            "removing the impostor must hand the range back"
    finally:
        _wipe(made)


def test_an_empty_selection_reports_the_gallery_and_adds_nothing():
    chk = augment.check_impostors([])
    assert chk.after == -1.0 and chk.pair is None
    assert not chk.unusable
