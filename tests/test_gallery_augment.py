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


def test_a_candidate_that_looks_like_someone_else_is_refused():
    """The failure this feature must not cause: adding a crop that pushes two
    DIFFERENT people to match each other above the threshold."""
    from sqlalchemy import select
    from app.db.models import Employee, FaceEmbedding
    from app.db.session import session_scope
    rng = np.random.default_rng(7)

    a = rng.standard_normal(512).astype(np.float32); a /= np.linalg.norm(a)
    b = rng.standard_normal(512).astype(np.float32); b /= np.linalg.norm(b)
    made = []
    with session_scope() as s:
        for nm, v in (("Person A", a), ("Person B", b)):
            e = Employee(full_name=nm, is_active=True); s.add(e); s.flush()
            s.add(FaceEmbedding(employee_id=e.id, source_file="enrol",
                                vector=v.tobytes(), dim=512, model_name="m"))
            made.append(e.id)

    # A crop labelled A that is really almost B: exactly the poisoning case.
    bad = _cand(made[0], b * 0.99 + a * 0.01, "Person A")
    chk = augment.check_impostors([bad])
    assert not chk.safe
    assert chk.pair is not None and any("Person" in p for p in chk.pair)

    good = _cand(made[0], a * 0.98 + rng.standard_normal(512).astype(np.float32) * 0.02,
                 "Person A")
    assert augment.check_impostors([good]).safe

    with session_scope() as s:
        s.execute(FaceEmbedding.__table__.delete().where(
            FaceEmbedding.employee_id.in_(made)))
        s.execute(Employee.__table__.delete().where(Employee.id.in_(made)))


def test_an_empty_selection_changes_nothing():
    chk = augment.check_impostors([])
    assert chk.before == chk.after
