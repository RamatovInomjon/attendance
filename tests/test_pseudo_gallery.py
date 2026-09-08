"""Grouping the passes of people who were never enrolled.

The properties that matter are not "does it cluster well" - that was measured
on three days of real passes and is recorded in the module docstring. What
tests can pin are the rules that keep the feature from becoming something it
must not be:

* a pseudo-person is named by FACE and the enrolment gallery, NEVER by body;
* body evidence is confined to one business date, because clothing changes;
* body evidence is confined to one ReID model, because two models' features
  compare successfully and mean nothing;
* a face too small to trust is discarded rather than down-weighted;
* naming a group relabels it and writes no attendance.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from app.config import settings
from app.db.models import PseudoPerson, ReidPass
from app.db.session import session_scope
from app.services.pseudo_gallery import PseudoGallery, _pack, _push, _unpack

TODAY = date(2026, 9, 8)
YESTERDAY = date(2026, 9, 7)
NOW = datetime(2026, 9, 8, 9, 0, tzinfo=timezone.utc)
MODEL = "reid_osnet_x0_75_256x128_e512_fp16.onnx"


def _v(seed, d=8):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d).astype(np.float32)
    return v / np.linalg.norm(v)


def _near(v, seed, amount=0.02):
    """A vector the same person would plausibly produce: same direction, jittered."""
    rng = np.random.default_rng(seed)
    w = v + rng.standard_normal(len(v)).astype(np.float32) * amount
    return w / np.linalg.norm(w)


def _row(s, *, day=TODAY, when=NOW, camera=1, track=1, employee_id=None):
    r = ReidPass(camera_id=camera, camera_name="Entrance", track_id=track,
                 first_seen=when, last_seen=when, business_date=day,
                 direction="ENTER", employee_id=employee_id, name="",
                 folder="", crops=5, dim=8, model_name=MODEL)
    s.add(r)
    s.flush()
    return r


def _clean(s):
    for p in s.query(PseudoPerson).all():
        s.delete(p)
    for r in s.query(ReidPass).all():
        s.delete(r)
    s.flush()


# --- the template store ----------------------------------------------------

def test_the_ring_buffer_keeps_the_newest_k():
    blob, dim = None, 4
    for i in range(8):
        blob = _push(blob, dim, np.full(4, float(i), np.float32), k=5)
    got = _unpack(blob, dim)
    assert got.shape == (5, 4)
    assert [row[0] for row in got] == [3.0, 4.0, 5.0, 6.0, 7.0]


def test_a_blob_that_does_not_divide_by_the_dimension_is_discarded():
    """Written under another model. Reshaping it anyway would produce finite,
    unit-ish, entirely meaningless vectors - which is worse than none."""
    blob = _pack(np.zeros((3, 5), np.float32))
    assert len(_unpack(blob, 5)) == 3
    assert len(_unpack(blob, 4)) == 0


# --- grouping --------------------------------------------------------------

def test_two_passes_of_one_face_land_on_one_pseudo_person():
    face = _v(1)
    with session_scope() as s:
        _clean(s)
        g = PseudoGallery()
        a = g.place(s, _row(s, track=1), face=face, body=_v(2),
                    face_ipd=30.0, body_model=MODEL)
        b = g.place(s, _row(s, track=2), face=_near(face, 3), body=_v(4),
                    face_ipd=30.0, body_model=MODEL)
        assert a is not None and b is not None
        assert a.id == b.id
        assert a.code.startswith("P-")
        assert a.n_passes == 2
        _clean(s)


def test_two_different_faces_open_two_pseudo_people():
    with session_scope() as s:
        _clean(s)
        g = PseudoGallery()
        a = g.place(s, _row(s, track=1), face=_v(10), body=_v(11),
                    face_ipd=30.0, body_model=MODEL)
        b = g.place(s, _row(s, track=2), face=_v(12), body=_v(13),
                    face_ipd=30.0, body_model=MODEL)
        assert a.id != b.id
        _clean(s)


def test_body_links_a_pass_that_has_no_usable_face():
    """This is body's whole contribution, and the reason it is kept: 27% of
    passes contain no face at all, and linking those is what halves the number
    of pseudo-people."""
    body = _v(20)
    with session_scope() as s:
        _clean(s)
        g = PseudoGallery()
        a = g.place(s, _row(s, track=1), face=_v(21), body=body,
                    face_ipd=30.0, body_model=MODEL)
        b = g.place(s, _row(s, track=2), face=None, body=_near(body, 22, 0.01),
                    body_model=MODEL)
        assert a.id == b.id
        assert b.n_passes == 2
        _clean(s)


def test_a_small_face_is_discarded_rather_than_trusted():
    """Below `pseudo_face_ipd_min` a face embedding reaches 0.35 against
    strangers, so it is a source of WRONG links. The pass falls through to
    body, which is the honest answer for a face that far away."""
    face = _v(30)
    with session_scope() as s:
        _clean(s)
        g = PseudoGallery()
        a = g.place(s, _row(s, track=1), face=face, body=_v(31),
                    face_ipd=40.0, body_model=MODEL)
        # Same face, but too small to trust, and an unrelated body.
        b = g.place(s, _row(s, track=2), face=face, body=_v(32),
                    face_ipd=settings.pseudo_face_ipd_min - 1.0,
                    body_model=MODEL)
        assert a.id != b.id, "a sub-threshold face must not link two passes"
        _clean(s)


def test_yesterdays_body_template_does_not_link_todays_pass():
    """Clothing changes overnight. A stale body template links whoever is
    wearing a similar coat today, which is the failure that makes body-only
    grouping useless in the first place."""
    body = _v(40)
    with session_scope() as s:
        _clean(s)
        g = PseudoGallery()
        a = g.place(s, _row(s, day=YESTERDAY, when=NOW - timedelta(days=1),
                            track=1),
                    face=None, body=body, body_model=MODEL)
        b = g.place(s, _row(s, day=TODAY, track=2), face=None,
                    body=_near(body, 41, 0.005), body_model=MODEL)
        assert a.id != b.id
        _clean(s)


def test_a_face_template_still_links_across_days():
    """The other half of the same rule: a face is the same face tomorrow, and
    it is the only key that works across a day boundary."""
    face = _v(50)
    with session_scope() as s:
        _clean(s)
        g = PseudoGallery()
        a = g.place(s, _row(s, day=YESTERDAY, when=NOW - timedelta(days=1),
                            track=1),
                    face=face, body=_v(51), face_ipd=30.0, body_model=MODEL)
        b = g.place(s, _row(s, day=TODAY, track=2), face=_near(face, 52),
                    body=_v(53), face_ipd=30.0, body_model=MODEL)
        assert a.id == b.id
        _clean(s)


def test_body_templates_from_another_model_are_not_compared():
    """Two ReID models' features have the same shape, the same norm and
    plausible cosines. Nothing errors; the matching is simply wrong."""
    body = _v(60)
    with session_scope() as s:
        _clean(s)
        g = PseudoGallery()
        a = g.place(s, _row(s, track=1), face=None, body=body,
                    body_model="reid_resnet101_ibn_256x128_e2048_fp16.onnx")
        b = g.place(s, _row(s, track=2), face=None, body=_near(body, 61, 0.005),
                    body_model=MODEL)
        assert a.id != b.id
        _clean(s)


# --- naming ----------------------------------------------------------------

def test_a_group_is_named_by_the_registry_and_its_history_is_relabelled():
    face = _v(70)
    with session_scope() as s:
        _clean(s)
        g = PseudoGallery()
        first = _row(s, track=1)
        p = g.place(s, first, face=face, body=_v(71), face_ipd=30.0,
                    body_model=MODEL)
        assert p.employee_id is None
        # A later pass of the same person that the face path DID name.
        named = _row(s, track=2, employee_id=42)
        p2 = g.place(s, named, face=_near(face, 72), body=_v(73),
                     face_ipd=30.0, body_model=MODEL,
                     registry_match=(42, 0.41, "Ismoilov Botir"), create=False)
        assert p2 is not None and p2.id == p.id
        assert p2.employee_id == 42
        # Flush before refresh: `refresh` re-reads the row, and an unflushed
        # change would be silently reverted rather than checked.
        s.flush()
        s.refresh(first)
        assert first.employee_id == 42, "the earlier pass must be relabelled"
        assert first.name == "Ismoilov Botir"
        _clean(s)


def test_a_named_pass_never_mints_a_pseudo_person():
    """A recognised employee is not a new visitor. Letting them open a
    pseudo-identity would make the distinct-visitor count meaningless."""
    with session_scope() as s:
        _clean(s)
        g = PseudoGallery()
        got = g.place(s, _row(s, track=1, employee_id=9), face=_v(80),
                      body=_v(81), face_ipd=30.0, body_model=MODEL,
                      registry_match=(9, 0.5, "Somebody"), create=False)
        assert got is None
        assert s.query(PseudoPerson).count() == 0
        _clean(s)


def test_body_similarity_alone_never_names_anybody():
    """The rule the whole module exists to protect. At this corridor's base
    rate - ~45 recoverable employees among ~1100 unknown tracks a day - a body
    rule writes more wrong attendance rows than right ones at EVERY threshold
    measured."""
    body = _v(90)
    with session_scope() as s:
        _clean(s)
        g = PseudoGallery()
        a = g.place(s, _row(s, track=1), face=None, body=body, body_model=MODEL)
        # A second pass joins by body, and carries a registry match for nobody.
        b = g.place(s, _row(s, track=2), face=None,
                    body=_near(body, 91, 0.005), body_model=MODEL)
        assert a.id == b.id                 # grouped...
        assert b.employee_id is None        # ...and still nameless
        _clean(s)


def test_a_pass_with_neither_feature_is_not_placed():
    with session_scope() as s:
        _clean(s)
        g = PseudoGallery()
        assert g.place(s, _row(s), face=None, body=None) is None
        _clean(s)


def test_the_pass_records_how_it_was_grouped():
    """`pseudo_by` is what makes a later audit possible: a group assembled by
    body within one day and a group assembled by face across three are not the
    same kind of claim, and after the fact nothing else can tell them apart."""
    face = _v(100)
    with session_scope() as s:
        _clean(s)
        g = PseudoGallery()
        r1 = _row(s, track=1)
        g.place(s, r1, face=face, body=_v(101), face_ipd=30.0, body_model=MODEL)
        r2 = _row(s, track=2)
        g.place(s, r2, face=_near(face, 102), body=_v(103), face_ipd=30.0,
                body_model=MODEL)
        assert r1.pseudo_by == "new"
        assert r2.pseudo_by == "face"
        assert r2.pseudo_score >= settings.pseudo_face_threshold
        _clean(s)
