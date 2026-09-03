"""`Gallery.match` decides every identity, and had no test.

Two changes are pinned here: the per-person reduction is now vectorised and
batched (it was a Python loop over every enrolment image, run once per face per
frame), and the constructor no longer normalises the caller's array in place.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.gallery import Gallery

THR, MAR = 0.18, 0.045


def _unit(rng, n=1):
    v = rng.standard_normal((n, 512)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _gallery(rng, people=6, per=5):
    """Each person is a tight cluster, so genuine and impostor are separable."""
    centres = _unit(rng, people)
    rows, owners = [], []
    for i, c in enumerate(centres):
        for _ in range(per):
            v = c + rng.normal(0, 0.05, 512).astype(np.float32)
            rows.append(v / np.linalg.norm(v))
            owners.append(100 + i)
    return (Gallery(np.stack(rows), np.array(owners, np.int64),
                    {100 + i: f"P{i}" for i in range(people)}),
            centres)


# --- C5: the constructor must not touch the caller's data ------------------

def test_construction_does_not_normalise_the_callers_array():
    """`np.asarray` returns the SAME object for a float32 input, and the `/=`
    then rewrote the caller's data. load_gallery survived on a fresh stack;
    enrolment and the benchmarks keep what they pass."""
    rng = np.random.default_rng(0)
    v = (rng.standard_normal((10, 512)) * 3).astype(np.float32)
    before = v.copy()
    Gallery(v, np.arange(10, dtype=np.int64), {})
    assert np.array_equal(v, before)


def test_vectors_are_normalised_internally_regardless():
    rng = np.random.default_rng(1)
    v = (rng.standard_normal((10, 512)) * 7).astype(np.float32)
    g = Gallery(v, np.arange(10, dtype=np.int64), {})
    assert np.allclose(np.linalg.norm(g.M, axis=1), 1.0, atol=1e-5)


# --- the two acceptance rules ---------------------------------------------

def test_a_genuine_query_matches_its_own_person():
    rng = np.random.default_rng(2)
    g, centres = _gallery(rng)
    m = g.match(centres[3], THR, MAR)
    assert m.employee_id == 103
    assert m.score > THR


def test_a_score_below_the_threshold_is_refused():
    rng = np.random.default_rng(3)
    g, _ = _gallery(rng)
    m = g.match(_unit(rng)[0], THR, 0.0)
    assert m.employee_id is None
    assert m.runner_up is not None, "a near miss should still say who it nearly was"


def test_the_margin_rule_refuses_a_lookalike():
    """Two people whose enrolments are nearly identical. The top score clears
    the threshold; the margin is what refuses it."""
    rng = np.random.default_rng(4)
    c = _unit(rng)[0]
    rows = [c, c + rng.normal(0, 0.001, 512).astype(np.float32)]
    rows = [r / np.linalg.norm(r) for r in rows]
    g = Gallery(np.stack(rows), np.array([1, 2], np.int64), {1: "A", 2: "B"})
    assert g.match(c, THR, 0.0).employee_id == 1        # threshold alone accepts
    assert g.match(c, THR, MAR).employee_id is None     # the margin refuses


def test_the_score_is_the_best_image_of_a_person_not_the_average():
    rng = np.random.default_rng(5)
    c = _unit(rng)[0]
    far = _unit(rng)[0]
    g = Gallery(np.stack([c, far]), np.array([1, 1], np.int64), {1: "A"})
    assert g.match(c, THR, 0.0).score == pytest.approx(1.0, abs=1e-4)


# --- D3: batching must not change a single answer -------------------------

def test_match_batch_agrees_with_match_face_by_face():
    rng = np.random.default_rng(6)
    g, centres = _gallery(rng)
    queries = np.stack(
        [centres[i % len(centres)] + rng.normal(0, 0.3, 512).astype(np.float32)
         for i in range(60)] + list(_unit(rng, 40)))
    batch = g.match_batch(queries, THR, MAR)
    for i, q in enumerate(queries):
        one = g.match(q, THR, MAR)
        assert batch[i].employee_id == one.employee_id
        assert batch[i].score == pytest.approx(one.score, abs=1e-5)
        assert batch[i].margin == pytest.approx(one.margin, abs=1e-5)
        assert batch[i].runner_up == one.runner_up


def test_match_batch_accepts_a_single_row():
    rng = np.random.default_rng(7)
    g, centres = _gallery(rng)
    assert len(g.match_batch(centres[0][None], THR, MAR)) == 1


# --- degenerate galleries -------------------------------------------------

def test_an_empty_gallery_matches_nobody():
    g = Gallery(np.zeros((0, 512), np.float32), np.zeros((0,), np.int64), {})
    assert g.match(np.ones(512, np.float32), THR, MAR).employee_id is None
    assert len(g.match_batch(np.ones((3, 512), np.float32), THR, MAR)) == 3


def test_a_one_person_gallery_has_no_runner_up_to_beat():
    """With nobody to compare against, the margin rule cannot fire - it must
    not accidentally reject the only enrolled person."""
    rng = np.random.default_rng(8)
    c = _unit(rng)[0]
    g = Gallery(c[None], np.array([1], np.int64), {1: "Only"})
    m = g.match(c, THR, MAR)
    assert m.employee_id == 1


# --- per-row acceptance floors --------------------------------------------
#
# A corridor crop added to the gallery does not inherit the global threshold;
# it carries its own, measured against how close another person already gets to
# it. See app/services/augment.py for why. These pin the mechanism that makes
# that a real restriction and not decoration.

def test_a_gallery_with_no_floors_is_untouched():
    """The overwhelmingly common case - nothing has been augmented - must take
    the same path and produce the same answers it always did."""
    rng = np.random.default_rng(21)
    g, centres = _gallery(rng)
    zeroed = Gallery(g.M, g.owner, g.names, np.zeros(len(g.M), np.float32))
    assert zeroed._floor is None
    for c in centres:
        a, b = zeroed.match(c, THR, MAR), g.match(c, THR, MAR)
        assert (a.employee_id, a.runner_up) == (b.employee_id, b.runner_up)
        assert a.score == pytest.approx(b.score, abs=1e-6)
        assert a.margin == pytest.approx(b.margin, abs=1e-6)


def test_a_row_below_its_own_floor_names_nobody():
    """The whole point: a query that clears the GLOBAL threshold against a
    floored row is still refused, because that row is only trusted higher up."""
    rng = np.random.default_rng(22)
    a, b, drift = _unit(rng, 3)
    # ~0.71 against a's row and ~0 against b's: comfortably over THR, with a
    # clear margin, and under a 0.9 floor. `drift` rather than `b` because a
    # query halfway between the two people is refused by the margin rule
    # instead, which would prove nothing about floors.
    q = a * 0.5 + drift * 0.5
    q /= np.linalg.norm(q)

    plain = Gallery(np.stack([a, b]), np.array([1, 2], np.int64), {1: "A", 2: "B"})
    assert plain.match(q, THR, MAR).employee_id == 1

    floored = Gallery(np.stack([a, b]), np.array([1, 2], np.int64), {1: "A", 2: "B"},
                      np.array([0.90, 0.0], np.float32))
    assert floored.match(q, THR, MAR).employee_id is None


def test_a_row_above_its_own_floor_still_names_its_person():
    """The floor must not be a wall: clearing it accepts, as an enrolment row
    clearing the global threshold does."""
    rng = np.random.default_rng(23)
    a, b = _unit(rng, 2)
    g = Gallery(np.stack([a, b]), np.array([1, 2], np.int64), {1: "A", 2: "B"},
                np.array([0.70, 0.0], np.float32))
    assert g.match(a, THR, MAR).employee_id == 1        # similarity 1.0


def test_a_persons_unfloored_row_still_answers_for_them():
    """Floors are per IMAGE, so an enrolment photo keeps working even when a
    corridor crop of the same person is trusted only at 0.7. Augmentation adds;
    it must never make a person harder to recognise than before."""
    rng = np.random.default_rng(24)
    a, b, drift = _unit(rng, 3)
    live = a * 0.6 + drift * 0.4
    live /= np.linalg.norm(live)
    g = Gallery(np.stack([a, live, b]), np.array([1, 1, 2], np.int64),
                {1: "A", 2: "B"}, np.array([0.0, 0.90, 0.0], np.float32))
    m = g.match(a, THR, MAR)
    assert m.employee_id == 1 and m.score == pytest.approx(1.0, abs=1e-5)


def test_a_floor_below_the_threshold_cannot_make_a_row_easier():
    """A stored floor may only ever tighten. Otherwise a row written with a low
    value would be a private back door past the recognition threshold - the
    exact failure the floors exist to prevent."""
    rng = np.random.default_rng(25)
    a, b = _unit(rng, 2)
    q = a * 0.3 + b * 0.7            # under THR against a's row
    q /= np.linalg.norm(q)
    g = Gallery(np.stack([a, b]), np.array([1, 2], np.int64), {1: "A", 2: "B"},
                np.array([0.01, 0.0], np.float32))
    lax = g.match(q, THR, MAR)
    strict = Gallery(np.stack([a, b]), np.array([1, 2], np.int64),
                     {1: "A", 2: "B"}).match(q, THR, MAR)
    assert lax.employee_id == strict.employee_id
    assert lax.score == pytest.approx(strict.score, abs=1e-6)


def test_batched_matching_applies_the_floors_too():
    """match_batch is the path the pipeline actually uses. A floor honoured
    only in match() would be honoured only in the tests."""
    rng = np.random.default_rng(26)
    a, b, drift = _unit(rng, 3)
    q = a * 0.5 + drift * 0.5
    q /= np.linalg.norm(q)
    g = Gallery(np.stack([a, b]), np.array([1, 2], np.int64), {1: "A", 2: "B"},
                np.array([0.90, 0.0], np.float32))
    batch = g.match_batch(np.stack([a, q]), THR, MAR)
    assert [m.employee_id for m in batch] == [1, None]
    for i, one in enumerate((g.match(a, THR, MAR), g.match(q, THR, MAR))):
        assert batch[i].employee_id == one.employee_id
        assert batch[i].score == pytest.approx(one.score, abs=1e-5)


def test_a_mismatched_floor_array_is_refused():
    """The floors are POSITIONAL. Silently accepting the wrong length would
    apply one row's floor to another - wrong in a way nothing would reveal."""
    rng = np.random.default_rng(27)
    v = _unit(rng, 4)
    with pytest.raises(ValueError, match="positional"):
        Gallery(v, np.array([1, 1, 2, 2], np.int64), {}, np.zeros(3, np.float32))
