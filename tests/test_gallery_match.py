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
