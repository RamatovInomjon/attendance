"""The second chance at naming a pass the per-frame vote could not.

Two rules meet here and they must not be confused with each other:

* the LIVE vote judges each frame and needs `vote_min_recognitions` of them to
  agree, so it fails on a short pass however good its frames were;
* the TRACKLET rule combines the same embeddings into one query and asks the
  same gallery once, at the SAME threshold the live matcher uses per frame.

What these tests pin is that the second rule cannot become a back door into the
first: it may never accept a similarity the live matcher rejects, it must
respect the margin, it must refuse a pass with too little evidence, and it must
leave a pass the live vote already named completely alone.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.config import settings
from app.core.gallery import Gallery
from app.core.pipeline import _unit


def _unit_rows(rng, n, d=512):
    v = rng.standard_normal((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class _Track:
    """The parts of a CompletedTrack the second chance reads."""

    def __init__(self, template, frames=4, ipd=30.0, employee_id=None):
        self.face_template = template
        self.face_frames = frames
        self.best_ipd = ipd
        self.employee_id = employee_id
        self.track_id = 7
        self.name = ""
        self.best_score = 0.0
        self.best_margin = 0.0
        self.embedded_frames = frames
        self.direction = "ENTER"


def _worker(gallery):
    """A CameraWorker without a camera, a database or a thread.

    `_second_chance` reads only `self.pipeline.gallery`, so the rule can be
    tested as the rule it is rather than through a running pipeline.
    """
    from app.services.worker import CameraWorker
    w = CameraWorker.__new__(CameraWorker)
    w.name = "test"
    w.pipeline = type("P", (), {"gallery": gallery})()
    return w


# --- the template itself ---------------------------------------------------

def test_the_template_is_a_unit_vector():
    rng = np.random.default_rng(0)
    E = _unit_rows(rng, 5, 16)
    w = np.array([0.9, 0.5, 1.0, 0.2, 0.7], np.float32)
    f = _unit((E * w[:, None]).sum(axis=0))
    assert np.linalg.norm(f) == pytest.approx(1.0, abs=1e-5)


def test_a_degenerate_sum_yields_no_template_rather_than_noise():
    """Two embeddings that cancel have no direction. Normalising that would
    scale numerical dust up to unit length and hand it to the gallery as a
    query, which is how a pass with no evidence gets a name."""
    v = np.array([1.0, 0.0, 0.0], np.float32)
    assert _unit(v - v) is None
    assert _unit(None) is None


def test_the_weighting_follows_the_frames_that_showed_a_face():
    """`w = aligner_score * sqrt(ipd)`. A close, confident frame must move the
    template further than a distant, doubtful one, or the whole point of
    aggregating a pass is lost - a plain mean lets the worst frames drag the
    template toward everybody."""
    rng = np.random.default_rng(1)
    close, far = _unit_rows(rng, 2, 32)
    w_close = 0.99 * np.sqrt(40.0)
    w_far = 0.91 * np.sqrt(8.0)
    f = _unit(close * w_close + far * w_far)
    assert float(f @ close) > float(f @ far)


# --- the rule --------------------------------------------------------------

def test_the_tracklet_threshold_is_never_below_the_live_one():
    """The safety property of the whole feature, and the one that survived
    contact with production.

    The rule used to be "stricter than live", set at 0.30 against a live 0.215.
    Measured on the deployment, that threw away half the genuine passes of
    everybody enrolled from a WEBCAM photograph, because a CCTV query scores
    ~0.2 lower against a webcam gallery row than against a CCTV one - so 0.30
    sat on the median of the population the feature exists to recover.

    What must hold is weaker and is the part that actually protects
    attendance: the second chance may never accept a similarity the live
    per-frame matcher would REJECT. Everything it adds beyond that is
    aggregation over the pass and the runner-up margin.
    """
    live = settings.threshold_for(settings.recognizer_model)
    assert settings.tracklet_face_threshold >= live


def test_a_pass_the_live_vote_named_is_left_alone():
    rng = np.random.default_rng(2)
    M = _unit_rows(rng, 3)
    g = Gallery(M, np.array([1, 2, 3]), {1: "A", 2: "B", 3: "C"})
    w = _worker(g)
    # Identical to person 1, so the rule WOULD fire - but the caller only asks
    # about passes with no identity, and the worker's branch is what enforces
    # that. Here we prove the rule itself has no opinion about a named pass:
    # it is the caller's guard, and the caller is tested by the branch below.
    t = _Track(M[0].copy(), employee_id=2)
    assert w._second_chance(t) is not None       # the rule fires...
    # ...and the worker only ever calls it when employee_id is None.
    import inspect
    from app.services import worker as wm
    src = inspect.getsource(wm.CameraWorker._persist_completed)
    assert "if ct.employee_id is None:" in src
    assert src.index("_second_chance") > src.index("if ct.employee_id is None:")


def test_a_clear_match_names_the_person():
    rng = np.random.default_rng(3)
    M = _unit_rows(rng, 3)
    g = Gallery(M, np.array([1, 2, 3]), {1: "A", 2: "B", 3: "C"})
    hit = _worker(g)._second_chance(_Track(M[1].copy()))
    assert hit is not None
    emp, score, margin = hit
    assert emp == 2
    assert score == pytest.approx(1.0, abs=1e-5)


def test_a_weak_template_names_nobody():
    rng = np.random.default_rng(4)
    M = _unit_rows(rng, 3)
    g = Gallery(M, np.array([1, 2, 3]), {1: "A", 2: "B", 3: "C"})
    # A random query against random rows scores far below 0.30.
    q = _unit_rows(rng, 1)[0]
    assert _worker(g)._second_chance(_Track(q)) is None


def test_two_people_the_template_resembles_equally_name_nobody():
    """The margin rule, and the reason it exists: a top score alone does not
    separate 'this person' from 'somebody between two people'."""
    d = 64
    a = np.zeros(d, np.float32); a[0] = 1.0
    b = np.zeros(d, np.float32); b[1] = 1.0
    g = Gallery(np.stack([a, b]), np.array([1, 2]), {1: "A", 2: "B"})
    between = _unit(a + b)
    hit = _worker(g)._second_chance(_Track(between))
    assert hit is None, "a query equidistant from two people must not name one"


def test_a_single_frame_is_not_a_second_chance():
    """One frame's template IS that frame, and the live matcher has already
    judged it. Re-asking at a different threshold is not new evidence."""
    rng = np.random.default_rng(5)
    M = _unit_rows(rng, 2)
    g = Gallery(M, np.array([1, 2]), {1: "A", 2: "B"})
    w = _worker(g)
    assert w._second_chance(_Track(M[0].copy(), frames=1)) is None
    assert w._second_chance(_Track(M[0].copy(), frames=2)) is not None


def test_a_pass_with_no_face_at_all_is_not_matched():
    rng = np.random.default_rng(6)
    M = _unit_rows(rng, 2)
    g = Gallery(M, np.array([1, 2]), {1: "A", 2: "B"})
    assert _worker(g)._second_chance(_Track(None, frames=9)) is None


def test_setting_the_threshold_to_zero_disables_the_feature():
    """The escape hatch has to work: a deployment that does not want the second
    chance must be able to turn it off without editing code."""
    rng = np.random.default_rng(7)
    M = _unit_rows(rng, 2)
    g = Gallery(M, np.array([1, 2]), {1: "A", 2: "B"})
    w = _worker(g)
    old = settings.tracklet_face_threshold
    try:
        settings.tracklet_face_threshold = 0.0
        assert w._second_chance(_Track(M[0].copy())) is None
    finally:
        settings.tracklet_face_threshold = old


def test_a_recovered_pass_loses_to_a_live_one_in_the_arbiter():
    """Both cameras see one walk. When one of them named the person by the live
    vote and the other only by the template, the live view must carry the group
    - its direction evidence and its snapshot are the better ones."""
    from app.services.arbiter import PendingPass
    track = _Track(None, frames=4)
    live = PendingPass(employee_id=1, camera_id=1, role=None, ts=None,
                       monotonic=0.0, track=track, snapshot=None, source="live")
    late = PendingPass(employee_id=1, camera_id=2, role=None, ts=None,
                       monotonic=0.0, track=track, snapshot=None,
                       source="tracklet")
    assert live.strength > late.strength
