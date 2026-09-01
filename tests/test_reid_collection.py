"""Body-crop collection and cross-camera ReID.

Two properties carry the weight here:

* **Collection must not touch the frame budget.** The pipeline produces crops on
  a list; the worker's queue drops rather than blocks. A data-collection feature
  that can delay recognition is worse than no feature.
* **The crop cadence and cap must hold**, because this writes images of every
  person who passes, recognised or not, and an unbounded version fills a disk
  that is already at 91%.
"""
from __future__ import annotations

import queue
from types import SimpleNamespace

import numpy as np
import pytest

from app.config import settings
from app.core.reid import aggregate


# --- the tracklet feature --------------------------------------------------

def _unit(rng, n, d=8):
    v = rng.standard_normal((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_a_single_crop_aggregates_to_itself():
    rng = np.random.default_rng(0)
    e = _unit(rng, 1)
    assert np.allclose(aggregate(e), e[0], atol=1e-6)


def test_the_tracklet_feature_is_unit_length():
    rng = np.random.default_rng(1)
    f = aggregate(_unit(rng, 6), scores=[0.4, 0.9, 0.5, 0.8, 0.3, 0.7],
                  sharpness=[10, 900, 40, 700, 5, 600])
    assert np.linalg.norm(f) == pytest.approx(1.0, abs=1e-5)


def test_quality_weighting_favours_the_better_crops():
    """The weighting is not decoration: a PLAIN mean lets blurred crops drag the
    template toward a generic body, which makes FNIR worse. The research
    project measured that; this pins the behaviour that avoids it."""
    rng = np.random.default_rng(2)
    good = _unit(rng, 1)[0]
    bad = _unit(rng, 1)[0]
    embs = np.stack([good, bad])

    weighted = aggregate(embs, scores=[0.95, 0.20], sharpness=[900.0, 5.0])
    plain = aggregate(embs, scores=None, sharpness=None)
    assert float(weighted @ good) > float(plain @ good), (
        "a high-confidence, sharp crop must dominate a weak one")


def test_aggregation_ignores_absolute_sharpness_scale():
    """Laplacian variance depends on crop size and lighting, so only the
    ordering within a pass is meaningful."""
    rng = np.random.default_rng(3)
    e = _unit(rng, 4)
    a = aggregate(e, [0.5] * 4, [10.0, 20.0, 30.0, 40.0])
    b = aggregate(e, [0.5] * 4, [1000.0, 2000.0, 3000.0, 4000.0])
    assert float(a @ b) == pytest.approx(1.0, abs=1e-5)


# --- the crop cadence ------------------------------------------------------

class _FakeTrack:
    def __init__(self):
        self.last_body_save = 0.0
        self.body_saves = 0


def _cadence(times, interval, cap):
    """The rule as the pipeline applies it, in isolation."""
    st, kept = _FakeTrack(), []
    for t in times:
        if (interval > 0 and st.body_saves < cap
                and t - st.last_body_save >= interval):
            kept.append(t)
            st.last_body_save, st.body_saves = t, st.body_saves + 1
    return kept


def test_crops_are_taken_on_the_interval_not_every_frame():
    t0 = 1_700_000_000.0
    frames = [t0 + i / 20.0 for i in range(200)]      # 10 s at 20 fps
    kept = _cadence(frames, 2.0, 8)
    gaps = np.diff(kept)
    # 200 frames at 20 fps spans t=0..9.95, so crops land at 0,2,4,6,8.
    assert len(kept) == 5, kept
    assert all(g >= 2.0 - 1e-6 for g in gaps)


def test_the_per_pass_cap_bounds_a_loiterer():
    """13% of live passes run over 60 s - people standing still. Without a cap
    one of them writes crops for as long as they stay."""
    t0 = 1_700_000_000.0
    frames = [t0 + i / 20.0 for i in range(20 * 600)]   # 10 minutes
    assert len(_cadence(frames, 2.0, 8)) == 8


def test_zero_interval_disables_collection():
    t0 = 1_700_000_000.0
    assert _cadence([t0 + i for i in range(50)], 0.0, 8) == []


def test_the_configured_cadence_is_bounded():
    assert settings.body_crop_interval_s > 0
    assert 0 < settings.body_crop_max_per_pass <= 32


# --- the queue must drop, never block --------------------------------------

def test_submitting_crops_never_blocks_the_capture_thread():
    """This is the whole reason ReID is on another thread. If the queue could
    block, a slow embed would delay the next frame."""
    from app.services.reid_worker import ReidWorker

    w = ReidWorker(model_path="unused", queue_size=4)
    w.enabled = True                       # no model loaded; we only test intake
    crops = [SimpleNamespace(track_id=1, ts=float(i), image=np.zeros((4, 4, 3), np.uint8),
                             score=0.5, first_seen=0.0) for i in range(50)]
    w.submit_crops(1, "Entrance", crops)   # must return, not hang
    assert w.q.qsize() <= 4
    assert w.crops_dropped >= 40, "a full queue must drop, not grow"


def test_the_newest_crop_survives_a_full_queue():
    """Drop-oldest, not drop-newest: the later crop is the closer one."""
    from app.services.reid_worker import ReidWorker

    w = ReidWorker(model_path="unused", queue_size=2)
    w.enabled = True
    for i in range(6):
        w.submit_crops(1, "Entrance", [SimpleNamespace(
            track_id=1, ts=float(i), image=np.zeros((4, 4, 3), np.uint8),
            score=0.5, first_seen=0.0)])
    seen = []
    while not w.q.empty():
        seen.append(w.q.get_nowait()[3].ts)
    assert max(seen) == 5.0, seen


def test_a_disabled_worker_accepts_and_discards():
    from app.services.reid_worker import ReidWorker
    w = ReidWorker(model_path="unused")
    w.enabled = False
    w.submit_crops(1, "Entrance", [SimpleNamespace(
        track_id=1, ts=0.0, image=np.zeros((4, 4, 3), np.uint8),
        score=0.5, first_seen=0.0)])
    assert w.q.qsize() == 0


# --- pass identity ---------------------------------------------------------

def test_pass_keys_separate_cameras_and_reused_track_ids():
    """Track ids restart on tracker.reset() after a stream gap and are numbered
    independently per camera, so `track_id` alone would merge two people."""
    from app.services.reid_worker import ReidWorker
    w = ReidWorker(model_path="unused")
    a = w._key(1, 3, 1_700_000_000.0)
    b = w._key(2, 3, 1_700_000_000.0)          # other camera, same id
    c = w._key(1, 3, 1_700_000_900.0)          # same camera, id reused later
    assert a != b and a != c and b != c


def test_folder_names_survive_real_employee_names():
    from app.services.reid_worker import slug
    assert slug("Ilxom G'aforov") == "Ilxom_Gaforov"
    assert slug("Bo‘ronov Nazim") == "Boronov_Nazim"
    assert slug("") == "Unnamed"
    assert "/" not in slug("a/b") and "\\" not in slug("a\\b")


# --- an unrecognised pass must still show a person -------------------------
# Reported from the dashboard: unknown sightings displayed a 112x112 aligned
# face - the one crop a human cannot judge, and of the one group where a human
# review is the whole point. Recognised passes had shown a body since 6a9ecda;
# unknowns were left behind because `TrackVote.best_person` is scoped to a
# COMMITTED identity, and an unknown pass has none.

class _FakeVote:
    def __init__(self, person=None):
        self.best_person = person


def _completed(vote_person, disp_person):
    """`_prune`'s choice of person_crop, in isolation."""
    return (vote_person if vote_person is not None else disp_person)


def test_an_unknown_pass_falls_back_to_the_quality_best_body():
    disp = np.full((40, 16, 3), 7, np.uint8)
    assert _completed(None, disp) is disp


def test_a_recognised_pass_still_shows_the_committed_identity_s_body():
    """The fallback must not override the scoped one: a track holding two
    people would otherwise show the wrong body under the voted name."""
    voted = np.full((40, 16, 3), 1, np.uint8)
    disp = np.full((40, 16, 3), 2, np.uint8)
    assert _completed(voted, disp) is voted


def test_a_pass_with_no_body_at_all_is_not_an_error():
    """A head tracked with no associated person box - the crop is simply
    absent, and the caller falls back to the face."""
    assert _completed(None, None) is None


def test_track_state_carries_the_display_body():
    from app.core.pipeline import TrackState
    from app.core.gallery import TrackVote
    st = TrackState(track_id=1, first_seen=0.0, last_seen=0.0,
                    box=np.zeros(4, np.float32), vote=TrackVote(window=5, required=3))
    assert hasattr(st, "disp_person") and st.disp_person is None


# --- crops must span the pass, and must not be fragments -------------------

def test_kept_crops_span_the_whole_pass():
    """Taking the first ten would cluster every kept crop at the start, when
    the person is furthest away and at one angle. The point of a tracklet is
    that its crops disagree usefully."""
    from app.services.reid_worker import _spread
    idx = _spread(list(range(30)), 10)
    assert len(idx) == 10
    assert idx[0] == 0 and idx[-1] == 29, "the pass endpoints must be kept"
    gaps = np.diff(idx)
    assert gaps.min() >= 2, f"crops bunched together: {idx}"


def test_a_short_pass_keeps_everything_it_collected():
    from app.services.reid_worker import _spread
    assert _spread(list(range(4)), 10) == [0, 1, 2, 3]


def test_selection_never_returns_duplicates_or_overruns():
    from app.services.reid_worker import _spread
    for n in range(1, 40):
        for k in (1, 3, 10, 25):
            idx = _spread(list(range(n)), k)
            assert len(idx) == len(set(idx)), (n, k, idx)
            assert len(idx) == min(n, k), (n, k, idx)
            assert all(0 <= i < n for i in idx)


def test_a_person_half_out_of_frame_is_not_kept():
    """A track begins the instant somebody appears at the edge of view, so its
    first box is a head and one shoulder. Those came out wider than tall and
    were saved as the first image of every pass - the worst possible ReID
    sample, since the model squashes whatever it gets to 256x128."""
    from app.core.pipeline import _whole_body
    W, H = 3840, 2160
    assert _whole_body(np.array([1800, 800, 1950, 1600], np.float32), W, H)
    # Sideways truncation is a fragment.
    assert not _whole_body(np.array([0, 800, 150, 1600], np.float32), W, H)
    assert not _whole_body(np.array([3700, 800, 3840, 1600], np.float32), W, H)


def test_top_and_bottom_edges_are_normal_in_this_corridor():
    """Measured over one clip: 46% of person boxes touch the TOP of the frame
    and 10% the bottom - the cameras look down a corridor, so somebody walking
    toward one has their head near the top and their feet out of shot.
    Rejecting those discarded 56% of crops including the closest and largest."""
    from app.core.pipeline import _whole_body
    W, H = 3840, 2160
    assert _whole_body(np.array([1800, 0, 1950, 900], np.float32), W, H), (
        "a head at the top of frame is normal here, not truncation")
    assert _whole_body(np.array([1800, 1200, 1950, 2160], np.float32), W, H), (
        "feet out of shot still leaves a usable body")


def test_a_box_wider_than_tall_is_not_a_standing_person():
    """Well inside the frame, but the wrong shape - a bad detection rather
    than somebody truncated."""
    from app.core.pipeline import _whole_body
    assert not _whole_body(np.array([1000, 900, 1400, 1100], np.float32), 3840, 2160)


# --- a completed track must be matchable to its own crops ------------------

def test_completed_track_carries_its_own_first_seen():
    """Reconstructing it as `completion_time - duration_s` is wrong by
    track_max_age_s, because a track is pruned three seconds AFTER it was last
    seen. That mismatch left every identified pass open until shutdown, where
    it was written as an UNKNOWN with no name - recognised people appearing in
    the unknown folders."""
    from app.core.pipeline import CompletedTrack
    ct = CompletedTrack(track_id=1, employee_id=7, name="X", best_score=0.5,
                        best_margin=0.1, embedded_frames=9, gated_frames=0,
                        direction="ENTER", direction_reason="", face_px=120,
                        duration_s=7.0, first_seen=1_700_000_000.0)
    assert ct.first_seen == 1_700_000_000.0
    # the old derivation, for contrast
    pruned_at = ct.first_seen + ct.duration_s + 3.0      # track_max_age_s
    assert pruned_at - ct.duration_s != ct.first_seen


def test_the_pass_key_uses_first_seen_not_the_completion_time():
    from app.services.reid_worker import ReidWorker
    w = ReidWorker(model_path="unused")
    first_seen = 1_700_000_000.0
    assert w._key(1, 3, first_seen) == w._key(1, 3, first_seen)
    assert w._key(1, 3, first_seen) != w._key(1, 3, first_seen + 3.0)


# --- a crop without a head is not worth keeping ----------------------------
# Reported from the collected folders: some crops were headless torsos. The
# cause was mine - I had relaxed the frame-edge rule to allow boxes touching
# the TOP of frame, because 46% of person boxes do in this corridor. Some of
# those are somebody standing close with their head fully in view; others are
# somebody whose head is above the frame entirely, and geometry cannot tell
# them apart. The head DETECTOR can, and the pipeline already runs it.

def test_a_crop_is_only_kept_when_a_head_was_detected_in_the_box():
    from app.core.pipeline import _head_in
    person = np.array([1000, 400, 1200, 1000], np.float32)
    inside = np.array([[1060, 410, 1140, 500]], np.float32)
    assert _head_in(person, inside) is not None

    # Head above the frame: the detector finds nothing in the box, so no crop.
    none_found = np.zeros((0, 4), np.float32)
    assert _head_in(person, none_found) is None

    # Somebody else's head, elsewhere in frame, must not qualify this box.
    elsewhere = np.array([[2600, 410, 2680, 500]], np.float32)
    assert _head_in(person, elsewhere) is None


def test_the_collection_cadence_gives_a_short_pass_something_to_choose_from():
    """The median pass is under ten seconds. At a 2 s cadence that is four
    crops and no room to select between them, which is why the interval is 1 s
    and the SELECTION does the spreading."""
    assert settings.body_crop_interval_s <= 1.0
    short_pass_s = 7.0
    collected = int(short_pass_s / settings.body_crop_interval_s) + 1
    assert collected >= 7, "a 7 s pass should yield enough crops to choose from"


def test_collection_covers_a_long_pass_before_the_cap_bites():
    span = settings.body_crop_max_per_pass * settings.body_crop_interval_s
    assert span >= 25.0, (
        f"collection stops after {span:.0f}s, so crops kept from a longer pass "
        f"could not span it")
