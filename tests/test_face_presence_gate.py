"""The gate that separates a face from the back of a head.

A recognition was committed from an aligned crop that contained no face at all
- the back of somebody's head, matched to a real person at 0.219, above the
0.18 threshold. Every other gate passed it:

    face_px    302     large, so the size gate was happy
    sharpness  473.3   HAIR TEXTURE, sharper than any real face that day (50-270)
    yaw 6.4    pitch 3.5   "frontal" - from landmarks regressed onto a hairline
    aligner    0.604   the only signal that knew, and the gate was at 0.40

The aligner is not a face detector. Given a head it returns five landmarks
whatever is in the crop, and symmetric nonsense landmarks then yield a
confident-looking frontal pose. Its own confidence score is the one thing that
distinguishes the two cases, which is why the threshold matters so much.

Measured when the gate was set:

    enrolment faces (n=220)   min 0.9929, none below 0.99
    live crops < 0.70         backs and tops of heads, no usable face
    live crops 0.70-0.90      steep top-down views, face barely visible
    live crops 0.90-0.99      face visible, looking down
    live crops >= 0.99        clear near-frontal faces

Over 174 live crops, twelve above-threshold matches came from sub-0.90 crops,
the worst being aligner 0.545 matching at 0.274.
"""
from __future__ import annotations

import pytest

from app.config import settings
from app.core.quality import assess

import numpy as np


# Real captures, taken from the debug sidecars of one live session. The two
# rejects are the reported false accept and a second back-of-head that was
# being shown as the "best" image for a check-in.
CAPTURES = [
    # (label,                   aligner, sharpness, yaw,  pitch, is_a_face)
    ("Baxtiyor  clear face",      0.992,  180.0,    12.0,  10.0, True),
    ("Fayzullayev clear face",    0.993,  120.0,    -8.0,  14.0, True),
    ("Fayzullayev frontal",       0.998,   95.0,     2.0,   6.0, True),
    ("Murodullayev looking down", 0.981,  122.6,   -23.3,  24.5, True),
    ("Nishonov  clear face",      0.981,  270.8,    22.0,  24.1, True),
    ("Nishonov  head down",       0.991,  118.1,    -2.3,  35.2, True),
    ("Nodira    clear face",      0.988,   54.5,    22.5,  12.4, True),
    ("Nodira    frontal",         0.996,  140.0,     5.0,   8.0, True),
    ("Nodira    BACK OF HEAD",    0.604,  473.3,     6.4,   3.5, False),
    ("Nodira    top of head",     0.810,  200.0,     4.0,   5.0, False),
]


def _assess(aligner, sharpness, yaw, pitch):
    """Drive the real gate with a synthetic crop of a known sharpness.

    A flat crop has zero Laplacian variance, so sharpness is injected by
    monkeypatching rather than by trying to draw an image of a given blur.
    """
    box = np.array([0.0, 0.0, 200.0, 200.0])          # comfortably over min_face_px
    aligned = np.zeros((3, 112, 112), np.float32)
    landmarks = np.array([[38.0, 51.0], [74.0, 51.0], [56.0, 71.0],
                          [42.0, 92.0], [70.0, 92.0]], np.float32)
    import app.core.quality as Q
    real = Q.sharpness_of
    Q.sharpness_of = lambda _a: sharpness
    try:
        return assess(box, aligned, landmarks, aligner,
                      min_face_px=settings.min_face_px,
                      min_laplacian_var=settings.min_laplacian_var,
                      min_aligner_score=settings.min_aligner_score,
                      max_yaw_deg=90.0, max_pitch_deg=90.0)   # isolate the aligner gate
    finally:
        Q.sharpness_of = real


def test_the_gate_is_high_enough_to_be_a_face_check():
    """0.40 was below every observed value and rejected nothing that mattered.

    Enrolment faces never scored under 0.99 and the reported false accept was
    0.604, so anything at or below 0.60 cannot separate the two.
    """
    assert settings.min_aligner_score >= 0.85, (
        "the aligner score is the ONLY signal that distinguishes a face from "
        "the back of a head; below ~0.85 it stops discriminating"
    )


@pytest.mark.parametrize("label,aligner,sharp,yaw,pitch,is_face", CAPTURES)
def test_real_captures_are_judged_correctly(label, aligner, sharp, yaw, pitch, is_face):
    q = _assess(aligner, sharp, yaw, pitch)
    assert q.ok is is_face, f"{label}: expected ok={is_face}, got {q.ok} ({q.reason})"


def test_the_reported_false_accept_is_rejected():
    """The exact capture that started this: back of a head, matched at 0.219."""
    q = _assess(0.604, 473.3, 6.4, 3.5)
    assert not q.ok
    assert "aligner" in q.reason


def test_sharpness_cannot_rescue_a_non_face():
    """Hair is sharper than skin. The back-of-head crop scored 473 where real
    faces that day ran 50-270, so a high sharpness must not imply a face."""
    q = _assess(0.604, 2000.0, 0.0, 0.0)
    assert not q.ok, "an extremely sharp non-face must still be rejected"


def test_a_confident_looking_pose_cannot_rescue_a_non_face():
    """Landmarks regressed onto a hairline are symmetric, so the pose estimate
    reads as perfectly frontal. Pose must not override face presence."""
    q = _assess(0.604, 100.0, 0.0, 0.0)
    assert not q.ok


def test_a_genuine_face_looking_down_still_passes():
    """The gate must not become so strict that ordinary corridor traffic - people
    walking with their head slightly down - stops being recognised."""
    q = _assess(0.981, 122.6, -23.3, 24.5)
    assert q.ok


def test_gate_boundary():
    assert _assess(settings.min_aligner_score, 100.0, 0.0, 0.0).ok
    assert not _assess(settings.min_aligner_score - 0.001, 100.0, 0.0, 0.0).ok
