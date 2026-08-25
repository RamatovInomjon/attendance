"""Hold the fast OpenCV warp against the bit-exact reference.

`warp_batch_reference` is the numpy implementation verified element-for-element
against CVLFace's `affine_grid` + `grid_sample`. The deployed `warp_batch` uses
`cv2.warpAffine`: 11x faster (1.694 ms -> 0.156 ms per face, more than the
aligner network itself costs) but interpolating in 5-bit fixed point, so the two
agree closely rather than exactly.

These tests pin down *how* closely, so an OpenCV upgrade or a change to the
transform cannot quietly drift the alignment every embedding depends on.

The tolerances are measured, not guessed, and they differ by content because
fixed-point error scales with local contrast:

    max |cv2 - reference|, source size -> 112
      identity, 112 px  : 0.0000  (no resampling happens at all)
      identity, 39-224  : 0.0008 - 0.0071 smooth,  0.039 - 0.043 noise
      rotate+scale      : 0.0185 smooth
      REAL faces        : 0.0121 - 0.0168  (40 samples)

Faces are smooth, so SMOOTH_TOL is the tolerance that describes production.
Random noise is the pathological case - adjacent pixels swinging the full
[-1, 1] range - and is kept as a stress bound, not as a claim about real input.
Downstream this is what matters: over 40 real faces the two paths produced
embeddings with cosine similarity >= 0.999922, gallery scores differing by at
most 0.00095 against a 0.18 threshold, and 40/40 identical identity decisions.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core import geometry as G

# Face-like content. Set from the real measurement rather than synthetic
# images: over 40 real corridor faces the worst pixel disagreement was
# 1.68e-02, and a rotate+scale+translate on synthetic smooth content gives
# 1.85e-02. 2.5e-2 leaves headroom without letting a genuine regression pass -
# a broken transform disagrees by O(1), not by hundredths.
SMOOTH_TOL = 2.5e-2
NOISE_TOL = 6e-2       # pathological high-frequency content; observed worst 0.0427


def _identity_theta(n: int = 1) -> np.ndarray:
    t = np.zeros((n, 2, 3), np.float64)
    t[:, 0, 0] = 1.0
    t[:, 1, 1] = 1.0
    return t


def _smooth(n=1, s=112, seed=0) -> np.ndarray:
    """Low-frequency image standing in for a face: the regime that matters."""
    xs = np.linspace(-1, 1, s)
    planes = [np.outer(np.sin((3 + k + seed) * xs), np.cos((2 + k) * xs))
              for k in range(3)]
    return np.stack([np.stack(planes)] * n).astype(np.float32)


def _noise(n=1, s=112, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((n, 3, s, s), dtype=np.float32) * 2 - 1).astype(np.float32)


def test_identity_at_native_size_is_exact():
    """No resampling means no fixed-point error at all - a useful canary: if
    this ever drifts, the transform itself changed, not the interpolation."""
    img = _noise(s=G.OUTPUT_SIZE, seed=1)
    out = G.warp_batch(img, _identity_theta(), G.OUTPUT_SIZE)
    ref = G.warp_batch_reference(img, _identity_theta(), G.OUTPUT_SIZE)
    assert np.array_equal(out, ref)


@pytest.mark.parametrize("size", [39, 64, 109, 112, 160, 224])
def test_matches_reference_on_face_like_content(size):
    """Crop size varies with how far away the person is, so every size the
    pipeline actually produces must agree."""
    img = _smooth(s=size, seed=size % 5)
    theta = _identity_theta()
    fast = G.warp_batch(img, theta, G.OUTPUT_SIZE)
    ref = G.warp_batch_reference(img, theta, G.OUTPUT_SIZE)
    assert fast.shape == ref.shape
    assert np.abs(fast - ref).max() < SMOOTH_TOL


@pytest.mark.parametrize("size", [39, 64, 109, 160, 224])
def test_stays_bounded_even_on_pathological_noise(size):
    img = _noise(s=size, seed=size)
    theta = _identity_theta()
    fast = G.warp_batch(img, theta, G.OUTPUT_SIZE)
    ref = G.warp_batch_reference(img, theta, G.OUTPUT_SIZE)
    assert np.abs(fast - ref).max() < NOISE_TOL


def test_matches_reference_for_a_realistic_similarity_transform():
    """Rotate + scale + translate, as a real landmark alignment produces."""
    img = _smooth(s=112, seed=3)
    ang, sc = np.deg2rad(12.0), 0.85
    theta = np.zeros((1, 2, 3))
    theta[0, 0, 0] = sc * np.cos(ang)
    theta[0, 0, 1] = -sc * np.sin(ang)
    theta[0, 1, 0] = sc * np.sin(ang)
    theta[0, 1, 1] = sc * np.cos(ang)
    theta[0, :, 2] = [0.06, -0.04]
    fast = G.warp_batch(img, theta, G.OUTPUT_SIZE)
    ref = G.warp_batch_reference(img, theta, G.OUTPUT_SIZE)
    assert np.abs(fast - ref).max() < SMOOTH_TOL


def test_batch_of_several_faces_matches_one_at_a_time():
    """Several faces in one frame are warped together; each row must equal what
    that face produces alone."""
    img = _smooth(n=3, s=112, seed=2)
    theta = _identity_theta(3)
    theta[1, 0, 2] = 0.1
    theta[2, 1, 1] = 0.8
    batched = G.warp_batch(img, theta, G.OUTPUT_SIZE)
    for i in range(3):
        one = G.warp_batch(img[i:i + 1], theta[i:i + 1], G.OUTPUT_SIZE)
        assert np.array_equal(batched[i], one[0])


def test_out_of_image_samples_are_black_not_grey():
    """The reference samples x+1 and subtracts 1 so anything outside the source
    returns -1. Mid-grey padding would leak a bright border into the crop and
    move the embedding. Scale > 1 samples beyond the source edges."""
    img = np.zeros((1, 3, 64, 64), np.float32)
    theta = _identity_theta()
    theta[0, 0, 0] = theta[0, 1, 1] = 3.0
    fast = G.warp_batch(img, theta, G.OUTPUT_SIZE)
    ref = G.warp_batch_reference(img, theta, G.OUTPUT_SIZE)
    assert fast.min() < -0.99
    # and it must black out the same region the reference does
    assert abs((fast < -0.9).mean() - (ref < -0.9).mean()) < 0.01


def test_output_is_float32_and_correct_shape():
    out = G.warp_batch(_smooth(n=2, s=90), _identity_theta(2), G.OUTPUT_SIZE)
    assert out.dtype == np.float32
    assert out.shape == (2, 3, G.OUTPUT_SIZE, G.OUTPUT_SIZE)
