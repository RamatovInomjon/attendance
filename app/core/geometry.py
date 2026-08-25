"""Similarity-warp geometry for the CVLFace DFA aligner.

Ported from the verified `cvlface_test/inference.py` (which matches the
reference PyTorch pipeline to ~5e-5), with the per-image Python loop replaced
by a batched implementation so a whole frame's faces warp in one pass.

The two numerically delicate parts are kept exactly as verified:

* `umeyama` is the scikit-image least-squares similarity fit, in float64.
* `bilinear_sample` is a direct gather, not `cv2.warpAffine`.  OpenCV's
  INTER_LINEAR quantizes interpolation weights to 5 fractional bits, worth
  ~3/255 of crop error; the gather reproduces `grid_sample` exactly.
"""
from __future__ import annotations

import cv2
import numpy as np

ALIGNER_INPUT_SIZE = 160  # fixed by the DFA aggregator's 1050-anchor input
OUTPUT_SIZE = 112

# Canonical ArcFace 5-point template in the 112x112 output frame.
REFERENCE_LANDMARKS = np.array(
    [
        [38.29459953, 51.69630051],  # left eye
        [73.53179932, 51.50139999],  # right eye
        [56.02519989, 71.73660278],  # nose tip
        [41.54930115, 92.36550140],  # left mouth corner
        [70.72990036, 92.20410156],  # right mouth corner
    ],
    dtype=np.float64,
)


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (scale + rotation + translation).

    Port of ``skimage.transform._geometric._umeyama`` with ``estimate_scale=True``
    so the runtime needs no scikit-image.  Returns the 3x3 homogeneous matrix
    mapping ``src`` onto ``dst``.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    num, dim = src.shape

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    A = dst_demean.T @ src_demean / num

    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.float64)
    U, S, V = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)

    if rank == 0:
        return np.full((dim + 1, dim + 1), np.nan)
    if rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V

    scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale
    return T


def _estimate_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Exact affine fit through 3 correspondences."""
    A = np.hstack([src, np.ones((src.shape[0], 1))])
    params, *_ = np.linalg.lstsq(A, dst, rcond=None)
    return params.T  # (2, 3)


def landmarks_to_theta(
    landmarks_norm: np.ndarray,
    in_size: int = ALIGNER_INPUT_SIZE,
    out_size: int = OUTPUT_SIZE,
) -> np.ndarray:
    """Normalized landmarks -> ``affine_grid`` theta, mirroring the reference."""
    ldmk_px = landmarks_norm.astype(np.float64) * np.array([[in_size, in_size]])
    M = umeyama(ldmk_px, REFERENCE_LANDMARKS)[0:2, :]

    srcs = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.float64)
    dsts = srcs @ M[:, :2].T + M[:, 2]

    srcs_n = srcs / in_size * 2 - 1
    dsts_n = dsts / out_size * 2 - 1
    return _estimate_affine(dsts_n, srcs_n)  # (2, 3)


def theta_to_pixel_matrix(
    theta: np.ndarray,
    in_size: int = ALIGNER_INPUT_SIZE,
    out_size: int = OUTPUT_SIZE,
) -> np.ndarray:
    """``affine_grid`` theta -> plain pixel-space inverse map (output px -> source px).

    ``F.affine_grid`` + ``F.grid_sample`` with ``align_corners=True`` compose to
    an affine function of the output pixel indices; this collects the terms.
    """
    s = (in_size - 1) / (out_size - 1)
    half = (in_size - 1) / 2.0
    A = np.empty((2, 3), dtype=np.float64)
    for r in range(2):
        A[r, 0] = s * theta[r, 0]
        A[r, 1] = s * theta[r, 1]
        A[r, 2] = half * (theta[r, 2] - theta[r, 0] - theta[r, 1] + 1.0)
    return A


def warp_batch_reference(
    images_chw: np.ndarray,
    thetas: np.ndarray,
    out_size: int = OUTPUT_SIZE,
) -> np.ndarray:
    """Warp a batch of CHW images in [-1, 1] to canonical crops.

    ``images_chw``: (N, 3, S, S) float32.  ``thetas``: (N, 2, 3).
    Returns (N, 3, out_size, out_size) float32.

    The reference samples ``x + 1`` and subtracts 1 afterwards so out-of-image
    pixels come out black (-1) rather than mid-grey; we do the same.
    """
    n, c, h, w = images_chw.shape
    ys, xs = np.meshgrid(
        np.arange(out_size, dtype=np.float64),
        np.arange(out_size, dtype=np.float64),
        indexing="ij",
    )

    sx = np.empty((n, out_size, out_size), dtype=np.float64)
    sy = np.empty((n, out_size, out_size), dtype=np.float64)
    for i in range(n):
        A = theta_to_pixel_matrix(thetas[i], h, out_size)
        sx[i] = A[0, 0] * xs + A[0, 1] * ys + A[0, 2]
        sy[i] = A[1, 0] * xs + A[1, 1] * ys + A[1, 2]

    src = images_chw.astype(np.float64) + 1.0

    x0 = np.floor(sx).astype(np.int64)
    y0 = np.floor(sy).astype(np.int64)
    wx = sx - x0
    wy = sy - y0

    out = np.zeros((n, c, out_size, out_size), dtype=np.float64)
    bidx = np.arange(n)[:, None, None]
    for xi, wxi in ((x0, 1.0 - wx), (x0 + 1, wx)):
        for yi, wyi in ((y0, 1.0 - wy), (y0 + 1, wy)):
            inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            vals = src[bidx, :, np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)]
            out += np.transpose(vals, (0, 3, 1, 2)) * (wxi * wyi * inside)[:, None]

    return (out - 1.0).astype(np.float32)


def to_normalized_chw(rgb_uint8: np.ndarray) -> np.ndarray:
    """HWC uint8 RGB -> CHW float32 in [-1, 1] (ToTensor + Normalize(0.5, 0.5))."""
    x = rgb_uint8.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return np.ascontiguousarray(np.transpose(x, (2, 0, 1)))


def aligned_to_uint8(aligned_chw: np.ndarray) -> np.ndarray:
    """Canonical crop in [-1, 1] -> HWC uint8 RGB, for saving/visualising."""
    hwc = np.transpose(aligned_chw, (1, 2, 0))
    return np.clip((hwc * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)


def warp_batch(
    images_chw: np.ndarray,
    thetas: np.ndarray,
    out_size: int = OUTPUT_SIZE,
) -> np.ndarray:
    """The deployed warp: same transform as `warp_batch_reference`, via OpenCV.

    The reference does the bilinear gather in numpy, which measured 1.694 ms per
    face - MORE than the aligner network it feeds (1.417 ms), and the single
    largest cost in alignment. `cv2.warpAffine` is the same affine map in SIMD
    C++ at 0.156 ms, an 11x reduction.

    It is not bit-exact: OpenCV interpolates in fixed point, so pixels differ by
    up to 1.7e-02 on a [-1, 1] scale. What matters is whether that survives to
    the embedding, and measured over 40 real faces it does not - cosine
    similarity between the two paths is 0.999922 at worst, gallery scores move
    by at most 0.00095 against a 0.18 threshold, and all 40 identity decisions
    were identical.

    `warp_batch_reference` is kept as the bit-exact CVLFace-equivalent
    implementation, and `tests/test_geometry_warp.py` holds the two together.
    """
    n, c, h, w = images_chw.shape
    out = np.empty((n, c, out_size, out_size), np.float32)
    for i in range(n):
        # theta_to_pixel_matrix maps OUTPUT pixels -> SOURCE pixels, which is
        # what WARP_INVERSE_MAP expects; without that flag OpenCV would invert
        # it again and the crop would come out mirrored about the transform.
        A = theta_to_pixel_matrix(thetas[i], h, out_size)
        hwc = np.ascontiguousarray(images_chw[i].transpose(1, 2, 0))
        # Sample x+1 and subtract 1 so out-of-image pixels land at -1 (black)
        # rather than mid-grey, matching the reference.
        warped = cv2.warpAffine(
            hwc + 1.0, A, (out_size, out_size),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0.0, 0.0, 0.0),
        ) - 1.0
        if warped.ndim == 2:          # single-channel input keeps its axis
            warped = warped[:, :, None]
        out[i] = warped.transpose(2, 0, 1)
    return out
