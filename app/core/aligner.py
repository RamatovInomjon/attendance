"""DFA-mobilenet face aligner (CVLFace) on ONNX Runtime.

Why the aligner is fed *crops*, never a whole frame
---------------------------------------------------
The exported graph resizes its input to 160x160 internally and the reference
pipeline warps the canonical 112x112 crop out of that resized tensor.  Hand it a
3840x2160 frame and a 200 px face becomes ~8 px before the warp ever runs.

So the detector localises and the aligner refines: a square with margin around
each detected box is cut from the *full-resolution* frame, and only that crop
reaches the aligner.  The aligner runs its own RetinaFace head inside the crop,
which is why the crop needs context around the face rather than a tight box.

Choosing the margin
-------------------
The margin sets how many pixels of face survive into the 112x112 output.  Inside
the aligner the crop is 160 px wide, so the face spans 160/margin px, and the
warp resamples that to 112:

    margin 1.00 -> face 160 px -> 112   downsample, full detail
    margin 1.30 -> face 123 px -> 112   slight downsample          <- deployed
    margin 1.43 -> face 112 px -> 112   1:1, the break-even point
    margin 1.80 -> face  89 px -> 112   upsampling, inventing detail
    margin 2.30 -> face  70 px -> 112   heavy upsampling

Measured on the 268-image gallery (bench/sweep_crop_pipeline.py), d-prime is
flat from 1.0 to 1.35 and falls away from 1.43 exactly as that predicts:

    1.00  12.44    1.30  12.47    1.50  12.33    1.80  11.96    2.00  11.69
    1.20  12.46    1.35  12.50    1.43  12.29

Three independent lines of evidence agree on ~1.3: the sweep above, the
break-even arithmetic, and the CVLFace authors' own reference `input.png`, whose
face occupies 57% of the frame - an implied margin of 1.30.

Two-stage warp
--------------
`ALIGN_MODE_SHARP` (default) uses the aligner only for landmarks - batched at
160x160, which is all the 1050-anchor aggregator can consume - and then warps
the canonical crop out of the **native-resolution** crop instead of the
aligner's internal 160 tensor.

That removes a resample: the reference path sends a 292 px crop down to 160 and
then to 112, and sends a small 126 px crop *up* to 160 and back down to 112.
Warping from native is one downsample either way.  Measured gain on the gallery
(bench/test_sharp_warp.py): d-prime 12.53 -> 12.64, and the weakest genuine pair
0.547 -> 0.571 - the hardest case, which is where it matters.

This deliberately departs from the reference PyTorch numerics that the ONNX
export was verified against.  That is only sound because enrolment and inference
share this code path, so the gallery and the query are always produced the same
way.  Set `mode="reference"` to get the bit-exact reference behaviour back.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from app.core.onnx_env import best_providers

from app.core import geometry as G

ALIGN_MODE_SHARP = "sharp"
ALIGN_MODE_REFERENCE = "reference"

# Fixed by the DFA aggregator's 1050-anchor input; the graph resizes to this.
ALIGNER_NATIVE = G.ALIGNER_INPUT_SIZE   # 160

# Warping from a crop far larger than the 112 output buys nothing and costs
# gather time, so native crops are capped here before the warp.
MAX_WARP_SOURCE = 384


@dataclass
class AlignedFace:
    aligned: np.ndarray      # (3, 112, 112) float32 in [-1, 1]
    landmarks: np.ndarray    # (5, 2) pixel coords in the ORIGINAL frame
    score: float             # aligner's own face confidence
    crop_box: tuple          # (x1, y1, x2, y2) of the crop in the original frame


class FaceAligner:
    def __init__(
        self,
        model_path: str | Path,
        providers=None,
        crop_size: int = ALIGNER_NATIVE,
        margin: float = 1.3,
        mode: str = ALIGN_MODE_SHARP,
    ):
        """``crop_size`` is what the aligner network sees.  It is 160 by default
        because the graph resizes to 160 regardless, so anything larger is a
        resample for nothing - measured d-prime across 144..320 is 12.37-12.53,
        i.e. flat.  The *warp* source is separate and controlled by ``mode``."""
        if providers is None:
            providers = best_providers()

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
        self.crop_size = crop_size
        self.margin = margin
        self.mode = mode

    @property
    def provider(self) -> str:
        return self.session.get_providers()[0]

    def _cut(self, frame_rgb: np.ndarray, box) -> tuple[np.ndarray, tuple]:
        """Square crop with margin around ``box``, padded rather than clamped so
        the face stays centred even at the frame edge."""
        h, w = frame_rgb.shape[:2]
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        half = max(x2 - x1, y2 - y1) * self.margin / 2.0

        cx1, cy1 = int(round(cx - half)), int(round(cy - half))
        cx2, cy2 = int(round(cx + half)), int(round(cy + half))

        pl, pt = max(0, -cx1), max(0, -cy1)
        pr, pb = max(0, cx2 - w), max(0, cy2 - h)
        sx1, sy1 = max(0, cx1), max(0, cy1)
        sx2, sy2 = min(w, cx2), min(h, cy2)

        crop = frame_rgb[sy1:sy2, sx1:sx2]
        if crop.size and (pl or pt or pr or pb):
            crop = cv2.copyMakeBorder(crop, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return crop, (cx1, cy1, cx2, cy2)

    def align(self, frame_rgb: np.ndarray, boxes, is_bgr: bool = False) -> list[AlignedFace]:
        """Align every ``box``; one batched session run for the landmarks.

        ``is_bgr`` lets the caller hand over the camera frame untouched. The
        alignment only ever reads small crops, so converting the whole 4K frame
        to RGB first costs 5.89 ms and a 24 MB allocation to feed a couple of
        200 px windows - converting each crop instead costs 0.010 ms. Callers
        that already hold an RGB frame keep the old behaviour by default.
        """
        if len(boxes) == 0:
            return []

        natives, small, crop_boxes = [], [], []
        for b in boxes:
            crop, cb = self._cut(frame_rgb, b)
            if crop.size == 0:
                continue
            if is_bgr:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            # Cap the warp source: past ~384 px the extra detail cannot reach a
            # 112 px output, and the bilinear gather gets needlessly expensive.
            n = crop
            if n.shape[0] > MAX_WARP_SOURCE:
                n = cv2.resize(n, (MAX_WARP_SOURCE, MAX_WARP_SOURCE), interpolation=cv2.INTER_AREA)
            natives.append(n)
            small.append(G.to_normalized_chw(
                cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)))
            crop_boxes.append(cb)

        if not natives:
            return []

        batch = np.stack(small).astype(np.float32)
        ldmk, _bbox, score, resized = self.session.run(None, {"image": batch})

        if self.mode == ALIGN_MODE_REFERENCE:
            thetas = np.stack([
                G.landmarks_to_theta(ldmk[i], ALIGNER_NATIVE, G.OUTPUT_SIZE)
                for i in range(len(natives))
            ])
            aligned = G.warp_batch(resized, thetas, G.OUTPUT_SIZE)
        else:
            # Landmarks are normalized to the square crop, so they carry over to
            # the native crop unchanged.  Group equal sizes to keep warps batched.
            aligned = [None] * len(natives)
            # Group on the FULL shape, not just height: a crop clamped at a frame
            # edge can share a height with another while differing in width, and
            # np.stack then raises on the mismatch. Grouping by height alone
            # crashed the pipeline whenever two such faces appeared together.
            by_size: dict[tuple, list[int]] = {}
            for i, n in enumerate(natives):
                by_size.setdefault(n.shape[:2], []).append(i)
            for shape, idxs in by_size.items():
                size = shape[0]
                imgs = np.stack([G.to_normalized_chw(natives[i]) for i in idxs]).astype(np.float32)
                th = np.stack([G.landmarks_to_theta(ldmk[i], size, G.OUTPUT_SIZE) for i in idxs])
                out = G.warp_batch(imgs, th, G.OUTPUT_SIZE)
                for k, i in enumerate(idxs):
                    aligned[i] = out[k]
            aligned = np.stack(aligned)

        faces = []
        for i, cb in enumerate(crop_boxes):
            side = cb[2] - cb[0]
            lm = ldmk[i] * side + np.array([[cb[0], cb[1]]])
            faces.append(AlignedFace(aligned[i], lm.astype(np.float32), float(score[i][0]), cb))
        return faces
