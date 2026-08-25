"""Face quality scoring and gating.

Two jobs:

* **Gate** — reject a face that is too small, too blurred, too far off-axis, or
  that the aligner itself was unsure about.  A rejected face costs nothing; a
  bad embedding costs a wrong name in the attendance report.
* **Score** — rank the frames of a track so the pipeline can recognize the
  *best* view of a person rather than the first one it happened to see.

Pose comes from the aligner's 5 landmarks.  It is a geometric approximation, not
a calibrated head-pose model, which is all the gate needs: it separates "looking
roughly at the camera" from "profile" reliably enough to filter on.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.core.geometry import aligned_to_uint8


@dataclass
class Quality:
    face_px: float
    sharpness: float
    yaw: float
    pitch: float
    roll: float
    aligner_score: float
    score: float          # combined, higher is better
    ok: bool
    reason: str = ""


def estimate_pose(landmarks: np.ndarray) -> tuple[float, float, float]:
    """Approximate (yaw, pitch, roll) in degrees from the 5-point landmarks.

    landmarks order: left eye, right eye, nose, left mouth, right mouth.
    """
    le, re, nose, lm, rm = landmarks[:5].astype(np.float64)

    eye_c = (le + re) / 2.0
    mouth_c = (lm + rm) / 2.0

    # Roll: tilt of the inter-ocular line.
    d = re - le
    roll = np.degrees(np.arctan2(d[1], d[0]))

    eye_dist = np.linalg.norm(d) + 1e-6

    # Yaw: how far the nose sits off the eye-centre, along the eye axis.
    axis = d / eye_dist
    off = float(np.dot(nose - eye_c, axis)) / eye_dist
    yaw = float(np.clip(off * 180.0, -90.0, 90.0))

    # Pitch: where the nose sits between the eye line and the mouth line.
    vert = mouth_c - eye_c
    vlen = np.linalg.norm(vert) + 1e-6
    vaxis = vert / vlen
    t = float(np.dot(nose - eye_c, vaxis)) / vlen     # ~0.5 when frontal
    pitch = float(np.clip((t - 0.5) * 180.0, -90.0, 90.0))

    return yaw, pitch, roll


def sharpness_of(aligned_chw: np.ndarray) -> float:
    """Laplacian variance of the aligned crop (grayscale)."""
    rgb = aligned_to_uint8(aligned_chw)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess(
    box: np.ndarray,
    aligned_chw: np.ndarray,
    landmarks: np.ndarray,
    aligner_score: float,
    *,
    min_face_px: int,
    min_laplacian_var: float,
    min_aligner_score: float,
    max_yaw_deg: float,
    max_pitch_deg: float,
) -> Quality:
    face_px = float(max(box[2] - box[0], box[3] - box[1]))
    sharp = sharpness_of(aligned_chw)
    yaw, pitch, roll = estimate_pose(landmarks)

    reason = ""
    if face_px < min_face_px:
        reason = f"small({face_px:.0f}px)"
    elif aligner_score < min_aligner_score:
        reason = f"aligner({aligner_score:.2f})"
    elif sharp < min_laplacian_var:
        reason = f"blur({sharp:.0f})"
    elif abs(yaw) > max_yaw_deg:
        reason = f"yaw({yaw:.0f})"
    elif abs(pitch) > max_pitch_deg:
        reason = f"pitch({pitch:.0f})"

    # Combined best-shot score: size x sharpness x frontality, each saturating.
    size_t = min(face_px / 160.0, 1.0)
    sharp_t = min(sharp / 400.0, 1.0)
    front_t = max(0.0, 1.0 - (abs(yaw) / 60.0)) * max(0.0, 1.0 - (abs(pitch) / 60.0))
    combined = size_t * (0.35 + 0.65 * sharp_t) * (0.25 + 0.75 * front_t) * max(aligner_score, 0.0)

    return Quality(face_px, sharp, yaw, pitch, roll, aligner_score,
                   combined, reason == "", reason)
