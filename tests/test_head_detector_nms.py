"""Per-class suppression in the head/person detector.

The pipeline asks for both classes in one pass. A person box that is only
head and shoulders overlaps its own head well above the IoU limit, and
class-agnostic NMS dropped whichever scored lower - no head, no recognition
on that frame; no person, a hole in the track.
"""
from __future__ import annotations

import numpy as np

from app.core.head_detector import CLS_HEAD, CLS_PERSON, _nms_per_class


def _rect(x, y, w, h):
    return [float(x), float(y), float(w), float(h)]


def test_a_head_inside_a_shoulder_box_survives():
    rects = [_rect(100, 100, 80, 90),      # head-and-shoulders person box
             _rect(110, 100, 60, 70)]      # its head: IoU ~0.58 with the person
    conf = np.array([0.9, 0.6], np.float32)
    cls = np.array([CLS_PERSON, CLS_HEAD])
    keep = sorted(_nms_per_class(rects, conf, cls, 0.35, 0.5))
    assert keep == [0, 1]


def test_duplicates_of_the_same_class_are_still_suppressed():
    rects = [_rect(100, 100, 80, 90), _rect(104, 102, 80, 90), _rect(400, 100, 60, 70)]
    conf = np.array([0.9, 0.7, 0.8], np.float32)
    cls = np.array([CLS_PERSON, CLS_PERSON, CLS_HEAD])
    keep = sorted(_nms_per_class(rects, conf, cls, 0.35, 0.5))
    assert keep == [0, 2]


def test_empty_input_is_empty_output():
    assert _nms_per_class([], np.zeros(0, np.float32), np.zeros(0, int), 0.35, 0.5) == []
