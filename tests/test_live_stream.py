"""The camera stream's two economies, and the shape of what leaves it.

One JPEG encode per processed frame however many viewers are watching; no
repainting of a source that has stopped delivering frames; and snapshot paths
that leave /api/events and the attendance socket as URLs carrying the
deployment prefix, the way every rendered page hands them out.
"""
from __future__ import annotations

import base64
import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api import ws as W
from app.api.main import app
from app.core.security import COOKIE_NAME, sign_session


class _Worker:
    """Just enough of CameraWorker for the render path."""

    def __init__(self, camera_id: int, *, stale: bool = False, latest: bool = True):
        self.camera_id = camera_id
        self.source = SimpleNamespace(is_stale=stale, fps=12.34)
        self.latest = SimpleNamespace(frame=SimpleNamespace(ts=100.0)) if latest else None
        self.renders = 0

    def render(self):
        self.renders += 1
        return b"\xff\xd8jpeg-%d" % self.renders


def test_one_encode_per_frame_however_many_viewers():
    w = _Worker(7)
    assert [W.shared_jpeg(w) for _ in range(3)] == [b"\xff\xd8jpeg-1"] * 3
    assert w.renders == 1
    # A new processed frame is a new encode...
    w.latest = SimpleNamespace(frame=SimpleNamespace(ts=101.0))
    assert W.shared_jpeg(w) == b"\xff\xd8jpeg-2" and w.renders == 2
    # ...and the cache is per camera.
    other = _Worker(8)
    assert W.shared_jpeg(other) == b"\xff\xd8jpeg-1" and other.renders == 1


def test_viewers_arriving_together_share_one_encode():
    w = _Worker(9)
    started, release = threading.Event(), threading.Event()

    def slow_render():
        w.renders += 1
        started.set()
        release.wait(2)
        return b"jpg"

    w.render = slow_render
    results: list = []
    threads = [threading.Thread(target=lambda: results.append(W.shared_jpeg(w)))
               for _ in range(4)]
    for t in threads:
        t.start()
    assert started.wait(2)
    release.set()
    for t in threads:
        t.join(2)
    assert results == [b"jpg"] * 4 and w.renders == 1


def test_a_stale_source_is_announced_without_a_frame():
    """A camera down for hours used to be re-encoded and re-sent at 10 fps,
    so it looked live to anyone watching."""
    w = _Worker(10, stale=True)
    msg = W.frame_message(w, 10)
    assert msg == {"type": "frame", "stale": True, "camera_id": 10, "fps": 12.3}
    assert w.renders == 0
    # A camera that has never produced a frame says nothing while it connects;
    # "Kadr kutilmoqda" is the page's own state for that.
    assert W.frame_message(_Worker(11, stale=True, latest=False), 11) is None


def test_a_live_source_sends_the_shared_jpeg():
    w = _Worker(12)
    msg = W.frame_message(w, 12)
    assert msg["stale"] is False and msg["camera_id"] == 12
    assert base64.b64decode(msg["data"]) == b"\xff\xd8jpeg-1"


def test_api_events_hands_out_prefixed_snapshot_urls(monkeypatch):
    """The worker stores `snapshots/x.jpg`; raw, the live page built
    `/media/...` off the domain root, which under /faceid is another project."""
    from app.config import settings
    from app.runtime import runtime
    fake = SimpleNamespace(_lock=threading.Lock(), recent_events=[
        {"ts": "08:15:30", "sort_ts": 2.0, "name": "A", "employee_id": 1,
         "camera": "Kirish", "snapshot": "snapshots/evt_1.jpg"},
        {"ts": "08:15:29", "sort_ts": 1.0, "name": "B", "employee_id": 2,
         "camera": "Kirish", "snapshot": None},
    ])
    monkeypatch.setattr(runtime, "workers", {1: fake})
    monkeypatch.setattr(settings, "url_prefix", "/faceid", raising=False)
    client = TestClient(app, follow_redirects=False)
    tok = sign_session(user_id=1, username="inomjon", is_admin=True)
    r = client.get("/api/events", headers={"Cookie": f"{COOKIE_NAME}={tok}"})
    assert r.status_code == 200
    assert [e["snapshot"] for e in r.json()] == ["/faceid/media/snapshots/evt_1.jpg", None]
