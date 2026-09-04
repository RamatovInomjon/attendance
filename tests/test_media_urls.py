"""Media URLs under a sub-path deployment.

This whole module exists because of a failure mode that is INVISIBLE at the
root. `url_prefix` is empty on the laptop, so applying the prefix twice is a
no-op and every template renders correctly; on gpu6, served under `/faceid`,
the same code produced `/faceid/media/faceid/media/...` and every thumbnail
404'd. The click-through modal kept working, because it used the raw value -
which made it look like a rendering quirk rather than a broken URL.

So every test here runs with a prefix set, which is the configuration the bugs
live in.
"""
from __future__ import annotations

import pytest

from app.config import settings
from app.web.django_compat import media_path, media_url


@pytest.fixture
def prefixed():
    was = settings.url_prefix
    settings.url_prefix = "/faceid"
    yield "/faceid"
    settings.url_prefix = was


def test_a_stored_path_gets_the_deployment_prefix(prefixed):
    assert media_path("snapshots/x.jpg") == "/faceid/media/snapshots/x.jpg"
    assert media_path("/media/snapshots/x.jpg") == "/faceid/media/snapshots/x.jpg"


def test_applying_it_twice_changes_nothing(prefixed):
    """The actual bug. A route builds the URL, then a template filter builds it
    again, and the second application has to be a no-op."""
    once = media_path("snapshots/x.jpg")
    assert media_path(once) == once
    assert media_path(media_path(once)) == once


def test_the_filter_and_the_helper_agree(prefixed):
    """`media_url` delegates to `media_path`, so a value that has been through
    either must survive the other. The template that broke did exactly this."""
    once = media_path("snapshots/x.jpg")
    assert media_url(once) == once
    assert media_url(media_url(once)) == once


def test_absolute_and_static_urls_are_left_alone(prefixed):
    for url in ("https://cdn.example/x.jpg", "http://x/y.png", "data:image/png;base64,AAA"):
        assert media_url(url) == url
    assert media_url("/static/img/a.png") == "/faceid/static/img/a.png"


def test_at_the_root_it_still_works():
    was = settings.url_prefix
    settings.url_prefix = ""
    try:
        assert media_path("snapshots/x.jpg") == "/media/snapshots/x.jpg"
        assert media_path("/media/snapshots/x.jpg") == "/media/snapshots/x.jpg"
        assert media_path(media_path("snapshots/x.jpg")) == "/media/snapshots/x.jpg"
    finally:
        settings.url_prefix = was


def test_the_unknown_page_renders_usable_image_urls(prefixed):
    """End to end, because the unit above passed for a year while the page was
    broken - the double application lived in the template, not the helper."""
    import re
    from fastapi.testclient import TestClient
    from app.api.main import app
    from app.core.security import COOKIE_NAME, sign_session
    client = TestClient(app, follow_redirects=False)
    tok = sign_session(user_id=1, username="inomjon", is_admin=True)
    # The app mounts its routes at import, so flipping `url_prefix` in a
    # fixture changes what the TEMPLATES build without moving the routes. Take
    # whichever path this process actually serves; the assertion is about the
    # rendered URLs, not about the mount point.
    h = {"Cookie": f"{COOKIE_NAME}={tok}"}
    r = client.get("/attendance/unknown", headers=h)
    if r.status_code == 404:
        r = client.get(f"{prefixed}/attendance/unknown", headers=h)
    assert r.status_code == 200, r.status_code
    for src in re.findall(r'<img[^>]+src="([^"]+)"', r.text):
        assert "/media/faceid/" not in src and src.count("/media/") <= 1, src
        assert src.count("/faceid/") <= 1, src
