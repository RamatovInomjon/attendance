"""The AIRI mark in the browser tab.

`/favicon.ico` has been in the public list in `app/api/auth.py` since that
list existed, but nothing ever served it: there was no file and no route, so
the request a browser makes on its own was answered with a JSON 404 - which
is what faceid.airi.uz returned when this was checked against the live
deployment.

Both page templates declare the icon explicitly. That is what makes it work
under a URL prefix: a browser that has parsed no page asks the DOMAIN root
for /favicon.ico, and under /faceid the domain root is the proxy's, not this
app's. The route is here for the request that does reach us.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.config import settings

ICON = settings.root / "static" / "favicon.ico"
client = TestClient(app)


def test_the_icon_ships_with_the_project():
    assert ICON.is_file(), "static/favicon.ico is the asset the templates point at"
    head = ICON.read_bytes()[:4]
    assert head == b"\x00\x00\x01\x00", "a real .ico, not a PNG or an HTML error page"


def test_it_carries_the_sizes_a_browser_picks_from():
    """16 for the tab, 32 for the task bar, 48 for a desktop shortcut. One
    scaled-down 16x16 looks soft everywhere else."""
    from PIL import Image
    with Image.open(ICON) as im:
        assert {(16, 16), (32, 32), (48, 48)} <= set(im.ico.sizes())


def test_the_route_serves_it():
    r = client.get("/favicon.ico")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/x-icon"
    assert r.content == ICON.read_bytes()


def test_it_is_public():
    """Signed out, on the login page, the tab still shows the mark - so the
    icon must not be behind the session gate."""
    client.cookies.clear()
    assert client.get("/favicon.ico").status_code == 200


@pytest.fixture
def prefixed():
    was = settings.url_prefix
    settings.url_prefix = "/faceid"
    yield "/faceid"
    settings.url_prefix = was


def test_both_documents_declare_it_with_the_deployment_prefix(prefixed):
    """The app shell and the login page are separate root documents. The
    login page is the only one an unauthenticated visitor sees, and it had
    been missed by every earlier head-of-document change."""
    from app.api.pages import render

    for template in ("base.html", "auth/base.html"):
        html = Path(f"templates/{template}").read_text()
        assert 'rel="icon"' in html, template
        assert "{{ PREFIX }}/static/favicon.ico" in html, template

    body = render("auth/login.html", request=None, current_view="auth:login",
                  error="", next_url="/").body.decode()
    assert '/faceid/static/favicon.ico' in body
    assert '/faceid/faceid/' not in body, "the prefix must not be applied twice"
