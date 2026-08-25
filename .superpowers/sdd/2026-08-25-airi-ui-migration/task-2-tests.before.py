"""Route contracts for the shared HTML page context."""
from __future__ import annotations

import warnings

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

warnings.filterwarnings(
    "ignore",
    message="distutils Version classes are deprecated. Use packaging.version instead.",
    category=DeprecationWarning,
    module="thop.profile",
)

from app.api.pages import router
from app.db.session import init_db


@pytest.fixture(scope="module")
def client():
    init_db()
    test_app = FastAPI()
    test_app.include_router(router)
    with TestClient(test_app) as test_client:
        yield test_client


@pytest.mark.parametrize(
    "path",
    ["/", "/employees", "/attendance", "/recognition", "/attendance/unknown", "/cameras"],
)
def test_page_routes_return_html(client: TestClient, path: str):
    """A missing page context must not stop any primary page from rendering."""
    response = client.get(path)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_dashboard_exposes_current_view_to_the_base_template(client: TestClient):
    """The dashboard shell must expose its explicit navigation identifier."""
    response = client.get("/")

    assert response.status_code == 200
    assert 'data-current-view="dashboard:home"' in response.text


def test_employee_registration_uses_the_navigation_view_identifier(client: TestClient):
    """The registration page must activate the registration navigation item."""
    response = client.get("/employees/add")

    assert response.status_code == 200
    assert 'data-current-view="employees:register"' in response.text
