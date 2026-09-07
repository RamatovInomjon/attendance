"""Route contracts for the shared HTML page context."""
from __future__ import annotations

import base64
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from html import unescape
import io
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace
import warnings

import cv2
import numpy as np
import pytest
from PIL import Image
from fastapi import FastAPI
from fastapi import Request
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

warnings.filterwarnings(
    "ignore",
    message="distutils Version classes are deprecated. Use packaging.version instead.",
    category=DeprecationWarning,
    module="thop.profile",
)
warnings.filterwarnings(
    "ignore",
    message="Support for class-based `config` is deprecated, use ConfigDict instead.",
    category=DeprecationWarning,
    module="pydantic._internal._config",
)

from app.api import pages
from app.api import main as api_main
from app.api.pages import router
from app.core.stream import RtspSource
from app.db.models import (
    Base, Camera, CameraRole, DailyAttendance, Employee, FaceEmbedding,
    RecognitionEvent, UnknownSighting,
)
from app.db.session import init_db
from app.services import enrollment as enrollment_service
from app.web.viewmodels import EventVM


def _app_as(role: str) -> FastAPI:
    """The pages router with a signed-in user of `role` attached.

    These tests mount the router without `auth_middleware`, so nothing would
    populate `request.state.user` and every page would render its logged-out
    shell - which is not the shell anybody sees. Standing a session in for it
    keeps the assertions about the navigation honest, and lets the same tests
    be run once per role.
    """
    init_db()
    test_app = FastAPI()

    @test_app.middleware("http")
    async def _as_user(request, call_next):
        request.state.user = {"uid": 1, "u": role, "adm": role == "admin", "r": role}
        return await call_next(request)

    test_app.include_router(router)
    return test_app


@pytest.fixture(scope="module")
def client():
    with TestClient(_app_as("admin")) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def viewer_client():
    """Reads the record, changes nothing."""
    with TestClient(_app_as("viewer")) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def operator_client():
    """A Davomat operatori: corrects attendance, never sees the cameras."""
    with TestClient(_app_as("operator")) as test_client:
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


def test_enrollment_page_keeps_submission_and_capture_contracts(monkeypatch):
    """Removing an enrollment field or capture hook must break the two-stage workflow."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        _isolated_employee_session(monkeypatch, session)
        response = pages.employee_add(_employee_page_request("/employees/add"))

    html = response.body.decode()
    assert 'action="/api/employees/"' in html
    for label in ("1. Xodim ma'lumotlari", "2. Yuz namunasini olish", "Sifat tekshiruvi"):
        assert label in html
    for name, element_id in (
        ("full_name", "full_name"), ("position", "position"),
        ("department", "department"), ("phone_number", "phone_number"),
        ("email", "email"), ("notes", "notes"),
        ("captured_image", "id_captured_image"),
        ("captured_images", "id_captured_images"),
        ("uploaded_images", "uploadedImages"),
    ):
        assert f'name="{name}"' in html
        assert f'id="{element_id}"' in html
    for element_id in (
        "registrationForm", "captureStatus", "webcam", "ipCameraStream",
        "ipCameraCanvas", "snapshot", "previewContainer", "previewImage",
        "progressContainer", "progressBar", "progressText", "angleInstruction",
        "angleInstructionText", "captureTimer", "startCaptureBtn", "cancelCaptureBtn",
        "captureComplete", "captureCount", "enrollmentResult",
        "uploadSelectionError", "uploadPreviewList",
    ):
        assert f'id="{element_id}"' in html
    assert "fetch(registrationForm.action" in html
    assert 'aria-live="polite"' in html
    assert "image/jpeg,image/png,image/webp" in html
    assert "renderUploadPreviews" in html
    assert "Olib tashlash" in html
    assert "cameraDataUrlToBlob" in html
    assert "formData.append('camera_images'" in html
    assert "hiddenImagesInput.value = JSON.stringify(capturedImages)" not in html


def test_enrollment_inline_script_has_valid_javascript(monkeypatch):
    """The rendered capture/upload workflow must remain parseable by a real JS engine."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        _isolated_employee_session(monkeypatch, session)
        response = pages.employee_add(_employee_page_request("/employees/add"))

    scripts = re.findall(r"<script(?:\s[^>]*)?>(.*?)</script>", response.body.decode(), re.DOTALL)
    result = subprocess.run(
        ["node", "--check", "-"], input="\n".join(scripts), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def _capture_data_url(pixel: int) -> str:
    """Return a real decodable JPEG while keeping inference stubbed in route tests."""
    payload = _encoded_test_image(".jpg", pixel)
    return "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")


def _encoded_test_image(extension: str, pixel: int) -> bytes:
    image = np.full((8, 8, 3), pixel, dtype=np.uint8)
    encoded, payload = cv2.imencode(extension, image)
    assert encoded
    return payload.tobytes()


def _realistic_camera_jpeg() -> bytes:
    """Return a production-shaped camera JPEG large enough to exercise multipart limits."""
    image = np.random.default_rng(7).integers(0, 256, (540, 960, 3), dtype=np.uint8)
    encoded, payload = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
    assert encoded
    raw = payload.tobytes()
    assert 100 * 1024 < len(raw) < enrollment_service.MAX_CAPTURE_BYTES
    return raw


def _pillow_image(format_name: str, size: tuple[int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, (80, 120, 160)).save(output, format=format_name)
    return output.getvalue()


def _registration_payload(images: list[str]) -> dict[str, str]:
    return {
        "full_name": "Aziza Karimova",
        "position": "Muhandis",
        "department": "AI",
        "phone_number": "+998901234567",
        "email": "aziza@example.test",
        "notes": "Sinov",
        "captured_image": images[0] if images else "",
        "captured_images": json.dumps(images),
    }


def _isolated_enrollment_database(monkeypatch, tmp_path):
    """Point both the page and enrollment service at one temporary v3 database."""
    engine = create_engine(f"sqlite:///{tmp_path / 'enrollment.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)

    @contextmanager
    def isolated_session_scope():
        session = session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr(pages, "session_scope", isolated_session_scope)
    monkeypatch.setattr(enrollment_service, "session_scope", isolated_session_scope)
    return session_factory


class _StubCaptureEnroller:
    """Inference boundary double; decoding, persistence, and HTTP behavior stay real."""

    def __init__(self, reject_at: int | set[int] | None = None):
        if isinstance(reject_at, int):
            reject_at = {reject_at}
        self.reject_at = reject_at or set()
        self.calls = 0

    def embed_capture(self, image_bgr):
        self.calls += 1
        assert image_bgr.ndim == 3 and image_bgr.shape[2] == 3
        if self.calls in self.reject_at:
            raise enrollment_service.EnrollmentCaptureError(
                "Yuz topilmadi. Yuzni kadr markaziga olib, namunani qayta oling."
            )
        vector = np.zeros(512, dtype=np.float32)
        vector[self.calls - 1] = 1.0
        return SimpleNamespace(vector=vector, quality=0.91)


def test_native_enrollment_persists_v3_embeddings_and_returns_fetch_json(monkeypatch, tmp_path):
    """A fetch submission must persist real v3 rows instead of legacy metadata only."""
    session_factory = _isolated_enrollment_database(monkeypatch, tmp_path)
    extractor = _StubCaptureEnroller()
    monkeypatch.setattr(pages, "Enroller", lambda: extractor, raising=False)
    reloads = []

    def reload_gallery():
        gallery = enrollment_service.load_gallery()
        reloads.append((len(gallery), gallery.n_people))
        return gallery

    monkeypatch.setattr(pages.runtime, "reload_gallery", reload_gallery)
    api = FastAPI()
    api.include_router(router)
    images = [_capture_data_url(30), _capture_data_url(220)]

    with TestClient(api) as test_client:
        response = test_client.post(
            "/api/employees/",
            data=_registration_payload(images),
            headers={"Accept": "application/json"},
        )

    assert response.status_code == 201
    assert response.json()["embeddings"] == 2
    assert response.json()["redirect_url"].startswith("/employees/")
    with session_factory() as session:
        employees = session.execute(select(Employee)).scalars().all()
        embeddings = session.execute(
            select(FaceEmbedding).order_by(FaceEmbedding.id)
        ).scalars().all()

    assert len(employees) == 1
    assert employees[0].full_name == "Aziza Karimova"
    assert employees[0].department == "AI"
    assert employees[0].position == "Muhandis"
    assert employees[0].phone == "+998901234567"
    assert employees[0].external_id.startswith("WEB-")
    assert len(embeddings) == 2
    assert [row.source_file for row in embeddings] == ["browser/001.jpg", "browser/002.jpg"]
    assert all(row.dim == 512 and row.model_name for row in embeddings)
    np.testing.assert_array_equal(
        np.frombuffer(embeddings[1].vector, dtype=np.float32),
        np.eye(2, 512, dtype=np.float32)[1],
    )
    assert reloads == [(2, 1)]
    assert extractor.calls == 2


def test_native_enrollment_rolls_back_and_renders_html_on_capture_rejection(monkeypatch, tmp_path):
    """A rejected capture must leave no employee/embedding rows and explain recovery in HTML."""
    session_factory = _isolated_enrollment_database(monkeypatch, tmp_path)
    extractor = _StubCaptureEnroller(reject_at={1, 2})
    monkeypatch.setattr(pages, "Enroller", lambda: extractor, raising=False)
    reloads = []
    monkeypatch.setattr(pages.runtime, "reload_gallery", lambda: reloads.append(True))
    api = FastAPI()
    api.include_router(router)

    with TestClient(api) as test_client:
        response = test_client.post(
            "/api/employees/",
            data=_registration_payload([_capture_data_url(50), _capture_data_url(100)]),
            headers={"Accept": "text/html"},
        )

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("text/html")
    assert "2-namuna rad etildi" in response.text
    assert "Yuz topilmadi" in response.text
    assert "Yuzni kadr markaziga olib" in response.text
    with session_factory() as session:
        assert session.execute(select(Employee)).scalars().all() == []
        assert session.execute(select(FaceEmbedding)).scalars().all() == []
    assert reloads == []


def test_native_enrollment_database_failure_rolls_back_employee_and_embeddings(monkeypatch, tmp_path):
    """A FaceEmbedding write failure after employee flush must roll back the whole v3 transaction."""
    session_factory = _isolated_enrollment_database(monkeypatch, tmp_path)
    real_face_embedding = enrollment_service.FaceEmbedding

    def invalid_face_embedding(**values):
        values["vector"] = None
        return real_face_embedding(**values)

    monkeypatch.setattr(enrollment_service, "FaceEmbedding", invalid_face_embedding)
    capture = enrollment_service.EnrollmentImage(
        source_file="browser/001.jpg",
        image_bgr=np.full((8, 8, 3), 80, dtype=np.uint8),
    )

    with pytest.raises(IntegrityError):
        enrollment_service.enroll_employee_captures(
            full_name="Rollback Employee",
            position="Tester",
            department="QA",
            phone_number="+998900000000",
            captures=[capture],
            enroller=_StubCaptureEnroller(),
        )

    with session_factory() as session:
        assert session.execute(select(Employee)).scalars().all() == []
        assert session.execute(select(FaceEmbedding)).scalars().all() == []


def test_native_enrollment_no_js_submission_redirects_to_v3_detail(monkeypatch, tmp_path):
    """A regular HTML form post must redirect to a rendered detail page, never raw JSON."""
    session_factory = _isolated_enrollment_database(monkeypatch, tmp_path)
    monkeypatch.setattr(pages, "Enroller", _StubCaptureEnroller, raising=False)
    monkeypatch.setattr(pages.runtime, "reload_gallery", enrollment_service.load_gallery)
    api = FastAPI()
    api.include_router(router)

    with TestClient(api) as test_client:
        response = test_client.post(
            "/api/employees/",
            data=_registration_payload([_capture_data_url(125)]),
            headers={"Accept": "text/html"},
            follow_redirects=False,
        )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/employees/")
    assert response.headers["content-type"].startswith("text/html")
    with session_factory() as session:
        employee = session.execute(select(Employee)).scalar_one()
        embedding = session.execute(select(FaceEmbedding)).scalar_one()
    assert response.headers["location"] == f"/employees/{employee.id}"
    assert embedding.employee_id == employee.id


def test_multipart_uploads_persist_v3_embeddings_through_the_capture_service(monkeypatch, tmp_path):
    """JPEG/PNG/WebP uploads must use the same extractor and v3 embedding transaction."""
    session_factory = _isolated_enrollment_database(monkeypatch, tmp_path)
    extractor = _StubCaptureEnroller()
    monkeypatch.setattr(pages, "Enroller", lambda: extractor, raising=False)
    monkeypatch.setattr(pages.runtime, "reload_gallery", enrollment_service.load_gallery)
    api = FastAPI()
    api.include_router(router)
    files = [
        ("uploaded_images", ("front.jpg", _encoded_test_image(".jpg", 40), "image/jpeg")),
        ("uploaded_images", ("left.png", _encoded_test_image(".png", 90), "image/png")),
        ("uploaded_images", ("right.webp", _encoded_test_image(".webp", 180), "image/webp")),
    ]

    with TestClient(api) as test_client:
        response = test_client.post(
            "/api/employees/",
            data=_registration_payload([]),
            files=files,
            headers={"Accept": "application/json"},
        )

    assert response.status_code == 201
    assert response.json()["embeddings"] == 3
    assert response.json()["rejected"] == []
    with session_factory() as session:
        employee = session.execute(select(Employee)).scalar_one()
        embeddings = session.execute(
            select(FaceEmbedding).order_by(FaceEmbedding.id)
        ).scalars().all()
    assert employee.full_name == "Aziza Karimova"
    assert [row.source_file for row in embeddings] == [
        "upload/001_front.jpg", "upload/002_left.png", "upload/003_right.webp",
    ]
    assert extractor.calls == 3


def test_camera_captures_submit_24_realistic_jpeg_file_parts(monkeypatch, tmp_path):
    """The production camera count must reach the route without a 1 MiB text-field failure."""
    session_factory = _isolated_enrollment_database(monkeypatch, tmp_path)
    extractor = _StubCaptureEnroller()
    monkeypatch.setattr(pages, "Enroller", lambda: extractor, raising=False)
    monkeypatch.setattr(pages.runtime, "reload_gallery", enrollment_service.load_gallery)
    api = FastAPI()
    api.include_router(router)
    camera_jpeg = _realistic_camera_jpeg()
    files = [
        ("camera_images", (f"camera-{index:03d}.jpg", camera_jpeg, "image/jpeg"))
        for index in range(1, 25)
    ]

    with TestClient(api) as test_client:
        response = test_client.post(
            "/api/employees/",
            data=_registration_payload([]),
            files=files,
            headers={"Accept": "application/json"},
        )

    assert response.status_code == 201
    assert response.json()["embeddings"] == 24
    assert extractor.calls == 24
    with session_factory() as session:
        embeddings = session.execute(
            select(FaceEmbedding).order_by(FaceEmbedding.id)
        ).scalars().all()
    assert len(embeddings) == 24
    assert embeddings[0].source_file == "browser/001.jpg"
    assert embeddings[-1].source_file == "browser/024.jpg"


def test_enrollment_parser_rejects_more_than_32_files_before_inference(monkeypatch, tmp_path):
    """The multipart parser itself must stop excess files instead of spooling 1000 of them."""
    session_factory = _isolated_enrollment_database(monkeypatch, tmp_path)
    monkeypatch.setattr(pages, "Enroller", lambda: pytest.fail("inference must not start"))
    api = FastAPI()
    api.include_router(router)
    files = [
        ("uploaded_images", (f"face-{index}.jpg", _encoded_test_image(".jpg", index), "image/jpeg"))
        for index in range(33)
    ]

    with TestClient(api) as test_client:
        response = test_client.post(
            "/api/employees/",
            data=_registration_payload([]),
            files=files,
            headers={"Accept": "application/json"},
        )

    assert response.status_code == 413
    assert response.json()["ok"] is False
    assert "32" in response.json()["error"]
    with session_factory() as session:
        assert session.execute(select(Employee)).scalars().all() == []


def test_enrollment_parser_rejects_a_file_over_4_mib_with_accessible_html(monkeypatch, tmp_path):
    """Per-file enforcement must happen while parsing and return a controlled HTML 413."""
    session_factory = _isolated_enrollment_database(monkeypatch, tmp_path)
    monkeypatch.setattr(pages, "Enroller", lambda: pytest.fail("inference must not start"))
    api = FastAPI()
    api.include_router(router)

    with TestClient(api) as test_client:
        response = test_client.post(
            "/api/employees/",
            data=_registration_payload([]),
            files=[("uploaded_images", (
                "oversize.jpg",
                b"x" * (enrollment_service.MAX_CAPTURE_BYTES + 1),
                "image/jpeg",
            ))],
            headers={"Accept": "text/html"},
        )

    assert response.status_code == 413
    assert response.headers["content-type"].startswith("text/html")
    assert 'role="alert"' in response.text
    assert "4 MB" in response.text
    with session_factory() as session:
        assert session.execute(select(Employee)).scalars().all() == []


def test_enrollment_parser_streaming_total_limit_without_content_length(monkeypatch, tmp_path):
    """Chunked bodies must be capped at 10 MiB even when Content-Length is unavailable."""
    session_factory = _isolated_enrollment_database(monkeypatch, tmp_path)
    monkeypatch.setattr(pages, "Enroller", lambda: pytest.fail("inference must not start"))
    api = FastAPI()
    api.include_router(router)
    boundary = "airi-stream-limit"
    chunks = [
        (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"full_name\""
            "\r\n\r\nAziza Karimova\r\n"
        ).encode(),
    ]
    for index in range(3):
        chunks.extend([
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"uploaded_images\"; "
                f"filename=\"face-{index}.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n"
            ).encode(),
            b"x" * 3_600_000,
            b"\r\n",
        ])
    chunks.append(f"--{boundary}--\r\n".encode())

    with TestClient(api) as test_client:
        response = test_client.post(
            "/api/employees/",
            content=iter(chunks),
            headers={
                "Accept": "application/json",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )

    assert response.status_code == 413
    assert "10 MB" in response.json()["error"]
    with session_factory() as session:
        assert session.execute(select(Employee)).scalars().all() == []


def test_enrollment_content_length_over_10_mib_short_circuits_before_parsing(monkeypatch, tmp_path):
    """A declared oversized body must get a controlled 413 without parser or inference work."""
    session_factory = _isolated_enrollment_database(monkeypatch, tmp_path)
    monkeypatch.setattr(pages, "Enroller", lambda: pytest.fail("inference must not start"))
    api = FastAPI()
    api.include_router(router)

    with TestClient(api) as test_client:
        response = test_client.post(
            "/api/employees/",
            content=b"",
            headers={
                "Accept": "application/json",
                "Content-Type": "multipart/form-data; boundary=unused",
                "Content-Length": str(enrollment_service.MAX_ENROLLMENT_BODY_BYTES + 1),
            },
        )

    assert response.status_code == 413
    assert "10 MB" in response.json()["error"]
    with session_factory() as session:
        assert session.execute(select(Employee)).scalars().all() == []


@pytest.mark.parametrize(
    ("filename", "content_type", "payload", "expected_error"),
    [
        ("disguised.jpg", "image/jpeg", _pillow_image("TIFF", (8, 8)), "haqiqiy formati"),
        ("huge.png", "image/png", _pillow_image("PNG", (5000, 3000)), "piksel"),
    ],
    ids=["disguised-tiff", "high-pixel-png"],
)
def test_enrollment_rejects_disguised_or_high_pixel_images_before_inference(
    monkeypatch, tmp_path, filename, content_type, payload, expected_error,
):
    """Header validation must reject spoofed formats and decompression-bomb dimensions."""
    session_factory = _isolated_enrollment_database(monkeypatch, tmp_path)
    extractor = _StubCaptureEnroller()
    monkeypatch.setattr(pages, "Enroller", lambda: extractor, raising=False)
    reloads = []
    monkeypatch.setattr(pages.runtime, "reload_gallery", lambda: reloads.append(True))
    api = FastAPI()
    api.include_router(router)

    with TestClient(api) as test_client:
        response = test_client.post(
            "/api/employees/",
            data=_registration_payload([]),
            files=[("uploaded_images", (filename, payload, content_type))],
            headers={"Accept": "application/json"},
        )

    assert response.status_code == 422
    assert expected_error in response.json()["rejected"][0]["error"]
    assert extractor.calls == 0
    assert reloads == []
    with session_factory() as session:
        assert session.execute(select(Employee)).scalars().all() == []
        assert session.execute(select(FaceEmbedding)).scalars().all() == []


@pytest.mark.parametrize("accept", ["application/json", "text/html"])
def test_enrollment_route_sanitizes_persistence_failure_and_skips_reload(
    monkeypatch, tmp_path, accept,
):
    """A failed embedding insert must roll back and never leak database internals to either client."""
    session_factory = _isolated_enrollment_database(monkeypatch, tmp_path)
    monkeypatch.setattr(pages, "Enroller", _StubCaptureEnroller, raising=False)
    real_face_embedding = enrollment_service.FaceEmbedding

    def invalid_face_embedding(**values):
        values["vector"] = None
        return real_face_embedding(**values)

    monkeypatch.setattr(enrollment_service, "FaceEmbedding", invalid_face_embedding)
    reloads = []
    monkeypatch.setattr(pages.runtime, "reload_gallery", lambda: reloads.append(True))
    api = FastAPI()
    api.include_router(router)

    with TestClient(api, raise_server_exceptions=False) as test_client:
        response = test_client.post(
            "/api/employees/",
            data=_registration_payload([]),
            files=[("uploaded_images", ("face.jpg", _encoded_test_image(".jpg", 90), "image/jpeg"))],
            headers={"Accept": accept},
        )

    assert response.status_code == 500
    assert "NOT NULL" not in response.text
    assert "IntegrityError" not in response.text
    if accept == "application/json":
        assert response.json() == {
            "ok": False,
            "error": "Xodimni saqlab bo'lmadi. Qayta urinib ko'ring; muammo takrorlansa administratorga murojaat qiling.",
        }
    else:
        assert response.headers["content-type"].startswith("text/html")
        assert 'role="alert"' in response.text
        assert "Xodimni saqlab bo&#39;lmadi" in response.text
    with session_factory() as session:
        assert session.execute(select(Employee)).scalars().all() == []
        assert session.execute(select(FaceEmbedding)).scalars().all() == []
    assert reloads == []


@pytest.mark.parametrize("accept", ["application/json", "text/html"])
def test_gallery_reload_failure_is_visible_after_successful_commit(monkeypatch, tmp_path, accept):
    """Operators must see a reload warning in both fetch JSON and normal HTML workflows."""
    session_factory = _isolated_enrollment_database(monkeypatch, tmp_path)
    monkeypatch.setattr(pages, "Enroller", _StubCaptureEnroller, raising=False)

    def failed_reload():
        raise RuntimeError("private gallery detail")

    monkeypatch.setattr(pages.runtime, "reload_gallery", failed_reload)
    api = FastAPI()
    api.include_router(router)

    with TestClient(api) as test_client:
        response = test_client.post(
            "/api/employees/",
            data=_registration_payload([]),
            files=[("uploaded_images", ("face.jpg", _encoded_test_image(".jpg", 120), "image/jpeg"))],
            headers={"Accept": accept},
            follow_redirects=False,
        )

    assert response.status_code == 201
    assert "private gallery detail" not in response.text
    if accept == "application/json":
        assert response.json()["ok"] is True
        assert "gallery yangilanmadi" in response.json()["warning"]
    else:
        assert response.headers["content-type"].startswith("text/html")
        assert "Location" not in response.headers
        assert 'role="alert"' in response.text
        assert "gallery yangilanmadi" in response.text
        assert "Xodim sahifasini ochish" in response.text
    with session_factory() as session:
        assert session.execute(select(Employee)).scalars().all()
        assert session.execute(select(FaceEmbedding)).scalars().all()


def test_multipart_uploads_report_unsupported_and_oversize_images_safely(monkeypatch, tmp_path):
    """Bad uploads must be reported per file while valid images still enroll."""
    session_factory = _isolated_enrollment_database(monkeypatch, tmp_path)
    extractor = _StubCaptureEnroller()
    monkeypatch.setattr(pages, "Enroller", lambda: extractor, raising=False)
    monkeypatch.setattr(pages.runtime, "reload_gallery", enrollment_service.load_gallery)
    monkeypatch.setattr(enrollment_service, "MAX_CAPTURE_BYTES", 1024)
    monkeypatch.setattr(enrollment_service, "MAX_BROWSER_CAPTURES", 3)
    api = FastAPI()
    api.include_router(router)
    files = [
        ("uploaded_images", ("valid.jpg", _encoded_test_image(".jpg", 70), "image/jpeg")),
        ("uploaded_images", ("animation.gif", b"GIF89a", "image/gif")),
        ("uploaded_images", ("huge.png", b"x" * 1025, "image/png")),
        ("uploaded_images", ("extra.webp", _encoded_test_image(".webp", 130), "image/webp")),
    ]

    with TestClient(api) as test_client:
        response = test_client.post(
            "/api/employees/",
            data=_registration_payload([]),
            files=files,
            headers={"Accept": "application/json"},
        )

    assert response.status_code == 201
    assert response.json()["embeddings"] == 1
    rejected = response.json()["rejected"]
    assert [item["source_file"] for item in rejected] == [
        "upload/002_animation.gif", "upload/003_huge.png", "upload/004_extra.webp",
    ]
    assert "JPEG, PNG yoki WebP" in rejected[0]["error"]
    assert "hajmi" in rejected[1]["error"]
    assert "ko'pi bilan 3 ta" in rejected[2]["error"]
    with session_factory() as session:
        assert session.execute(select(Employee)).scalars().all()
        embeddings = session.execute(select(FaceEmbedding)).scalars().all()
    assert len(embeddings) == 1
    assert embeddings[0].source_file == "upload/001_valid.jpg"
    assert extractor.calls == 1


def test_multipart_uploads_render_all_rejections_and_leave_no_employee(monkeypatch, tmp_path):
    """When every upload fails, HTML must explain each failure and the transaction stays empty."""
    session_factory = _isolated_enrollment_database(monkeypatch, tmp_path)
    extractor = _StubCaptureEnroller(reject_at=1)
    monkeypatch.setattr(pages, "Enroller", lambda: extractor, raising=False)
    reloads = []
    monkeypatch.setattr(pages.runtime, "reload_gallery", lambda: reloads.append(True))
    api = FastAPI()
    api.include_router(router)
    files = [
        ("uploaded_images", ("wrong.gif", b"GIF89a", "image/gif")),
        ("uploaded_images", ("no-face.jpg", _encoded_test_image(".jpg", 110), "image/jpeg")),
    ]

    with TestClient(api) as test_client:
        response = test_client.post(
            "/api/employees/",
            data=_registration_payload([]),
            files=files,
            headers={"Accept": "text/html"},
        )

    assert response.status_code == 422
    assert 'role="alert"' in response.text
    assert "Hech bir yuz namunasi yaroqli embedding bermadi" in response.text
    assert "wrong.gif" in response.text
    assert "no-face.jpg" in response.text
    assert "Yuz topilmadi" in response.text
    with session_factory() as session:
        assert session.execute(select(Employee)).scalars().all() == []
        assert session.execute(select(FaceEmbedding)).scalars().all() == []
    assert reloads == []


def test_v3_app_does_not_mount_the_legacy_employee_router():
    """The v3 app must not expose the metadata-only legacy database boundary."""
    assert not any(
        getattr(getattr(route, "endpoint", None), "__module__", "")
        == "fast_api.routers.employees"
        for route in api_main.app.routes
    )


def _unknown_page_response(monkeypatch, session: Session):
    """Render unknown-sighting review with a caller-owned database transaction."""
    _isolated_employee_session(monkeypatch, session)
    return pages.attendance_unknown(_employee_page_request("/attendance/unknown"))


def test_unknown_review_renders_evidence_and_localized_empty_state(monkeypatch):
    """Review must show the persisted track evidence and keep an actionable empty state."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        response = _unknown_page_response(monkeypatch, session)
        empty_html = response.body.decode()

        session.add(UnknownSighting(
            camera_id=17, track_id=42, business_date=date(2026, 8, 25), frames=9,
            first_seen=datetime(2026, 8, 25, 3, 5, tzinfo=timezone.utc),
            last_seen=datetime(2026, 8, 25, 3, 7, tzinfo=timezone.utc),
            snapshot="unknown/quality-best.jpg",
        ))
        session.flush()
        response = _unknown_page_response(monkeypatch, session)

    html = response.body.decode()
    assert "Noma'lum kuzatuvlar" in empty_html
    assert "Hozircha noma'lum kuzatuvlar yo'q" in empty_html
    for label in ("Eng yaxshi kadr", "Kamera", "Birinchi ko'rilgan", "Oxirgi ko'rilgan", "Kadrlar soni"):
        assert label in html
    assert 'data-evidence-src="/media/unknown/quality-best.jpg"' in html
    assert "Kamera #17" in html
    assert "08:05:00" in html
    assert "08:07:00" in html
    assert "9 ta" in html
    assert "/delete" not in html


def _camera_settings_response(monkeypatch, session: Session):
    """Render diagnostics against isolated configured cameras and worker statistics."""
    _isolated_employee_session(monkeypatch, session)
    return pages.cameras(_employee_page_request("/cameras"))


def test_camera_diagnostics_separate_configuration_observation_pipeline_and_drift(monkeypatch):
    """An absent or stale worker must remain unavailable while configuration stays visible."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        online = Camera(
            name="Sharqiy kirish", role=CameraRole.IN,
            rtsp_url="rtsp://operator:very-secret@10.0.0.17:554/stream/primary", ip="10.0.0.17",
        )
        offline = Camera(
            name="G'arbiy chiqish", role=CameraRole.OUT, rtsp_url="rtsp://west", ip="10.0.0.18",
        )
        session.add_all([online, offline])
        session.flush()
        online_id = online.id
        # What CameraWorker.stats() really carries: stream fps and the last
        # frame's timings. The algorithm rate is derived from the stream rate
        # and process_every_nth; nothing emits it directly.
        monkeypatch.setattr(pages.settings, "process_every_nth", 2)
        monkeypatch.setattr(pages.runtime, "workers", {
            online_id: SimpleNamespace(camera_id=online_id, stats=lambda: {
                "stream": {"connected": True, "stale": False, "resolution": "1920x1080", "fps": 13.0},
                "timings": {"total": 44.0},
                "pipeline_errors": 1, "last_error": "Dekoder kechikmoqda",
            }),
        })
        response = _camera_settings_response(monkeypatch, session)

    html = response.body.decode()
    for label in ("Sozlangan", "Kuzatilgan", "Pipeline", "Drift ogohlantirishi"):
        assert label in html
    assert "NVR encoder sozlamalariga egalik qiladi" in html
    assert "1920x1080" in html
    assert "13.0" in html
    assert "6.5" in html                     # 13 fps, every second frame
    assert "44.0 ms" in html
    assert "Tushib qolgan kadrlar" not in html   # nothing ever emitted it
    assert "Dekoder kechikmoqda" in html
    assert "operator" not in html
    assert "very-secret" not in html
    assert "10.0.0.17:554/stream/primary" in html
    assert 'data-drift-state="nvr-owned"' in html
    assert "Ma'lumotlarni solishtirib bo'lmaydi" in unescape(html)
    assert 'data-camera-state="online"' in html
    assert 'data-camera-state="offline"' in html
    assert "Mavjud emas" in html


def test_camera_diagnostics_survive_a_worker_stats_failure(monkeypatch):
    """One failed worker stats call must leave its configured camera explicitly unavailable."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        camera = Camera(name="Nosoz kamera", role=CameraRole.IN, rtsp_url="rtsp://broken")
        session.add(camera)
        session.flush()
        monkeypatch.setattr(pages.runtime, "workers", {
            camera.id: SimpleNamespace(camera_id=camera.id, stats=lambda: (_ for _ in ()).throw(RuntimeError("boom"))),
        })
        response = _camera_settings_response(monkeypatch, session)

    html = response.body.decode()
    assert 'data-camera-state="offline"' in html
    assert "Ishchi statistikasi olinmadi" in html


def test_camera_diagnostics_exposes_nvr_owned_non_comparable_drift(monkeypatch):
    """Removing the drift presentation object must make NVR-owned values look comparable."""
    camera = Camera(id=8, name="NVR kamera", role=CameraRole.IN, rtsp_url="rtsp://nvr")
    monkeypatch.setattr(pages.runtime, "workers", {})

    drift = pages._camera_diagnostics_context([camera])[0]["drift"]

    assert drift == {
        "state": "nvr-owned", "comparison": "unavailable",
        "label": "Ma'lumotlarni solishtirib bo'lmaydi",
        "detail": "Encoder qiymatlari NVR tomonidan boshqariladi",
    }


def test_shared_shell_exposes_localized_primary_navigation_once(client: TestClient):
    """The base shell must expose every primary destination in Uzbek exactly once."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.text.count('id="airiPrimaryNavigation"') == 1
    assert response.text.count('aria-label="Asosiy navigatsiya"') == 1
    for label, href in (
        ("Boshqaruv", "/"),
        ("Xodimlar", "/employees"),
        ("Davomat", "/attendance"),
        ("Jonli kuzatuv", "/recognition"),
        ("Ro'yxatdan o'tkazish", "/employees/add"),
        ("Noma'lumlar", "/attendance/unknown"),
        ("Kameralar", "/cameras"),
    ):
        assert label in response.text
        assert f'href="{href}"' in response.text


def test_shared_shell_provides_theme_control_and_evidence_modal(client: TestClient):
    """The shell must include the shared theme control and reusable evidence dialog."""
    response = client.get("/")

    assert response.status_code == 200
    assert 'id="themeToggle"' in response.text
    assert 'aria-pressed="false"' in response.text
    assert 'id="evidenceModal"' in response.text
    assert 'id="evidenceModalImage"' in response.text
    assert 'id="evidenceModalTitle"' in response.text
    assert 'id="evidenceModalSubtitle"' in response.text
    assert 'class="btn-close" data-bs-dismiss="modal" aria-label="Yopish"' in response.text


def test_dashboard_renders_operational_sections_for_the_current_database_state(client: TestClient):
    """The dashboard must remain useful whether the current database has events or not."""
    response = client.get("/")

    assert response.status_code == 200
    for heading in ("Kunlik davomat", "Tezkor amallar", "So'nggi tanishlar", "Tizim holati"):
        assert heading in response.text
    assert ('data-evidence-src=' in response.text or "Hozircha tanishlar yo'q" in response.text)


def test_dashboard_embeds_chart_data_and_never_exposes_debug_evidence_paths(client: TestClient):
    """Dashboard charts consume server-provided JSON and recognition evidence stays quality-best."""
    response = client.get("/")

    assert response.status_code == 200
    assert 'id="weekly-chart-data" type="application/json"' in response.text
    assert 'id="monthly-chart-data" type="application/json"' in response.text
    assert "why_" not in response.text


def _dashboard_response(monkeypatch, session: Session, day: date):
    """Render the real handler against an isolated database and fixed business date."""
    @contextmanager
    def isolated_session_scope():
        yield session

    monkeypatch.setattr(pages, "session_scope", isolated_session_scope)
    monkeypatch.setattr(pages, "today", lambda: day)
    request = Request({
        "type": "http", "method": "GET", "path": "/", "headers": [],
        "query_string": b"", "server": ("testserver", 80),
        "client": ("testclient", 50000), "scheme": "http",
    })
    return pages.dashboard(request)


def _chart_payload(html: str, element_id: str):
    match = re.search(
        rf'<script id="{element_id}" type="application/json">(.*?)</script>', html,
        flags=re.DOTALL,
    )
    assert match, element_id
    return json.loads(match.group(1))


def test_dashboard_excludes_direction_undetermined_rows_from_metrics_and_charts(monkeypatch):
    """Rows without either attendance transition cannot count as recorded presence."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    day = date(2026, 8, 25)
    with Session(engine) as session:
        employees = [Employee(full_name=f"Predicate employee {index}") for index in range(3)]
        session.add_all(employees)
        session.flush()
        session.add_all([
            DailyAttendance(employee_id=employees[0].id, business_date=day,
                            check_in_time=datetime(2026, 8, 25, 3, tzinfo=timezone.utc)),
            DailyAttendance(employee_id=employees[1].id, business_date=day,
                            check_out_time=datetime(2026, 8, 25, 9, tzinfo=timezone.utc)),
            DailyAttendance(employee_id=employees[2].id, business_date=day),
        ])
        session.commit()

        response = _dashboard_response(monkeypatch, session, day)

    assert "Davomat: 2/3 (67%)" in response.body.decode()
    weekly = _chart_payload(response.body.decode(), "weekly-chart-data")
    monthly = _chart_payload(response.body.decode(), "monthly-chart-data")
    assert weekly[-1]["present"] == 2
    assert monthly[-1]["present"] == 2


def test_dashboard_table_is_limited_to_the_most_recent_recorded_rows(monkeypatch):
    """A busy day cannot make the dashboard render an unbounded attendance table."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    day = date(2026, 8, 25)
    with Session(engine) as session:
        employees = [Employee(full_name=f"Dashboard employee {index}") for index in range(51)]
        session.add_all(employees)
        session.flush()
        session.add_all([
            DailyAttendance(
                employee_id=employee.id, business_date=day,
                check_in_time=datetime(2026, 8, 25, tzinfo=timezone.utc) + timedelta(minutes=index),
            )
            for index, employee in enumerate(employees)
        ])
        session.commit()

        response = _dashboard_response(monkeypatch, session, day)

    html = response.body.decode()
    assert "Davomat: 51/51 (100%)" in html
    assert html.count("Dashboard employee") == 50
    assert "Dashboard employee 0" not in html
    assert html.index("Dashboard employee 50") < html.index("Dashboard employee 49")


def _employee_page_request(path: str) -> Request:
    """Build the smallest complete ASGI request scope for direct page rendering."""
    return Request({
        "type": "http", "method": "GET", "path": path, "headers": [],
        "query_string": b"", "server": ("testserver", 80),
        "client": ("testclient", 50000), "scheme": "http",
    })


def _isolated_employee_session(monkeypatch, session: Session):
    """Make employee pages use the caller-owned in-memory transaction."""
    @contextmanager
    def isolated_session_scope():
        yield session

    monkeypatch.setattr(pages, "session_scope", isolated_session_scope)


def test_employee_roster_filters_round_trip_and_uses_grouped_enrollment_counts(monkeypatch):
    """Roster matches active names/IDs and exposes one real grouped enrollment result."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    statements = []
    listener = lambda _conn, _cursor, statement, _parameters, _context, _executemany: statements.append(statement)
    event.listen(engine, "before_cursor_execute", listener)
    with Session(engine) as session:
        by_name = Employee(
            full_name="Dilshod Murodov", external_id="HR-01", department="IT",
            position="Muhandis", phone="+998901112233",
        )
        by_external_id = Employee(
            full_name="Madina Karimova", external_id="AIRI-77", department="IT",
            position="Tahlilchi",
        )
        other = Employee(full_name="Sardor Aliyev", external_id="FIN-03", department="Moliya")
        inactive = Employee(
            full_name="Old AIRI record", external_id="AIRI-OLD", department="IT", is_active=False,
        )
        session.add_all([by_name, by_external_id, other, inactive])
        session.flush()
        session.add_all([
            FaceEmbedding(employee_id=by_name.id, vector=b"one"),
            FaceEmbedding(employee_id=by_external_id.id, vector=b"two"),
            FaceEmbedding(employee_id=by_external_id.id, vector=b"three"),
        ])
        session.flush()
        statements.clear()
        _isolated_employee_session(monkeypatch, session)

        response = pages.employees_list(
            _employee_page_request("/employees"), query="AIRI-77", department="IT",
        )
        external_html = response.body.decode()
        response = pages.employees_list(_employee_page_request("/employees"), query="Dilshod")
        name_html = response.body.decode()

    event.remove(engine, "before_cursor_execute", listener)

    assert 'value="AIRI-77"' in external_html
    assert '<option value="IT" selected>' in external_html
    assert "Madina Karimova" in external_html
    assert "AIRI-77" in external_html
    assert "2 ta namuna" in external_html
    assert "Sardor Aliyev" not in external_html
    assert "Old AIRI record" not in external_html
    assert "Dilshod Murodov" in name_html
    assert "Xodimlar ro'yxati" in external_html
    assert "/delete" not in external_html
    assert "bi-trash" not in external_html
    assert sum("face_embedding" in statement.lower() for statement in statements) == 2


def test_employee_detail_uses_actual_enrollment_count_and_evidence_empty_states(monkeypatch):
    """Profile shows its own embedding rows plus actionable IN/OUT evidence states."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        employee = Employee(
            full_name="Nodira Xasanova", external_id="OPS-09", department="Operatsiyalar",
            position="Navbatchi", phone="+998909998877",
        )
        session.add(employee)
        session.flush()
        session.add_all([
            FaceEmbedding(employee_id=employee.id, vector=b"face-one"),
            FaceEmbedding(employee_id=employee.id, vector=b"face-two"),
        ])
        session.add(DailyAttendance(
            employee_id=employee.id, business_date=date(2026, 8, 25),
            check_in_time=datetime(2026, 8, 25, 3, tzinfo=timezone.utc),
            check_in_snapshot="evidence/nodira-in.jpg",
        ))
        session.flush()
        _isolated_employee_session(monkeypatch, session)

        response = pages.employee_detail(_employee_page_request(f"/employees/{employee.id}"), employee.id)

    html = response.body.decode()
    assert "Nodira Xasanova" in html
    assert "OPS-09" in html
    assert "Operatsiyalar" in html
    assert "Navbatchi" in html
    assert "2 ta namuna" in html
    assert "Ro'yxatdan o'tgan" in html
    assert 'data-evidence-src="/media/evidence/nodira-in.jpg"' in html
    assert "IN dalili" in html
    assert "OUT dalili mavjud emas" in html
    assert "Profil rasmi mavjud emas" in html
    assert "Davomat tarixi" in html
    assert "/delete" not in html
    assert "bi-trash" not in html


def _attendance_page_response(monkeypatch, session: Session, day: date, **filters):
    """Render the attendance list against one caller-owned, isolated database."""
    @contextmanager
    def isolated_session_scope():
        yield session

    monkeypatch.setattr(pages, "session_scope", isolated_session_scope)
    monkeypatch.setattr(pages, "today", lambda: day)
    request = Request({
        "type": "http", "method": "GET", "path": "/attendance", "headers": [],
        "query_string": (
            b"query=AIRI-25&department=IT&start_date=2026-08-25&end_date=2026-08-25&status=PRESENT"
        ),
        "server": ("testserver", 80), "client": ("testclient", 50000), "scheme": "http",
    })
    return pages.attendance_list(request, **filters)


def test_attendance_filters_round_trip_in_sql_once_and_render_evidence(monkeypatch):
    """Breaking the joined date/name/department/status predicates must expose unrelated rows."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    statements = []
    listener = lambda _conn, _cursor, statement, _parameters, _context, _executemany: statements.append(statement)
    event.listen(engine, "before_cursor_execute", listener)
    day = date(2026, 8, 25)
    with Session(engine) as session:
        matching = Employee(full_name="Dilshod Murodov", external_id="AIRI-25", department="IT")
        later_match = Employee(full_name="Dilshod Rahimov", external_id="AIRI-26", department="IT")
        no_checkin = Employee(full_name="Dilshod Jo'rayev", external_id="AIRI-27", department="IT")
        wrong_status = Employee(full_name="Nodira Xasanova", external_id="OPS-25", department="IT")
        wrong_department = Employee(full_name="Sardor Aliyev", external_id="MOL-26", department="Moliya")
        wrong_date = Employee(full_name="Madina Karimova", external_id="AIRI-24", department="IT")
        session.add_all([matching, later_match, no_checkin, wrong_status, wrong_department, wrong_date])
        session.flush()
        session.add_all([
            DailyAttendance(
                employee_id=matching.id, business_date=day, status="PRESENT", worked_seconds=28800,
                check_in_time=datetime(2026, 8, 25, 3, tzinfo=timezone.utc),
                check_out_time=datetime(2026, 8, 25, 11, tzinfo=timezone.utc),
                check_in_snapshot="evidence/dilshod-in.jpg", check_out_snapshot="evidence/dilshod-out.jpg",
            ),
            DailyAttendance(
                employee_id=later_match.id, business_date=day, status="PRESENT",
                check_in_time=datetime(2026, 8, 25, 4, tzinfo=timezone.utc),
                check_out_time=datetime(2026, 8, 25, 12, tzinfo=timezone.utc),
            ),
            DailyAttendance(
                employee_id=no_checkin.id, business_date=day, status="NO_CHECKIN",
                check_out_time=datetime(2026, 8, 25, 10, tzinfo=timezone.utc),
            ),
            DailyAttendance(employee_id=wrong_status.id, business_date=day, status="NO_CHECKOUT"),
            DailyAttendance(employee_id=wrong_department.id, business_date=day, status="PRESENT"),
            DailyAttendance(employee_id=wrong_date.id, business_date=date(2026, 8, 24), status="PRESENT"),
        ])
        session.flush()
        statements.clear()
        response = _attendance_page_response(
            monkeypatch, session, day, query="AIRI-25", department="IT",
            start_date="2026-08-25", end_date="2026-08-25", status="PRESENT",
        )
        external_html = response.body.decode()
        response = _attendance_page_response(
            monkeypatch, session, day, query="Dilshod", department="IT",
            start_date="2026-08-25", end_date="2026-08-25", status="PRESENT",
        )
        name_html = response.body.decode()
        response = _attendance_page_response(
            monkeypatch, session, day, query="Dilshod", department="IT",
            start_date="2026-08-25", end_date="2026-08-25", status="NO_CHECKIN",
        )

    event.remove(engine, "before_cursor_execute", listener)
    no_checkin_html = response.body.decode()
    html = external_html

    assert 'value="AIRI-25"' in html
    assert '<option value="IT" selected>' in html
    assert '<option value="Moliya"' in html
    assert 'value="2026-08-25"' in html
    assert '<option value="PRESENT" selected>' in html
    assert "Dilshod Murodov" in html
    assert "Nodira Xasanova" not in html
    assert "Sardor Aliyev" not in html
    assert "Madina Karimova" not in html
    assert name_html.index("Dilshod Rahimov") < name_html.index("Dilshod Murodov")
    assert '<option value="NO_CHECKIN" selected>' in no_checkin_html
    assert "Kelish qayd etilmagan" in no_checkin_html
    assert 'data-evidence-src="/media/evidence/dilshod-in.jpg"' in html
    assert 'data-evidence-src="/media/evidence/dilshod-out.jpg"' in html
    assert "data-airi-evidence" in html
    assert "onclick=" not in html
    assert 'href="/api/attendance/export?query=AIRI-25&amp;department=IT&amp;start_date=2026-08-25&amp;end_date=2026-08-25&amp;status=PRESENT"' in html
    assert sum("from daily_attendance join employee" in statement.lower() for statement in statements) == 3
    assert len(statements) == 6
    assert not any("where employee.id =" in statement.lower() for statement in statements)


def test_attendance_normalizes_an_inverted_date_range(monkeypatch):
    """Removing inverted-range normalization would leave the date controls inconsistent with the query."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    day = date(2026, 8, 25)
    with Session(engine) as session:
        response = _attendance_page_response(
            monkeypatch, session, day, start_date="2026-08-25", end_date="2026-08-20",
        )

    html = response.body.decode()
    assert 'value="2026-08-20"' in html
    assert 'value="2026-08-25"' in html
    assert "Sana oralig'i tartibga keltirildi" in unescape(html)


@pytest.mark.parametrize(
    "params",
    [
        {"start_date": "not-a-date"},
        {"end_date": "2026-99-99"},
        {"target_date": "invalid"},
        {"start_date": "2025-08-24", "end_date": "2026-08-25"},
    ],
)
def test_attendance_rejects_invalid_or_overlong_date_ranges(client: TestClient, params):
    """Removing date validation must never turn bad attendance filters into a server error."""
    response = client.get("/attendance", params=params)

    assert response.status_code == 422


def test_attendance_export_applies_the_visible_list_filters(monkeypatch):
    """Removing a CSV predicate must expose a row that the matching list would hide."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    day = date(2026, 8, 25)
    with Session(engine) as session:
        matching = Employee(full_name="Export Dilshod", external_id="EXPORT-25", department="IT")
        wrong_department = Employee(full_name="Export Sardor", external_id="EXPORT-26", department="Moliya")
        wrong_status = Employee(full_name="Export Nodira", external_id="EXPORT-27", department="IT")
        wrong_date = Employee(full_name="Export Madina", external_id="EXPORT-24", department="IT")
        session.add_all([matching, wrong_department, wrong_status, wrong_date])
        session.flush()
        session.add_all([
            DailyAttendance(employee_id=matching.id, business_date=day, status="PRESENT", worked_seconds=3600,
                            check_in_time=datetime(2026, 8, 25, 3, tzinfo=timezone.utc)),
            DailyAttendance(employee_id=wrong_department.id, business_date=day, status="PRESENT"),
            DailyAttendance(employee_id=wrong_status.id, business_date=day, status="NO_CHECKOUT"),
            DailyAttendance(employee_id=wrong_date.id, business_date=date(2026, 8, 24), status="PRESENT"),
        ])
        session.flush()

        @contextmanager
        def isolated_session_scope():
            yield session

        monkeypatch.setattr(api_main, "session_scope", isolated_session_scope)
        monkeypatch.setattr(pages, "today", lambda: day)
        test_app = FastAPI()
        test_app.get("/api/attendance/export")(api_main.export)
        with TestClient(test_app) as test_client:
            response = test_client.get("/api/attendance/export", params={
                "query": "EXPORT-25", "department": "IT", "start_date": "2026-08-25",
                "end_date": "2026-08-25", "status": "PRESENT",
            })
            legacy_response = test_client.get("/api/attendance/export", params={"day": "2026-08-25"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "Export Dilshod" in response.text
    assert "Export Sardor" not in response.text
    assert "Export Nodira" not in response.text
    assert "Export Madina" not in response.text
    assert legacy_response.status_code == 200
    assert "Export Sardor" in legacy_response.text
    assert "filename=attendance_2026-08-25.csv" in legacy_response.headers["content-disposition"]


def _recognition_page_response(monkeypatch, session: Session, day: date):
    """Render the live console against isolated cameras, events, and runtime workers."""
    @contextmanager
    def isolated_session_scope():
        yield session

    monkeypatch.setattr(pages, "session_scope", isolated_session_scope)
    monkeypatch.setattr(pages, "today", lambda: day)
    request = Request({
        "type": "http", "method": "GET", "path": "/recognition", "headers": [],
        "query_string": b"", "server": ("testserver", 80),
        "client": ("testclient", 50000), "scheme": "http",
    })
    return pages.recognition_live(request)


def test_live_console_normalizes_role_cameras_and_missing_workers_as_offline(monkeypatch):
    """Dropping role/state normalization must not make an absent worker look healthy."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    day = date(2026, 8, 25)
    with Session(engine) as session:
        entrance = Camera(
            name="Asosiy kirish", role=CameraRole.IN, rtsp_url="rtsp://entrance", enabled=True,
        )
        exit_camera = Camera(
            name="Asosiy chiqish", role=CameraRole.OUT, rtsp_url="rtsp://exit", enabled=True,
        )
        session.add_all([entrance, exit_camera])
        session.flush()
        entrance_id = entrance.id
        exit_id = exit_camera.id
        monkeypatch.setattr(pages.settings, "process_every_nth", 2)
        worker = SimpleNamespace(
            camera_id=entrance_id,
            # The source's own clock is where the last frame time comes from;
            # stats() carries no such key.
            source=SimpleNamespace(
                last_frame_ts=datetime(2026, 8, 25, 3, 14, tzinfo=timezone.utc).timestamp()),
            stats=lambda: {
                "camera_id": entrance_id,
                "name": "Asosiy kirish",
                "role": "IN",
                "stream": {
                    "connected": True, "stale": False, "fps": 25.0,
                    "frames": 120, "resolution": "1920x1080",
                },
                "frames_processed": 100,
                "pipeline_errors": 2,
                "last_error": "Decoder navbati to'ldi",
                "timings": {"total": 40.0},
            },
        )
        monkeypatch.setattr(pages.runtime, "workers", {entrance_id: worker})

        response = _recognition_page_response(monkeypatch, session, day)

    html = response.body.decode()
    assert "Kirish kamerasi" in html
    assert "Chiqish kamerasi" in html
    assert f'data-camera-id="{entrance_id}"' in html
    assert f'data-camera-id="{exit_id}"' in html
    assert 'data-camera-role="IN"' in html
    assert 'data-camera-role="OUT"' in html
    assert 'data-stream-canvas' in html
    assert 'data-stream-state="online"' in html
    assert 'data-stream-state="offline"' in html
    for label in ("Kamera FPS", "Algoritm FPS", "Kechikish", "Oxirgi kadr"):
        assert label in html
    assert "Tushib qolgan kadrlar" not in html      # nothing ever emitted it
    assert "1920x1080" in html
    assert "25.0" in html                            # camera fps
    assert "12.5" in html                            # 25 fps, every second frame
    assert "40.0 ms" in html
    assert "08:14:00" in html
    assert "Decoder navbati to'ldi" in unescape(html)
    assert "Mavjud emas" in html


def test_live_console_and_logs_use_quality_best_evidence_with_stable_contract(monkeypatch):
    """Live evidence must remain the accepted event snapshot and polling keys must not drift."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    day = date(2026, 8, 25)
    with Session(engine) as session:
        employee = Employee(full_name="Madina Karimova", external_id="AIRI-06", department="AI")
        camera = Camera(
            name="Sharqiy kirish", role=CameraRole.IN, rtsp_url="rtsp://east", enabled=True,
        )
        session.add_all([employee, camera])
        session.flush()
        session.add(RecognitionEvent(
            employee_id=employee.id, camera_id=camera.id, role=CameraRole.IN,
            ts=datetime(2026, 8, 25, 3, 15, tzinfo=timezone.utc), business_date=day,
            score=0.934, snapshot="evidence/quality-best.jpg", transition="CHECK_IN",
        ))
        session.add(UnknownSighting(
            camera_id=camera.id, track_id=42,
            first_seen=datetime(2026, 8, 25, 3, 20, tzinfo=timezone.utc),
            last_seen=datetime(2026, 8, 25, 3, 21, tzinfo=timezone.utc),
            business_date=day, frames=7, snapshot="unknown/quality-best.jpg",
        ))
        session.flush()
        monkeypatch.setattr(pages.runtime, "workers", {})

        page_response = _recognition_page_response(monkeypatch, session, day)
        payload = pages.recognition_logs()

    html = page_response.body.decode()
    assert 'data-evidence-src="/media/evidence/quality-best.jpg"' in html
    assert "Madina Karimova" in html
    assert "Sharqiy kirish" in html
    assert "93.4%" in html
    assert "/media/unknown/quality-best.jpg" in html
    assert "why_" not in html
    assert set(payload) == {"employees", "events", "unknown_attempts", "stats"}
    assert payload["events"][0]["snapshot"] == "/media/evidence/quality-best.jpg"
    assert payload["unknown_attempts"][0]["snapshot"] == "/media/unknown/quality-best.jpg"
    assert "why_" not in json.dumps(payload)


def test_live_feed_script_is_safe_bounded_and_preserves_stream_contract():
    """Unsafe interpolation, unbounded nodes, or transport drift must fail the UI contract."""
    root = Path(__file__).resolve().parents[1]
    script = (root / "static/js/camera_stream.js").read_text()
    template = (root / "templates/recognition/live.html").read_text()

    assert "MAX_RECENT_FEED_NODES = 30" in script
    assert ".slice(0, MAX_RECENT_FEED_NODES)" in script
    assert "replaceChildren" in script
    assert "textContent" in script
    assert ".innerHTML" not in script
    assert "onclick=" not in template
    assert "data-airi-evidence" in template
    assert "document.addEventListener('click'" in script
    assert "setInterval(refreshLogs, 10000)" in script
    assert "/ws/camera/${streamEndpoint}/" in script
    assert "data:image/jpeg;base64," in script


def test_camera_socket_only_promotes_valid_non_stale_frames():
    """Socket open/stale metadata must not claim a usable camera or refresh frame metrics."""
    root = Path(__file__).resolve().parents[1]
    contract = r"""
const assert = require('node:assert/strict');
global.window = {
    location: { protocol: 'http:', host: 'testserver', origin: 'http://testserver' },
    setTimeout, clearTimeout,
};
class FakeSocket {
    static instances = [];
    constructor(url) { this.url = url; this.listeners = {}; FakeSocket.instances.push(this); }
    addEventListener(type, handler) { this.listeners[type] = handler; }
    emit(type, data = {}) { this.listeners[type]?.(data); }
    close() {}
}
class FakeImage {
    static instances = [];
    constructor() { FakeImage.instances.push(this); }
    addEventListener(type, handler) { if (type === 'load') this.loadHandler = handler; }
    set src(value) { this.width = 640; this.height = 360; }
}
global.WebSocket = FakeSocket;
global.Image = FakeImage;

const { CameraStreamManager } = require('./static/js/camera_stream.js');
const values = {
    fps: { textContent: 'Mavjud emas' },
    last: { textContent: 'Mavjud emas' },
    latency: { textContent: 'Mavjud emas' },
};
const status = {
    dataset: {}, textContent: '',
    classList: { toggle() {} },
};
const placeholder = { setAttribute() {} };
const canvas = {
    dataset: { streamEndpoint: '7' }, width: 0, height: 0,
    getContext() { return { drawImage() {} }; },
};
const card = {
    dataset: { cameraId: '7', streamState: 'unavailable' },
    querySelector(selector) {
        return {
            '[data-stream-canvas]': canvas,
            '.live-camera-card__header [data-stream-state]': status,
            '[data-stream-placeholder]': placeholder,
            '[data-camera-metric="camera-fps"]': values.fps,
            '[data-camera-metric="last-frame"]': values.last,
            '[data-camera-metric="latency"]': values.latency,
        }[selector] || null;
    },
};

const manager = new CameraStreamManager();
manager.connect(card);
const socket = FakeSocket.instances[0];
socket.emit('open');
assert.notEqual(card.dataset.streamState, 'online');

socket.emit('message', { data: JSON.stringify({ type: 'frame', data: 'jpeg', stale: true, fps: 8.5 }) });
assert.notEqual(card.dataset.streamState, 'online');
assert.equal(values.fps.textContent, 'Mavjud emas');
assert.equal(values.last.textContent, 'Mavjud emas');

socket.emit('message', { data: JSON.stringify({ type: 'frame', data: 'older', stale: false, fps: 7.5 }) });
socket.emit('message', { data: JSON.stringify({ type: 'frame', data: 'newer-stale', stale: true, fps: 8.0 }) });
FakeImage.instances[0].loadHandler();
assert.notEqual(card.dataset.streamState, 'online');
assert.equal(values.fps.textContent, 'Mavjud emas');
assert.equal(values.last.textContent, 'Mavjud emas');

socket.emit('message', { data: JSON.stringify({ type: 'frame', data: 'jpeg', stale: false, fps: 8.5 }) });
FakeImage.instances[1].loadHandler();
assert.equal(card.dataset.streamState, 'online');
assert.equal(values.fps.textContent, '8.5');
assert.notEqual(values.last.textContent, 'Mavjud emas');
"""
    result = subprocess.run(
        ["node", "-e", contract], cwd=root, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_live_script_normalizes_actual_attendance_websocket_events():
    """The live push path must consume `/ws/attendance/` event envelopes safely."""
    root = Path(__file__).resolve().parents[1]
    contract = r"""
const assert = require('node:assert/strict');
const { normalizeAttendanceEvent } = require('./static/js/camera_stream.js');
assert.deepEqual(normalizeAttendanceEvent({
    type: 'event', ts: '08:15:30', name: '<img src=x>', camera: 'Kirish',
    role: 'OUT', transition: 'CHECK_IN', score: 0.91,
    snapshot: 'snapshots/evt_1.jpg',
}), {
    name: '<img src=x>', department: '', camera: 'Kirish', action: 'IN',
    transition: 'CHECK_IN', score: 0.91, time: '08:15:30',
    snapshot: '/media/snapshots/evt_1.jpg',
});
assert.equal(normalizeAttendanceEvent({ type: 'pong' }), null);
"""
    result = subprocess.run(
        ["node", "-e", contract], cwd=root, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    script = (root / "static/js/camera_stream.js").read_text()
    assert "/ws/recognition/" not in script
    assert "/ws/attendance/" in script
    assert "setInterval(refreshLogs, 10000)" in script


def test_live_camera_context_treats_real_zero_rtsp_stats_as_unavailable(monkeypatch):
    """A fresh RtspSource's `0` FPS and `0x0` resolution are unknown, not healthy metrics."""
    camera = Camera(id=17, name="Sinov kamera", role=CameraRole.IN, rtsp_url="rtsp://unused")
    source = RtspSource("rtsp://unused", name="Sinov kamera")
    worker_stats = {
        "camera_id": camera.id,
        "stream": source.stats(),
        "pipeline_errors": 0,
        "timings": {},
    }
    worker = SimpleNamespace(camera_id=camera.id, stats=lambda: worker_stats)
    monkeypatch.setattr(pages.runtime, "workers", {camera.id: worker})

    context = pages._live_camera_context([camera])[0]

    assert context["state"] == "unavailable"
    assert context["camera_fps"] is None
    assert context["resolution"] is None


@pytest.mark.parametrize(
    ("role", "transition", "expected"),
    [
        (CameraRole.OUT, "CHECK_IN", "IN"),
        (CameraRole.IN, "CHECK_OUT", "OUT"),
        (CameraRole.OUT, "RE_SIGHTING", "OUT"),
        # The losing half of a cross-camera pair moved no state, so it must not
        # be drawn as a check-in or check-out.
        (CameraRole.IN, "DUPLICATE_VIEW", "SEEN"),
        (CameraRole.OUT, "DUPLICATE_VIEW", "SEEN"),
    ],
)
def test_event_vm_action_prefers_attendance_transition_over_camera_role(role, transition, expected):
    """A persisted attendance transition is authoritative when camera role disagrees."""
    event_row = SimpleNamespace(
        id=1, role=role, transition=transition,
        ts=datetime(2026, 8, 25, 3, tzinfo=timezone.utc),
        snapshot=None, score=0.9,
    )

    event = EventVM.of(event_row, "Xodim", "AI", "Kamera")

    assert event.action_type == expected


# --- media URLs must carry the deployment prefix ---------------------------
# Found in a browser against the deployed site: every snapshot on the
# dashboard 404'd because "/media/x.jpg" resolves against the DOMAIN root,
# which on aiscan.airi.uz belongs to a different project entirely.

import pytest as _pytest


@_pytest.mark.parametrize("stored,expected", [
    ("snapshots/evt_1.jpg", "/faceid/media/snapshots/evt_1.jpg"),
    ("/media/snapshots/a.jpg", "/faceid/media/snapshots/a.jpg"),
    ("media/snapshots/a.jpg", "/faceid/media/snapshots/a.jpg"),
])
def test_media_path_carries_the_prefix(stored, expected, monkeypatch):
    from app.config import settings
    from app.web.django_compat import media_path
    monkeypatch.setattr(settings, "url_prefix", "/faceid", raising=False)
    assert media_path(stored) == expected


def test_media_path_at_the_root_is_unchanged(monkeypatch):
    from app.config import settings
    from app.web.django_compat import media_path
    monkeypatch.setattr(settings, "url_prefix", "", raising=False)
    assert media_path("snapshots/evt_1.jpg") == "/media/snapshots/evt_1.jpg"


def test_media_url_filter_prefixes_placeholder_and_files(monkeypatch):
    from app.config import settings
    from app.web.django_compat import media_url
    monkeypatch.setattr(settings, "url_prefix", "/faceid", raising=False)
    assert media_url("") == "/faceid/static/img/avatar-placeholder.png"
    assert media_url("faces/a.jpg") == "/faceid/media/faces/a.jpg"
    # absolute and data URLs are left alone
    assert media_url("https://x/y.jpg") == "https://x/y.jpg"
    assert media_url("data:image/png;base64,AA") == "data:image/png;base64,AA"


# --- WebSocket URLs must carry the deployment prefix -----------------------
# The same class of bug as the media URLs above, and it survived that fix
# because .js files cannot interpolate {{ PREFIX }}. Under /faceid a bare
# "/ws/camera/1/" resolves against the DOMAIN root, which on aiscan.airi.uz
# belongs to a different project: the live console's camera feed and its
# attendance push both 404, while the page itself looks fine.

def test_camera_socket_urls_carry_the_deployment_prefix():
    import subprocess
    root = Path(__file__).resolve().parent.parent
    contract = r"""
const assert = require('node:assert/strict');
global.window = { location: { protocol: 'https:', host: 'aiscan.airi.uz' },
                  addEventListener() {} };
global.document = {
    querySelector: (sel) => sel === '[data-url-prefix]'
        ? { dataset: { urlPrefix: '/faceid' } } : null,
    querySelectorAll: () => [],
    addEventListener() {},
};
const src = require('fs').readFileSync('static/js/camera_stream.js', 'utf8');
eval(src);
assert.equal(airiPrefix(), '/faceid');
assert.equal(airiSocketUrl('/ws/camera/1/'),
             'wss://aiscan.airi.uz/faceid/ws/camera/1/');
assert.equal(airiSocketUrl('/ws/attendance/'),
             'wss://aiscan.airi.uz/faceid/ws/attendance/');
console.log('ok');
"""
    r = subprocess.run(["node", "-e", contract], cwd=root,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_socket_urls_are_bare_when_served_at_the_root():
    """No prefix attribute means the app is at the domain root, which is how it
    runs locally - the helper must not invent a path segment."""
    import subprocess
    root = Path(__file__).resolve().parent.parent
    contract = r"""
const assert = require('node:assert/strict');
global.window = { location: { protocol: 'http:', host: '127.0.0.1:8000' },
                  addEventListener() {} };
global.document = { querySelector: () => null, querySelectorAll: () => [],
                    addEventListener() {} };
eval(require('fs').readFileSync('static/js/camera_stream.js', 'utf8'));
assert.equal(airiSocketUrl('/ws/attendance/'), 'ws://127.0.0.1:8000/ws/attendance/');
console.log('ok');
"""
    r = subprocess.run(["node", "-e", contract], cwd=root,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_no_javascript_points_at_a_route_that_does_not_exist():
    """`/ws/recognition/` had two clients and no server route, one of them over
    a hardcoded ws:// that fails on the HTTPS deployment."""
    root = Path(__file__).resolve().parent.parent
    routes = (root / "app" / "api" / "ws.py").read_text()
    for js in (root / "static" / "js").glob("*.js"):
        body = js.read_text()
        for path in re.findall(r"/ws/[a-z]+/", body):
            assert path.rstrip("/") in routes or "{camera_id}" in routes, (
                f"{js.name} connects to {path}, which app/api/ws.py does not serve")


def test_no_javascript_hardcodes_insecure_websockets():
    """`ws://` on an HTTPS page is blocked by the browser as mixed content."""
    root = Path(__file__).resolve().parent.parent
    for js in (root / "static" / "js").glob("*.js"):
        body = js.read_text()
        for line in body.splitlines():
            if "new WebSocket(" in line:
                assert "ws://" not in line, (
                    f"{js.name} hardcodes ws://; derive it from "
                    f"window.location.protocol: {line.strip()}")


# ---- what each role's shell shows ---------------------------------------

def test_operator_shell_hides_every_infrastructure_destination(operator_client: TestClient):
    """The five things a Davomat operatori was asked not to see. Hiding the
    link is only the cosmetic half - `tests/test_roles.py` proves the paths
    themselves refuse - but a link to a 403 is its own kind of broken."""
    body = operator_client.get("/").text
    for label in ("Pipeline holati", "Jonli kuzatuv", "Kameralar",
                  "Galereya", "Foydalanuvchilar"):
        assert label not in body, f"operator shell still offers {label!r}"


def test_operator_shell_keeps_the_pages_the_job_needs(operator_client: TestClient):
    body = operator_client.get("/").text
    for label in ("Boshqaruv", "Xodimlar", "Davomat", "Noma'lumlar"):
        assert label in body


def test_operator_dashboard_hides_the_pipeline_panel(operator_client: TestClient):
    """"Tizim holati" is per-camera FPS and pipeline error counts - the same
    content as the live page, in a card."""
    body = operator_client.get("/").text
    assert "Tizim holati" not in body
    assert "Kunlik davomat" in body       # the rest of the page is untouched


def test_operator_gets_exactly_the_same_correction_controls_as_an_admin(
        client: TestClient, operator_client: TestClient, viewer_client: TestClient):
    """The control the role exists for: "bu u emas" on the event feed.

    Counted rather than merely looked for, so the test still means something
    if the seeded events change - and compared against the admin's page so it
    cannot pass by both being empty."""
    from app.db.session import session_scope
    from app.services.attendance import business_date

    with session_scope() as session:
        employee = Employee(full_name="Void Control Subject", external_id="AIRI-VC")
        camera = Camera(name="Nazorat", role=CameraRole.IN,
                        rtsp_url="rtsp://vc", enabled=True)
        session.add_all([employee, camera])
        session.flush()
        now = datetime.now(timezone.utc)
        session.add(RecognitionEvent(
            employee_id=employee.id, camera_id=camera.id, role=CameraRole.IN,
            ts=now, business_date=business_date(now), score=0.91,
            snapshot="evidence/vc.jpg", transition="CHECK_IN"))

    def voids(c):
        return c.get("/").text.count("/attendance/event/")

    assert voids(client) > 0, "no events seeded: this test would prove nothing"
    assert voids(operator_client) == voids(client)
    assert voids(viewer_client) == 0


def test_live_stream_falls_back_to_mjpeg_when_the_socket_never_delivers_a_frame():
    """The live page showed "Mavjud emas" on aiscan.airi.uz because uvicorn was
    installed without `websockets`, so every handshake was served as ordinary
    HTTP and answered with a login redirect. A socket that closes before a
    single frame is not a flaky link - it is a deployment where that transport
    cannot work - so the page moves to the MJPEG endpoint, which is plain HTTP
    and needs neither the library nor an Upgrade-aware proxy.
    """
    root = Path(__file__).resolve().parents[1]
    contract = r"""
const assert = require('assert');
global.window = {
    location: { protocol: 'https:', host: 'aiscan.airi.uz' },
    setTimeout: () => 1, clearTimeout: () => {},
};
// A document is needed only so `airiPrefix()` finds the deployment prefix -
// this suite drives the manager directly, not the DOMContentLoaded path.
global.document = {
    querySelector: (s) => (s === '[data-url-prefix]'
        ? { dataset: { urlPrefix: '/faceid' } } : null),
    addEventListener() {},
};

class FakeSocket {
    static instances = [];
    constructor(url) { this.url = url; this.handlers = {}; FakeSocket.instances.push(this); }
    addEventListener(t, h) { (this.handlers[t] = this.handlers[t] || []).push(h); }
    emit(t, e) { (this.handlers[t] || []).forEach((h) => h(e)); }
    close() {}
}
class FakeImage {
    static instances = [];
    constructor() { this.handlers = {}; FakeImage.instances.push(this); }
    addEventListener(t, h) { (this.handlers[t] = this.handlers[t] || []).push(h); }
    emit(t) { (this.handlers[t] || []).forEach((h) => h()); }
    remove() { this.removed = true; }
    set src(v) { this._src = v; }
    get src() { return this._src; }
}
global.WebSocket = FakeSocket;
global.Image = FakeImage;

const { CameraStreamManager } = require('./static/js/camera_stream.js');
const placeholder = { hidden: false,
    setAttribute() { this.hidden = true; }, removeAttribute() { this.hidden = false; } };
const status = { dataset: {}, textContent: '', classList: { toggle() {} } };
const canvas = {
    dataset: { streamEndpoint: '7' }, attrs: {},
    getAttribute(k) { return k === 'aria-label' ? 'Kirish kamerasi' : null; },
    setAttribute(k) { this.attrs[k] = true; },
    removeAttribute(k) { delete this.attrs[k]; },
    after(node) { this.sibling = node; },
    getContext() { return { drawImage() {} }; },
};
const card = {
    dataset: { cameraId: '7', streamState: 'unavailable' },
    querySelector(sel) {
        return { '[data-stream-canvas]': canvas,
                 '.live-camera-card__header [data-stream-state]': status,
                 '[data-stream-placeholder]': placeholder }[sel] || null;
    },
};

const manager = new CameraStreamManager();
manager.connect(card);
assert.equal(FakeSocket.instances[0].url,
    'wss://aiscan.airi.uz/faceid/ws/camera/7/', 'socket is tried first');

FakeSocket.instances[0].emit('close');
const img = FakeImage.instances[0];
assert.ok(img, 'a frameless close must fall back rather than only retry');
assert.equal(img.src, '/faceid/video/7', 'MJPEG carries the deployment prefix');
assert.ok(canvas.attrs.hidden, 'the canvas is hidden while the img is showing');
assert.equal(canvas.sibling, img);

img.emit('load');
assert.equal(card.dataset.streamState, 'online');
assert.equal(placeholder.hidden, true);

// A camera that is not running 404s here exactly as the socket closed 4004.
img.emit('error');
assert.equal(card.dataset.streamState, 'offline');
assert.equal(placeholder.hidden, false);

// Teardown must end the multipart request, or its JPEG encode runs for the
// life of the page - one more every time the operator hits reconnect.
manager.disconnect('7');
assert.equal(img.src, '');
assert.ok(img.removed);
assert.ok(!canvas.attrs.hidden, 'the canvas comes back for the next attempt');

// A socket that DID deliver frames is a flaky link, not a dead transport:
// that one keeps reconnecting instead of downgrading.
FakeImage.instances.length = 0;
manager.connect(card);
const live = FakeSocket.instances[FakeSocket.instances.length - 1];
live.emit('message', { data: JSON.stringify({ type: 'frame', data: 'jpeg', stale: false }) });
live.emit('close');
// decodeFrame builds an Image per frame, so count only MJPEG ones.
assert.equal(FakeImage.instances.filter((i) => String(i.src).includes('/video/')).length, 0,
    'a stream that has shown frames must not downgrade on one drop');
"""
    result = subprocess.run(
        ["node", "-e", contract], cwd=root, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


# ---- review fixes -----------------------------------------------------------

def test_escapejs_neutralises_everything_that_could_end_a_string_literal():
    """json.dumps left the apostrophe alone, so a name closed the confirm('...')
    literal an inline handler had put it in and the rest ran as code."""
    from app.web.django_compat import escapejs, json_script
    out = escapejs("x'); alert(1); //")
    assert "'" not in out and ";" not in out
    assert out == "x\\u0027)\\u003B alert(1)\\u003B //"
    assert escapejs('a"b<c>&d\\e') == "a\\u0022b\\u003Cc\\u003E\\u0026d\\u005Ce"
    assert escapejs(None) == ""
    # The JSON island cannot close its own <script> block either.
    block = str(json_script({"name": "</script><img src=x onerror=alert(1)>&"}, "x"))
    assert "</script><img" not in block
    assert block.count("</script>") == 1
    assert json.loads(re.search(r">(\{.*\})<", block).group(1)) == {
        "name": "</script><img src=x onerror=alert(1)>&"}


def test_a_hostile_employee_name_cannot_break_out_of_the_confirm_handler(client: TestClient):
    """Rendered end to end: the dashboard feed and the day view both put the
    name inside onsubmit="return confirm('...')"."""
    from app.db.session import session_scope
    from app.services.attendance import business_date

    name = "Ali'); alert(1); //"
    with session_scope() as session:
        employee = Employee(full_name=name, external_id="AIRI-XSS")
        camera = Camera(name="Nazorat-XSS", role=CameraRole.IN, rtsp_url="rtsp://xss", enabled=True)
        session.add_all([employee, camera])
        session.flush()
        now = datetime.now(timezone.utc)
        session.add(RecognitionEvent(
            employee_id=employee.id, camera_id=camera.id, role=CameraRole.IN,
            ts=now, business_date=business_date(now), score=0.9,
            snapshot="evidence/xss.jpg", transition="CHECK_IN"))
        emp_id, day = employee.id, business_date(now)

    for path in ("/", f"/attendance/day/{emp_id}/{day}"):
        html = client.get(path).text
        handlers = [unescape(v) for v in re.findall(r'onsubmit="([^"]*)"', html)]
        ours = [h for h in handlers if "alert(1)" in h]
        assert ours, f"{path}: the seeded event did not render with a void control"
        for h in ours:
            # Exactly the two delimiting quotes: the name's own is ' now.
            assert h.count("'") == 2, h
            assert re.fullmatch(r"return confirm\('[^']*'\)", h), h
            assert "\\u0027)\\u003B alert(1)" in h


def test_enrolment_is_offered_only_to_an_admin(client: TestClient, operator_client: TestClient,
                                              viewer_client: TestClient):
    """The route is admin-only now (tests/test_roles.py); a link to a 403 is
    its own kind of broken."""
    for path in ("/", "/employees"):
        assert "Ro'yxatdan o'tkazish" in client.get(path).text
        for other in (operator_client, viewer_client):
            assert "Ro'yxatdan o'tkazish" not in other.get(path).text, path


def test_enrolment_uses_the_browser_camera_even_with_cameras_configured(monkeypatch):
    """The IP-camera branch opens /ws/camera/registration/, which app/api/ws.py
    has never resolved - so with cameras configured, i.e. on every real
    deployment, the capture never produced a frame."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Camera(name="Kirish", role=CameraRole.IN, rtsp_url="rtsp://in", enabled=True))
        session.flush()
        _isolated_employee_session(monkeypatch, session)
        html = pages.employee_add(_employee_page_request("/employees/add")).body.decode()
    assert "const useIpCamera = false;" in html
    assert 'id="cameraSelector"' not in html
    assert "registration" not in (Path(__file__).resolve().parents[1] / "app/api/ws.py").read_text()


def test_the_attendance_table_stops_at_a_month_and_says_so(monkeypatch):
    """Up to a year of rows used to be materialised for one page. The table
    keeps the newest month and says so; the CSV export keeps the range."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    day = date(2026, 8, 25)
    with Session(engine) as session:
        emp = Employee(full_name="Oy Chegarasi", external_id="AIRI-31", department="IT")
        session.add(emp)
        session.flush()
        for d in (date(2026, 6, 15), date(2026, 7, 25), date(2026, 8, 20)):
            session.add(DailyAttendance(
                employee_id=emp.id, business_date=d, status="PRESENT",
                check_in_time=datetime(d.year, d.month, d.day, 3, tzinfo=timezone.utc)))
        session.flush()
        html = _attendance_page_response(
            monkeypatch, session, day, start_date="2026-06-01", end_date="2026-08-25",
        ).body.decode()
        # A calendar month - the page's own "Bir oy" shortcut - is never cut.
        month = _attendance_page_response(
            monkeypatch, session, day, start_date="2026-07-25", end_date="2026-08-25",
        ).body.decode()

    assert "data-range-cap" in html
    assert "31 kun" in unescape(html)
    assert 'value="2026-07-25"' in html and 'value="2026-08-25"' in html
    # One "Hodisalar" link per rendered row.
    rows = re.findall(r"/attendance/day/\d+/(\d{4}-\d{2}-\d{2})", html)
    assert sorted(rows) == ["2026-07-25", "2026-08-20"]
    assert "data-range-cap" not in month
    assert sorted(re.findall(r"/attendance/day/\d+/(\d{4}-\d{2}-\d{2})", month)) == [
        "2026-07-25", "2026-08-20"]


def test_the_day_view_lists_the_debug_directory_once(monkeypatch, tmp_path):
    """`capture_stem` globs data/debug per call, so a day of events was a
    directory walk per row. One listing, the same answers."""
    import glob
    from app.config import settings
    from app.services import corrections
    from app.services.attendance import business_date

    monkeypatch.setattr(settings, "debug_dir", tmp_path, raising=False)
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        emp = Employee(full_name="Indeks Sinovi", external_id="AIRI-IDX")
        cam = Camera(name="Exit", role=CameraRole.OUT, rtsp_url="rtsp://exit", enabled=True)
        session.add_all([emp, cam])
        session.flush()
        base = datetime(2026, 9, 3, 8, 0, 30, tzinfo=timezone.utc)
        (tmp_path / "Indeks Sinovi").mkdir()
        stems = []
        for i in range(3):
            ts, score = base + timedelta(minutes=i), 0.25 + i / 100
            session.add(RecognitionEvent(
                employee_id=emp.id, camera_id=cam.id, role=CameraRole.OUT, ts=ts,
                business_date=business_date(ts), score=score, transition="CHECK_OUT"))
            stem = f"{ts.astimezone(settings.tz):%Y%m%d_%H%M%S}_{score:.3f}_Exit_001"
            (tmp_path / "Indeks Sinovi" / f"{stem}.json").write_text("{}")
            stems.append((ts, score, stem))
        session.flush()
        _isolated_employee_session(monkeypatch, session)

        # Parity with the per-event helper, the miss included.
        index = pages._capture_index()
        for ts, score, stem in stems:
            assert pages._capture_stem_from(index, ts, score, "Exit") == stem
            assert corrections.capture_stem(ts, score, "Exit") == stem
        assert pages._capture_stem_from(index, base, 0.999, "Exit") == ""
        assert corrections.capture_stem(base, 0.999, "Exit") == ""

        calls: list = []
        real_glob = glob.glob
        monkeypatch.setattr(glob, "glob", lambda *a, **k: calls.append(a) or real_glob(*a, **k))
        html = pages.attendance_day(_employee_page_request("/attendance/day"), emp.id,
                                    business_date(base).isoformat()).body.decode()
    assert len(calls) == 1
    assert html.count("/evidence/face") == 3


def test_unknown_review_preselects_the_resolved_employee(monkeypatch):
    """The picker reads `attempt.resolved_employee_id`, which the row omitted."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        emp = Employee(full_name="Belgilangan Xodim", external_id="AIRI-RS")
        session.add(emp)
        session.flush()
        session.add(UnknownSighting(
            camera_id=1, track_id=7, business_date=date(2026, 8, 25), frames=3,
            first_seen=datetime(2026, 8, 25, 3, 5, tzinfo=timezone.utc),
            last_seen=datetime(2026, 8, 25, 3, 7, tzinfo=timezone.utc),
            resolved_kind="employee", resolved_employee_id=emp.id, resolved_by="inomjon"))
        session.flush()
        _isolated_employee_session(monkeypatch, session)
        request = Request({
            "type": "http", "method": "GET", "path": "/attendance/unknown", "headers": [],
            "query_string": b"show=all", "server": ("testserver", 80),
            "client": ("testclient", 50000), "scheme": "http",
            "state": {"user": {"uid": 1, "u": "inomjon", "adm": True, "r": "admin"}},
        })
        html = pages.attendance_unknown(request, show="all").body.decode()
    assert f'data-selected="{emp.id}"' in html


def test_the_forbidden_page_link_carries_the_prefix(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "url_prefix", "/faceid", raising=False)
    html = pages.render("auth/forbidden.html",
                        request=_employee_page_request("/cameras")).body.decode()
    assert 'href="/faceid/"' in html
    assert 'href="/"' not in html


def test_live_script_prefixes_bare_snapshot_paths_under_a_sub_path():
    root = Path(__file__).resolve().parents[1]
    contract = r"""
const assert = require('node:assert/strict');
global.window = { location: { protocol: 'https:', host: 'aiscan.airi.uz', origin: 'https://aiscan.airi.uz' },
                  addEventListener() {} };
global.document = {
    querySelector: (sel) => sel === '[data-url-prefix]' ? { dataset: { urlPrefix: '/faceid' } } : null,
    querySelectorAll: () => [],
    addEventListener() {},
};
const { normalizeAttendanceEvent } = require('./static/js/camera_stream.js');
assert.equal(normalizeAttendanceEvent({ type: 'event', snapshot: 'snapshots/evt_1.jpg' }).snapshot,
             '/faceid/media/snapshots/evt_1.jpg');
// Already a URL - which is what the server sends now - passes through.
assert.equal(normalizeAttendanceEvent({ type: 'event', snapshot: '/faceid/media/snapshots/evt_1.jpg' }).snapshot,
             '/faceid/media/snapshots/evt_1.jpg');
"""
    result = subprocess.run(["node", "-e", contract], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_camera_socket_marks_a_stale_notice_that_carries_no_frame():
    """The server no longer re-sends a quiet camera's last frame; it says
    `stale` without `data`, and the page must act on that rather than drop it."""
    root = Path(__file__).resolve().parents[1]
    contract = r"""
const assert = require('node:assert/strict');
global.window = { location: { protocol: 'http:', host: 'testserver', origin: 'http://testserver' },
                  setTimeout: () => 1, clearTimeout() {} };
class FakeSocket {
    static instances = [];
    constructor(url) { this.url = url; this.listeners = {}; FakeSocket.instances.push(this); }
    addEventListener(type, handler) { this.listeners[type] = handler; }
    emit(type, data = {}) { this.listeners[type]?.(data); }
    close() {}
}
class FakeImage {
    static instances = [];
    constructor() { FakeImage.instances.push(this); }
    addEventListener(type, handler) { if (type === 'load') this.loadHandler = handler; }
    set src(value) { this._src = value; this.width = 640; this.height = 360; }
    get src() { return this._src; }
}
global.WebSocket = FakeSocket;
global.Image = FakeImage;
const { CameraStreamManager } = require('./static/js/camera_stream.js');
const status = { dataset: {}, textContent: '', classList: { toggle() {} } };
const canvas = { dataset: { streamEndpoint: '7' }, width: 0, height: 0,
                 getContext() { return { drawImage() {} }; },
                 setAttribute() {}, removeAttribute() {}, after() {}, getAttribute() { return ''; } };
const card = {
    dataset: { cameraId: '7', streamState: 'unavailable' },
    querySelector(selector) {
        return { '[data-stream-canvas]': canvas,
                 '.live-camera-card__header [data-stream-state]': status,
                 '[data-stream-placeholder]': { setAttribute() {}, removeAttribute() {} } }[selector] || null;
    },
};
const manager = new CameraStreamManager();
manager.connect(card);
const socket = FakeSocket.instances[0];
socket.emit('open');
socket.emit('message', { data: JSON.stringify({ type: 'frame', data: 'jpeg', stale: false, fps: 8.5 }) });
FakeImage.instances[0].loadHandler();
assert.equal(card.dataset.streamState, 'online');

socket.emit('message', { data: JSON.stringify({ type: 'frame', stale: true, fps: 8.5 }) });
assert.equal(card.dataset.streamState, 'unavailable');
assert.equal(status.textContent, 'Oqim eskirgan');
assert.equal(FakeImage.instances.length, 1, 'a frameless notice decodes nothing');

// A stale notice proves the transport works: a drop afterwards reconnects
// rather than downgrading to MJPEG.
socket.emit('close');
assert.equal(FakeImage.instances.filter((i) => String(i.src).includes('/video/')).length, 0);
"""
    result = subprocess.run(["node", "-e", contract], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ---- the registered photograph on a profile -------------------------------

PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c63000100000500010d0a2db40000000049454e44ae"
    "426082")


@pytest.fixture
def enrolled(tmp_path, monkeypatch):
    """An employee with a registered photograph on disk, cleaned up after.

    `gallery_dir` is redirected at tmp_path so the test never reads - or is
    satisfied by - the real face_id_users export.
    """
    from app.config import settings
    from app.db.session import session_scope
    from sqlalchemy import delete
    monkeypatch.setattr(settings, "gallery_dir", tmp_path)
    made = []

    def make(name="Portrait Person", folder="900_Portrait",
             filename="image_01.png", *, on_disk=True, source=None):
        if on_disk:
            (tmp_path / folder).mkdir(parents=True, exist_ok=True)
            (tmp_path / folder / filename).write_bytes(PNG_1PX)
        with session_scope() as s:
            e = Employee(full_name=name, folder=folder, is_active=True)
            s.add(e); s.flush()
            if source is not False:
                s.add(FaceEmbedding(employee_id=e.id,
                                    source_file=source or filename,
                                    vector=b"\x00" * 2048, dim=512))
            made.append(e.id)
            return e.id

    yield make

    with session_scope() as s:
        if made:
            s.execute(delete(FaceEmbedding).where(
                FaceEmbedding.employee_id.in_(made)))
            s.execute(delete(Employee).where(Employee.id.in_(made)))


def test_a_profile_shows_the_photograph_the_person_was_registered_with(
        client, enrolled):
    emp_id = enrolled()

    page = client.get(f"/employees/{emp_id}")
    assert page.status_code == 200
    assert f"/employees/{emp_id}/photo" in page.text
    assert "Profil rasmi mavjud emas" not in page.text

    shot = client.get(f"/employees/{emp_id}/photo")
    assert shot.status_code == 200
    assert shot.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_clicking_the_photograph_opens_the_evidence_view(client, enrolled):
    """The photo must be a control wired to the same modal as the IN/OUT
    evidence buttons beside it - and wired ONCE. This page binds every
    [data-evidence-src] in its own script, so carrying the global
    `data-airi-evidence` marker as well would open the modal twice per click."""
    emp_id = enrolled(name="Clickable", folder="901_Clickable")

    html = client.get(f"/employees/{emp_id}").text
    at = html.index(f"/employees/{emp_id}/photo")
    button = html[at - 300:at + 300]
    assert "data-evidence-src" in button
    assert "<button" in button
    assert "data-airi-evidence" not in html


def test_a_person_with_no_photograph_still_renders(client, enrolled):
    """The placeholder is the correct answer, not a broken image."""
    emp_id = enrolled(name="No Photo", folder="904_None", on_disk=False,
                      source=False)

    page = client.get(f"/employees/{emp_id}")
    assert page.status_code == 200
    assert "Profil rasmi mavjud emas" in page.text
    assert f"/employees/{emp_id}/photo" not in page.text
    assert client.get(f"/employees/{emp_id}/photo").status_code == 404


def test_a_corridor_crop_is_never_shown_as_somebody_portrait(client, enrolled):
    """An augmented `live:` row recognises well and looks nothing like a
    portrait, and its source_file is a capture key, not a path."""
    from app.services import augment
    emp_id = enrolled(name="Only Corridor", folder="902_Corridor",
                      on_disk=False, source=f"{augment.TAG}somekey")

    assert augment.profile_image(emp_id) is None
    assert client.get(f"/employees/{emp_id}/photo").status_code == 404


def test_the_photograph_route_cannot_be_walked_out_of_the_gallery(
        client, enrolled, tmp_path):
    """`folder` and `source_file` come from the database, but the resolved path
    is still checked: a row saying '..' must not read the filesystem."""
    from app.services import augment
    (tmp_path.parent / "secret.txt").write_text("not a face")
    emp_id = enrolled(name="Traversal", folder="..", on_disk=False,
                      source="secret.txt")

    assert augment.profile_image(emp_id) is None
    assert client.get(f"/employees/{emp_id}/photo").status_code == 404


def test_every_signed_in_role_may_see_the_profile_photograph(
        viewer_client, operator_client, enrolled):
    """It follows the profile page, which every role can already open. Gating it
    at admin would leave a broken image on a page they are meant to read."""
    emp_id = enrolled(name="Seen By All", folder="903_All")

    for c in (viewer_client, operator_client):
        assert c.get(f"/employees/{emp_id}/photo").status_code == 200
