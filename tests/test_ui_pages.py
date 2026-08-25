"""Route contracts for the shared HTML page context."""
from __future__ import annotations

import base64
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from html import unescape
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace
import warnings

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi import Request
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
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
    ):
        assert f'name="{name}"' in html
        assert f'id="{element_id}"' in html
    for element_id in (
        "registrationForm", "captureStatus", "webcam", "ipCameraStream",
        "ipCameraCanvas", "snapshot", "previewContainer", "previewImage",
        "progressContainer", "progressBar", "progressText", "angleInstruction",
        "angleInstructionText", "captureTimer", "startCaptureBtn", "cancelCaptureBtn",
        "captureComplete", "captureCount",
    ):
        assert f'id="{element_id}"' in html


def _capture_data_url(pixel: int) -> str:
    """Return a real decodable JPEG while keeping inference stubbed in route tests."""
    image = np.full((8, 8, 3), pixel, dtype=np.uint8)
    encoded, payload = cv2.imencode(".jpg", image)
    assert encoded
    return "data:image/jpeg;base64," + base64.b64encode(payload.tobytes()).decode("ascii")


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

    def __init__(self, reject_at: int | None = None):
        self.reject_at = reject_at
        self.calls = 0

    def embed_capture(self, image_bgr):
        self.calls += 1
        assert image_bgr.shape == (8, 8, 3)
        if self.calls == self.reject_at:
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
    monkeypatch.setattr(pages, "Enroller", lambda: extractor)
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
    extractor = _StubCaptureEnroller(reject_at=2)
    monkeypatch.setattr(pages, "Enroller", lambda: extractor)
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


def test_native_enrollment_no_js_submission_redirects_to_v3_detail(monkeypatch, tmp_path):
    """A regular HTML form post must redirect to a rendered detail page, never raw JSON."""
    session_factory = _isolated_enrollment_database(monkeypatch, tmp_path)
    monkeypatch.setattr(pages, "Enroller", _StubCaptureEnroller)
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


def test_v3_app_does_not_mount_the_legacy_employee_router():
    """The v3 app must not expose the metadata-only legacy database boundary."""
    from fast_api.routers import employees as legacy_employee_api

    assert not any(
        getattr(route, "original_router", None) is legacy_employee_api.router
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
        monkeypatch.setattr(pages.runtime, "workers", {
            online_id: SimpleNamespace(camera_id=online_id, stats=lambda: {
                "stream": {"connected": True, "stale": False, "resolution": "1920x1080", "fps": 12.5},
                "processed_fps": 9.5, "latency_ms": 44.0, "dropped_frames": 3,
                "pipeline_errors": 1, "last_error": "Dekoder kechikmoqda",
            }),
        })
        response = _camera_settings_response(monkeypatch, session)

    html = response.body.decode()
    for label in ("Sozlangan", "Kuzatilgan", "Pipeline", "Drift ogohlantirishi"):
        assert label in html
    assert "NVR encoder sozlamalariga egalik qiladi" in html
    assert "1920x1080" in html
    assert "12.5" in html
    assert "9.5" in html
    assert "44" in html
    assert "3" in html
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
        worker = SimpleNamespace(
            camera_id=entrance_id,
            stats=lambda: {
                "camera_id": entrance_id,
                "name": "Asosiy kirish",
                "role": "IN",
                "stream": {
                    "connected": True, "stale": False, "fps": 12.5,
                    "frames": 120, "resolution": "1920x1080",
                },
                "frames_processed": 100,
                "processed_fps": 25.0,
                "latency_ms": 40.0,
                "dropped_frames": 20,
                "last_frame_time": datetime(2026, 8, 25, 3, 14, tzinfo=timezone.utc),
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
    for label in ("Kamera FPS", "Algoritm FPS", "Kechikish", "Tushib qolgan kadrlar", "Oxirgi kadr"):
        assert label in html
    assert "1920x1080" in html
    assert "12.5" in html
    assert "25" in html
    assert "20" in html
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
