"""Adapters from the v3 schema to the shape the original templates expect.

The Django templates were written against the old models and read
`record.date`, `record.working_hours` (a timedelta), `employee.phone_number`.
The v3 schema deliberately uses different names — `business_date`,
`worked_seconds`, `phone` — because they are more accurate.

Rather than rename the schema to suit the markup, or edit 6,000 lines of
markup to suit the schema, the translation lives here, at the presentation
boundary, where it belongs.  Times are converted to local (`Asia/Tashkent`)
here too, so templates never deal with timezones.
"""
from __future__ import annotations

from app.web.django_compat import media_path

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.config import settings


def _local(dt: datetime | None):
    return dt.astimezone(settings.tz) if dt else None


@dataclass
class EmployeeVM:
    """Matches `employee.*` in employees/list.html and detail.html."""
    id: int
    employee_id: str
    full_name: str
    department: str
    position: str
    phone_number: str          # schema calls it `phone`
    email: str
    image: str | None
    is_active: bool
    created_at: datetime | None
    notes: str = ""
    enrollment_count: int = 0
    enrollment_state: str = "not_enrolled"

    @classmethod
    def of(cls, e, enrollment_count: int = 0, image: str | None = None):
        # `image` is passed in rather than resolved here: finding it touches the
        # filesystem, and this classmethod is called once per row on the
        # employee list and the dashboard. Only the profile page, which renders
        # one person, pays for it.
        return cls(
            id=e.id, employee_id=e.external_id or "", full_name=e.full_name,
            department=e.department or "", position=e.position or "",
            phone_number=e.phone or "", email="", image=image,
            is_active=bool(e.is_active), created_at=_local(e.created_at),
            enrollment_count=enrollment_count,
            enrollment_state="enrolled" if enrollment_count else "not_enrolled",
        )


@dataclass
class DailyVM:
    """Matches `record.*` in dashboard/index.html and attendance/list.html."""
    id: int
    employee: EmployeeVM | None
    date: object                       # schema calls it `business_date`
    check_in_time: datetime | None
    check_out_time: datetime | None
    working_hours: timedelta | None    # schema stores `worked_seconds`
    status: str
    check_in_snapshot: str | None
    check_out_snapshot: str | None
    recognition_count: int = 0
    presence: str = "OUTSIDE"

    @classmethod
    def of(cls, d, emp=None):
        secs = d.worked_seconds or 0
        return cls(
            id=d.id, employee=EmployeeVM.of(emp) if emp else None,
            date=d.business_date,
            check_in_time=_local(d.check_in_time), check_out_time=_local(d.check_out_time),
            working_hours=timedelta(seconds=secs) if secs else None,
            status=d.status or "PRESENT",
            check_in_snapshot=media_path(d.check_in_snapshot) if d.check_in_snapshot else None,
            check_out_snapshot=media_path(d.check_out_snapshot) if d.check_out_snapshot else None,
            recognition_count=d.event_count or 0,
            presence=d.presence.value if hasattr(d.presence, "value") else str(d.presence),
        )

    # dashboard/index.html reads these on the "latest records" list
    @property
    def employee_name(self):
        return self.employee.full_name if self.employee else "Unknown"

    @property
    def employee_department(self):
        return self.employee.department if self.employee else ""

    @property
    def action_type(self):
        return "OUT" if self.check_out_time else "IN"


@dataclass
class EventVM:
    """Matches the recent-activity rows."""
    id: int
    employee_name: str
    employee_department: str
    action_type: str
    timestamp: datetime
    snapshot: str | None
    confidence: float
    camera: str
    transition: str
    # An admin said this is not that person. The row stays visible, struck
    # through, because hiding a correction makes the log agree with itself by
    # forgetting - and an operator who cannot see that a fix landed will apply
    # it again.
    voided: bool = False
    void_reason: str = ""

    @classmethod
    def of(cls, e, name, dept, cam):
        transition = e.transition or ""
        role = e.role.value if hasattr(e.role, "value") else str(e.role)
        # DUPLICATE_VIEW is the losing half of a cross-camera pair: both
        # cameras really saw the person, but only one may move attendance
        # state. Falling through to the camera ROLE labelled it "IN" or "OUT"
        # on the dashboard, which reads as a check-in that never happened.
        # Named for what it is instead.
        action_type = {
            "CHECK_IN": "IN",
            "CHECK_OUT": "OUT",
            "DUPLICATE_VIEW": "SEEN",
            "DEBOUNCED": "SEEN",
        }.get(transition, role)
        return cls(
            id=e.id, employee_name=name or "Unknown", employee_department=dept or "",
            action_type=action_type,
            timestamp=_local(e.ts),
            snapshot=media_path(e.snapshot) if e.snapshot else None,
            confidence=e.score or 0.0, camera=cam or "", transition=transition,
            voided=getattr(e, "voided_at", None) is not None,
            void_reason=getattr(e, "void_reason", "") or "",
        )


def live_event(ev: dict) -> dict:
    """A worker's recent-event dict, with its snapshot made into a URL.

    The worker records the stored path (`snapshots/x.jpg`). The attendance
    socket and /api/events used to hand that out raw, and the live page then
    built `/media/...` off the DOMAIN root - which under /faceid belongs to
    another project. Built here, once, for both consumers, through the same
    helper every rendered page uses.
    """
    out = dict(ev)
    if out.get("snapshot"):
        out["snapshot"] = media_path(out["snapshot"])
    return out


@dataclass
class Page:
    """Stand-in for Django's `page_obj`, which the list templates paginate on."""
    object_list: list = field(default_factory=list)
    number: int = 1
    num_pages: int = 1

    @property
    def has_previous(self): return self.number > 1

    @property
    def has_next(self): return self.number < self.num_pages

    @property
    def previous_page_number(self): return max(1, self.number - 1)

    @property
    def next_page_number(self): return min(self.num_pages, self.number + 1)

    @property
    def paginator(self): return self

    @property
    def count(self): return len(self.object_list)

    def __iter__(self): return iter(self.object_list)

    def __len__(self): return len(self.object_list)
