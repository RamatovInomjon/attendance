"""Jinja2 environment that renders the project's original Django templates.

The UI in `templates/` was written for Django.  Rather than rewrite 6,000 lines
of working, well-designed Bootstrap markup, this supplies the handful of Django
constructs it depends on: the `{% load %}` tag, `{% url %}`, and the filters the
templates actually use (`date`, `default`, `duration_hm`, `floatformat`,
`json_script`, `pluralize`, `escapejs`, `yesno`, `add_class`, `media_url`,
`clean_phone`, `length`).

Ported from the previous `fast_api/routers/pages.py`, with the URL map pointed
at the v3 routes and the filters made tolerant of the types the new view models
hand them.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, nodes
from jinja2.ext import Extension
from markupsafe import Markup

from app.config import settings

TEMPLATES_DIR = settings.root / "templates"

# Django |date / |time format strings -> strftime
DJANGO_FORMATS = {
    "H:i": "%H:%M", "H:i:s": "%H:%M:%S", "Y-m-d": "%Y-%m-%d",
    "M d, Y": "%b %d, %Y", "d/m/Y": "%d/%m/%Y", "F j, Y": "%B %d, %Y",
    "d M Y": "%d %b %Y", "M j": "%b %j", "N j, Y": "%b %d, %Y",
    "D, d M Y": "%a, %d %b %Y", "j F Y": "%d %B %Y",
}

URL_MAP = {
    "dashboard:index": "/", "dashboard:home": "/",
    "employees:list": "/employees", "employees:add": "/employees/add",
    "employees:register": "/employees/add", "employees:detail": "/employees/{pk}",
    "employees:edit": "/employees/{pk}/edit", "employees:delete": "/employees/{pk}/delete",
    "recognition:live": "/recognition", "recognition:register": "/employees/add",
    "recognition:logs": "/recognition/logs", "recognition:stream": "/recognition/stream",
    "camera:settings": "/cameras", "camera:add": "/cameras/add",
    "camera:rtsp_register": "/cameras/rtsp",
    "attendance:list": "/attendance", "attendance:history": "/attendance/history",
    "attendance:unknown_attempts": "/attendance/unknown",
    "attendance:unknown": "/attendance/unknown",
    "attendance:employee_detail": "/attendance/employee/{pk}",
    "attendance:export": "/api/attendance/export",
    "auth:login": "/login", "auth:logout": "/logout", "auth:register": "/register",
}


class LoadExtension(Extension):
    """`{% load ... %}` is a Django no-op here."""
    tags = {"load"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        while parser.stream.current.test("name"):
            next(parser.stream)
        return nodes.Output([nodes.Const("")]).set_lineno(lineno)


def _fmt(value, spec, fallback):
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    py = DJANGO_FORMATS.get(spec, spec)
    try:
        return value.strftime(py if "%" in py else fallback)
    except Exception:
        return str(value)


def date_filter(value, spec="Y-m-d"):
    return _fmt(value, spec, "%Y-%m-%d")


def time_filter(value, spec="H:i"):
    return _fmt(value, spec, "%H:%M")


def duration_hm(value):
    """Render a timedelta (or seconds) as HH:MM."""
    if value is None:
        return ""
    try:
        secs = int(value.total_seconds()) if isinstance(value, timedelta) else int(value)
    except Exception:
        return str(value)
    return f"{secs // 3600:02d}:{(secs % 3600) // 60:02d}"


def default_filter(value, fallback=""):
    return fallback if value is None or value == "" or value == [] else value


def floatformat(value, arg=2):
    try:
        return f"{float(value):.{int(arg)}f}"
    except Exception:
        return value


def pluralize(value, arg="s"):
    try:
        n = int(value if not hasattr(value, "__len__") else len(value))
    except Exception:
        return ""
    if n == 1:
        return ""
    return arg.split(",", 1)[1] if "," in arg else arg


def yesno(value, arg="yes,no,maybe"):
    parts = arg.split(",")
    if value:
        return parts[0]
    if value is None and len(parts) > 2:
        return parts[2]
    return parts[1] if len(parts) > 1 else "no"


def escapejs(value):
    return "" if value is None else json.dumps(str(value))[1:-1]


def json_script(value, element_id=""):
    return Markup(f'<script id="{element_id}" type="application/json">'
                  f'{json.dumps(value, default=str)}</script>')


def media_url(value):
    if not value:
        return "/static/img/avatar-placeholder.png"
    if str(value).startswith(("http://", "https://", "/media/", "/static/", "data:")):
        return value
    return f"/media/{value}"


def clean_phone(value):
    return re.sub(r"[^\d+]", "", str(value)) if value else ""


def get_item(d, key):
    if d is None:
        return None
    return d.get(key) if hasattr(d, "get") else getattr(d, key, None)


def add_class(field, css):
    """Templates use this on Django form fields; the v3 pages pass plain values."""
    return field.as_widget(attrs={"class": css}) if hasattr(field, "as_widget") else field


def url(name, *args, **kwargs):
    path = URL_MAP.get(name, f"/{name}")
    pk = kwargs.get("pk") or kwargs.get("id") or (args[0] if args else None)
    if pk is not None:
        path = path.replace("{pk}", str(pk))
    return path


class _User:
    """The templates check `user.is_authenticated`; auth is not built yet."""
    is_authenticated = True
    is_staff = True
    is_superuser = True
    username = "admin"


def build_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        extensions=[LoadExtension],
        autoescape=True,
    )
    env.filters.update(
        date=date_filter, time=time_filter, duration_hm=duration_hm,
        default=default_filter, floatformat=floatformat, pluralize=pluralize,
        yesno=yesno, escapejs=escapejs, json_script=json_script,
        media_url=media_url, clean_phone=clean_phone, get_item=get_item,
        add_class=add_class, length=lambda v: len(v) if hasattr(v, "__len__") else 0,
        safe=lambda v: Markup(v), add=lambda v, a: (int(v) + int(a)) if str(v).lstrip("-").isdigit() else v,
    )
    env.globals.update(
        url=url,
        now=lambda: datetime.now(settings.tz),
        today=lambda: datetime.now(settings.tz).date(),
        user=_User(),
        DEBUG=False,
        STATIC_VERSION="3.0.0",
        current_view="",
    )
    return env
