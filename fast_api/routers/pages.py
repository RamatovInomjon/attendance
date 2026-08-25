"""
Page routes for Jinja2 template rendering.
Serves HTML pages from existing Django templates.
"""
from fastapi import APIRouter, Request, Depends, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from datetime import date, datetime
import os
import re

from jinja2 import Environment, FileSystemLoader, pass_context
from jinja2.ext import Extension
from jinja2 import nodes

from fast_api.database import get_db
from fast_api import crud
from fast_api.services import active_cameras

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")


class DjangoCompatExtension(Extension):
    """Jinja2 extension to handle Django's {% load %} tag."""
    tags = {'load'}
    
    def parse(self, parser):
        # Skip the 'load' token
        lineno = next(parser.stream).lineno
        # Parse all arguments (the filter library names)
        while parser.stream.current.test('name'):
            next(parser.stream)
        # Return an empty output node (load does nothing in Jinja2)
        return nodes.Output([nodes.Const('')]).set_lineno(lineno)


def create_jinja2_env():
    """Create a Jinja2 environment with Django compatibility."""
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        extensions=[DjangoCompatExtension],
        autoescape=True,
    )
    
    # Django date format to Python strftime mapping
    DJANGO_DATE_FORMATS = {
        'H:i': '%H:%M',
        'H:i:s': '%H:%M:%S',
        'Y-m-d': '%Y-%m-%d',
        'M d, Y': '%b %d, %Y',
        'd/m/Y': '%d/%m/%Y',
        'F j, Y': '%B %d, %Y',
    }
    
    def date_filter(value, format_str="Y-m-d"):
        """Django's |date filter."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        # Convert Django format to strftime
        py_format = DJANGO_DATE_FORMATS.get(format_str, format_str)
        # Handle strftime formats directly
        if '%' in py_format:
            return value.strftime(py_format)
        return value.strftime('%Y-%m-%d')
    
    def time_filter(value, format_str="H:i"):
        """Django's |time filter."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        py_format = DJANGO_DATE_FORMATS.get(format_str, format_str)
        if '%' in py_format:
            return value.strftime(py_format)
        return value.strftime('%H:%M')
    
    def duration_hm(value):
        """Format timedelta as HH:MM."""
        if value is None:
            return ""
        try:
            total_seconds = int(value.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours:02d}:{minutes:02d}"
        except:
            return str(value)
    
    def default_filter(value, default_value=""):
        """Django's |default filter."""
        if value is None or value == "" or value == []:
            return default_value
        return value
    
    def get_item(dictionary, key):
        """Django's |get_item filter."""
        if dictionary is None:
            return None
        if hasattr(dictionary, 'get'):
            return dictionary.get(key)
        return getattr(dictionary, key, None)
    
    def clean_phone(value):
        """Clean phone number for tel: links."""
        if not value:
            return ""
        import re
        return re.sub(r'[^\d+]', '', str(value))
    
    def json_script(value, element_id=""):
        """Django json_script filter - outputs safe JSON in script tag."""
        import json
        safe_json = json.dumps(value)
        return f'<script id="{element_id}" type="application/json">{safe_json}</script>'
    
    def add_filter(value, arg):
        """Django add filter."""
        try:
            return int(value) + int(arg)
        except (ValueError, TypeError):
            return value
    
    def escapejs(value):
        """Escape string for JavaScript."""
        if value is None:
            return ""
        import json
        return json.dumps(str(value))[1:-1]  # Remove quotes
    
    def floatformat(value, arg=2):
        """Django floatformat filter."""
        try:
            return f"{float(value):.{int(arg)}f}"
        except:
            return value
    
    def yesno(value, arg="yes,no,maybe"):
        """Django yesno filter."""
        parts = arg.split(",")
        if value:
            return parts[0] if parts else "yes"
        elif value is None and len(parts) > 2:
            return parts[2]
        else:
            return parts[1] if len(parts) > 1 else "no"
    
    def pluralize(value, arg="s"):
        """Django pluralize filter."""
        try:
            count = int(value)
        except (ValueError, TypeError):
            return ""
        if count == 1:
            return ""
        if "," in arg:
            singular, plural = arg.split(",", 1)
            return plural
        return arg
    
    def length(value):
        """Get length of value."""
        try:
            return len(value)
        except:
            return 0
    
    # URL reverse function (maps Django URL names to FastAPI paths)
    URL_MAP = {
        'dashboard:index': '/',
        'dashboard:home': '/',
        'dashboard:charts': '/api/charts',
        'employees:list': '/employees',
        'employees:add': '/employees/add',
        'employees:detail': '/employees/{pk}',
        'employees:edit': '/employees/{pk}/edit',
        'employees:delete': '/employees/{pk}/delete',
        'employees:register': '/employees/add',
        'recognition:live': '/recognition',
        'recognition:register': '/recognition/register',
        'recognition:logs': '/recognition/logs',
        'recognition:stream': '/recognition/stream',
        'camera:settings': '/cameras',
        'camera:add': '/cameras/add',
        'camera:rtsp_register': '/cameras/rtsp',
        'attendance:list': '/attendance',
        'attendance:history': '/attendance/history',
        'attendance:unknown_attempts': '/attendance/unknown',
        'attendance:unknown': '/attendance/unknown',
        'attendance:employee_detail': '/attendance/employee/{pk}',
        'attendance:export': '/attendance/export',
        'auth:login': '/login',
        'auth:logout': '/logout',
        'auth:register': '/register',
    }
    
    def url(name, *args, **kwargs):
        """Django's {% url %} tag equivalent."""
        path = URL_MAP.get(name, f'/{name}')
        # Replace {pk} with actual pk if provided
        if 'pk' in kwargs:
            path = path.replace('{pk}', str(kwargs['pk']))
        elif args:
            path = path.replace('{pk}', str(args[0]))
        return path
    
    # Register filters
    env.filters['date'] = date_filter
    env.filters['time'] = time_filter
    env.filters['duration_hm'] = duration_hm
    env.filters['default'] = default_filter
    env.filters['get_item'] = get_item
    env.filters['clean_phone'] = clean_phone
    env.filters['json_script'] = json_script
    env.filters['add'] = add_filter
    env.filters['escapejs'] = escapejs
    env.filters['floatformat'] = floatformat
    env.filters['yesno'] = yesno
    env.filters['pluralize'] = pluralize
    env.filters['length'] = length
    
    # Django form-related filters
    def add_class(field, css_class):
        """Django add_class filter for adding CSS classes to form fields."""
        if hasattr(field, 'as_widget'):
            return field.as_widget(attrs={'class': css_class})
        # If it's just a string (value), return it unchanged
        return field
    
    def safe_filter(value):
        """Mark value as safe (no escaping)."""
        from markupsafe import Markup
        return Markup(value)
    
    def media_url(value):
        """Prepend /media/ to relative image paths."""
        if not value:
            return "/static/images/default-avatar.png"
        if value.startswith(('http://', 'https://', '/media/', '/static/')):
            return value
        return f"/media/{value}"
    
    env.filters['add_class'] = add_class
    env.filters['safe'] = safe_filter
    env.filters['media_url'] = media_url
    
    # Mock user object for Django compatibility
    class MockUser:
        is_authenticated = False
        username = "Guest"
        is_staff = False
        is_superuser = False
    
    # Register globals
    env.globals['url'] = url
    env.globals['now'] = lambda: datetime.now()
    env.globals['user'] = MockUser()
    env.globals['DEBUG'] = os.environ.get('DEBUG', 'false').lower() == 'true'
    env.globals['STATIC_VERSION'] = '2.0.0'
    env.globals['current_view'] = ''
    
    return env


# Create custom templates with Django compatibility
class DjangoCompatTemplates(Jinja2Templates):
    def __init__(self, directory: str):
        super().__init__(directory=directory)
        self.env = create_jinja2_env()


templates = DjangoCompatTemplates(directory=TEMPLATES_DIR)

router = APIRouter(tags=["pages"])


@router.get("/", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Dashboard page - main entry point."""
    stats = await crud.get_attendance_stats(db, target_date=date.today())
    employees = await crud.get_employees(db, limit=10, is_active=True)
    recent_attendance = await crud.get_attendance_records(db, limit=10)
    
    return templates.TemplateResponse("dashboard/index.html", {
        "request": request,
        "stats": stats,
        "employees": employees,
        "recent_attendance": recent_attendance,
        "active_cameras": len(active_cameras),
    })


@router.get("/employees", response_class=HTMLResponse)
async def employees_list(
    request: Request,
    query: Optional[str] = None,
    department: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Employees list page."""
    employees = await crud.get_employees(
        db, is_active=True, department=department, search=query
    )
    
    return templates.TemplateResponse("employees/list.html", {
        "request": request,
        "employees": employees,
        "query": query or "",
        "department": department or "",
        "is_paginated": False,
    })


@router.get("/employees/add", response_class=HTMLResponse)
async def employee_add_form(request: Request):
    """Employee add form page."""
    return templates.TemplateResponse("employees/register.html", {
        "request": request,
        "employee": None,
        "action": "add",
    })


@router.get("/employees/{employee_id}", response_class=HTMLResponse)
async def employee_detail(
    request: Request,
    employee_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Employee detail page."""
    employee = await crud.get_employee(db, employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    
    # Get attendance history
    attendance = await crud.get_daily_attendance(
        db, employee_id=employee_id, limit=30
    )
    
    return templates.TemplateResponse("employees/detail.html", {
        "request": request,
        "employee": employee,
        "attendance_history": attendance,
    })


@router.get("/employees/{employee_id}/edit", response_class=HTMLResponse)
async def employee_edit_form(
    request: Request,
    employee_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Employee edit form page."""
    employee = await crud.get_employee(db, employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    
    return templates.TemplateResponse("employees/edit.html", {
        "request": request,
        "employee": employee,
        "action": "edit",
    })


@router.get("/attendance", response_class=HTMLResponse)
async def attendance_list(
    request: Request,
    target_date: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Attendance list page."""
    if target_date:
        filter_date = datetime.strptime(target_date, "%Y-%m-%d").date()
    else:
        filter_date = date.today()
    
    daily_records = await crud.get_daily_attendance(db, target_date=filter_date)
    stats = await crud.get_attendance_stats(db, target_date=filter_date)
    
    return templates.TemplateResponse("attendance/list.html", {
        "request": request,
        "records": daily_records,
        "daily_records": daily_records,
        "stats": stats,
        "selected_date": filter_date,
        "start_date": "",
        "end_date": "",
        "department": "",
        "employee": "",
        "status": "",
        "is_paginated": False,
        "weekly_chart_data": [],
    })


@router.get("/attendance/history", response_class=HTMLResponse)
async def attendance_history(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Attendance history page."""
    records = await crud.get_attendance_records(db, limit=100)
    
    return templates.TemplateResponse("attendance/list.html", {
        "request": request,
        "records": records,
    })


@router.get("/cameras", response_class=HTMLResponse)
async def cameras_list(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Cameras list page."""
    cameras = await crud.get_cameras(db)
    
    # Get first camera for settings form
    camera = cameras[0] if cameras else None
    
    return templates.TemplateResponse("camera/settings.html", {
        "request": request,
        "cameras": cameras,
        "camera": camera,
        "active_cameras": active_cameras,
    })


@router.get("/cameras/add", response_class=HTMLResponse)
async def camera_add_form(request: Request):
    """Camera add form page."""
    return templates.TemplateResponse("camera/rtsp_register.html", {
        "request": request,
        "camera": None,
        "action": "add",
    })


@router.get("/recognition", response_class=HTMLResponse)
async def recognition_page(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Face recognition streaming page."""
    cameras = await crud.get_cameras(db)
    stats = await crud.get_attendance_stats(db, target_date=date.today())
    today_records = await crud.get_daily_attendance(db, target_date=date.today())
    
    # Get unknown attempts from today
    unknown_attempts = []  # Would need crud function for this
    
    # Camera profile defaults
    camera_profile = {
        "device": "Primary Camera",
        "resolution": "1920x1080",
        "fps": 30,
        "type": "RTSP/USB"
    }
    
    # Get primary camera
    primary_camera_id = 1
    secondary_streams = []
    
    if cameras:
        primary_camera_id = cameras[0].id if cameras else 1
        # Use correct CameraConfig attributes
        secondary_streams = [
            {
                "id": cam.id,
                "label": cam.camera_type or f"Camera {cam.id}",
                "recognition_enabled": True,
                "resolution": f"{cam.width}x{cam.height}",
                "fps": cam.fps
            }
            for cam in cameras[1:]
        ]
        if cameras:
            camera_profile["device"] = cameras[0].camera_type or "Primary Camera"
            camera_profile["resolution"] = f"{cameras[0].width}x{cameras[0].height}"
            camera_profile["fps"] = cameras[0].fps
    
    # Merge CRUD stats with template-required defaults
    template_stats = {
        "recognized_today": 0,
        "unknown_attempts": 0,
        "total_employees": 0,
        "checked_in_today": 0,
        "checked_out_today": 0,
        "last_event": None,
    }
    if stats:
        template_stats.update({
            "recognized_today": stats.get("present", 0),
            "total_employees": stats.get("total_employees", 0),
            "checked_in_today": stats.get("present", 0),
        })
    
    return templates.TemplateResponse("recognition/live.html", {
        "request": request,
        "cameras": cameras,
        "active_cameras": active_cameras,
        "stats": template_stats,
        "today_records": today_records or [],
        "unknown_attempts": unknown_attempts,
        "camera_profile": camera_profile,
        "primary_camera_id": primary_camera_id,
        "secondary_streams": secondary_streams,
        "attendance_summary": f"{len(today_records or [])} employees checked in today",
        "employee_logs_serialized": [],
        "unknown_attempts_serialized": [],
    })


@router.get("/recognition/register", response_class=HTMLResponse)
async def registration_page(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Face registration page."""
    cameras = await crud.get_cameras(db)
    
    # Check if there's an IP camera
    use_ip_camera = any(c.camera_type == "ip" for c in cameras) if cameras else False
    
    return templates.TemplateResponse("employees/register.html", {
        "request": request,
        "cameras": cameras,
        "active_cameras": active_cameras,
        "use_ip_camera": use_ip_camera,
        "employee": None,
        "available_cameras": [
            {"url": c.rtsp_url, "label": c.camera_type or f"Camera {c.id}"}
            for c in cameras if c.rtsp_url
        ],
        "default_camera_url": cameras[0].rtsp_url if cameras and cameras[0].rtsp_url else "",
    })


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page."""
    return templates.TemplateResponse("auth/login.html", {
        "request": request,
    })


@router.get("/recognition/logs")
async def recognition_logs(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """API endpoint for recognition logs (used by JS for refreshing feed)."""
    from datetime import date
    
    stats = await crud.get_attendance_stats(db, target_date=date.today())
    today_records = await crud.get_daily_attendance(db, target_date=date.today())
    
    # Format employees for JSON response
    employees = []
    for record in (today_records or []):
        employees.append({
            "employee_id": record.employee.employee_id if record.employee else "",
            "name": record.employee.full_name if record.employee else "",
            "department": record.employee.department if record.employee else "",
            "check_in": record.check_in_time.isoformat() if record.check_in_time else None,
            "check_out": record.check_out_time.isoformat() if record.check_out_time else None,
            "check_in_snapshot": f"/media/{record.check_in_snapshot}" if record.check_in_snapshot else None,
            "check_out_snapshot": f"/media/{record.check_out_snapshot}" if record.check_out_snapshot else None,
            "status": "checked_out" if record.check_out_time else ("checked_in" if record.check_in_time else "waiting"),
            "working_hours_seconds": record.working_hours if record.working_hours else 0,
            "recognition_count": record.recognition_count if hasattr(record, 'recognition_count') else 0,
        })
    
    return {
        "employees": employees,
        "unknown_attempts": [],
        "stats": {
            "recognized_today": stats.get("present", 0) if stats else 0,
            "total_employees": stats.get("total_employees", 0) if stats else 0,
            "checked_in_today": stats.get("present", 0) if stats else 0,
            "summary": f"{len(employees)} employees checked in today",
        }
    }


@router.get("/recognition/stream", response_class=HTMLResponse)
async def recognition_stream(request: Request):
    """Pop-out camera stream page."""
    return templates.TemplateResponse("recognition/stream.html", {
        "request": request,
    })


@router.get("/cameras/rtsp", response_class=HTMLResponse)
async def cameras_rtsp(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """RTSP camera registration page."""
    cameras = await crud.get_cameras(db)
    
    return templates.TemplateResponse("camera/rtsp_register.html", {
        "request": request,
        "cameras": cameras,
        "camera": None,
    })


@router.get("/attendance/export")
async def attendance_export(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Export attendance records as CSV."""
    from fastapi.responses import StreamingResponse
    from datetime import date
    import io
    import csv
    
    records = await crud.get_daily_attendance(db, target_date=date.today())
    
    # Create CSV
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Employee ID", "Name", "Department", "Check In", "Check Out", "Working Hours", "Date"])
    
    for record in (records or []):
        writer.writerow([
            record.employee.employee_id if record.employee else "",
            record.employee.full_name if record.employee else "",
            record.employee.department if record.employee else "",
            record.check_in_time.strftime("%H:%M") if record.check_in_time else "",
            record.check_out_time.strftime("%H:%M") if record.check_out_time else "",
            f"{(record.working_hours or 0) / 3600:.1f}h",
            record.date.strftime("%Y-%m-%d") if record.date else "",
        ])
    
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=attendance_{date.today()}.csv"}
    )


@router.get("/attendance/unknown", response_class=HTMLResponse)
async def attendance_unknown(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Unknown face attempts page."""
    # Would need crud function to get unknown attempts
    unknown_attempts = []
    
    return templates.TemplateResponse("attendance/unknown.html", {
        "request": request,
        "unknown_attempts": unknown_attempts,
    })
