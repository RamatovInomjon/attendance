"""
CRUD operations for all models.
Async SQLAlchemy operations for FastAPI.
"""
from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from datetime import datetime, date, timedelta
from typing import Optional, List, Tuple
import numpy as np

from fast_api.models import (
    Employee, EmployeeImage, AttendanceRecord, DailyAttendance,
    UnknownFaceAttempt, CameraConfig, User
)


# ============= EMPLOYEES =============

async def get_employees(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
    is_active: Optional[bool] = None,
    department: Optional[str] = None,
    search: Optional[str] = None,
) -> List[Employee]:
    """Get list of employees with optional filters."""
    query = select(Employee)
    
    if is_active is not None:
        query = query.where(Employee.is_active == is_active)
    if department:
        query = query.where(Employee.department == department)
    if search:
        query = query.where(
            or_(
                Employee.full_name.ilike(f"%{search}%"),
                Employee.employee_id.ilike(f"%{search}%"),
            )
        )
    
    query = query.order_by(Employee.full_name).offset(skip).limit(limit)
    result = await db.execute(query)
    return result.scalars().all()


async def get_employee(db: AsyncSession, employee_id: int) -> Optional[Employee]:
    """Get employee by ID."""
    result = await db.execute(select(Employee).where(Employee.id == employee_id))
    return result.scalar_one_or_none()


async def get_employee_by_employee_id(db: AsyncSession, employee_id: str) -> Optional[Employee]:
    """Get employee by employee_id (EMP-XXXXXX)."""
    result = await db.execute(select(Employee).where(Employee.employee_id == employee_id))
    return result.scalar_one_or_none()


async def create_employee(
    db: AsyncSession,
    full_name: str,
    position: str,
    department: str,
    phone_number: str,
    email: Optional[str] = None,
    notes: Optional[str] = None,
    image: Optional[str] = None,
    face_embeddings: Optional[bytes] = None,
) -> Employee:
    """Create new employee."""
    employee = Employee(
        employee_id=Employee.generate_employee_id(),
        full_name=full_name,
        position=position,
        department=department,
        phone_number=phone_number,
        email=email,
        notes=notes,
        image=image,
        face_embeddings=face_embeddings,
    )
    db.add(employee)
    await db.commit()
    await db.refresh(employee)
    return employee


async def update_employee(
    db: AsyncSession,
    employee_id: int,
    **kwargs
) -> Optional[Employee]:
    """Update employee fields."""
    employee = await get_employee(db, employee_id)
    if not employee:
        return None
    
    for key, value in kwargs.items():
        if hasattr(employee, key):
            setattr(employee, key, value)
    
    employee.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(employee)
    return employee


async def delete_employee(db: AsyncSession, employee_id: int) -> bool:
    """Delete employee."""
    employee = await get_employee(db, employee_id)
    if not employee:
        return False
    
    await db.delete(employee)
    await db.commit()
    return True


async def get_all_embeddings(db: AsyncSession) -> List[Tuple[int, bytes]]:
    """Get all employee embeddings for face recognition."""
    query = select(Employee.id, Employee.face_embeddings).where(
        and_(
            Employee.is_active == True,
            Employee.face_embeddings.isnot(None),
        )
    )
    result = await db.execute(query)
    return result.all()


# ============= ATTENDANCE =============

async def get_attendance_records(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
    employee_id: Optional[int] = None,
    action_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> List[AttendanceRecord]:
    """Get attendance records with optional filters."""
    query = select(AttendanceRecord).options(selectinload(AttendanceRecord.employee))
    
    if employee_id:
        query = query.where(AttendanceRecord.employee_id == employee_id)
    if action_type:
        query = query.where(AttendanceRecord.action_type == action_type)
    if start_date:
        query = query.where(AttendanceRecord.timestamp >= start_date)
    if end_date:
        query = query.where(AttendanceRecord.timestamp <= end_date)
    
    query = query.order_by(desc(AttendanceRecord.timestamp)).offset(skip).limit(limit)
    result = await db.execute(query)
    return result.scalars().all()


async def create_attendance_record(
    db: AsyncSession,
    employee_id: Optional[int] = None,
    employee_name: str = "",
    employee_department: str = "",
    action_type: str = "IN",
    snapshot: Optional[str] = None,
    confidence: float = 0.0,
    notes: str = "",
) -> AttendanceRecord:
    """Create attendance record."""
    record = AttendanceRecord(
        employee_id=employee_id,
        employee_name=employee_name,
        employee_department=employee_department,
        action_type=action_type,
        snapshot=snapshot,
        confidence=confidence,
        notes=notes,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record


async def get_daily_attendance(
    db: AsyncSession,
    target_date: Optional[date] = None,
    employee_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
) -> List[DailyAttendance]:
    """Get daily attendance records."""
    query = select(DailyAttendance).options(selectinload(DailyAttendance.employee))
    
    if target_date:
        query = query.where(DailyAttendance.date == target_date)
    if employee_id:
        query = query.where(DailyAttendance.employee_id == employee_id)
    
    query = query.order_by(desc(DailyAttendance.date)).offset(skip).limit(limit)
    result = await db.execute(query)
    return result.scalars().all()


async def get_or_create_daily_attendance(
    db: AsyncSession,
    employee_id: int,
    target_date: date,
) -> Tuple[DailyAttendance, bool]:
    """Get or create daily attendance record."""
    query = select(DailyAttendance).where(
        and_(
            DailyAttendance.employee_id == employee_id,
            DailyAttendance.date == target_date,
        )
    )
    result = await db.execute(query)
    daily = result.scalar_one_or_none()
    
    if daily:
        return daily, False
    
    daily = DailyAttendance(
        employee_id=employee_id,
        date=target_date,
        status="ABS",
    )
    db.add(daily)
    await db.commit()
    await db.refresh(daily)
    return daily, True


async def update_daily_attendance(
    db: AsyncSession,
    daily_id: int,
    **kwargs
) -> Optional[DailyAttendance]:
    """Update daily attendance record."""
    query = select(DailyAttendance).where(DailyAttendance.id == daily_id)
    result = await db.execute(query)
    daily = result.scalar_one_or_none()
    
    if not daily:
        return None
    
    for key, value in kwargs.items():
        if hasattr(daily, key):
            setattr(daily, key, value)
    
    daily.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(daily)
    return daily


# ============= CAMERA =============

async def get_cameras(db: AsyncSession) -> List[CameraConfig]:
    """Get all camera configurations."""
    query = select(CameraConfig).order_by(desc(CameraConfig.updated_at))
    result = await db.execute(query)
    return result.scalars().all()


async def get_camera(db: AsyncSession, camera_id: int) -> Optional[CameraConfig]:
    """Get camera by ID."""
    result = await db.execute(select(CameraConfig).where(CameraConfig.id == camera_id))
    return result.scalar_one_or_none()


async def create_camera(
    db: AsyncSession,
    camera_type: str = "laptop",
    device_id: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    rtsp_url: Optional[str] = None,
    **kwargs
) -> CameraConfig:
    """Create camera configuration."""
    camera = CameraConfig(
        camera_type=camera_type,
        device_id=device_id,
        width=width,
        height=height,
        fps=fps,
        rtsp_url=rtsp_url,
        **kwargs
    )
    db.add(camera)
    await db.commit()
    await db.refresh(camera)
    return camera


async def update_camera(
    db: AsyncSession,
    camera_id: int,
    **kwargs
) -> Optional[CameraConfig]:
    """Update camera configuration."""
    camera = await get_camera(db, camera_id)
    if not camera:
        return None
    
    for key, value in kwargs.items():
        if hasattr(camera, key):
            setattr(camera, key, value)
    
    camera.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(camera)
    return camera


async def delete_camera(db: AsyncSession, camera_id: int) -> bool:
    """Delete camera configuration."""
    camera = await get_camera(db, camera_id)
    if not camera:
        return False
    
    await db.delete(camera)
    await db.commit()
    return True


# ============= UNKNOWN FACES =============

async def find_similar_unknown_face(
    db: AsyncSession,
    embedding: np.ndarray,
    threshold: float = 0.6,
    camera_id: str = "",
) -> Optional[UnknownFaceAttempt]:
    """Find similar unknown face by embedding similarity."""
    query = select(UnknownFaceAttempt)
    if camera_id:
        query = query.where(UnknownFaceAttempt.camera_id == camera_id)
    
    result = await db.execute(query.order_by(desc(UnknownFaceAttempt.last_seen)))
    attempts = result.scalars().all()
    
    for attempt in attempts:
        stored_emb = np.array(attempt.embedding)
        similarity = np.dot(embedding, stored_emb) / (
            np.linalg.norm(embedding) * np.linalg.norm(stored_emb)
        )
        if similarity >= threshold:
            return attempt
    
    return None


async def create_unknown_face_attempt(
    db: AsyncSession,
    embedding: list,
    camera_id: str = "",
    latest_record_id: Optional[int] = None,
) -> UnknownFaceAttempt:
    """Create unknown face attempt record."""
    attempt = UnknownFaceAttempt(
        embedding=embedding,
        camera_id=camera_id,
        latest_record_id=latest_record_id,
    )
    db.add(attempt)
    await db.commit()
    await db.refresh(attempt)
    return attempt


async def update_unknown_face_attempt(
    db: AsyncSession,
    attempt_id: int,
    **kwargs
) -> Optional[UnknownFaceAttempt]:
    """Update unknown face attempt."""
    query = select(UnknownFaceAttempt).where(UnknownFaceAttempt.id == attempt_id)
    result = await db.execute(query)
    attempt = result.scalar_one_or_none()
    
    if not attempt:
        return None
    
    for key, value in kwargs.items():
        if hasattr(attempt, key):
            setattr(attempt, key, value)
    
    attempt.attempt_count += 1
    await db.commit()
    await db.refresh(attempt)
    return attempt


# ============= STATISTICS =============

async def get_attendance_stats(
    db: AsyncSession,
    target_date: Optional[date] = None,
) -> dict:
    """Get attendance statistics for a date."""
    if target_date is None:
        target_date = date.today()
    
    # Total employees
    total_query = select(func.count(Employee.id)).where(Employee.is_active == True)
    total_result = await db.execute(total_query)
    total_employees = total_result.scalar()
    
    # Present employees
    present_query = select(func.count(DailyAttendance.id)).where(
        and_(
            DailyAttendance.date == target_date,
            DailyAttendance.check_in_time.isnot(None),
        )
    )
    present_result = await db.execute(present_query)
    present_count = present_result.scalar()
    
    # Late employees
    late_query = select(func.count(DailyAttendance.id)).where(
        and_(
            DailyAttendance.date == target_date,
            DailyAttendance.status.contains("LATE"),
        )
    )
    late_result = await db.execute(late_query)
    late_count = late_result.scalar()
    
    return {
        "date": target_date.isoformat(),
        "total_employees": total_employees,
        "present": present_count,
        "absent": total_employees - present_count,
        "late": late_count,
    }
