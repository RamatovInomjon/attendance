"""
Attendance API endpoints.
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from datetime import datetime, date

from fast_api.database import get_db
from fast_api import crud
from fast_api.schemas import AttendanceRecordResponse, DailyAttendanceResponse, AttendanceStatsResponse

router = APIRouter(prefix="/api/attendance", tags=["attendance"])


@router.get("/", response_model=List[AttendanceRecordResponse])
async def list_attendance(
    skip: int = 0,
    limit: int = 100,
    employee_id: Optional[int] = None,
    action_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: AsyncSession = Depends(get_db),
):
    """Get attendance records with optional filters."""
    records = await crud.get_attendance_records(
        db, skip=skip, limit=limit,
        employee_id=employee_id, action_type=action_type,
        start_date=start_date, end_date=end_date
    )
    return records


@router.get("/daily", response_model=List[DailyAttendanceResponse])
async def list_daily_attendance(
    target_date: Optional[date] = Query(None, description="Filter by date (YYYY-MM-DD)"),
    employee_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    """Get daily attendance summaries."""
    records = await crud.get_daily_attendance(
        db, target_date=target_date, employee_id=employee_id,
        skip=skip, limit=limit
    )
    return records


@router.get("/stats", response_model=AttendanceStatsResponse)
async def get_attendance_stats(
    target_date: Optional[date] = Query(None, description="Stats for date (YYYY-MM-DD)"),
    db: AsyncSession = Depends(get_db),
):
    """Get attendance statistics for a specific date."""
    stats = await crud.get_attendance_stats(db, target_date=target_date)
    return stats


@router.get("/today")
async def get_today_attendance(
    db: AsyncSession = Depends(get_db),
):
    """Get today's attendance records."""
    today = date.today()
    start = datetime.combine(today, datetime.min.time())
    end = datetime.combine(today, datetime.max.time())
    
    records = await crud.get_attendance_records(
        db, start_date=start, end_date=end, limit=1000
    )
    return records
