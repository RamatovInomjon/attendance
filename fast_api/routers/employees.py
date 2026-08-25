"""
Employee management API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import os
import shutil

from fast_api.database import get_db
from fast_api import crud
from fast_api.schemas import (
    EmployeeCreate, EmployeeUpdate, EmployeeResponse, EmployeeListResponse
)

router = APIRouter(prefix="/api/employees", tags=["employees"])

MEDIA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "media")


@router.get("/", response_model=List[EmployeeListResponse])
async def list_employees(
    skip: int = 0,
    limit: int = 100,
    is_active: Optional[bool] = None,
    department: Optional[str] = None,
    search: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Get list of employees with optional filters."""
    employees = await crud.get_employees(
        db, skip=skip, limit=limit,
        is_active=is_active, department=department, search=search
    )
    return employees


@router.get("/{employee_id}", response_model=EmployeeResponse)
async def get_employee(
    employee_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get employee by ID."""
    employee = await crud.get_employee(db, employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    return employee


@router.post("/", response_model=EmployeeResponse)
async def create_employee(
    full_name: str = Form(...),
    position: str = Form(...),
    department: str = Form(...),
    phone_number: str = Form(...),
    email: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db),
):
    """Create new employee."""
    image_path = None
    if image:
        # Save uploaded image
        os.makedirs(os.path.join(MEDIA_ROOT, "employees/profile"), exist_ok=True)
        image_path = f"employees/profile/{image.filename}"
        with open(os.path.join(MEDIA_ROOT, image_path), "wb") as f:
            shutil.copyfileobj(image.file, f)
    
    employee = await crud.create_employee(
        db,
        full_name=full_name,
        position=position,
        department=department,
        phone_number=phone_number,
        email=email,
        notes=notes,
        image=image_path,
    )
    return employee


@router.put("/{employee_id}", response_model=EmployeeResponse)
async def update_employee(
    employee_id: int,
    full_name: Optional[str] = Form(None),
    position: Optional[str] = Form(None),
    department: Optional[str] = Form(None),
    phone_number: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    is_active: Optional[bool] = Form(None),
    image: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db),
):
    """Update employee."""
    update_data = {}
    if full_name is not None:
        update_data["full_name"] = full_name
    if position is not None:
        update_data["position"] = position
    if department is not None:
        update_data["department"] = department
    if phone_number is not None:
        update_data["phone_number"] = phone_number
    if email is not None:
        update_data["email"] = email
    if notes is not None:
        update_data["notes"] = notes
    if is_active is not None:
        update_data["is_active"] = is_active
    
    if image:
        os.makedirs(os.path.join(MEDIA_ROOT, "employees/profile"), exist_ok=True)
        image_path = f"employees/profile/{image.filename}"
        with open(os.path.join(MEDIA_ROOT, image_path), "wb") as f:
            shutil.copyfileobj(image.file, f)
        update_data["image"] = image_path
    
    employee = await crud.update_employee(db, employee_id, **update_data)
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    return employee


@router.delete("/{employee_id}")
async def delete_employee(
    employee_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Delete employee."""
    success = await crud.delete_employee(db, employee_id)
    if not success:
        raise HTTPException(status_code=404, detail="Employee not found")
    return {"message": "Employee deleted successfully"}
