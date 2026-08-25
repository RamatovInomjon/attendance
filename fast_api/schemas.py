"""
Pydantic schemas for API request/response models.
"""
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime, date, timedelta


# ============= EMPLOYEES =============

class EmployeeBase(BaseModel):
    full_name: str
    position: str
    department: str
    phone_number: str
    email: Optional[str] = None
    notes: Optional[str] = None


class EmployeeCreate(EmployeeBase):
    pass


class EmployeeUpdate(BaseModel):
    full_name: Optional[str] = None
    position: Optional[str] = None
    department: Optional[str] = None
    phone_number: Optional[str] = None
    email: Optional[str] = None
    notes: Optional[str] = None
    is_active: Optional[bool] = None


class EmployeeListResponse(BaseModel):
    id: int
    employee_id: str
    full_name: str
    department: str
    position: str
    is_active: bool
    image: Optional[str] = None
    
    class Config:
        from_attributes = True


class EmployeeResponse(EmployeeListResponse):
    phone_number: str
    email: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# ============= ATTENDANCE =============

class AttendanceRecordResponse(BaseModel):
    id: int
    employee_id: Optional[int] = None
    employee_name: str
    employee_department: str
    action_type: str
    timestamp: datetime
    snapshot: Optional[str] = None
    confidence: float
    notes: str
    
    class Config:
        from_attributes = True


class DailyAttendanceResponse(BaseModel):
    id: int
    employee_id: int
    date: date
    check_in_time: Optional[datetime] = None
    check_out_time: Optional[datetime] = None
    working_hours: Optional[timedelta] = None
    recognition_count: int
    status: str
    
    class Config:
        from_attributes = True


class AttendanceStatsResponse(BaseModel):
    date: str
    total_employees: int
    present: int
    absent: int
    late: int


# ============= CAMERA =============

class CameraConfigBase(BaseModel):
    camera_type: str = "laptop"
    device_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    brightness: float = 0.5
    contrast: float = 0.5
    sharpness: float = 0.5
    rtsp_url: Optional[str] = None


class CameraConfigCreate(CameraConfigBase):
    pass


class CameraConfigUpdate(BaseModel):
    camera_type: Optional[str] = None
    device_id: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[int] = None
    brightness: Optional[float] = None
    contrast: Optional[float] = None
    sharpness: Optional[float] = None
    rtsp_url: Optional[str] = None


class CameraConfigResponse(CameraConfigBase):
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# ============= REGISTRATION =============

class EmployeeRegisterRequest(BaseModel):
    first_name: str
    last_name: str
    department: str
    camera_id: int


class EmployeeRegisterResponse(BaseModel):
    id: int
    full_name: str
    message: str
    face_count: int


# ============= AUTH =============

class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
