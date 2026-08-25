"""
SQLAlchemy models matching Django models for data continuity.
Table names match Django's default naming convention (app_model).
"""
from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean, DateTime, Date,
    ForeignKey, LargeBinary, Index, Interval, JSON
)
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from fast_api.database import Base


# ============= EMPLOYEES =============

class Employee(Base):
    """Represents a single employee within the organization."""
    __tablename__ = "employees_employee"
    
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String(12), unique=True, nullable=False)
    full_name = Column(String(255), nullable=False)
    position = Column(String(255), nullable=False)
    department = Column(String(255), nullable=False)
    phone_number = Column(String(20), nullable=False)
    email = Column(String(254), nullable=True)
    notes = Column(Text, nullable=True)
    image = Column(String(100), nullable=True)  # Image file path
    face_embeddings = Column(LargeBinary, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    captured_images = relationship("EmployeeImage", back_populates="employee", cascade="all, delete-orphan")
    attendance_records = relationship("AttendanceRecord", back_populates="employee", cascade="all, delete-orphan")
    daily_attendance = relationship("DailyAttendance", back_populates="employee", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('ix_employees_active_created', 'is_active', 'created_at'),
        Index('ix_employees_department', 'department'),
        Index('ix_employees_full_name', 'full_name'),
    )
    
    @staticmethod
    def generate_employee_id() -> str:
        return f"EMP-{uuid.uuid4().hex[:6].upper()}"


class EmployeeImage(Base):
    """Stores all captured images for an employee during registration."""
    __tablename__ = "employees_employeeimage"
    
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees_employee.id", ondelete="CASCADE"), nullable=False)
    image = Column(String(100), nullable=False)  # Image file path
    capture_order = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    employee = relationship("Employee", back_populates="captured_images")


# ============= ATTENDANCE =============

class AttendanceRecord(Base):
    """Check-in/check-out events."""
    __tablename__ = "attendance_attendancerecord"
    
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees_employee.id", ondelete="CASCADE"), nullable=True)
    employee_name = Column(String(255), default="")
    employee_department = Column(String(255), default="")
    action_type = Column(String(3), default="IN")  # IN, OUT, UNK
    timestamp = Column(DateTime, default=datetime.utcnow)
    snapshot = Column(String(100), nullable=True)  # Image file path
    confidence = Column(Float, default=0.0)
    notes = Column(Text, default="")
    
    # Relationships
    employee = relationship("Employee", back_populates="attendance_records")
    unknown_face_attempts = relationship("UnknownFaceAttempt", back_populates="latest_record")
    
    __table_args__ = (
        Index('ix_attendance_timestamp', 'timestamp'),
        Index('ix_attendance_employee_timestamp', 'employee_id', 'timestamp'),
    )


class DailyAttendance(Base):
    """Daily attendance summary."""
    __tablename__ = "attendance_dailyattendance"
    
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees_employee.id", ondelete="CASCADE"), nullable=False)
    date = Column(Date, nullable=False)
    check_in_time = Column(DateTime, nullable=True)
    check_out_time = Column(DateTime, nullable=True)
    working_hours = Column(Interval, nullable=True)
    recognition_count = Column(Integer, default=0)
    last_seen_time = Column(DateTime, nullable=True)
    check_in_snapshot = Column(String(100), nullable=True)
    check_out_snapshot = Column(String(100), nullable=True)
    status = Column(String(100), default="ABS")  # NORMAL, LATE, EARLY_LEAVE, OVERTIME, PARTIAL_SHIFT, ABS
    late_reason = Column(String(255), default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    employee = relationship("Employee", back_populates="daily_attendance")
    
    __table_args__ = (
        Index('ix_daily_date_checkin', 'date', 'check_in_time'),
        Index('ix_daily_employee_date', 'employee_id', 'date'),
        Index('ix_daily_status_date', 'status', 'date'),
    )


class UnknownFaceAttempt(Base):
    """Track unique unknown faces with embeddings to avoid duplicate records."""
    __tablename__ = "attendance_unknownfaceattempt"
    
    id = Column(Integer, primary_key=True, index=True)
    embedding = Column(JSON, nullable=False)  # Face embedding vector (512-dim)
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    attempt_count = Column(Integer, default=1)
    camera_id = Column(String(100), default="")
    latest_record_id = Column(Integer, ForeignKey("attendance_attendancerecord.id", ondelete="SET NULL"), nullable=True)
    
    # Relationships
    latest_record = relationship("AttendanceRecord", back_populates="unknown_face_attempts")
    
    __table_args__ = (
        Index('ix_unknown_last_seen', 'last_seen'),
        Index('ix_unknown_camera_last_seen', 'camera_id', 'last_seen'),
    )


# ============= CAMERA =============

class CameraConfig(Base):
    """Camera configuration."""
    __tablename__ = "camera_cameraconfig"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_type = Column(String(20), default="laptop")  # laptop, usb, ip
    device_id = Column(Integer, default=0)
    width = Column(Integer, default=1280)
    height = Column(Integer, default=720)
    fps = Column(Integer, default=30)
    brightness = Column(Float, default=0.5)
    contrast = Column(Float, default=0.5)
    sharpness = Column(Float, default=0.5)
    rtsp_url = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self) -> dict:
        return {
            "camera_type": self.camera_type,
            "device_id": self.device_id,
            "resolution": {"width": self.width, "height": self.height},
            "fps": self.fps,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "sharpness": self.sharpness,
            "rtsp_url": self.rtsp_url,
        }


# ============= AUTH (Optional) =============

class User(Base):
    """User authentication model."""
    __tablename__ = "auth_app_customuser"
    
    id = Column(Integer, primary_key=True, index=True)
    password = Column(String(128), nullable=False)
    last_login = Column(DateTime, nullable=True)
    is_superuser = Column(Boolean, default=False)
    username = Column(String(150), unique=True, nullable=False)
    first_name = Column(String(150), default="")
    last_name = Column(String(150), default="")
    email = Column(String(254), default="")
    is_staff = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    date_joined = Column(DateTime, default=datetime.utcnow)
