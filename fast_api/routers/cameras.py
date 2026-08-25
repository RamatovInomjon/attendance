"""
Camera management API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional

from fast_api.database import get_db
from fast_api import crud
from fast_api.schemas import CameraConfigCreate, CameraConfigUpdate, CameraConfigResponse

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


@router.get("/", response_model=List[CameraConfigResponse])
async def list_cameras(
    db: AsyncSession = Depends(get_db),
):
    """Get all camera configurations."""
    cameras = await crud.get_cameras(db)
    return cameras


@router.get("/{camera_id}", response_model=CameraConfigResponse)
async def get_camera(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get camera by ID."""
    camera = await crud.get_camera(db, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


@router.post("/", response_model=CameraConfigResponse)
async def create_camera(
    data: CameraConfigCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create new camera configuration."""
    camera = await crud.create_camera(
        db,
        camera_type=data.camera_type,
        device_id=data.device_id,
        width=data.width,
        height=data.height,
        fps=data.fps,
        rtsp_url=data.rtsp_url,
        brightness=data.brightness,
        contrast=data.contrast,
        sharpness=data.sharpness,
    )
    return camera


@router.put("/{camera_id}", response_model=CameraConfigResponse)
async def update_camera(
    camera_id: int,
    data: CameraConfigUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update camera configuration."""
    update_data = data.model_dump(exclude_unset=True)
    camera = await crud.update_camera(db, camera_id, **update_data)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


@router.delete("/{camera_id}")
async def delete_camera(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Delete camera configuration."""
    success = await crud.delete_camera(db, camera_id)
    if not success:
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"message": "Camera deleted successfully"}
