"""
FastAPI Application - Face Recognition Attendance System
Migrated from Django for improved performance.
"""
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import cv2
import time
import os
from typing import List
from fastapi import UploadFile, File, Form

# Local imports
from fast_api.schemas import EmployeeRegisterRequest, EmployeeRegisterResponse
from fast_api import services
from fast_api.services import active_cameras
from fast_api.routers import employees, attendance, cameras, pages, websocket

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(PROJECT_ROOT, "staticfiles")
MEDIA_DIR = os.path.join(PROJECT_ROOT, "media")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown events."""
    # Startup
    print("🚀 Initializing Face Recognition Service...")
    
    # Initialize cameras from database
    print("📹 Loading cameras from database...")
    await services.initialize_cameras_from_db()
    
    # Warmup AI models
    print("🧠 Warming up YOLO model...")
    services.get_yolo_model()
    
    print("✅ Service ready!")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down cameras...")
    for cam_id, instance in active_cameras.items():
        instance.stop_event.set()
    print("👋 Goodbye!")


# Create FastAPI app
app = FastAPI(
    title="EmAtSy - Face Recognition Attendance System",
    description="High-performance face recognition attendance system powered by FastAPI",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if os.path.exists(MEDIA_DIR):
    app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")

# Include routers
app.include_router(pages.router)  # HTML pages (must be first for / route)
app.include_router(employees.router)  # /api/employees
app.include_router(attendance.router)  # /api/attendance
app.include_router(cameras.router)  # /api/cameras
app.include_router(websocket.router)  # WebSocket endpoints


# ============= VIDEO STREAMING =============

def generate_frames(camera_id: int):
    """Generator for MJPEG stream - uses display_queue for smooth output."""
    if camera_id not in active_cameras:
        return
        
    instance = active_cameras[camera_id]
    
    while not instance.stop_event.is_set():
        if not instance.display_queue.empty():
            frame = instance.display_queue.get()
            
            # Draw bounding boxes from active tracks
            for track_id, track in list(instance.active_tracks.items()):
                # Only draw if track was seen recently (within last 2 seconds)
                if time.time() - track.last_seen < 2.0 and track.bbox is not None:
                    x1, y1, x2, y2 = track.bbox
                    
                    # Color: Green for recognized, Red for unknown
                    color = (0, 255, 0) if track.employee_id else (0, 0, 255)
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with name and confidence
                    label = f"{track.name}"
                    if track.confidence > 0:
                        label += f" ({track.confidence*100:.1f}%)"
                    
                    # Background for text
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        frame, 
                        (x1, y1 - text_height - 10), 
                        (x1 + text_width, y1), 
                        color, 
                        -1
                    )
                    
                    # Draw text
                    cv2.putText(
                        frame, 
                        label, 
                        (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (255, 255, 255), 
                        2
                    )
            
            # Encode frame - High quality JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.01)


@app.get("/video_feed/{camera_id}")
async def video_feed(camera_id: int):
    """MJPEG Stream Endpoint for specific camera."""
    if camera_id not in active_cameras:
        raise HTTPException(status_code=404, detail="Camera not found or not active")
    return StreamingResponse(
        generate_frames(camera_id), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ============= REGISTRATION ENDPOINTS =============

@app.post("/api/v1/register", response_model=EmployeeRegisterResponse)
async def register_employee(data: EmployeeRegisterRequest, background_tasks: BackgroundTasks):
    """
    Start process to register a new employee.
    Captures frames from the active camera stream.
    """
    if data.camera_id not in active_cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
         
    instance = active_cameras[data.camera_id]
    
    if instance.capture_mode:
        raise HTTPException(status_code=409, detail="Registration already in progress on this camera")
    
    full_name = f"{data.first_name} {data.last_name}"
    
    # Reset capture state
    instance.captured_frames = []
    instance.capture_mode = True
    
    # Wait for frames
    start_time = time.time()
    while len(instance.captured_frames) < instance.capture_target:
        if time.time() - start_time > 30:  # 30s timeout
            instance.capture_mode = False
            raise HTTPException(status_code=408, detail="Timeout capturing faces")
        time.sleep(0.1)
    
    instance.capture_mode = False
    
    # Process and Save
    success = services.process_registration(full_name, data.department, instance.captured_frames)
    
    if success:
        return {
            "id": 0,
            "full_name": full_name,
            "message": "Successfully registered",
            "face_count": len(instance.captured_frames)
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to extract embeddings")


@app.post("/api/v1/register-batch")
async def register_batch(
    first_name: str = Form(...),
    last_name: str = Form(...),
    department: str = Form(...),
    employee_id: int = Form(None),
    files: List[UploadFile] = File(...)
):
    """Register employee using uploaded images."""
    full_name = f"{first_name} {last_name}"
    
    image_bytes_list = []
    for file in files:
        content = await file.read()
        image_bytes_list.append(content)
        
    success = services.process_registration_from_bytes(
        full_name, department, image_bytes_list, employee_id
    )
    
    if success:
        return {"message": "Successfully registered", "count": len(image_bytes_list)}
    else:
        raise HTTPException(status_code=400, detail="Could not extract valid face embeddings")


# ============= PERFORMANCE & HEALTH =============

@app.get("/api/v1/performance")
async def get_performance():
    """Get real-time performance metrics for all cameras."""
    metrics = {}
    for cam_id, instance in active_cameras.items():
        metrics[cam_id] = {
            "fps_display": instance.fps_display,
            "fps_process": instance.fps_process,
            "active_tracks": len(instance.active_tracks),
            "display_queue_size": instance.display_queue.qsize(),
            "process_queue_size": instance.process_queue.qsize(),
            "frame_skip_rate": instance.process_every_n_frames
        }
    return metrics


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_cameras": len(active_cameras),
        "version": "2.0.0"
    }
