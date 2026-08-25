"""
WebSocket routes for real-time camera streaming.
Replaces Django Channels functionality.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import cv2
import asyncio
import time
import base64

from fast_api.services import active_cameras

router = APIRouter(tags=["websocket"])


async def send_frames(websocket: WebSocket, camera_id: int):
    """Send frames via WebSocket."""
    if camera_id not in active_cameras:
        await websocket.close(code=4004, reason="Camera not found")
        return
    
    instance = active_cameras[camera_id]
    
    try:
        while not instance.stop_event.is_set():
            if not instance.display_queue.empty():
                frame = instance.display_queue.get()
                
                # Draw bounding boxes
                for track_id, track in list(instance.active_tracks.items()):
                    if time.time() - track.last_seen < 2.0 and track.bbox is not None:
                        x1, y1, x2, y2 = track.bbox
                        color = (0, 255, 0) if track.employee_id else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        label = f"{track.name}"
                        if track.confidence > 0:
                            label += f" ({track.confidence*100:.1f}%)"
                        
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Encode as JPEG and base64
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    await websocket.send_json({
                        "type": "frame",
                        "data": frame_base64,
                        "timestamp": time.time()
                    })
            
            await asyncio.sleep(0.033)  # ~30 FPS
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")


@router.websocket("/ws/camera/{camera_id}/")
async def websocket_camera(websocket: WebSocket, camera_id: str):
    """WebSocket endpoint for camera streaming."""
    await websocket.accept()
    
    # Map camera IDs
    if camera_id == "primary":
        cam_id = 1
    elif camera_id.isdigit():
        cam_id = int(camera_id)
    else:
        await websocket.close(code=4004, reason="Invalid camera ID")
        return
    
    if cam_id not in active_cameras:
        await websocket.close(code=4004, reason="Camera not active")
        return
    
    print(f"📡 WebSocket connected: Camera {cam_id}")
    
    try:
        await send_frames(websocket, cam_id)
    except WebSocketDisconnect:
        print(f"📴 WebSocket disconnected: Camera {cam_id}")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")


@router.websocket("/ws/camera/primary/")
async def websocket_primary(websocket: WebSocket):
    """WebSocket endpoint for primary camera (Camera 1)."""
    await websocket.accept()
    
    if 1 not in active_cameras:
        await websocket.close(code=4004, reason="Primary camera not active")
        return
    
    print("📡 WebSocket connected: Primary camera")
    
    try:
        await send_frames(websocket, 1)
    except WebSocketDisconnect:
        print("📴 WebSocket disconnected: Primary camera")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")


@router.websocket("/ws/attendance/")
async def websocket_attendance(websocket: WebSocket):
    """WebSocket for real-time attendance notifications."""
    await websocket.accept()
    
    print("📡 Attendance WebSocket connected")
    
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(30)
            await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        print("📴 Attendance WebSocket disconnected")
