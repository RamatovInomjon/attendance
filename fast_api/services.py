"""
Face Recognition Services - SQLAlchemy Version
Core AI logic with YOLO detection, InsightFace recognition, and async database operations.
"""
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import insightface
from insightface.app import FaceAnalysis
import os
import threading
import time
import queue
import asyncio
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import io

# SQLAlchemy imports
from sqlalchemy import select, and_
from sqlalchemy.orm import Session
from fast_api.database import async_session_maker
from fast_api.models import Employee, AttendanceRecord, DailyAttendance, CameraConfig, UnknownFaceAttempt

# Thread pool for async database operations
db_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="DB-Worker")

# Global Models
_yolo_model = None
_insightface_app = None

# ---- FIX: Disable CUDA stream capturing to avoid conflicts ----
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA_DIR = os.path.join(PROJECT_ROOT, "media")


# ============= 1. MODELS INITIALIZATION (GPU) =============

def get_yolo_model():
    """Lazy load YOLOv8-face model on GPU."""
    global _yolo_model
    if _yolo_model is None:
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'yolov8n-face.pt')
            _yolo_model = YOLO(model_path)
            if torch.cuda.is_available():
                _yolo_model.to('cuda')
                print("⚡ YOLOv8-face loaded on CUDA")
            else:
                print("⚠️ CUDA not available for YOLO!")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
    return _yolo_model


def get_insightface():
    """Lazy load InsightFace (Alignment + Embedding)."""
    global _insightface_app
    if _insightface_app is None:
        try:
            _insightface_app = FaceAnalysis(
                name='buffalo_l', 
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            _insightface_app.prepare(ctx_id=0, det_size=(640, 640))
            print("⚡ InsightFace loaded on CUDA")
        except Exception as e:
            print(f"❌ Error loading InsightFace: {e}")
    return _insightface_app


# ============= 2. TRACKING & STATE MANAGEMENT =============

class TrackState:
    """Stores state for a single Track ID (Person)."""
    def __init__(self, track_id):
        self.track_id = track_id
        self.employee_id: Optional[int] = None
        self.name: str = "Unknown"
        self.confidence: float = 0.0
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.recognition_attempted = False
        self.face_image = None
        
        # Re-recognition timing
        self.last_recognition_time = 0
        self.re_recognize_interval = 3600  # 1 hour
        self.frames_since_last_seen = 0
        
        # Bounding box for visualization
        self.bbox = None

    def update(self, bbox=None):
        self.last_seen = time.time()
        self.frames_since_last_seen = 0
        if bbox is not None:
            self.bbox = bbox
    
    def should_re_recognize(self):
        """Check if enough time has passed to re-recognize."""
        return (time.time() - self.last_recognition_time) > self.re_recognize_interval


# Global State
known_embeddings_cache = []
known_ids_cache = []


def refresh_known_faces_sync():
    """Refresh cache of known faces from DB (synchronous version for thread)."""
    global known_embeddings_cache, known_ids_cache
    
    # Run async function in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_refresh_known_faces_async())
    finally:
        loop.close()


async def _refresh_known_faces_async():
    """Async refresh of known faces from DB."""
    global known_embeddings_cache, known_ids_cache
    
    temp_emb = []
    temp_ids = []
    
    async with async_session_maker() as session:
        query = select(Employee.id, Employee.face_embeddings).where(
            and_(
                Employee.is_active == True,
                Employee.face_embeddings.isnot(None),
            )
        )
        result = await session.execute(query)
        rows = result.all()
        
        for emp_id, emb_bytes in rows:
            if emb_bytes:
                try:
                    emb = np.frombuffer(emb_bytes, dtype=np.float32)
                    temp_emb.append(emb)
                    temp_ids.append(emp_id)
                except:
                    pass
    
    known_embeddings_cache = temp_emb
    known_ids_cache = temp_ids
    print(f"🧠 Knowledge Base Updated: {len(known_embeddings_cache)} faces loaded.")


def refresh_known_faces():
    """Refresh cache of known faces from DB."""
    refresh_known_faces_sync()


def get_known_faces():
    """Return current known faces cache."""
    global known_embeddings_cache, known_ids_cache
    if not known_embeddings_cache:
        refresh_known_faces()
    return known_embeddings_cache, known_ids_cache


# ============= 3. CORE LOGIC: RECOGNITION & ATTENDANCE =============

def align_and_recognize(frame, box):
    """
    1. Crop face from frame using box.
    2. Pass to InsightFace for Alignment + Embedding.
    3. Match against DB.
    Returns: (employee_id, confidence, face_crop, embedding)
    """
    app = get_insightface()
    
    # Crop with padding
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    pad_w = int((x2-x1)*0.1)
    pad_h = int((y2-y1)*0.1)
    crop = frame[max(0, y1-pad_h):min(h, y2+pad_h), max(0, x1-pad_w):min(w, x2+pad_w)]
    
    if crop.size == 0:
        return None, 0.0, None, None
        
    # Get Embedding
    faces = app.get(crop)
    if not faces:
        return None, 0.0, None, None
    
    faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
    embedding = faces[0].embedding
    embedding = embedding / np.linalg.norm(embedding)
    
    # Match
    best_score = 0
    best_id = None
    
    global known_embeddings_cache, known_ids_cache
    if not known_embeddings_cache:
        refresh_known_faces()
        
    for idx, known_emb in enumerate(known_embeddings_cache):
        known_emb_norm = known_emb / np.linalg.norm(known_emb)
        score = np.dot(embedding, known_emb_norm)
        if score > best_score:
            best_score = score
            best_id = known_ids_cache[idx]
            
    return best_id, best_score, crop, embedding


def process_attendance_logic(employee_id, camera_id=None, snapshot=None):
    """
    Business Rules:
    - 1st Time: Check-In.
    - 5 mins later: Check-Out.
    - 1 hour later: Update previous Check-Out.
    """
    # Run async function in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            _process_attendance_async(employee_id, camera_id, snapshot)
        )
        return result
    finally:
        loop.close()


async def _process_attendance_async(employee_id, camera_id=None, snapshot=None):
    """Async attendance processing."""
    now = datetime.utcnow()
    today = now.date()
    
    async with async_session_maker() as session:
        try:
            # Get employee
            result = await session.execute(
                select(Employee).where(Employee.id == employee_id)
            )
            employee = result.scalar_one_or_none()
            if not employee:
                return "Unknown"
            
            # Get or create daily attendance
            result = await session.execute(
                select(DailyAttendance).where(
                    and_(
                        DailyAttendance.employee_id == employee_id,
                        DailyAttendance.date == today,
                    )
                )
            )
            daily = result.scalar_one_or_none()
            
            if not daily:
                daily = DailyAttendance(
                    employee_id=employee_id,
                    date=today,
                    status="ABS"
                )
                session.add(daily)
                await session.commit()
                await session.refresh(daily)
            
            # Rule 1: First Check-in
            if daily.check_in_time is None:
                daily.check_in_time = now
                daily.status = "NORMAL"
                daily.last_seen_time = now
                
                # Save snapshot
                if snapshot is not None:
                    snapshot_path = await _save_snapshot(snapshot, employee_id, "checkin")
                    daily.check_in_snapshot = snapshot_path
                
                await session.commit()
                
                # Create attendance record
                record = AttendanceRecord(
                    employee_id=employee_id,
                    employee_name=employee.full_name,
                    employee_department=employee.department,
                    action_type="IN",
                    confidence=1.0,
                )
                session.add(record)
                await session.commit()
                
                print(f"✅ CHECK-IN: {employee.full_name}")
                return employee.full_name
            
            # Rule 2 & 3: Updates
            daily.last_seen_time = now
            daily.recognition_count = (daily.recognition_count or 0) + 1
            
            # Get last attendance record
            result = await session.execute(
                select(AttendanceRecord)
                .where(AttendanceRecord.employee_id == employee_id)
                .order_by(AttendanceRecord.timestamp.desc())
                .limit(1)
            )
            last_log = result.scalar_one_or_none()
            
            if last_log:
                time_diff = (now - last_log.timestamp).total_seconds()
                
                # If > 1 hour, update check-out
                if time_diff > 3600:
                    if last_log.action_type == "OUT":
                        last_log.timestamp = now
                        last_log.notes = "Updated Check-out (1hr+)"
                        await session.commit()
                        print(f"🔄 UPDATED CHECK-OUT: {employee.full_name}")
                    else:
                        await _create_checkout(session, employee, now, snapshot)
                
                # If > 5 mins, create check-out
                elif time_diff > 300:
                    if last_log.action_type != "OUT":
                        await _create_checkout(session, employee, now, snapshot)
            
            await session.commit()
            return employee.full_name
            
        except Exception as e:
            print(f"⚠️ Logic Error: {e}")
            return "Unknown"


async def _create_checkout(session, employee, timestamp, snapshot):
    """Create checkout record."""
    snapshot_path = None
    if snapshot is not None:
        snapshot_path = await _save_snapshot(snapshot, employee.id, "checkout")
    
    record = AttendanceRecord(
        employee_id=employee.id,
        employee_name=employee.full_name,
        employee_department=employee.department,
        action_type="OUT",
        confidence=1.0,
        snapshot=snapshot_path,
    )
    session.add(record)
    await session.commit()
    print(f"🚪 CHECK-OUT: {employee.full_name}")


async def _save_snapshot(frame, employee_id, action_type):
    """Save snapshot to media directory."""
    try:
        os.makedirs(os.path.join(MEDIA_DIR, "attendance/snapshots"), exist_ok=True)
        filename = f"{action_type}_{employee_id}_{int(time.time())}.jpg"
        filepath = os.path.join(MEDIA_DIR, "attendance/snapshots", filename)
        ret, buf = cv2.imencode('.jpg', frame)
        if ret:
            with open(filepath, 'wb') as f:
                f.write(buf.tobytes())
            return f"attendance/snapshots/{filename}"
    except Exception as e:
        print(f"⚠️ Error saving snapshot: {e}")
    return None


def process_unknown_face(frame, embedding, camera_id):
    """Log unknown face detection to database."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            _process_unknown_face_async(frame, embedding, camera_id)
        )
    finally:
        loop.close()


async def _process_unknown_face_async(frame, embedding, camera_id):
    """Async unknown face processing."""
    try:
        if isinstance(embedding, np.ndarray):
            embedding_list = embedding.tolist()
        else:
            embedding_list = list(embedding)
        
        async with async_session_maker() as session:
            # Find similar unknown face
            result = await session.execute(
                select(UnknownFaceAttempt)
                .where(UnknownFaceAttempt.camera_id == str(camera_id))
                .order_by(UnknownFaceAttempt.last_seen.desc())
                .limit(50)
            )
            attempts = result.scalars().all()
            
            similar_attempt = None
            for attempt in attempts:
                stored_emb = np.array(attempt.embedding)
                query_emb = np.array(embedding_list)
                similarity = np.dot(query_emb, stored_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(stored_emb)
                )
                if similarity >= 0.6:
                    similar_attempt = attempt
                    break
            
            # Save snapshot
            snapshot_path = await _save_snapshot(frame, "unknown", "detection")
            
            # Create attendance record for unknown
            record = AttendanceRecord(
                employee_name="Unknown",
                action_type="UNK",
                snapshot=snapshot_path,
            )
            session.add(record)
            await session.commit()
            await session.refresh(record)
            
            if similar_attempt:
                similar_attempt.attempt_count += 1
                similar_attempt.latest_record_id = record.id
                await session.commit()
                print(f"📸 Unknown face updated: Attempt #{similar_attempt.attempt_count}")
            else:
                new_attempt = UnknownFaceAttempt(
                    embedding=embedding_list,
                    camera_id=str(camera_id),
                    latest_record_id=record.id,
                )
                session.add(new_attempt)
                await session.commit()
                print(f"📸 New unknown face logged")
                
    except Exception as e:
        import traceback
        print(f"⚠️ Error logging unknown face: {e}")
        traceback.print_exc()


# ============= 4. CAMERA & PROCESSING LOOP =============

class CameraInstance:
    def __init__(self, camera_id, source):
        self.camera_id = camera_id
        self.source = source
        # Dual-queue system for performance
        self.display_queue = queue.Queue(maxsize=2)
        self.process_queue = queue.Queue(maxsize=10)
        self.frame_queue = queue.Queue(maxsize=2)
        
        self.stop_event = threading.Event()
        self.capture_mode = False
        self.captured_frames = []
        self.capture_target = 24
        
        # Performance settings
        self.process_every_n_frames = 3
        self.frame_counter = 0
        
        # Performance metrics
        self.fps_display = 0
        self.fps_process = 0
        self.last_fps_update = time.time()
        self.display_frame_count = 0
        self.process_frame_count = 0
        
        # Active tracks for this camera
        self.active_tracks = {}


active_cameras: Dict[int, CameraInstance] = {}


def capture_thread(instance: CameraInstance):
    """Thread 1: Capture frames and feed display queue."""
    print(f"🎥 Starting Capture Thread for Camera {instance.camera_id}...")
    
    cap = None
    
    # Connect to camera
    if isinstance(instance.source, str) and instance.source.startswith("rtsp://"):
        # Try GStreamer first
        try:
            gst_pipeline = (
                f"rtspsrc location={instance.source} "
                "latency=100 timeout=10000000 tcp-timeout=10000000 "
                "protocols=tcp buffer-mode=auto drop-on-latency=true ! "
                "queue max-size-buffers=3 leaky=downstream ! "
                "rtph265depay ! h265parse ! avdec_h265 ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink drop=1 max-buffers=2"
            )
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"✅ Camera {instance.camera_id}: Using GStreamer")
                else:
                    cap.release()
                    cap = None
            else:
                cap = None
        except:
            cap = None
        
        # Fallback to OpenCV
        if cap is None:
            try:
                cap = cv2.VideoCapture(instance.source, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if cap.isOpened():
                    print(f"✅ Camera {instance.camera_id}: Using OpenCV FFMPEG")
                else:
                    cap = None
            except:
                cap = None
    else:
        cap = cv2.VideoCapture(instance.source)
        if isinstance(instance.source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if cap is None or not cap.isOpened():
        print(f"❌ Camera {instance.camera_id}: Failed to initialize")
        return
    
    last_good_frame = None
    consecutive_failures = 0
    
    while not instance.stop_event.is_set():
        try:
            ret, frame = cap.read()
        except:
            ret, frame = False, None
        
        if not ret or frame is None:
            consecutive_failures += 1
            if last_good_frame is not None:
                frame = last_good_frame
            else:
                time.sleep(0.05)
                if consecutive_failures >= 50:
                    print(f"⚠️ Camera {instance.camera_id}: Reconnecting...")
                    cap.release()
                    time.sleep(2.0)
                    cap = cv2.VideoCapture(instance.source)
                    consecutive_failures = 0
                continue
        else:
            consecutive_failures = 0
            last_good_frame = frame.copy()
        
        if frame is None:
            continue
        
        instance.frame_counter += 1
        
        # Registration Mode
        if instance.capture_mode:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur > 120:
                instance.captured_frames.append(frame.copy())
            if len(instance.captured_frames) >= instance.capture_target:
                instance.capture_mode = False
        
        # Feed queues
        if instance.display_queue.full():
            try:
                instance.display_queue.get_nowait()
            except:
                pass
        instance.display_queue.put(frame.copy())
        
        if instance.frame_counter % instance.process_every_n_frames == 0:
            if instance.process_queue.full():
                try:
                    instance.process_queue.get_nowait()
                except:
                    pass
            instance.process_queue.put(frame.copy())
        
        # Update FPS
        instance.display_frame_count += 1
        if time.time() - instance.last_fps_update > 1.0:
            instance.fps_display = instance.display_frame_count
            instance.display_frame_count = 0
            instance.last_fps_update = time.time()
    
    cap.release()
    print(f"🛑 Capture Thread stopped for Camera {instance.camera_id}")


def processing_thread(instance: CameraInstance):
    """Thread 2: Process frames with YOLO + Recognition."""
    print(f"🧠 Starting Processing Thread for Camera {instance.camera_id}...")
    
    yolo = get_yolo_model()
    refresh_known_faces()
    
    process_start_time = time.time()
    
    while not instance.stop_event.is_set():
        try:
            frame = instance.process_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            results = yolo.track(
                frame, persist=True, verbose=False,
                conf=0.4, iou=0.4, imgsz=640, device=0
            )
        except RuntimeError as e:
            if "CUDA" in str(e):
                torch.cuda.empty_cache()
            continue
        except:
            continue
        
        for r in results:
            boxes = r.boxes
            if boxes.id is not None:
                track_ids = boxes.id.cpu().numpy().astype(int)
                coords = boxes.xyxy.cpu().numpy().astype(int)
                
                for i, track_id in enumerate(track_ids):
                    box = coords[i]
                    
                    if track_id not in instance.active_tracks:
                        instance.active_tracks[track_id] = TrackState(track_id)
                    
                    track = instance.active_tracks[track_id]
                    track.update(bbox=box)
                    
                    should_recognize = (
                        not track.recognition_attempted or 
                        track.should_re_recognize()
                    )
                    
                    if should_recognize:
                        try:
                            emp_id, score, face_crop, embedding = align_and_recognize(frame, box)
                            track.recognition_attempted = True
                            track.last_recognition_time = time.time()
                            
                            if emp_id and score > 0.5:
                                track.employee_id = emp_id
                                track.confidence = score
                                track.face_image = face_crop
                                
                                db_executor.submit(
                                    process_attendance_logic,
                                    emp_id, instance.camera_id, face_crop
                                )
                                
                                # Get employee name
                                track.name = f"Employee #{emp_id}"
                                print(f"✅ KNOWN: {track.name} (score: {score:.3f})")
                            else:
                                track.name = "Unknown"
                                if face_crop is not None and embedding is not None:
                                    db_executor.submit(
                                        process_unknown_face,
                                        face_crop, embedding, instance.camera_id
                                    )
                        except Exception as e:
                            print(f"⚠️ Recognition error: {e}")
                            track.name = "Unknown"
        
        instance.process_frame_count += 1
        if time.time() - process_start_time > 1.0:
            instance.fps_process = instance.process_frame_count
            instance.process_frame_count = 0
            process_start_time = time.time()
    
    print(f"🛑 Processing Thread stopped for Camera {instance.camera_id}")


def start_stream_for_camera(instance: CameraInstance):
    """Start dual-thread system: capture + processing."""
    capture_t = threading.Thread(
        target=capture_thread, args=(instance,),
        daemon=True, name=f"Capture-{instance.camera_id}"
    )
    capture_t.start()
    
    process_t = threading.Thread(
        target=processing_thread, args=(instance,),
        daemon=True, name=f"Process-{instance.camera_id}"
    )
    process_t.start()
    
    print(f"✅ Dual-thread system started for Camera {instance.camera_id}")


# ============= 5. REGISTRATION HELPER =============

def process_registration(name: str, department: str, frames: List[np.ndarray], employee_id: int = None) -> bool:
    """Process face registration from captured frames."""
    app = get_insightface()
    embeddings = []
    
    for frame in frames:
        faces = app.get(frame)
        if faces:
            faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
            embeddings.append(faces[0].embedding)
            
    if not embeddings:
        return False
        
    mean_embedding = np.mean(embeddings, axis=0)
    mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
    
    # Run async update in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            _update_employee_embedding(employee_id, name, department, mean_embedding)
        )
        return result
    finally:
        loop.close()


async def _update_employee_embedding(employee_id, name, department, embedding):
    """Update employee with face embedding."""
    async with async_session_maker() as session:
        if employee_id:
            result = await session.execute(
                select(Employee).where(Employee.id == employee_id)
            )
            emp = result.scalar_one_or_none()
            if emp:
                emp.face_embeddings = embedding.tobytes()
                await session.commit()
                print(f"✅ Updated embeddings for Employee {employee_id}")
                refresh_known_faces()
                return True
            return False
        else:
            emp = Employee(
                employee_id=Employee.generate_employee_id(),
                full_name=name,
                department=department,
                position="Employee",
                phone_number="",
                face_embeddings=embedding.tobytes(),
            )
            session.add(emp)
            await session.commit()
            print(f"✅ Created new employee: {name}")
            refresh_known_faces()
            return True


def process_registration_from_bytes(name: str, department: str, image_bytes_list: List[bytes], employee_id: int = None) -> bool:
    """Process registration from uploaded images."""
    frames = []
    for img_bytes in image_bytes_list:
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None:
            frames.append(frame)
    
    return process_registration(name, department, frames, employee_id)


async def initialize_cameras_from_db():
    """Load cameras from DB and start streams."""
    # Clear existing cameras
    for inst in active_cameras.values():
        inst.stop_event.set()
    active_cameras.clear()
    
    # Fetch cameras
    async with async_session_maker() as session:
        result = await session.execute(select(CameraConfig))
        cameras = result.scalars().all()
        
        for cam in cameras:
            source = 0
            if cam.camera_type == "ip":
                source = cam.rtsp_url
            elif cam.camera_type == "usb":
                source = cam.device_id
            
            instance = CameraInstance(cam.id, source)
            t = threading.Thread(
                target=start_stream_for_camera,
                args=(instance,),
                daemon=True
            )
            t.start()
            active_cameras[cam.id] = instance
            print(f"✅ Initialized Camera {cam.id}")
