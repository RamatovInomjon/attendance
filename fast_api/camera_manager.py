"""
IP Camera Manager for Employee Registration.
Migrated from employees/camera_manager.py
"""
import cv2
import threading
import time
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class IPCamera:
    """Thread-based IP camera handler for registration."""
    
    def __init__(self, url: str, timeout_ms: int = 5000):
        self.url = url
        self.timeout_ms = timeout_ms
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.stopped = True
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.last_frame_time = 0
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3
        
    def start(self) -> bool:
        """Start camera stream in background thread."""
        if not self.stopped:
            return self.connected
        
        logger.info(f"🎥 Starting IP camera: {self.url[:50]}...")
        self.stopped = False
        self.connected = False
        
        if not self._open_camera():
            logger.error(f"❌ Failed to open camera: {self.url[:50]}")
            self.stopped = True
            return False
        
        self.thread = threading.Thread(
            target=self._update, daemon=True, name="IPCamera-Thread"
        )
        self.thread.start()
        time.sleep(0.5)
        
        if self.connected:
            logger.info(f"✅ IP camera started: {self.url[:50]}")
        return self.connected
    
    def _open_camera(self) -> bool:
        """Open camera connection."""
        try:
            if self.url.startswith("rtsp://"):
                url_with_tcp = self.url
                if "rtsp_transport=tcp" not in self.url:
                    url_with_tcp = f"{self.url}?rtsp_transport=tcp"
                
                self.cap = cv2.VideoCapture(url_with_tcp, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.timeout_ms)
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.timeout_ms // 2)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                device_id = int(self.url) if self.url.isdigit() else 0
                self.cap = cv2.VideoCapture(device_id)
            
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                return True
            return False
        except Exception as e:
            logger.error(f"Error opening camera: {e}")
            return False
    
    def _update(self):
        """Background thread to read frames."""
        consecutive_failures = 0
        
        while not self.stopped:
            try:
                if self.cap is None or not self.cap.isOpened():
                    if self.reconnect_attempts < self.max_reconnect_attempts:
                        logger.info(f"🔄 Reconnecting camera...")
                        if self._open_camera():
                            self.reconnect_attempts = 0
                            consecutive_failures = 0
                        else:
                            self.reconnect_attempts += 1
                            time.sleep(1)
                            continue
                    else:
                        break
                
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    with self.lock:
                        self.frame = frame.copy()
                        self.last_frame_time = time.time()
                        if not self.connected:
                            self.connected = True
                            logger.info("✅ Camera connected")
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= 10:
                        if self.cap:
                            self.cap.release()
                            self.cap = None
                        self.connected = False
                        consecutive_failures = 0
                        time.sleep(0.5)
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error in camera thread: {e}")
                time.sleep(0.1)
        
        self._release()
    
    def read(self) -> Optional[np.ndarray]:
        """Get latest frame."""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def is_connected(self) -> bool:
        """Check if camera is connected."""
        return self.connected and not self.stopped
    
    def stop(self):
        """Stop camera."""
        if self.stopped:
            return
        
        logger.info("🛑 Stopping IP camera...")
        self.stopped = True
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        self._release()
    
    def _release(self):
        """Release camera resources."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            finally:
                self.cap = None
                self.frame = None
                self.connected = False


# Global camera instance (singleton)
_registration_camera: Optional[IPCamera] = None
_camera_lock = threading.Lock()


def get_registration_camera(url: Optional[str] = None) -> Optional[IPCamera]:
    """Get or create registration camera instance."""
    global _registration_camera
    
    with _camera_lock:
        if _registration_camera is not None and not _registration_camera.stopped:
            if url and _registration_camera.url != url:
                _registration_camera.stop()
                _registration_camera = None
            else:
                return _registration_camera
        
        if url:
            _registration_camera = IPCamera(url, timeout_ms=5000)
            if _registration_camera.start():
                return _registration_camera
            else:
                _registration_camera = None
    return None


def stop_registration_camera():
    """Stop registration camera."""
    global _registration_camera
    with _camera_lock:
        if _registration_camera is not None:
            _registration_camera.stop()
            _registration_camera = None
