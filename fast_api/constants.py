"""
Constants for face recognition and image processing.
Migrated from recognition/constants.py
"""

# JPEG Quality Settings
STREAM_JPEG_QUALITY = 50  # Lower quality for live stream (faster)
SNAPSHOT_JPEG_QUALITY = 85  # Higher quality for snapshots
REGISTRATION_JPEG_QUALITY = 90  # Highest quality for registration

# Face Recognition Settings
RECOGNITION_TOLERANCE_DEFAULT = 0.5  # Default tolerance for face matching
RECOGNITION_THRESHOLD = 0.62  # Optimal threshold for face matching (0.58-0.65)
UNKNOWN_FACE_SIMILARITY_THRESHOLD = 0.65  # Similarity threshold for grouping unknown faces

# Registration Quality Settings
MIN_FACE_SIZE_FOR_REGISTRATION = 90  # Minimum face size in pixels
MIN_EMBEDDING_NORM = 0.3  # Minimum embedding norm for quality
MAX_FACES_PER_IMAGE = 1  # Only 1 face allowed per registration image

# Image Processing
MIN_FACE_DETECTION_SIZE = 20  # Minimum face size in pixels
MAX_FACE_DETECTION_SIZE = 1000  # Maximum face size in pixels

# Distant Face Detection Settings
DISTANT_FACE_DETECTION_ENABLED = True
DISTANT_FACE_DETECTION_SIZE = 800  # Optimal balance (640-1280px)
DISTANT_FACE_DETECTION_THRESHOLD = 0.2  # Optimal threshold (0.15-0.3)
DISTANT_FACE_MIN_SIZE = 20  # Optimal minimum size (15-30px)

# SCRFD Detector Settings
DETECTOR = "scrfd"  # Options: scrfd (recommended), yolo, retinaface
SCRFD_MODEL = "buffalo_l"  # InsightFace model
SCRFD_DET_SIZE = (640, 640)  # Detection input size
SCRFD_CONF_THRESHOLD = 0.5  # Detection confidence threshold
SCRFD_MIN_FACE_SIZE = 80  # Minimum face size in pixels

# Registration Settings
MIN_REQUIRED_CAPTURES = 15  # Minimum images for registration
MIN_VALID_EMBEDDINGS = 10  # Minimum valid embeddings for training
CAPTURES_PER_ANGLE = 5  # Captures per angle step
TOTAL_CAPTURES = 20  # Total captures for multi-angle registration

# Snapshot Settings
UNKNOWN_FACE_SNAPSHOT_INTERVAL = 3  # Save snapshot every N attempts
SNAPSHOT_RESOLUTION_WIDTH = 1280
SNAPSHOT_RESOLUTION_HEIGHT = 720

# Performance Settings
FRAME_PROCESSING_TIMEOUT = 5.0  # Max seconds for frame processing
MAX_FRAMES_PER_SECOND = 30  # Maximum FPS
CHANNEL_CAPACITY = 1000  # Redis channel capacity

# Work Hours Settings
WORK_START_HOUR = 9  # 09:00
WORK_END_HOUR = 18  # 18:00
FULL_SHIFT_HOURS = 9  # 9 hours for full shift
