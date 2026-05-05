"""
ClarifEye System Configuration
All hardware pins, model paths, thresholds, and constants.
NEVER hardcode values elsewhere — always import from here.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
from enum import IntEnum, auto

# ─── BASE PATHS ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")    # Pre-recorded WAV files
VOICES_DIR = os.path.join(BASE_DIR, "voices")  # Piper TTS voice models
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")

# ─── GPIO PIN ASSIGNMENTS (BCM numbering) ───
# HC-SR04 Ultrasonic Sensor #1 (LEFT — mounted on left side of glasses)
ULTRASONIC_LEFT_TRIG = 17    # Physical Pin 11
ULTRASONIC_LEFT_ECHO = 27    # Physical Pin 13 (via 1kΩ/2kΩ voltage divider)

# HC-SR04 Ultrasonic Sensor #2 (RIGHT — mounted on right side of glasses)
ULTRASONIC_RIGHT_TRIG = 22   # Physical Pin 15
ULTRASONIC_RIGHT_ECHO = 23   # Physical Pin 16 (via 1kΩ/2kΩ voltage divider)

# VL53L0X ToF Sensor (CENTER — I2C bus 1)
TOF_I2C_BUS = 1
TOF_I2C_ADDRESS = 0x29       # Default VL53L0X address
TOF_TIMING_BUDGET_US = 33000 # 33ms measurement time (balance speed/accuracy)

# Push Buttons (with internal pull-up resistors enabled in software)
BUTTON_NEXT_MODE = 24        # Physical Pin 18 — connected to GND
BUTTON_LANGUAGE  = 25        # Physical Pin 22 — connected to GND
BUTTON_ACTION    = 5         # Physical Pin 29 — connected to GND
BUTTON_DEBOUNCE_MS = 300     # Debounce time in milliseconds
BUTTON_ACTION_LONG_PRESS_MS = 800  # Threshold (ms) for long vs short press on ACTION

# ─── CAMERA SETTINGS ───
CAMERA_RESOLUTION = (640, 640)     # Width x Height — square for YOLO input
CAMERA_FRAMERATE = 30              # Target FPS
CAMERA_FORMAT = "RGB888"           # 3-channel RGB
CAMERA_BUFFER_COUNT = 2            # Double buffering
FRAME_QUEUE_MAXSIZE = 2            # Drop old frames if processing is slow

# ─── AI MODEL SETTINGS ───
# Traffic Light Detection (custom-trained YOLOv8n, INT8 quantized)
TRAFFIC_LIGHT_MODEL_PATH = os.path.join(MODELS_DIR, "traffic_light_yolov8n.tflite")
TRAFFIC_LIGHT_CONFIDENCE_THRESHOLD = 0.55    # Minimum confidence to accept detection
TRAFFIC_LIGHT_IOU_THRESHOLD = 0.45           # Non-max suppression IoU
TRAFFIC_LIGHT_CLASSES = {0: "red", 1: "yellow", 2: "green"}  # Class ID → color name

# General Object Detection (YOLOv8n COCO INT8, 320×320 input)
# Run scripts/export_yolov8n_coco.py on a desktop to produce this file.
OBJECT_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n_coco_int8.tflite")
# Deprecated — superseded by OBJECT_CONFIDENCE_BY_CLASS below.
# Kept for backward compatibility but no longer used by the detector.
OBJECT_CONFIDENCE_THRESHOLD = 0.45
OBJECT_IOU_THRESHOLD = 0.45
# Mobility-focused whitelist of COCO class IDs.
OBJECT_CLASSES_OF_INTEREST = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    9:  "traffic_light",
    10: "fire_hydrant",
    11: "stop_sign",
    13: "bench",
}
# Per-class confidence thresholds (tuned for mobility relevance and false-positive cost).
OBJECT_CONFIDENCE_BY_CLASS: Dict[str, float] = {
    "person":        0.40,
    "bicycle":       0.45,
    "car":           0.40,
    "motorcycle":    0.45,
    "bus":           0.40,
    "truck":         0.40,
    "traffic_light": 0.55,
    "stop_sign":     0.55,
    "fire_hydrant":  0.60,
    "bench":         0.60,
}
OBJECT_CONFIDENCE_DEFAULT = 0.50  # fallback if class missing from above

# ─── HSV COLOR VERIFICATION RANGES ───
# (H_min, S_min, V_min, H_max, S_max, V_max)
HSV_RED_LOWER_1 = (0, 70, 50)
HSV_RED_UPPER_1 = (10, 255, 255)
HSV_RED_LOWER_2 = (170, 70, 50)     # Red wraps around in HSV
HSV_RED_UPPER_2 = (180, 255, 255)
HSV_YELLOW_LOWER = (15, 70, 50)
HSV_YELLOW_UPPER = (35, 255, 255)
HSV_GREEN_LOWER = (35, 70, 50)
HSV_GREEN_UPPER = (85, 255, 255)
COLOR_PIXEL_RATIO_THRESHOLD = 0.15   # Min ratio of color pixels to confirm

# ─── SENSOR FUSION SETTINGS ───
@dataclass
class KalmanConfig:
    process_noise: float = 0.1       # How much we expect position to change
    measurement_noise_camera: float = 0.5   # Camera distance is less precise
    measurement_noise_ultrasonic: float = 0.15  # Ultrasonic is quite precise
    measurement_noise_tof: float = 0.05     # ToF is most precise at close range
    initial_covariance: float = 1.0
    tof_priority_range_cm: float = 200.0    # ToF trusted below this distance
    ultrasonic_priority_range_cm: float = 300.0  # Ultrasonic trusted below this


KALMAN = KalmanConfig()

# ─── DISTANCE ESTIMATION FROM CAMERA ───
# Known real-world heights (meters) for distance calculation via pinhole model
KNOWN_HEIGHTS = {
    "person": 1.7,
    "car": 1.5,
    "bus": 3.0,
    "truck": 3.5,
    "bicycle": 1.0,
    "motorcycle": 1.1,
    "traffic_light": 0.3,  # Just the light head
}
CAMERA_FOCAL_LENGTH_PX = 500.0   # Approximate for Pi Camera Module 3 at 640px width
                                  # Calibrate this: f_px = (f_mm / sensor_width_mm) * image_width_px
                                  # Pi Camera v3: f=4.74mm, sensor=6.287mm → (4.74/6.287)*640 ≈ 482
                                  # Use 500 as safe starting point, refine with testing

# ─── PRIORITY ENGINE SETTINGS ───
class ThreatLevel(IntEnum):
    CRITICAL = 10     # Moving car/truck/bus approaching
    HIGH = 7          # Motorcycle/bicycle approaching
    MEDIUM = 5        # Person nearby
    LOW = 3           # Static obstacles (bench, hydrant, etc.)
    INFO = 1          # Non-threatening (traffic light color, text, etc.)


# Base threat scores by object class
THREAT_BASE_SCORES: Dict[str, int] = {
    "car": ThreatLevel.CRITICAL,
    "bus": ThreatLevel.CRITICAL,
    "truck": ThreatLevel.CRITICAL,
    "motorcycle": ThreatLevel.HIGH,
    "bicycle": ThreatLevel.HIGH,
    "person": ThreatLevel.MEDIUM,
    "fire_hydrant": ThreatLevel.LOW,
    "stop_sign": ThreatLevel.LOW,
    "bench": ThreatLevel.LOW,
    "chair": ThreatLevel.LOW,
    "traffic_light": ThreatLevel.INFO,
}

# Distance-based multipliers
DISTANCE_CRITICAL_CM = 100     # < 1m = maximum urgency
DISTANCE_CLOSE_CM = 200        # < 2m = high urgency
DISTANCE_MEDIUM_CM = 400       # < 4m = moderate urgency
MAX_NOTIFICATION_OBJECTS = 2   # Announce up to 2 objects per cycle
SECOND_ANNOUNCEMENT_MIN_SCORE = 5.0  # Second object must clear this threat score
NOTIFICATION_COOLDOWN_SEC = 6.0
# Tracker association: max pixel distance to match a detection to an existing tracker.
ASSOCIATION_MAX_PX = 120

# ─── AUDIO SETTINGS ───
TTS_ENGINE = "espeak-ng"          # Offline TTS engine
# Requires espeak-ng-data with Bulgarian support installed on the system.
# Proper Bulgarian TTS will be addressed in a later prompt (Piper TTS integration).
TTS_VOICE_BG = "bg"
TTS_VOICE_EN = "en"
TTS_SPEED = 155                   # Words per minute
AUDIO_PRIORITY_CRITICAL = 0       # Interrupts everything
AUDIO_PRIORITY_HIGH = 1
AUDIO_PRIORITY_NORMAL = 2
AUDIO_PRIORITY_LOW = 3

# ─── OCR SETTINGS ───
OCR_LANGUAGES = ["bg", "en"]      # Bulgarian + English
# EasyOCR confidence is better-calibrated than Tesseract's; 0.4 keeps noise out.
# EasyOCR downloads ~100MB of model weights on first run; ensure the Pi has
# internet for the initial run, then it works offline.
OCR_CONFIDENCE_THRESHOLD = 0.4    # Minimum OCR confidence
OCR_GPU = False                   # No GPU on Pi
TEXT_READING_MAX_BLOCKS = 3       # Max OCR result blocks spoken per action press

# ─── TRANSLATION / RAG SETTINGS ───
TRANSLATION_CACHE_PATH = os.path.join(DATA_DIR, "translation_cache.json")
COMMON_PHRASES_PATH = os.path.join(DATA_DIR, "common_phrases.json")
TRANSLATION_SOURCE_LANG = "en"
TRANSLATION_TARGET_LANG = "bg"

# ─── LOW-LIGHT ENHANCEMENT ───
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID_SIZE = (8, 8)
LOW_LIGHT_BRIGHTNESS_THRESHOLD = 60  # Mean brightness below this triggers enhancement

# ─── THREADING ───
THREAD_QUEUE_TIMEOUT_SEC = 0.05    # 50ms timeout for queue operations

# ─── OPERATING MODES ───
class Mode(IntEnum):
    TRAFFIC_LIGHT = 0
    NAVIGATION = 1
    TEXT_READING = 2
    CURRENCY = 3
    SCENE = 4


MODE_NAMES = {
    Mode.TRAFFIC_LIGHT: "Traffic light mode",
    Mode.NAVIGATION: "Navigation mode",
    Mode.TEXT_READING: "Text reading mode",
    Mode.CURRENCY: "Currency mode",
    Mode.SCENE: "Scene description mode",
}
MODE_NAMES_EN = {
    Mode.TRAFFIC_LIGHT: "Traffic light mode",
    Mode.NAVIGATION: "Navigation mode",
    Mode.TEXT_READING: "Text reading mode",
    Mode.CURRENCY: "Currency mode",
    Mode.SCENE: "Scene description mode",
}

DEFAULT_MODE = Mode.TRAFFIC_LIGHT
NUM_MODES = len(Mode)

# ─── CURRENCY RECOGNITION ───
CURRENCY_REFERENCES_DIR = os.path.join(DATA_DIR, "banknotes")
CURRENCY_MIN_MATCHES = 15       # Minimum ORB good-match count to accept a result
CURRENCY_RESIZE_WIDTH = 800     # Resize input frame long-edge before ORB matching

# ─── SCENE DESCRIPTION (SmolVLM-256M) ───
# SCENE_MODEL_PATH may be a directory (optimum-exported ONNX) or a single .onnx file.
# The download script (scripts/download_smolvlm.py) creates a directory at this path.
SCENE_MODEL_PATH = os.path.join(MODELS_DIR, "smolvlm_256m_int8.onnx")
SCENE_TOKENIZER_PATH = os.path.join(MODELS_DIR, "smolvlm_256m_tokenizer")
SCENE_MAX_WORDS = 25

# ─── LOGGING ───
LOG_FILE = os.path.join(LOGS_DIR, "clarifeye.log")
LOG_LEVEL = "INFO"   # DEBUG for development, INFO for production
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_MAX_BYTES = 10 * 1024 * 1024   # 10MB max log file
LOG_BACKUP_COUNT = 3
