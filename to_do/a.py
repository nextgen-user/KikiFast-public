import sys
import os
import time
import math
import faulthandler
import multiprocessing
import queue
import threading
import cv2
import numpy as np
import setproctitle
import json
import subprocess
import base64
import shutil
import signal
PET_MODE_ARG = True  # Set by --pet argument at startup
ZMQ_MOTOR_PORT = 5557   # REP socket — chassis motors + ultrasonic, owned here

# Import GPIO and motor control for full body movement mode
try:
    import gpiod
    from gpiod.line import Direction, Value
    import motor_control as mc
    MOTOR_CONTROL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Motor control not available: {e}")
    MOTOR_CONTROL_AVAILABLE = False

# Add paths to ensure we can import hailo_apps and local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../hailo-apps')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Neck rotation is now handled by a stepper motor driven directly via GPIO
# (no Arduino / webcam_move needed)

# Import ZeroMQ for 2-way communication
try:
    import zmq
    import zmq.asyncio
except ImportError:
    print("Error: pyzmq not installed. Run: pip install pyzmq")
    sys.exit(1)
try:
    import pet_mode as pet_mode_module
    PET_MODE_AVAILABLE = False
except ImportError:
    print("Warning: pet_mode.py not found. Pet mode disabled.")
    PET_MODE_AVAILABLE = False

# Hand landmark + gesture recognition (shares the device in-process; see the
# module docstring for why it cannot be a GStreamer stage).
try:
    from hand_gestures import HandState, HandGestureWorker, draw_hands
    HAND_GESTURES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: hand_gestures.py not available: {e}")
    HAND_GESTURES_AVAILABLE = False

# Posture / gesture / fall recognition over the pose keypoints. Runs entirely on
# its own worker thread off a ring buffer, so it cannot slow the pose branch.
try:
    from pose_recognition import (PoseRecognizer, PoseRecognitionState,
                                  draw_pose_labels)
    POSE_RECOGNITION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: pose_recognition.py not available: {e}")
    POSE_RECOGNITION_AVAILABLE = False

# Imports from hailo_apps
try:
    from hailo_apps.python.pipeline_apps.face_recognition.face_recognition_pipeline import GStreamerFaceRecognitionApp
    from hailo_apps.python.pipeline_apps.detection_simple.detection_simple_pipeline import GStreamerDetectionSimpleApp
    from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
    from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
        QUEUE, OVERLAY_PIPELINE, DISPLAY_PIPELINE,
        INFERENCE_PIPELINE, INFERENCE_PIPELINE_WRAPPER, TRACKER_PIPELINE,
        CROPPER_PIPELINE
    )
    from hailo_apps.python.core.common.core import (
        get_pipeline_parser, get_resource_path, resolve_hef_path
    )
    import hailo
    from hailo_apps.python.core.common.hailo_logger import get_logger
    from hailo_apps.python.core.common.buffer_utils import (
        get_caps_from_pad,
        get_numpy_from_buffer,
    )
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
except ImportError as e:
    print(f"Error importing hailo apps: {e}")
    sys.exit(1)

hailo_logger = get_logger(__name__)

# --- CONFIGURATION ---
DEFAULT_TARGET_PERSON = "Alex"  # Default person to follow

FRAME_WIDTH = 640
CENTER_X = FRAME_WIDTH // 2

# Dead Zone (Hysteresis) to prevent jitter when the person is roughly centered
DEAD_ZONE = 60 

# Webcam Step Configuration
STEP_SIZE = 50

# Neck Stepper Motor Configuration (GPIO-driven, matches test.py wiring)
NECK_STEPPER_CHIP = "/dev/gpiochip4"
NECK_STEPPER_PIN_1 = 21
NECK_STEPPER_PIN_2 = 16
NECK_STEPPER_PIN_3 = 12
NECK_STEPPER_PIN_4 = 26
NECK_STEPPER_LINE_OFFSETS = [NECK_STEPPER_PIN_1, NECK_STEPPER_PIN_3,
                             NECK_STEPPER_PIN_2, NECK_STEPPER_PIN_4]  # [21, 12, 16, 26]
NECK_STEPPER_STEP_SEQUENCE = [
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 1],
    [1, 0, 0, 1],
]
NECK_STEPPER_STEP_DELAY = 0.01   # Seconds between steps (controls speed)
NECK_STEPPER_STEPS_PER_CMD = 50  # Number of steps to take per TURN command

# MJPEG Stream Configuration
MJPEG_PORT = 5000
# Frames are served from a separate stdlib HTTP server because Flask's dev
# server cannot do keep-alive (see frame_server_thread).
FRAME_PORT = 5001
# Streaming the full 1280x720 at quality 85 is ~100 KB/frame; at 30 fps that is
# ~3 MB/s per viewer, which a Pi over WiFi cannot sustain. Once the client stops
# draining, frames queue in the socket and latency grows without bound — that is
# the "delayed then hangs" behaviour. Downscaling for the stream only (the
# display window and recognition still use full resolution) keeps it real-time.
MJPEG_STREAM_WIDTH = 640    # 0 = don't downscale
MJPEG_QUALITY = 80
MJPEG_MAX_FPS = 20          # cap encode rate; the pipeline still runs at 30

# Standing latency inside GStreamer. Each INFERENCE_PIPELINE_WRAPPER/CROPPER
# keeps a "bypass" queue holding the original frame while its branch runs
# inference, defaulting to 20 buffers. Once the pipeline is running at the
# source rate every queue sits permanently full, so those buffers become fixed
# delay, not headroom: measured 21 of 22 queues saturated, ~84 frames on the
# critical path ~= 2.8 s at 30 fps. Throughput still reads 30 fps the whole
# time, which is why the fps display looks fine while the picture is stale.
# Shrinking the bypass queues trades jitter tolerance for latency.
# Raise this if you see frames dropping or the aggregator stalling.
PIPELINE_BYPASS_BUFFERS = 5

# Drop stale frames at the source instead of queueing them. Every queue in the
# stock pipeline is leaky=no, i.e. a strict FIFO that preserves every frame at
# the cost of latency — so at steady state they sit permanently full and become
# pure delay. For a robot reacting to what it sees now, an old frame has no
# value: better to discard it and work on the newest one. Leaking at the source
# caps how far behind the whole pipeline can ever get.
# NOTE: only the source is leaked. The INFERENCE_PIPELINE_WRAPPER/CROPPER bypass
# queues must NOT leak — hailoaggregator pairs each bypass buffer with its
# inference result, and dropping one desynchronises metadata from frames.
SOURCE_LEAKY = True
SOURCE_QUEUE_BUFFERS = 2

# ZeroMQ Configuration
ZMQ_CMD_PORT = 5555   # REP socket for receiving commands
ZMQ_EVENT_PORT = 5556  # PUB socket for publishing events

# Training Configuration
TRAIN_DIR = os.path.expanduser("~/Kiki/hailo-apps/hailo_apps/python/pipeline_apps/face_recognition/train")
TRAIN_PHOTO_COUNT = 7
TRAIN_PHOTO_INTERVAL = 1.0  # seconds between photos

# Face-recognition app dir + the standalone DB admin helper (rename/forget). We
# shell out to it (like training) so this server never imports LanceDB.
FACE_RECON_DIR = os.path.expanduser("~/Kiki/hailo-apps/hailo_apps/python/pipeline_apps/face_recognition")
DB_ADMIN_SCRIPT = os.path.join(FACE_RECON_DIR, "db_admin.py")
TRAINING_SCRIPT = os.path.join(FACE_RECON_DIR, "face_recognition.py")

# --- AUTO-ENROLL UNKNOWN FACES ---
# When a single, clear unknown face lingers in front of the camera we give it a
# temporary name, capture clear images, train in the background, and publish an
# `unknown_face_enrolled` event (with a face crop) so Kiki can describe/remember
# them and later ask their real name. Kept intentionally conservative: ONE clear
# face only, so we never mix two strangers' images. All heavy work runs in a
# daemon thread — app_callback only flips lightweight flags.
#
# All tunables below can be overridden WITHOUT editing code by dropping a
# `face_server_config.json` next to this file, e.g.:
#   {"auto_enroll_linger_seconds": 3, "face_confirm_seconds": 1.5}
def _load_face_server_cfg():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_server_config.json")
    try:
        with open(path) as f:
            cfg = json.load(f)
        print(f"[FaceCfg] Loaded overrides from {path}: {cfg}")
        return cfg
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"[FaceCfg] Failed to load {path}: {e}")
        return {}

_FACE_CFG = _load_face_server_cfg()
AUTO_ENROLL_ENABLED = _FACE_CFG.get("auto_enroll_enabled", True)
AUTO_ENROLL_LINGER_SECONDS = _FACE_CFG.get("auto_enroll_linger_seconds", 6.0)   # unknown must persist continuously this long
AUTO_ENROLL_COOLDOWN_SECONDS = _FACE_CFG.get("auto_enroll_cooldown_seconds", 30.0)  # min gap between two auto-enrolls
AUTO_ENROLL_MIN_FACE_WIDTH_FRAC = _FACE_CFG.get("auto_enroll_min_face_width_frac", 0.12)  # bbox width / frame width — skip tiny/far faces
TEMP_NAME_PREFIX = _FACE_CFG.get("temp_name_prefix", "guest")
CROWD_EVENT_INTERVAL_SECONDS = _FACE_CFG.get("crowd_event_interval_seconds", 8.0)  # throttle `crowd_detected` events
# How long a face must be present before we emit `face_detected` (was 5.0 —
# lowered so Kiki recognises who it's talking to quickly), and how long absent
# before `face_lost`.
FACE_CONFIRM_SECONDS = _FACE_CFG.get("face_confirm_seconds", 2.0)
FACE_EXIT_SECONDS = _FACE_CFG.get("face_exit_seconds", 5.0)
# On startup, throw away leftover guest_* temp identities (images + DB records)
# and make sure every NAMED person in train/ is actually enrolled. Temp guests
# are disposable by design: an un-renamed guest is just noise, and keeping them
# lets a bad enrollment shadow the real person forever.
PURGE_GUESTS_ON_START = _FACE_CFG.get("purge_guests_on_start", True)
# Never auto-enroll a track that was recognised as a real person moments ago —
# stops a split-second "Unknown" flicker from enrolling someone we already know.
KNOWN_FACE_GRACE_SECONDS = _FACE_CFG.get("known_face_grace_seconds", 60.0)
# Safety valve: if this many auto-enrolls happen without any being renamed to a
# real name, stop enrolling (something is looping) until the next restart.
MAX_PENDING_GUESTS = _FACE_CFG.get("max_pending_guests", 3)

# --- FULL BODY MODE CONFIGURATION ---
# Dynamic Dead Zone Settings (for body detection)
BODY_DEAD_ZONE_MAX = 120  # Deadzone when Close (Stable)
BODY_DEAD_ZONE_MIN = 40   # Deadzone when Far (Responsive/Tight)

# Dynamic Calibration Ranges (Target Width in Pixels)
RANGE_WIDTH_FAR = 50
RANGE_WIDTH_CLOSE = 200

# Movement Thresholds (Tune based on camera FOV and person size)
TARGET_WIDTH_MIN = 150  # If person width is smaller than this, move forward
TARGET_WIDTH_MAX = 950  # If person width is larger than this, move backward

# Ultrasonic Pins
TRIG_PIN = 20
ECHO_PIN = 21
BACK_TRIG_PIN = 6
BACK_ECHO_PIN = 19

MIN_DIST_FRONT = 20  # cm (Stop distance)
MIN_DIST_REAR = 20   # cm (Stop distance)
BACKOFF_DIST = 10    # cm (If closer than this, back off actively)

# Motor Speeds for Body Mode
SPEED_NORMAL = 100
SPEED_TURN_MAX = 90   # Max turn speed (when Close)
SPEED_TURN_MIN = 30   # Min turn speed (when Far)
SPEED_SLOW = 60
SPEED_SEARCH = 30     # Speed for searching mode

TURN_DELAY = 0        # Seconds to wait before confirming a turn


# --- POSE ESTIMATION CONFIGURATION ---
# Pose runs as a second, independent branch off a tee on the same source, sharing
# the single Hailo-8 with the two face models. Every hailonet built by
# INFERENCE_PIPELINE carries vdevice-group-id=SHARED and hailonet's default
# scheduling-algorithm is ROUND_ROBIN, so the HailoRT scheduler time-slices all
# three models with no extra device setup.
POSE_HEF_NAME = "yolov8s_pose"   # 10.5 MB; the hailo8 default yolov8m_pose is 30 MB
POSE_POSTPROCESS_SO_NAME = "libyolov8pose_postprocess.so"
POSE_POSTPROCESS_FUNCTION = "filter_letterbox"
POSE_BATCH_SIZE = 2
# Below hailonet's default priority of 16, so face recognition wins scheduler
# arbitration; the timeout stops pose from ever being starved outright.
POSE_SCHEDULER_PRIORITY = 10
POSE_SCHEDULER_TIMEOUT_MS = 200

# COCO 17-keypoint ordering, as emitted by libyolov8pose_postprocess.so.
POSE_KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# Set True to restore the old either/or behaviour where full-body mode tore down
# the face pipeline and ran yolov8m detection instead. Left as an escape hatch.
LEGACY_BODY_PIPELINE = False

# Populated from the command line in main(); read when building the pipeline.
#   mode "overlay"  — pose is spliced into the main chain after the face callback,
#                     so hailooverlay draws skeletons on both the display window
#                     and the MJPEG stream. Runs at the source frame rate.
#   mode "parallel" — pose hangs off a tee in its own branch and cannot slow the
#                     face chain, and honours --pose-fps, but produces no overlay
#                     (its metadata is on a different buffer).
POSE_OPTIONS = {
    "enabled": True,
    "hef": POSE_HEF_NAME,
    "fps": None,       # None => run at the source frame rate (parallel mode only)
    "mode": "overlay",
}


# --- CLIP (ZERO-SHOT DESCRIPTION) CONFIGURATION ---
# CLIP adds two more models: a small person detector and the CLIP image encoder.
# Person crops are embedded and matched against pre-computed text embeddings, so
# no torch/openai-clip is needed at runtime — TextImageMatcher.match() is pure
# numpy over the embeddings loaded from JSON.
CLIP_POSTPROCESS_SO_NAME = "libclip_postprocess.so"
CLIP_CROPPER_SO_NAME = "libclip_croppers_postprocess.so"
CLIP_DETECTION_POSTPROCESS_SO_NAME = "libyolo_hailortpp_postprocess.so"
CLIP_POSTPROCESS_FUNCTION = "filter"
CLIP_DETECTION_POSTPROCESS_FUNCTION = "filter"
CLIP_CROPPER_FUNCTION = "person_cropper"
CLIP_PERSON_CLASS_ID = 1
CLIP_DETECTION_BATCH_SIZE = 8
CLIP_ENCODER_BATCH_SIZE = 8
# CLIP is the lowest-priority consumer of the device — face recognition and pose
# both matter more for responsiveness than a description that updates per-track.
CLIP_SCHEDULER_PRIORITY = 8
CLIP_SCHEDULER_TIMEOUT_MS = 1000

# Text-embedding prompt files, tried in order. Drop a clip_prompts.json next to
# this script (same schema as the stock embeddings.json) to use your own prompts.
CLIP_PROMPT_FILES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "clip_prompts.json"),
    os.path.expanduser(
        "~/Kiki/hailo-apps/hailo_apps/python/pipeline_apps/clip/embeddings.json"
    ),
]

# Vision mode, set from --vision-mode in main(). One of:
#   "pose"      face recognition + pose            (3 models — verified 30 fps)
#   "clip"      face recognition + CLIP            (4 models)
#   "pose+clip" face recognition + pose + CLIP     (4 models; CLIP reuses the
#               pose person boxes instead of running a second detector)
#   "none"      face recognition only              (2 models)
VISION_MODE = {"value": "pose"}


# --- PIPELINE STATE MANAGEMENT ---
class PipelineState:
    """Thread-safe state management for the pipeline."""
    def __init__(self):
        self._lock = threading.Lock()
        self._webcam_enabled = True
        self._mode = "eval"  # "eval" or "train"
        self._neck_movement_enabled = True
        self._full_body_mode_enabled = False  # Full body tracking with robot motors
        self._training_person = None
        self._training_in_progress = False
        self._target_person = DEFAULT_TARGET_PERSON
        self._pipeline_restart_requested = False
        self._pipeline_stop_requested = False
        
    @property
    def webcam_enabled(self):
        with self._lock:
            return self._webcam_enabled
    
    @webcam_enabled.setter
    def webcam_enabled(self, value):
        with self._lock:
            old_value = self._webcam_enabled
            self._webcam_enabled = value
            # Request pipeline action if state changed
            if old_value != value:
                if value:
                    self._pipeline_restart_requested = True
                else:
                    self._pipeline_stop_requested = True
            
    @property
    def mode(self):
        with self._lock:
            return self._mode
    
    @mode.setter
    def mode(self, value):
        with self._lock:
            self._mode = value
            
    @property
    def neck_movement_enabled(self):
        with self._lock:
            return self._neck_movement_enabled
    
    @neck_movement_enabled.setter
    def neck_movement_enabled(self, value):
        with self._lock:
            self._neck_movement_enabled = value
            
    @property
    def training_person(self):
        with self._lock:
            return self._training_person
    
    @training_person.setter
    def training_person(self, value):
        with self._lock:
            self._training_person = value
            
    @property
    def training_in_progress(self):
        with self._lock:
            return self._training_in_progress
    
    @training_in_progress.setter
    def training_in_progress(self, value):
        with self._lock:
            self._training_in_progress = value
    
    @property
    def target_person(self):
        with self._lock:
            return self._target_person
    
    @target_person.setter
    def target_person(self, value):
        with self._lock:
            self._target_person = value
            print(f"[State] Target person changed to: {value}")
    
    @property
    def pipeline_restart_requested(self):
        with self._lock:
            return self._pipeline_restart_requested
    
    def clear_restart_request(self):
        with self._lock:
            self._pipeline_restart_requested = False
    
    @property
    def pipeline_stop_requested(self):
        with self._lock:
            return self._pipeline_stop_requested
    
    def clear_stop_request(self):
        with self._lock:
            self._pipeline_stop_requested = False
    
    @property
    def full_body_mode_enabled(self):
        with self._lock:
            return self._full_body_mode_enabled
    
    @full_body_mode_enabled.setter
    def full_body_mode_enabled(self, value):
        with self._lock:
            old_value = self._full_body_mode_enabled
            self._full_body_mode_enabled = value
            # Request pipeline restart when mode changes
            if old_value != value:
                self._pipeline_stop_requested = True
                self._pipeline_restart_requested = True
                print(f"[State] Full body mode changed to: {'enabled' if value else 'disabled'}")
    
    def get_state_dict(self):
        with self._lock:
            return {
                "webcam": "on" if self._webcam_enabled else "off",
                "mode": self._mode,
                "neck_movement": "on" if self._neck_movement_enabled else "off",
                "full_body_mode": "on" if self._full_body_mode_enabled else "off",
                "training_in_progress": self._training_in_progress,
                "target_person": self._target_person
            }


# Global pipeline state
pipeline_state = PipelineState()


# --- FACE TRACKING FOR EVENTS ---
class FaceTracker:
    """Tracks faces to detect enter/exit events with 5-second buffer."""
    def __init__(self):
        self._lock = threading.Lock()
        self._active_faces = {}  # {track_id: {"name": str, "last_seen": float, "first_seen": float, "confirmed": bool}}
        self._exit_timeout = FACE_EXIT_SECONDS   # seconds before considering a face as exited
        self._enter_timeout = FACE_CONFIRM_SECONDS  # seconds before confirming a new face
        
    def update_face(self, track_id, name, confidence):
        """
        Update a face's presence. 
        Returns 'confirmed' if face was just confirmed after 1 second presence.
        Returns None otherwise.
        """
        with self._lock:
            current_time = time.time()
            
            if track_id not in self._active_faces:
                # New face - add as pending (not yet confirmed)
                self._active_faces[track_id] = {
                    "name": name,
                    "confidence": confidence,
                    "first_seen": current_time,
                    "last_seen": current_time,
                    "confirmed": False
                }
                return None  # Don't announce yet
            else:
                # Existing face - update last seen
                face_data = self._active_faces[track_id]
                face_data["last_seen"] = current_time
                face_data["name"] = name  # Update name in case recognition improved
                face_data["confidence"] = confidence
                
                # Check if we should confirm this face (seen for 1+ seconds)
                if not face_data["confirmed"]:
                    if current_time - face_data["first_seen"] >= self._enter_timeout:
                        face_data["confirmed"] = True
                        return "confirmed"  # Now announce the face
                
                return None
    
    def check_exits(self):
        """Check for faces that have exited (not seen for 1 second). Returns list of (track_id, name) tuples."""
        with self._lock:
            current_time = time.time()
            exited = []
            to_remove = []
            
            for track_id, data in self._active_faces.items():
                if current_time - data["last_seen"] > self._exit_timeout:
                    # Only announce exit for faces that were confirmed
                    if data["confirmed"]:
                        exited.append((track_id, data["name"]))
                    to_remove.append(track_id)
            
            for track_id in to_remove:
                del self._active_faces[track_id]
                
            return exited
    
    def get_active_faces(self):
        with self._lock:
            return dict(self._active_faces)
    
    def clear(self):
        """Clear all tracked faces."""
        with self._lock:
            self._active_faces.clear()


# Global face tracker
face_tracker = FaceTracker()


# --- RAW FRAME BUFFER FOR TRAINING ---
class RawFrameBuffer:
    """Thread-safe buffer for raw camera frames (without detection boxes)."""
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        
    def set_frame(self, frame):
        with self._lock:
            self._frame = frame.copy() if frame is not None else None
    
    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None
    
    def clear(self):
        with self._lock:
            self._frame = None


# Global raw frame buffer
raw_frame_buffer = RawFrameBuffer()


# --- POSE ESTIMATION STATE ---
class PoseState:
    """Thread-safe holder for the latest pose-estimation result.

    Written from the pose branch's own GStreamer streaming thread, read from
    anywhere else (ZMQ handlers, motor logic, MJPEG). All coordinates are
    frame-normalized 0..1, matching how bboxes are treated elsewhere in this file.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._persons = []
        self._updated_at = 0.0
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._fps = 0.0

    def update(self, persons):
        now = time.time()
        with self._lock:
            self._persons = persons
            self._updated_at = now
            self._frame_count += 1
            elapsed = now - self._last_fps_time
            if elapsed >= 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._last_fps_time = now

    def get_persons(self, max_age=1.0):
        """Latest persons, or [] if the data is older than `max_age` seconds."""
        with self._lock:
            if not self._persons or (time.time() - self._updated_at) > max_age:
                return []
            return list(self._persons)

    @property
    def fps(self):
        with self._lock:
            return self._fps

    def clear(self):
        with self._lock:
            self._persons = []
            self._updated_at = 0.0


# Global pose state
pose_state = PoseState()


# --- CLIP STATE ---
class ClipState:
    """Thread-safe holder for the latest CLIP zero-shot matches.

    Maps track_id -> {"text", "similarity", "updated_at"}. Written from the CLIP
    branch's streaming thread.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._tracks = {}
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._fps = 0.0

    def update(self, matches):
        """matches: list of (track_id, text, similarity)."""
        now = time.time()
        with self._lock:
            for track_id, text, similarity in matches:
                self._tracks[track_id] = {
                    "text": text,
                    "similarity": float(similarity),
                    "updated_at": now,
                }
            # Forget tracks we haven't seen for a while so the dict can't grow
            # without bound over a long run.
            stale = [t for t, v in self._tracks.items() if now - v["updated_at"] > 30.0]
            for t in stale:
                del self._tracks[t]

            self._frame_count += 1
            elapsed = now - self._last_fps_time
            if elapsed >= 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._last_fps_time = now

    def get_descriptions(self, max_age=10.0):
        now = time.time()
        with self._lock:
            return {
                t: dict(v) for t, v in self._tracks.items()
                if now - v["updated_at"] <= max_age
            }

    @property
    def fps(self):
        with self._lock:
            return self._fps

    def clear(self):
        with self._lock:
            self._tracks = {}


# Global CLIP state
clip_state = ClipState()

# Global hand-gesture state. Populated by HandGestureWorker when --hands is on;
# stays empty otherwise so consumers can read it unconditionally.
hand_state = HandState() if HAND_GESTURES_AVAILABLE else None

# Set from --hands in main().
HAND_OPTIONS = {"enabled": False, "fps": 10, "max_hands": 2, "hef": None}

# Global pose-recognition state. The recognizer itself is constructed in main()
# only when pose is active, so a face-only run never loads the classifier.
pose_recognition_state = PoseRecognitionState() if POSE_RECOGNITION_AVAILABLE else None
pose_recognizer = None

# Set from --pose-recognition in main().
POSE_RECOGNITION_OPTIONS = {"enabled": False, "overlay": True}

# Lazily-imported CLIP text matcher (only pulled in when CLIP mode is active, so
# face-only / pose runs never pay for it).
clip_text_matcher = None


def init_clip_text_matcher():
    """Load pre-computed text embeddings for zero-shot matching.

    Returns the matcher, or None if no usable prompt file was found. Only the
    JSON embeddings are needed — matching is numpy dot products, so torch and
    openai-clip are never imported.
    """
    global clip_text_matcher
    if clip_text_matcher is not None:
        return clip_text_matcher

    try:
        from hailo_apps.python.pipeline_apps.clip.text_image_matcher import (
            text_image_matcher,
        )
    except ImportError as e:
        print(f"[CLIP] Could not import text_image_matcher: {e}")
        return None

    for path in CLIP_PROMPT_FILES:
        if not os.path.isfile(path):
            continue
        text_image_matcher.load_embeddings(path)
        prompts = [e.text for e in text_image_matcher.entries]
        if not prompts:
            print(f"[CLIP] {path} has no entries — trying next")
            continue
        print(f"[CLIP] Loaded {len(prompts)} prompts from {path}: {prompts}")
        print(f"[CLIP] Match threshold: {text_image_matcher.threshold}")
        clip_text_matcher = text_image_matcher
        return clip_text_matcher

    print(f"[CLIP] No prompt file found (looked in {CLIP_PROMPT_FILES}) — "
          f"embeddings will be computed but not matched to any text")
    return None


# --- AUTO-ENROLL MANAGER ---
class UnknownEnroller:
    """Decides when a lone, clear unknown face should be auto-enrolled.

    app_callback calls `evaluate()` every frame with the current face list. This
    class only holds cheap timers/flags; the actual capture+train is dispatched
    to a daemon thread so the GStreamer callback is never blocked.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._candidate_track_id = None      # the single unknown track we're timing
        self._candidate_since = 0.0
        self._in_progress = False
        self._last_enroll_time = 0.0
        self._last_crowd_event = 0.0
        # track_id -> last time this track was recognised as a REAL person. A
        # recognised face that flickers to "Unknown" for a moment (bad angle,
        # motion blur, confidence dipping under the 0.4 threshold) must NEVER be
        # auto-enrolled as a stranger — that creates a duplicate identity for
        # somebody we already know.
        self._known_track_seen = {}

    def _make_temp_name(self):
        base = f"{TEMP_NAME_PREFIX}_{time.strftime('%Y%m%d_%H%M')}"
        name = base
        i = 2
        while os.path.isdir(os.path.join(TRAIN_DIR, name)):
            name = f"{base}_{i}"
            i += 1
        return name

    def enroll_finished(self):
        with self._lock:
            self._in_progress = False
            self._last_enroll_time = time.time()
            self._candidate_track_id = None

    def evaluate(self, faces, frame, state, event_pub):
        """faces: list of dicts {track_id, name, confidence, bbox_wfrac, crop}."""
        if not AUTO_ENROLL_ENABLED:
            return
        now = time.time()

        # Never interfere with an in-flight training (manual or auto).
        with self._lock:
            if self._in_progress or state.training_in_progress:
                return
            if now - self._last_enroll_time < AUTO_ENROLL_COOLDOWN_SECONDS:
                return

        # Remember which tracks have been recognised as real people, so a
        # momentary "Unknown" flicker on a known face can't trigger an enroll.
        with self._lock:
            for f in faces:
                if f["name"] != "Unknown":
                    self._known_track_seen[f["track_id"]] = now
            if len(self._known_track_seen) > 256:  # prune stale entries
                self._known_track_seen = {
                    k: v for k, v in self._known_track_seen.items()
                    if now - v < KNOWN_FACE_GRACE_SECONDS}

        unknown_faces = [f for f in faces if f["name"] == "Unknown"]

        # More than one face present → we can't isolate a single stranger.
        # Nudge the crowd to come one-by-one (throttled) and reset the timer.
        if len(faces) > 1:
            with self._lock:
                self._candidate_track_id = None
            if unknown_faces and event_pub and (now - self._last_crowd_event) > CROWD_EVENT_INTERVAL_SECONDS:
                self._last_crowd_event = now
                event_pub.publish_event("crowd_detected", {"face_count": len(faces)})
            return

        # Exactly one face, and it must be Unknown and reasonably large/clear.
        if len(unknown_faces) != 1:
            with self._lock:
                self._candidate_track_id = None
            return
        cand = unknown_faces[0]
        if cand["bbox_wfrac"] < AUTO_ENROLL_MIN_FACE_WIDTH_FRAC or cand["crop"] is None:
            with self._lock:
                self._candidate_track_id = None
            return

        with self._lock:
            # This track was recognised as a real person very recently → it's a
            # known face flickering, not a stranger. Don't enroll them again.
            last_known = self._known_track_seen.get(cand["track_id"])
            if last_known is not None and (now - last_known) < KNOWN_FACE_GRACE_SECONDS:
                self._candidate_track_id = None
                return

            if self._candidate_track_id != cand["track_id"]:
                # New candidate — start the linger timer.
                self._candidate_track_id = cand["track_id"]
                self._candidate_since = now
                return
            if now - self._candidate_since < AUTO_ENROLL_LINGER_SECONDS:
                return

            # Runaway guard: un-renamed guests piling up means the enroll→ask→
            # rename loop isn't completing (or recognition is broken and a known
            # person keeps reading Unknown). Stop making new identities rather
            # than spawning a folder every cooldown.
            pending = len(_guest_train_dirs())
            if pending >= MAX_PENDING_GUESTS:
                if now - self._last_enroll_time > 60:  # don't spam the log
                    self._last_enroll_time = now
                    print(f"[AutoEnroll] Skipping: {pending} un-renamed guest identities "
                          f"already pending (max {MAX_PENDING_GUESTS}). Recognition may be "
                          f"broken — check that named people have DB records.")
                self._candidate_track_id = None
                return

            # Lingered long enough → commit to enrolling this face.
            self._in_progress = True
            temp_name = self._make_temp_name()

        crop = cand["crop"]
        threading.Thread(
            target=run_auto_enroll,
            args=(temp_name, crop, raw_frame_buffer, state, event_pub, self),
            daemon=True,
        ).start()


unknown_enroller = UnknownEnroller()


def crop_face_from_frame(frame, bbox, pad=0.35):
    """Crop a face bbox (Hailo normalized xmin/ymin/width/height) from a BGR frame,
    with padding, returning a BGR image or None."""
    try:
        h, w = frame.shape[:2]
        x = bbox.xmin() * w
        y = bbox.ymin() * h
        bw = bbox.width() * w
        bh = bbox.height() * h
        px = bw * pad
        py = bh * pad
        x0 = max(0, int(x - px))
        y0 = max(0, int(y - py))
        x1 = min(w, int(x + bw + px))
        y1 = min(h, int(y + bh + py))
        if x1 <= x0 or y1 <= y0:
            return None
        return frame[y0:y1, x0:x1].copy()
    except Exception:
        return None


def _jpeg_b64(bgr_image, max_side=256):
    """Encode a BGR image to a base64 JPEG string (downscaled), or '' on failure."""
    try:
        img = bgr_image
        # An empty or 1px source makes cv2.resize raise a cv::Exception that can
        # escape as a std::terminate abort rather than a catchable Python error,
        # so validate rather than relying on the except below.
        if img is None or img.size == 0 or img.ndim < 2:
            return ""
        h, w = img.shape[:2]
        if h < 2 or w < 2:
            return ""
        scale = max_side / float(max(h, w)) if max(h, w) > max_side else 1.0
        if scale < 1.0:
            # Clamp to >=2px: int() can round a thin crop's short side to 0,
            # which is itself an invalid resize target.
            new_w = max(2, int(w * scale))
            new_h = max(2, int(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return ""
        return base64.b64encode(enc.tobytes()).decode("ascii")
    except Exception:
        return ""


# Latest largest-face crop, cached so a client can fetch a face image on demand
# (e.g. for Kiki's OLED face card) even when it missed the enroll event's crop.
_latest_face_crop = {"name": None, "bgr": None, "ts": 0.0}
_latest_face_crop_lock = threading.Lock()


def cache_latest_face_crop(name, bgr):
    if bgr is None:
        return
    with _latest_face_crop_lock:
        _latest_face_crop["name"] = name
        _latest_face_crop["bgr"] = bgr
        _latest_face_crop["ts"] = time.time()


def get_latest_face_crop_b64(max_age=8.0):
    """Return (name, b64_jpeg) of the most recent face crop, or (None, '') if
    stale/missing."""
    with _latest_face_crop_lock:
        name, bgr, ts = _latest_face_crop["name"], _latest_face_crop["bgr"], _latest_face_crop["ts"]
    if bgr is None or (time.time() - ts) > max_age:
        return None, ""
    return name, _jpeg_b64(bgr)


# --- MJPEG STREAMING SERVER ---
# Thread-safe frame buffer for MJPEG streaming
class FrameBuffer:
    """Latest-frame buffer for MJPEG streaming.

    Three things here matter for stream latency, and all three were wrong before:

    1. JPEG encoding does NOT happen in set_frame. set_frame runs on the
       GStreamer appsink thread; encoding a 1280x720 frame there (inside the
       lock, 30x a second) stalled the appsink, back-pressured the display
       branch, and showed up as the stream hanging. Encoding now happens on a
       dedicated thread.
    2. Frames are versioned with a sequence number and consumers *wait* on a
       condition instead of polling. The old generator polled every 16 ms and
       re-sent whatever was in the buffer, so at a 30 fps source roughly half
       of everything sent was a byte-identical duplicate. That doubled
       bandwidth and, once the client could not drain it, queued in the socket
       as unbounded, ever-growing delay.
    3. The encoded bytes are stored once as an immutable object rather than
       calling .tobytes() (a ~100 KB copy) per client per frame.
    """

    def __init__(self):
        self._lock = threading.Condition()
        self._frame = None
        self._frame_seq = 0
        self._frame_captured_at = 0.0
        self._jpeg = None
        self._jpeg_seq = -1
        self._jpeg_captured_at = 0.0
        self._last_encode_age = 0.0
        self._last_send_age = 0.0
        self._last_encode_ms = 0.0
        self._camera_off_jpeg = self._create_camera_off_image()
        self._encoder_thread = None
        self._stop = threading.Event()
    
    def _create_camera_off_image(self):
        """Create a 'Camera Off' static image."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)  # Dark background
        
        # Add text
        text = "CAMERA OFF"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (640 - text_size[0]) // 2
        text_y = (480 + text_size[1]) // 2
        cv2.putText(img, text, (text_x, text_y), font, font_scale, (100, 100, 100), thickness)
        
        _, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jpeg.tobytes()
    
    def start_encoder(self):
        """Start the background JPEG encoder thread (idempotent)."""
        if self._encoder_thread and self._encoder_thread.is_alive():
            return
        self._stop.clear()
        self._encoder_thread = threading.Thread(
            target=self._encode_loop, name="mjpeg_encoder", daemon=True)
        self._encoder_thread.start()

    def stop_encoder(self):
        self._stop.set()
        with self._lock:
            self._lock.notify_all()

    def set_frame(self, frame, captured_at=None):
        """Publish the current frame. Called from the GStreamer appsink thread."""
        with self._lock:
            self._frame = frame
            self._frame_captured_at = captured_at or time.time()
            self._frame_seq += 1
            self._lock.notify_all()

    def get_latency_stats(self):
        """Where time is being spent, so delay can be localised rather than guessed.

        age_at_encode  — how old a frame already is when it reaches the encoder.
                         Large here means the delay is UPSTREAM, inside GStreamer.
        age_at_send    — age when handed to the HTTP client. Large only here
                         means the delay is in the encode/serve path.
        """
        with self._lock:
            return {
                "age_at_encode_ms": round(self._last_encode_age * 1000, 1),
                "age_at_send_ms": round(self._last_send_age * 1000, 1),
                "encode_ms": round(self._last_encode_ms, 1),
                "frame_seq": self._frame_seq,
                "jpeg_seq": self._jpeg_seq,
                "seq_lag": self._frame_seq - self._jpeg_seq,
            }

    def _encode_loop(self):
        """Downscale + JPEG-encode the newest frame, at most MJPEG_MAX_FPS times/s."""
        last_encoded_seq = -1
        min_interval = 1.0 / max(1, MJPEG_MAX_FPS)

        while not self._stop.is_set():
            t0 = time.time()
            with self._lock:
                # Wait for a frame newer than the one we last encoded.
                while (not self._stop.is_set()) and self._frame_seq == last_encoded_seq:
                    self._lock.wait(timeout=0.5)
                if self._stop.is_set():
                    return
                frame = self._frame
                seq = self._frame_seq
                captured_at = self._frame_captured_at

            if frame is None:
                last_encoded_seq = seq
                continue

            encode_started = time.time()
            try:
                encoded = self._encode(frame)
            except Exception as e:
                print(f"[MJPEG] Encode error: {e}")
                encoded = None
            encode_ms = (time.time() - encode_started) * 1000.0

            if encoded is not None:
                with self._lock:
                    self._jpeg = encoded
                    self._jpeg_seq = seq
                    self._jpeg_captured_at = captured_at
                    self._last_encode_age = encode_started - captured_at
                    self._last_encode_ms = encode_ms
                    self._lock.notify_all()
            last_encoded_seq = seq

            # Cap the encode rate so streaming can never eat the CPU the
            # inference pipeline needs.
            elapsed = time.time() - t0
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

    @staticmethod
    def _encode(frame):
        """Downscale to the stream width and JPEG-encode. Returns bytes or None."""
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        # Guard cv2.resize: an empty or degenerate source aborts the whole
        # process via an uncaught cv::Exception, not a Python exception.
        if h < 2 or w < 2:
            return None

        if MJPEG_STREAM_WIDTH and w > MJPEG_STREAM_WIDTH:
            new_w = int(MJPEG_STREAM_WIDTH)
            new_h = max(2, int(round(h * (new_w / float(w)))))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ok, enc = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), MJPEG_QUALITY])
        if not ok:
            return None
        return enc.tobytes()

    def wait_for_jpeg(self, last_seq, timeout=1.0):
        """Block until a JPEG newer than last_seq exists. Returns (bytes, seq).

        This is what removes duplicate frames from the stream: a client that has
        already sent sequence N sleeps here until N+1 exists rather than
        re-sending N.
        """
        with self._lock:
            if self._jpeg_seq <= last_seq:
                self._lock.wait(timeout=timeout)
            if self._jpeg is not None and self._jpeg_captured_at:
                self._last_send_age = time.time() - self._jpeg_captured_at
            return self._jpeg, self._jpeg_seq

    def get_jpeg(self, camera_enabled=True):
        """Latest JPEG bytes (no waiting). Kept for compatibility."""
        with self._lock:
            if not camera_enabled:
                return self._camera_off_jpeg
            return self._jpeg if self._jpeg is not None else self._camera_off_jpeg

    @property
    def camera_off_jpeg(self):
        return self._camera_off_jpeg

    def clear(self):
        """Clear the frame buffer."""
        with self._lock:
            self._frame = None
            self._jpeg = None
            self._jpeg_seq = -1
            self._lock.notify_all()

# Global frame buffer for MJPEG streaming
mjpeg_frame_buffer = FrameBuffer()


# --- ZEROMQ EVENT PUBLISHER ---
class ZMQEventPublisher:
    """Publishes face events via ZeroMQ PUB socket."""
    def __init__(self, port=ZMQ_EVENT_PORT):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        print(f"[ZMQ] Event publisher started on port {port}")
        
    def publish_event(self, event_type, data):
        """Publish an event to subscribers."""
        event = {"event": event_type, **data}
        self.socket.send_json(event)
        print(f"[ZMQ] Published event: {event}")
        
    def close(self):
        self.socket.close()
        self.context.term()


# Global event publisher
event_publisher = None

# Global pipeline control
pipeline_control = {
    "app": None,
    "running": False,
    "loop": None,  # GLib.MainLoop reference
    "cmd_queue": None  # Queue for webcam motor commands (including relay)
}


# --- ZEROMQ COMMAND SERVER ---
def zmq_command_server(stop_event, state, event_pub, raw_buffer):
    """ZeroMQ REP server for receiving commands."""
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout for checking stop_event
    socket.bind(f"tcp://*:{ZMQ_CMD_PORT}")
    print(f"[ZMQ] Command server started on port {ZMQ_CMD_PORT}")
    
    while not stop_event.is_set():
        try:
            message = socket.recv_json()
            print(f"[ZMQ] Received command: {message}")
            response = handle_command(message, state, event_pub, raw_buffer)
            socket.send_json(response)
        except zmq.Again:
            # Timeout, check stop_event and continue
            continue
        except Exception as e:
            print(f"[ZMQ] Error handling command: {e}")
            try:
                socket.send_json({"status": "error", "message": str(e)})
            except:
                pass
    
    socket.close()
    context.term()
    print("[ZMQ] Command server stopped")


def handle_command(message, state, event_pub, raw_buffer):
    """Handle incoming ZeroMQ commands."""
    global pipeline_control
    
    cmd = message.get("cmd", "")
    
    if cmd == "get":
        # Currently-visible confirmed faces — lets a client that just connected
        # (and missed the one-shot face_detected PUB messages) sync who is in
        # front RIGHT NOW instead of waiting for the person to leave and return.
        if message.get("active_faces"):
            faces = face_tracker.get_active_faces()
            confirmed = [
                {"track_id": tid, "name": d.get("name", "Unknown"),
                 "confidence": d.get("confidence", 0.0)}
                for tid, d in faces.items() if d.get("confirmed")
            ]
            return {"status": "ok", "active_faces": confirmed}
        # On-demand face crop (for the OLED face card when no thumbnail exists).
        if message.get("face_crop"):
            nm, b64 = get_latest_face_crop_b64()
            return {"status": "ok", "name": nm, "image_b64": b64}
        # Return current state
        return state.get_state_dict()
    
    elif cmd == "set":
        # ── Rename a person (temp → real). No retraining needed: LanceDB just
        # updates the label. Also rename the train/ folder so a future full
        # retrain stays consistent. ──
        if "rename" in message:
            old = str(message["rename"].get("old", "")).strip()
            new = str(message["rename"].get("new", "")).strip()
            if not old or not new:
                return {"status": "error", "message": "rename needs old+new"}
            res = _run_db_admin("rename", old, new)
            if res.get("status") == "ok":
                old_dir = os.path.join(TRAIN_DIR, old)
                new_dir = os.path.join(TRAIN_DIR, new)
                try:
                    if os.path.isdir(old_dir) and not os.path.isdir(new_dir):
                        os.rename(old_dir, new_dir)
                except Exception as e:
                    print(f"[ZMQ] rename folder warning: {e}")
                if event_pub:
                    event_pub.publish_event("person_renamed", {"old": old, "new": new})
                # Reload the DB in the live pipeline so the new label is used NOW
                # (otherwise it keeps showing the old label until a manual restart).
                bounce_face_pipeline(reason=f"rename {old}->{new}")
            return res

        # ── Forget a person: drop their DB record + train folder. Used when a
        # freshly auto-enrolled temp identity turns out to be a duplicate. ──
        if "forget_person" in message:
            name = str(message["forget_person"]).strip()
            if not name:
                return {"status": "error", "message": "forget_person needs a name"}
            res = _run_db_admin("forget", name)
            try:
                d = os.path.join(TRAIN_DIR, name)
                if os.path.isdir(d):
                    shutil.rmtree(d)
            except Exception as e:
                print(f"[ZMQ] forget folder warning: {e}")
            if res.get("status") == "ok":
                # Reload the DB so the forgotten person stops being recognised now.
                bounce_face_pipeline(reason=f"forget {name}")
            return res

        # ── Merge a temp identity into an existing REAL person who was mis-read
        # as Unknown (too little training data). Move the temp's images into the
        # real person's folder, DELETE the real DB record (run_training skips
        # labels already present), forget the temp, then retrain so the real
        # person is re-embedded with the extra images. Runs in a bg thread. ──
        if "merge" in message:
            real = str(message["merge"].get("real", "")).strip()
            temp = str(message["merge"].get("temp", "")).strip()
            if not real or not temp:
                return {"status": "error", "message": "merge needs real+temp"}
            if state.training_in_progress:
                return {"status": "error", "message": "training in progress"}

            def _do_merge():
                try:
                    state.training_in_progress = True
                    real_dir = os.path.join(TRAIN_DIR, real)
                    temp_dir = os.path.join(TRAIN_DIR, temp)
                    os.makedirs(real_dir, exist_ok=True)
                    # Move temp images into the real person's folder (unique names).
                    if os.path.isdir(temp_dir):
                        for fn in os.listdir(temp_dir):
                            src = os.path.join(temp_dir, fn)
                            dst = os.path.join(real_dir, f"{temp}_{fn}")
                            try:
                                shutil.move(src, dst)
                            except Exception as e:
                                print(f"[Merge] move {fn} warning: {e}")
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    # Delete both DB records so retrain re-embeds `real` fresh
                    # from the (now larger) folder, and the temp is gone.
                    #
                    # This is the DANGEROUS window: `real` is now absent from the
                    # DB, so if the retrain silently no-ops (Hailo device busy —
                    # the trainer still exits 0) the real person is GONE and every
                    # sighting reads "Unknown", which then triggers a fresh
                    # auto-enroll → another merge → repeat. That feedback loop is
                    # what wiped 'Alex' (84 images, zero DB records). So the
                    # retrain is now VERIFIED, and a failure is loud + recoverable
                    # (the images remain, and startup reconcile re-enrolls them).
                    _run_db_admin("forget", temp)
                    _run_db_admin("forget", real)
                    if _retrain_all(expect_labels=[real]):
                        if event_pub:
                            event_pub.publish_event("training_complete", {"person": real})
                            event_pub.publish_event("person_merged", {"real": real, "temp": temp})
                    else:
                        print(f"[Merge] !! RETRAIN FAILED — '{real}' is missing from the DB. "
                              f"Its {len(os.listdir(real_dir)) if os.path.isdir(real_dir) else 0} "
                              f"images are intact and will be re-enrolled on next start.")
                        if event_pub:
                            event_pub.publish_event("training_failed",
                                                    {"person": real, "reason": "retrain_unverified"})
                except Exception as e:
                    print(f"[Merge] error: {e}")
                finally:
                    state.training_in_progress = False
                    state.mode = "eval"

            threading.Thread(target=_do_merge, daemon=True).start()
            return {"status": "ok", "message": f"merging {temp} → {real}"}

        # Handle target person
        if "target_person" in message:
            person_name = message["target_person"]
            state.target_person = person_name
            print(f"[ZMQ] Target person set to: {person_name}")
        
        # Handle webcam on/off
        if "webcam" in message:
            webcam_val = message["webcam"].lower()
            new_state = (webcam_val == "on")
            
            if new_state != state.webcam_enabled:
                state.webcam_enabled = new_state
                
                if not new_state:
                    # Stop pipeline - use GLib.idle_add for thread-safe quit
                    print(f"[ZMQ] Webcam OFF requested - stopping pipeline...")
                    app = pipeline_control.get("app")
                    if app and hasattr(app, 'loop') and app.loop:
                        # Stop pipeline state first, then quit loop
                        def stop_pipeline_safe():
                            print("[ZMQ] Stopping pipeline from GLib main thread...")
                            if app.pipeline:
                                app.pipeline.set_state(Gst.State.NULL)
                            if app.loop:
                                app.loop.quit()
                            return False  # Remove idle callback
                        GLib.idle_add(stop_pipeline_safe)
                    else:
                        print("[ZMQ] Warning: No pipeline reference available")
                else:
                    # Restart request is handled by the main loop
                    print(f"[ZMQ] Webcam ON requested - pipeline will restart...")
            
            print(f"[ZMQ] Webcam set to: {webcam_val}")
        
        # Handle mode (eval/train)
        if "mode" in message:
            mode_val = message["mode"].lower()
            
            if mode_val.startswith("train"):
                # Training mode: train(PersonName) or {"mode": "train", "person": "Name"}
                person_name = message.get("person")
                if not person_name and "(" in mode_val:
                    # Parse train(PersonName) format
                    person_name = mode_val.split("(")[1].rstrip(")")
                
                if person_name:
                    # Start training in a separate thread
                    if not state.training_in_progress:
                        state.training_in_progress = True
                        state.training_person = person_name
                        state.mode = "train"
                        
                        training_thread = threading.Thread(
                            target=run_training_capture,
                            args=(person_name, raw_buffer, state, event_pub)
                        )
                        training_thread.daemon = True
                        training_thread.start()
                        return {"status": "ok", "message": f"Training started for {person_name}"}
                    else:
                        return {"status": "error", "message": "Training already in progress"}
                else:
                    return {"status": "error", "message": "Person name required for training"}
            else:
                state.mode = mode_val
                print(f"[ZMQ] Mode set to: {mode_val}")
        
        # Handle neck movement on/off
        if "neck_movement" in message:
            neck_val = message["neck_movement"].lower()
            state.neck_movement_enabled = (neck_val == "on")
            print(f"[ZMQ] Neck movement set to: {neck_val}")
            
            # Control relay: off when neck movement disabled, on when enabled
            cmd_queue = pipeline_control.get("cmd_queue")
            if cmd_queue:
                if neck_val == "on":
                    cmd_queue.put("RELAY_ON")  # Enable motor power
                    print("[ZMQ] Sent RELAY_ON command")
                    if PET_MODE_ARG and PET_MODE_AVAILABLE and not state.full_body_mode_enabled:
                        if not pet_mode_module._running:
                            print("[ZMQ] Starting pet mode (neck ON + pet arg enabled)")
                            pet_mode_module.start_pet_mode()
                else:
                    cmd_queue.put("RELAY_OFF")  # Disable motor power
                    print("[ZMQ] Sent RELAY_OFF command")
                    if PET_MODE_AVAILABLE and pet_mode_module._running:
                        print("[ZMQ] Stopping pet mode (neck OFF)")
                        pet_mode_module.stop_pet_mode()

        # ── Explicit one-shot NECK turn for expression (added for KikiFast) ──
        # Distinct from neck_movement (autonomous face-tracking on/off): this is a
        # deliberate left/right/center head turn requested by Kiki (e.g. "turn away
        # when hurt"). Enqueued to the neck stepper process via cmd_queue; the loop
        # there guarantees LOOK_* commands run and tracks offset so center returns
        # to neutral. UNTESTED ON HARDWARE — verify direction/step count on the bot.
        if "look" in message:
            direction = str(message.get("look", "")).lower().strip()
            steps = message.get("look_steps")
            cmd_queue = pipeline_control.get("cmd_queue")
            if cmd_queue:
                if direction == "left":
                    cmd_queue.put(f"LOOK_LEFT:{int(steps)}" if steps else "LOOK_LEFT")
                elif direction == "right":
                    cmd_queue.put(f"LOOK_RIGHT:{int(steps)}" if steps else "LOOK_RIGHT")
                elif direction == "center":
                    cmd_queue.put("LOOK_CENTER")
                else:
                    print(f"[ZMQ] Unknown look direction '{direction}' — ignoring")
            print(f"[ZMQ] Explicit neck look: {direction} (steps={steps})")

        # Handle full body mode on/off
        if "full_body" in message:
            full_body_val = message["full_body"].lower()
            new_full_body_state = (full_body_val == "on")
            # if full_body_val == "on":
                # cmd_queue.put("RELAY_ON")  # Enable motor power
                # print("[ZMQ] Sent RELAY_ON command")
            # else:
                # cmd_queue.put("RELAY_OFF")  # Disable motor power
                # print("[ZMQ] Sent RELAY_OFF command")
            if not MOTOR_CONTROL_AVAILABLE and new_full_body_state:
                return {"status": "error", "message": "Motor control not available"}
            
            if new_full_body_state != state.full_body_mode_enabled:
                state.full_body_mode_enabled = new_full_body_state

                if LEGACY_BODY_PIPELINE:
                    # Old behaviour: tear the pipeline down and rebuild in the
                    # other mode. Face recognition stops while body mode runs.
                    print(f"[ZMQ] Full body mode {'ON' if new_full_body_state else 'OFF'} - switching pipeline...")
                    app = pipeline_control.get("app")
                    if app and hasattr(app, 'loop') and app.loop:
                        def stop_for_mode_switch():
                            print("[ZMQ] Stopping pipeline for mode switch...")
                            if app.pipeline:
                                app.pipeline.set_state(Gst.State.NULL)
                            if app.loop:
                                app.loop.quit()
                            return False
                        GLib.idle_add(stop_for_mode_switch)
                    else:
                        print("[ZMQ] Warning: No pipeline reference for mode switch")
                else:
                    # Combined pipeline: face recognition and pose keep running,
                    # so full-body mode only arms/disarms the chassis motors.
                    print(f"[ZMQ] Full body mode {'ON' if new_full_body_state else 'OFF'} - "
                          f"arming motors (pipeline stays up)")
                    threading.Thread(
                        target=_notify_motor_body_mode,
                        args=(new_full_body_state,),
                        daemon=True,
                    ).start()

            print(f"[ZMQ] Full body mode set to: {full_body_val}")
        
        return {"status": "ok"}
    
    else:
        return {"status": "error", "message": f"Unknown command: {cmd}"}


def _wait_pipeline_stopped(timeout=15.0):
    """Block until the face pipeline's run() has fully exited (device freed)."""
    t0 = time.time()
    while pipeline_control.get("running") and (time.time() - t0) < timeout:
        time.sleep(0.2)
    return not pipeline_control.get("running")


def pause_face_pipeline_for_training(timeout=15.0):
    """Stop the live face-recognition pipeline so the Hailo device is free for
    the training subprocess.

    The Hailo8 has ONE physical device; the running recognition pipeline and a
    separate training vdevice cannot coexist (HAILO_OUT_OF_PHYSICAL_DEVICES).
    So we stop recognition, let training use the device, then resume. Returns
    True once the device is freed. Reuses the exact webcam-off stop path."""
    app = pipeline_control.get("app")
    if app is None or not pipeline_control.get("running"):
        return True  # nothing running to pause
    print("[Training] Pausing recognition to free the Hailo device...")
    # Park the main pipeline loop (so it won't auto-restart), then quit the GLib
    # loop from the pipeline's own thread (idle_add), matching the webcam-off path.
    pipeline_state.webcam_enabled = False

    def _quit():
        try:
            if app.pipeline:
                app.pipeline.set_state(Gst.State.NULL)
            if app.loop:
                app.loop.quit()
        except Exception as e:
            print(f"[Training] pause quit error: {e}")
        return False
    GLib.idle_add(_quit)

    freed = _wait_pipeline_stopped(timeout)
    time.sleep(0.7)  # small settle so HailoRT fully releases the vdevice
    # Forget stale tracks. Otherwise the just-enrolled person, who was confirmed
    # as "Unknown" before training, keeps that confirmed track — and update_face
    # returns None when only the NAME changes (Unknown -> guest_x), so no fresh
    # face_detected event fires and Kiki never learns to ask their name.
    face_tracker.clear()
    print("[Training] Recognition paused; Hailo device free."
          if freed else
          "[Training] WARNING: pipeline did not stop in time; training may fail.")
    return freed


def resume_face_pipeline():
    """Restart the live face-recognition pipeline after training (the main loop
    parks while webcam_enabled is False and restarts when it flips back True)."""
    print("[Training] Resuming recognition pipeline...")
    pipeline_state.webcam_enabled = True


def bounce_face_pipeline(reason="db update"):
    """Stop + restart the recognition pipeline in a background thread so it
    REOPENS the LanceDB. The running pipeline holds a fixed LanceDB version from
    startup, so a rename/forget done by the db_admin subprocess isn't visible
    until the pipeline reloads — this makes it take effect immediately instead of
    after a manual restart. Non-blocking so the ZMQ command returns right away."""
    def _do():
        print(f"[Pipeline] Bouncing recognition to reload DB ({reason})...")
        if pause_face_pipeline_for_training():
            resume_face_pipeline()
        print(f"[Pipeline] Bounce complete ({reason}).")
    threading.Thread(target=_do, daemon=True).start()


def run_training_capture(person_name, raw_buffer, state, event_pub):
    """Capture photos and run training for a person."""
    try:
        print(f"[Training] Starting capture for {person_name}")
        
        # Create folder for the person
        person_dir = os.path.join(TRAIN_DIR, person_name)
        os.makedirs(person_dir, exist_ok=True)
        print(f"[Training] Created directory: {person_dir}")
        
        # Capture photos
        photos_captured = 0
        for i in range(TRAIN_PHOTO_COUNT):
            frame = raw_buffer.get_frame()
            if frame is not None:
                photo_path = os.path.join(person_dir, f"{i+1}.jpg")
                cv2.imwrite(photo_path, frame)
                photos_captured += 1
                print(f"[Training] Captured photo {i+1}/{TRAIN_PHOTO_COUNT}: {photo_path}")
            else:
                print(f"[Training] Warning: No frame available for photo {i+1}")
            
            if i < TRAIN_PHOTO_COUNT - 1:
                time.sleep(TRAIN_PHOTO_INTERVAL)
        
        print(f"[Training] Captured {photos_captured} photos. Running training...")

        # Run training subprocess
        training_script = os.path.expanduser(
            "~/Kiki/hailo-apps/hailo_apps/python/pipeline_apps/face_recognition/face_recognition.py"
        )
        training_cmd = [sys.executable, training_script, "--mode", "train"]

        # Free the single Hailo device: stop live recognition, train, then resume.
        paused = pause_face_pipeline_for_training()
        try:
            result = subprocess.run(
                training_cmd,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(training_script)
            )
            if result.returncode == 0:
                print(f"[Training] Training completed successfully for {person_name}")
            else:
                print(f"[Training] Training completed with warnings: {result.stderr}")
            if "OUT_OF_PHYSICAL_DEVICES" in (result.stderr or ""):
                print("[Training] ERROR: Hailo device was still busy — no embedding "
                      "created. (Recognition pipeline did not release the device.)")
        finally:
            # Always bring recognition back, even if training raised.
            if paused:
                resume_face_pipeline()

        # Publish training complete event
        if event_pub:
            event_pub.publish_event("training_complete", {"person": person_name})
        
    except Exception as e:
        print(f"[Training] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Reset state
        state.training_in_progress = False
        state.training_person = None
        state.mode = "eval"
        print(f"[Training] Switched back to eval mode")


def run_auto_enroll(temp_name, crop_bgr, raw_buffer, state, event_pub, enroller):
    """Background worker: capture clear images of a lone stranger, train, and
    publish `unknown_face_enrolled` (with a face crop) so Kiki can remember them.

    Reuses run_training_capture for the capture+train (it also flips
    state.training_in_progress, captures TRAIN_PHOTO_COUNT frames, and runs the
    training subprocess). We stash a face crop up front for the OLED/Gemini.
    """
    try:
        print(f"[AutoEnroll] New stranger → temp name '{temp_name}'")
        crop_b64 = _jpeg_b64(crop_bgr) if crop_bgr is not None else ""

        # Mark manual-training state so nothing else fights for the camera, then
        # capture + train exactly like a normal enrollment.
        state.training_in_progress = True
        state.training_person = temp_name
        state.mode = "train"
        run_training_capture(temp_name, raw_buffer, state, event_pub)

        # The trainer exits 0 even when it enrolled nothing (busy Hailo device),
        # so confirm the record really exists. Announcing a stranger who has no
        # embedding would make Kiki ask a name for an identity it can never
        # recognise again — and leave a junk folder behind.
        labels = _db_labels()
        if labels is not None and temp_name not in labels:
            print(f"[AutoEnroll] Training produced NO DB record for '{temp_name}' "
                  f"— discarding it instead of announcing a phantom identity.")
            shutil.rmtree(os.path.join(TRAIN_DIR, temp_name), ignore_errors=True)
            return

        # Announce the freshly enrolled stranger with a face crop for Kiki.
        if event_pub:
            event_pub.publish_event("unknown_face_enrolled", {
                "temp_name": temp_name,
                "image_b64": crop_b64,
                "timestamp": time.time(),
            })
    except Exception as e:
        print(f"[AutoEnroll] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        enroller.enroll_finished()


def _run_db_admin(*admin_args):
    """Run db_admin.py as a subprocess and return its parsed JSON result dict."""
    try:
        result = subprocess.run(
            [sys.executable, DB_ADMIN_SCRIPT, *admin_args],
            capture_output=True, text=True, cwd=FACE_RECON_DIR, timeout=60,
        )
        out = (result.stdout or "").strip().splitlines()
        for line in reversed(out):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        return {"status": "error", "message": f"no json from db_admin: {result.stderr[-300:]}"}
    except Exception as e:
        return {"status": "error", "message": f"db_admin call failed: {e}"}


def _db_labels():
    """Set of labels currently in the face DB, or None if it couldn't be read."""
    res = _run_db_admin("list")
    if res.get("status") != "ok":
        print(f"[DBAdmin] list failed: {res.get('message')}")
        return None
    return {r.get("label") for r in (res.get("records") or []) if r.get("label")}


def _named_train_dirs():
    """train/ subfolders that are REAL people (not guest_* temp identities)."""
    try:
        return sorted(
            d for d in os.listdir(TRAIN_DIR)
            if os.path.isdir(os.path.join(TRAIN_DIR, d))
            and not d.startswith(f"{TEMP_NAME_PREFIX}_")
            and any(f.lower().endswith((".jpg", ".jpeg", ".png"))
                    for f in os.listdir(os.path.join(TRAIN_DIR, d)))
        )
    except OSError:
        return []


def _retrain_all(expect_labels=None, attempts=2):
    """Run the training subprocess (enrolls any train/ folder not already in the
    DB). Callers that need an EXISTING person re-embedded must delete that DB
    record first (run_training skips labels already present). Frees the Hailo
    device (stops live recognition) for the duration, like normal enrollment.

    IMPORTANT: the trainer's exit status cannot be trusted to mean "enrolled".
    If the Hailo device is still held by the recognition pipeline it prints
    HAILO_OUT_OF_PHYSICAL_DEVICES for every frame and then SEGFAULTS (exit 139),
    and the old version of this function never looked at the return code at all
    — which is how a merge deleted a real person and reported success. Success is
    therefore verified by reading the DB back and confirming `expect_labels`
    actually landed. Returns
    True only if verified (or if there was nothing to verify and no device
    error). Retries up to `attempts` times, since the device may still be
    settling right after the pipeline was paused."""
    expect = set(expect_labels or ())
    for attempt in range(1, attempts + 1):
        paused = pause_face_pipeline_for_training()
        device_busy = False
        try:
            # Give HailoRT a moment to actually release the device after the
            # pipeline stopped, otherwise the trainer starts before it's free.
            time.sleep(1.5)
            proc = subprocess.run(
                [sys.executable, TRAINING_SCRIPT, "--mode", "train"],
                capture_output=True, text=True, cwd=FACE_RECON_DIR, timeout=600,
            )
            blob = (proc.stdout or "") + (proc.stderr or "")
            device_busy = ("HAILO_OUT_OF_PHYSICAL_DEVICES" in blob
                           or "not enough free devices" in blob)
            if device_busy:
                print(f"[DBAdmin] retrain attempt {attempt}: Hailo device was BUSY "
                      f"— nothing was enrolled (trainer rc={proc.returncode}).")
            elif proc.returncode != 0:
                # e.g. rc=-11/139 segfault. Not necessarily fatal (it may have
                # enrolled before dying), so fall through to DB verification.
                print(f"[DBAdmin] retrain attempt {attempt}: trainer exited "
                      f"abnormally (rc={proc.returncode}); verifying against the DB.")
        except Exception as e:
            print(f"[DBAdmin] retrain attempt {attempt} failed: {e}")
        finally:
            if paused:
                resume_face_pipeline()

        labels = _db_labels()
        if labels is None:
            print("[DBAdmin] retrain: could not verify (DB unreadable).")
        else:
            missing = expect - labels
            if not missing and not device_busy:
                print(f"[DBAdmin] retrain verified. DB labels: {sorted(labels)}")
                return True
            if missing:
                print(f"[DBAdmin] retrain attempt {attempt}: still missing {sorted(missing)}")
            elif not device_busy:
                return True
        if attempt < attempts:
            time.sleep(3.0)

    print("[DBAdmin] retrain FAILED after all attempts.")
    return False


def _guest_train_dirs():
    """train/ subfolders that are temp guest identities."""
    try:
        return sorted(d for d in os.listdir(TRAIN_DIR)
                      if os.path.isdir(os.path.join(TRAIN_DIR, d))
                      and d.startswith(f"{TEMP_NAME_PREFIX}_"))
    except OSError:
        return []


def reconcile_face_db_on_start(event_pub=None):
    """Startup housekeeping for the face DB. MUST run before the recognition
    pipeline claims the Hailo device, so the trainer can actually get it.

    1. Drop every leftover guest_* temp identity (images + DB record). An
       un-renamed guest is disposable noise, and a stale guest trained on a known
       person's face actively shadows that person.
    2. Re-enroll any NAMED person whose train/ folder exists but has no DB record.
       This self-heals the exact failure that erased 'Alex': a merge deleted
       the real record and the follow-up retrain silently no-op'd because the
       device was busy, leaving images with no embedding — so every sighting read
       "Unknown" forever.
    """
    print("[Reconcile] Checking face DB against train/ folders...")
    labels = _db_labels()
    if labels is None:
        print("[Reconcile] Skipped — DB unreadable.")
        return

    # ── 1. purge guests ────────────────────────────────────────────────
    if PURGE_GUESTS_ON_START:
        guests_on_disk = _guest_train_dirs()
        guest_labels = {l for l in labels if l.startswith(f"{TEMP_NAME_PREFIX}_")}
        for g in sorted(set(guests_on_disk) | guest_labels):
            if g in guest_labels:
                _run_db_admin("forget", g)
            d = os.path.join(TRAIN_DIR, g)
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
        if guests_on_disk or guest_labels:
            print(f"[Reconcile] Purged {len(set(guests_on_disk) | guest_labels)} "
                  f"leftover guest identity(ies): "
                  f"{sorted(set(guests_on_disk) | guest_labels)}")
            labels = _db_labels() or labels

    # ── 2. re-enroll named people missing from the DB ──────────────────
    named = _named_train_dirs()
    missing = [n for n in named if n not in labels]
    if not missing:
        print(f"[Reconcile] OK — all {len(named)} named people enrolled: {named}")
        return

    print(f"[Reconcile] Missing embeddings for {missing} — retraining "
          f"(recognition is paused while the Hailo device is used)...")
    if _retrain_all(expect_labels=missing):
        print(f"[Reconcile] Re-enrolled {missing}.")
        if event_pub:
            for n in missing:
                event_pub.publish_event("training_complete", {"person": n})
    else:
        print(f"[Reconcile] WARNING: could not re-enroll {missing}. They will "
              f"keep reading as Unknown until a retrain succeeds.")


# --- FACE EXIT CHECKER THREAD ---
def face_exit_checker(stop_event, tracker, event_pub):
    """Background thread to check for faces that have exited."""
    while not stop_event.is_set():
        exited_faces = tracker.check_exits()
        for track_id, name in exited_faces:
            if event_pub:
                event_pub.publish_event("face_lost", {
                    "track_id": track_id,
                    "name": name
                })
        time.sleep(0.5)  # Check every 500ms


def mjpeg_server_thread(stop_event):
    """Run Flask MJPEG server in a separate thread."""
    from flask import Flask, Response
    from werkzeug.serving import WSGIRequestHandler
    import logging

    # Suppress Flask logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    # Werkzeug's dev server speaks HTTP/1.0 by default, which means NO
    # keep-alive: every request opens and closes a TCP connection. The pull
    # viewer makes ~20 requests/second, so over the network that is 20
    # handshakes/second, each connection then sitting in TIME_WAIT, against a
    # browser limit of ~6 concurrent connections per host. The result is a few
    # seconds of smooth video followed by a multi-second stall while connection
    # slots free up — invisible on localhost, where handshakes are free.
    # HTTP/1.1 keeps one connection open and reuses it for every frame.
    WSGIRequestHandler.protocol_version = "HTTP/1.1"

    app = Flask(__name__)
    
    @app.route("/")
    def index():
        return """
        <html>
        <head>
            <title>Kiki Face Recognition Stream</title>
            <style>
                body { 
                    background: #1a1a2e; 
                    margin: 0; 
                    padding: 20px;
                    font-family: 'Segoe UI', sans-serif;
                    color: white;
                }
                .container {
                    max-width: 1280px;
                    margin: 0 auto;
                    text-align: center;
                }
                h1 {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-size: 2.5em;
                    margin-bottom: 20px;
                }
                img {
                    border-radius: 12px;
                    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
                    max-width: 100%;
                    height: auto;
                }
                .status {
                    margin-top: 15px;
                    padding: 10px 20px;
                    background: rgba(255,255,255,0.1);
                    border-radius: 25px;
                    display: inline-block;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 Kiki Face Recognition</h1>
                <img id="stream" alt="Live Stream" />
                <div class="status">
                    <span id="fps">-- fps</span> &nbsp;|&nbsp;
                    <span id="lat">-- ms</span> &nbsp;|&nbsp;
                    <a href="/mjpeg" style="color:#9ab">push stream</a>
                </div>
            </div>
            <script>
            // Pull, don't push. Each frame is requested only after the previous
            // one has finished decoding, so the browser paces itself and can
            // never build a backlog. An <img src="/mjpeg"> push stream drifts
            // further behind the longer it runs; this cannot.
            (function () {
                var img = document.getElementById('stream');
                var fpsEl = document.getElementById('fps');
                var latEl = document.getElementById('lat');
                var frames = 0, lastReport = performance.now(), inFlight = false;

                function nextFrame() {
                    if (inFlight) return;
                    inFlight = true;
                    var started = performance.now();
                    var probe = new Image();
                    probe.onload = function () {
                        // Swap only once fully decoded — avoids tearing.
                        img.src = probe.src;
                        inFlight = false;
                        frames++;
                        var now = performance.now();
                        latEl.textContent = Math.round(now - started) + ' ms';
                        if (now - lastReport >= 1000) {
                            fpsEl.textContent =
                                (frames * 1000 / (now - lastReport)).toFixed(1) + ' fps';
                            frames = 0; lastReport = now;
                        }
                        // setTimeout, not requestAnimationFrame: rAF is throttled
                        // hard (or paused entirely) when the tab is backgrounded
                        // or the browser is busy, which shows up as the stream
                        // freezing. A 0ms timeout keeps pacing driven by how fast
                        // frames actually arrive.
                        setTimeout(nextFrame, 0);
                    };
                    probe.onerror = function () {
                        inFlight = false;
                        setTimeout(nextFrame, 500);   // server restarting, back off
                    };
                    // Port FRAME_PORT, not this one: that server speaks
                    // HTTP/1.1 keep-alive so all frames share one connection.
                    probe.src = location.protocol + '//' + location.hostname +
                                ':__FRAME_PORT__/snapshot?t=' + Date.now();
                }
                nextFrame();
            })();
            </script>
        </body>
        </html>
        """.replace("__FRAME_PORT__", str(FRAME_PORT))

    def generate_mjpeg():
        """Yield each frame exactly once, blocking until a new one exists.

        Sending only new frames is what keeps the stream real-time: a slow
        client simply receives fewer frames instead of accumulating a backlog
        of stale ones.
        """
        last_seq = -1
        last_off_send = 0.0
        try:
            while not stop_event.is_set():
                if not pipeline_state.webcam_enabled:
                    # Resend the placeholder slowly rather than spinning.
                    now = time.time()
                    if now - last_off_send >= 1.0:
                        last_off_send = now
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n'
                               + mjpeg_frame_buffer.camera_off_jpeg + b'\r\n')
                    else:
                        time.sleep(0.1)
                    continue

                jpeg_bytes, seq = mjpeg_frame_buffer.wait_for_jpeg(last_seq, timeout=1.0)
                if jpeg_bytes is None or seq == last_seq:
                    continue    # timed out waiting; loop re-checks stop_event
                last_seq = seq
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
        except GeneratorExit:
            # Client disconnected — let the generator die instead of leaking it.
            return

    @app.route("/mjpeg")
    def mjpeg():
        resp = Response(
            generate_mjpeg(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
        # Proxies or the browser buffering this stream reintroduces exactly the
        # latency we just removed.
        resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['X-Accel-Buffering'] = 'no'
        return resp
    
    @app.route("/snapshot")
    def snapshot():
        """One current JPEG. The basis of the low-latency pull viewer.

        With an MJPEG push stream the server keeps writing whether or not the
        browser is keeping up; the surplus queues in the kernel socket buffer
        and inside the browser, so the displayed image falls further and
        further behind (measured server-side latency stays ~57 ms while the
        browser showed ~15 s). A pull model cannot accumulate: each request is
        answered with the newest frame, and a client that is slow simply asks
        less often.
        """
        from flask import Response as _Resp
        jpeg = mjpeg_frame_buffer.get_jpeg(pipeline_state.webcam_enabled)
        resp = _Resp(jpeg or b"", mimetype="image/jpeg")
        resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        resp.headers['Pragma'] = 'no-cache'
        return resp

    # Status endpoint for debugging
    @app.route("/status")
    def status():
        return pipeline_state.get_state_dict()

    @app.route("/latency")
    def latency():
        """Localise stream delay: upstream (GStreamer) vs downstream (encode/serve).

        Also reports every queue's fill level in the running pipeline — a queue
        pinned at its max is where frames are piling up.
        """
        stats = mjpeg_frame_buffer.get_latency_stats()

        queues = {}
        app_ref = pipeline_control.get("app")
        if app_ref is not None and getattr(app_ref, "pipeline", None):
            try:
                it = app_ref.pipeline.iterate_recurse()
                while True:
                    res, elem = it.next()
                    if res != Gst.IteratorResult.OK:
                        break
                    if elem.__class__.__name__ != "GstQueue":
                        continue
                    cur = elem.get_property("current-level-buffers")
                    mx = elem.get_property("max-size-buffers")
                    if cur > 0:
                        queues[elem.get_name()] = f"{cur}/{mx}"
            except Exception as e:
                queues["error"] = str(e)

        stats["nonempty_queues"] = queues
        stats["hint"] = ("age_at_encode_ms large => delay is UPSTREAM in GStreamer; "
                         "only age_at_send_ms large => delay is in the HTTP path")
        return stats
    
    mjpeg_frame_buffer.start_encoder()
    print(f"[MJPEG] Starting MJPEG server on http://0.0.0.0:{MJPEG_PORT}/mjpeg "
          f"({MJPEG_STREAM_WIDTH}px, q{MJPEG_QUALITY}, max {MJPEG_MAX_FPS} fps)")
    app.run(host='0.0.0.0', port=MJPEG_PORT, threaded=True, use_reloader=False)


def frame_server_thread(stop_event):
    """Serve video frames over a keep-alive HTTP/1.1 connection.

    Flask is not used for this. Werkzeug's development server hardcodes

        # Always close the connection. This disables HTTP/1.1
        self.send_header("Connection", "close")

    so every frame request costs a fresh TCP connection. At ~20 frames/second
    over WiFi that is 20 handshakes/second plus a pile of sockets in TIME_WAIT,
    against the browser's ~6-connections-per-host limit — which shows up as a
    couple of smooth seconds followed by a multi-second freeze, repeating.

    The stdlib ThreadingHTTPServer does support keep-alive, so the viewer holds
    one connection open and reuses it for every frame. Flask keeps serving the
    control endpoints on MJPEG_PORT; only the frame path moves here.
    """
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

    class FrameHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"     # the whole point: persistent connections

        def log_message(self, fmt, *args):
            pass                          # one request per frame; do not log

        def _send_jpeg(self, payload):
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path not in ("/snapshot", "/frame"):
                self.send_error(404)
                return
            try:
                jpeg = mjpeg_frame_buffer.get_jpeg(pipeline_state.webcam_enabled)
                self._send_jpeg(jpeg or b"")
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

    try:
        server = ThreadingHTTPServer(("0.0.0.0", FRAME_PORT), FrameHandler)
        server.daemon_threads = True
        print(f"[Frames] Keep-alive frame server on http://0.0.0.0:{FRAME_PORT}/snapshot")
        server.serve_forever(poll_interval=0.5)
    except Exception as e:
        print(f"[Frames] Frame server error: {e}")

# --- FACE BBOX SANITIZER ---
# Guards a crash inside TAPPAS, confirmed by gdb:
#
#   Thread "cropper_wrapper" received signal SIGABRT
#   #9  cv::resize()
#   #10 resize_letterbox_rgb()                 libgsthailotools.so
#   #11 resize_letterbox()                     libgsthailotools.so
#   #12 gst_hailocropper_resize_by_method(...) libgsthailotools.so
#
# hailocropper crops each face detection and letterbox-resizes it for the
# recognition model. If a detection's box has (near) zero area once clamped to
# the frame, the crop is an empty cv::Mat and cv::resize asserts. The exception
# is thrown on the element's own streaming thread with no handler, so it becomes
# std::terminate -> SIGABRT and takes the whole process down — there is nothing
# to catch in Python.
#
# The tracker is what makes this reachable: with keep_lost_frames/keep_past_metadata
# it emits Kalman-predicted boxes for faces that have left the frame, and those
# predictions drift outside the image. Hence the crash correlating with
# crowd_detected / face_lost. This callback runs between the tracker and the
# cropper and drops any box that cannot produce a valid crop.
MIN_FACE_BOX_NORM = 0.015   # ~19x11 px at 1280x720; below this is useless for recognition anyway


def face_sanitize_callback(element, buffer, app):
    if buffer is None:
        return
    try:
        roi = hailo.get_roi_from_buffer(buffer)
    except Exception:
        return
    if roi is None:
        return

    removed = 0
    for detection in roi.get_objects_typed(hailo.HAILO_DETECTION):
        bbox = detection.get_bbox()
        try:
            xmin, ymin = bbox.xmin(), bbox.ymin()
            w, h = bbox.width(), bbox.height()
        except Exception:
            roi.remove_object(detection)
            removed += 1
            continue

        bad = False
        # NaN/inf from a diverged Kalman prediction.
        if not all(math.isfinite(v) for v in (xmin, ymin, w, h)):
            bad = True
        elif w <= 0 or h <= 0:
            bad = True
        else:
            # Area actually inside the frame after clamping — this is what the
            # cropper will end up with.
            cx0, cy0 = max(0.0, xmin), max(0.0, ymin)
            cx1, cy1 = min(1.0, xmin + w), min(1.0, ymin + h)
            if (cx1 - cx0) < MIN_FACE_BOX_NORM or (cy1 - cy0) < MIN_FACE_BOX_NORM:
                bad = True

        if bad:
            roi.remove_object(detection)
            removed += 1

    if removed:
        face_sanitize_callback.total = getattr(face_sanitize_callback, "total", 0) + removed
        if face_sanitize_callback.total % 25 == 1:
            print(f"[Sanitize] Dropped {face_sanitize_callback.total} degenerate "
                  f"face box(es) that would crash hailocropper")


# --- POSE ESTIMATION CALLBACK ---
def pose_app_callback(element, buffer, app):
    """Handoff callback for the pose branch.

    Runs on the pose branch's own streaming thread — independent of the face
    callback's thread — so everything it touches must be thread-safe. Kept
    deliberately cheap: parse metadata into pose_state and return.
    """
    if buffer is None:
        return

    try:
        roi = hailo.get_roi_from_buffer(buffer)
    except Exception:
        return

    persons = []
    for detection in roi.get_objects_typed(hailo.HAILO_DETECTION):
        if detection.get_label() != "person":
            continue

        bbox = detection.get_bbox()

        track_id = 0
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()

        keypoints = {}
        landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if landmarks:
            points = landmarks[0].get_points()
            for name, index in POSE_KEYPOINTS.items():
                if index >= len(points):
                    continue
                point = points[index]
                # Landmark coords are normalized to the bbox, not the frame —
                # rescale so everything downstream is in frame-normalized 0..1.
                keypoints[name] = (
                    point.x() * bbox.width() + bbox.xmin(),
                    point.y() * bbox.height() + bbox.ymin(),
                )

        persons.append({
            "track_id": track_id,
            "confidence": detection.get_confidence(),
            "bbox": (bbox.xmin(), bbox.ymin(), bbox.width(), bbox.height()),
            "keypoints": keypoints,
        })

    pose_state.update(persons)

    # Hand off to the recognizer. This is a remap plus a deque append per
    # person — deliberately the only recognition work on the streaming thread.
    if pose_recognizer is not None:
        pose_recognizer.submit(persons)

    if getattr(app, "show_fps", False):
        pose_app_callback.counter = getattr(pose_app_callback, "counter", 0) + 1
        if pose_app_callback.counter % 30 == 0:
            summary = ""
            if pose_recognition_state is not None:
                labels = [f"{r['track_id']}:{r['label']}"
                          for r in pose_recognition_state.get().values()
                          if r.get("label") and r["label"] != "—"]
                if labels:
                    summary = " | " + ", ".join(labels)
            print(f"[Pose] fps: {pose_state.fps:.1f} | "
                  f"persons: {len(persons)}{summary}")


# --- CLIP CALLBACK ---
def clip_app_callback(element, buffer, app):
    """Handoff callback for the CLIP stage.

    Pulls the CLIP image embedding (a HAILO_MATRIX) off each person detection,
    matches it against the loaded text embeddings, and attaches the winning
    label as a HAILO_CLASSIFICATION so hailooverlay renders it on the frame.
    Mirrors GStreamerClipApp.matching_identity_callback, minus the GTK GUI.
    """
    if buffer is None:
        return

    matcher = clip_text_matcher
    try:
        roi = hailo.get_roi_from_buffer(buffer)
    except Exception:
        return
    if roi is None:
        return

    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Face detections carry a 512-d arcface embedding in the same HAILO_MATRIX
    # slot that CLIP uses for its 640-d image embedding, so filtering by label
    # alone is not enough — stacking the two shapes raises in np.vstack.
    embeddings = None
    used = []
    expected_dim = None
    for detection in detections:
        if detection.get_label() != "person":
            continue
        results = detection.get_objects_typed(hailo.HAILO_MATRIX)
        if len(results) == 0:
            continue
        embedding = np.array(results[0].get_data()).flatten()
        if expected_dim is None:
            expected_dim = embedding.shape[0]
        elif embedding.shape[0] != expected_dim:
            continue
        used.append(detection)
        embeddings = (embedding[np.newaxis, :] if embeddings is None
                      else np.vstack((embeddings, embedding)))

    if embeddings is None or matcher is None:
        return

    # Guard against a prompt file whose embeddings were made with a different
    # CLIP variant than the HEF (RN50x4 = 640-d).
    entries = matcher.get_embeddings()
    if entries and matcher.entries[entries[0]].embedding.shape[0] != embeddings.shape[1]:
        if not getattr(clip_app_callback, "warned_dim", False):
            print(f"[CLIP] Prompt embeddings are "
                  f"{matcher.entries[entries[0]].embedding.shape[0]}-d but the model "
                  f"produces {embeddings.shape[1]}-d — regenerate clip_prompts.json "
                  f"with the matching CLIP variant. Skipping matching.")
            clip_app_callback.warned_dim = True
        return

    matches = matcher.match(embeddings)
    results = []
    for match in matches:
        if match.negative or not match.passed_threshold:
            continue
        detection = used[match.row_idx]

        # Replace any previous CLIP label on this detection.
        for old in detection.get_objects_typed(hailo.HAILO_CLASSIFICATION):
            if old.get_classification_type() == 'clip':
                detection.remove_object(old)
        detection.add_object(
            hailo.HailoClassification('clip', match.text, match.similarity)
        )

        track_id = 0
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()
        results.append((track_id, match.text, match.similarity))

    clip_state.update(results)

    if results and getattr(app, "show_fps", False):
        clip_app_callback.counter = getattr(clip_app_callback, "counter", 0) + 1
        if clip_app_callback.counter % 15 == 0:
            summary = ", ".join(f"#{t}:{txt} {s:.2f}" for t, txt, s in results)
            print(f"[CLIP] fps: {clip_state.fps:.1f} | {summary}")


# --- CUSTOM FACE RECOGNITION APP WITH MJPEG STREAMING ---
class MJPEGStreamingFaceRecognitionApp(GStreamerFaceRecognitionApp):
    """Face Recognition App with MJPEG streaming and a parallel pose branch.

    Face recognition (scrfd_10g + arcface_mobilefacenet) and pose estimation
    (yolov8s_pose) run concurrently on the single Hailo-8: the source is tee'd,
    the face chain is left exactly as it was, and pose hangs off a second branch
    that terminates in its own fakesink. Keeping pose out of the face chain
    avoids feeding "person" boxes into the face tracker and lets the pose branch
    run at its own frame rate.
    """

    def __init__(self, app_callback, user_data, parser=None):
        # These must exist before super().__init__(), which calls
        # create_pipeline() -> get_pipeline_string() before it returns.
        mode = VISION_MODE["value"]
        self.pose_enabled = bool(POSE_OPTIONS.get("enabled", True)) and "pose" in mode
        self.clip_enabled = "clip" in mode
        self.pose_mode = POSE_OPTIONS.get("mode", "overlay")
        self.pose_hef_path = None
        self.pose_post_process_so = None
        self.pose_frame_rate = None
        self.clip_hef_path = None
        self.clip_detection_hef_path = None
        # In pose+clip the pose model already supplies "person" boxes, so CLIP
        # crops those instead of running a second detector — saves a whole model.
        self.clip_use_own_detector = self.clip_enabled and not self.pose_enabled
        super().__init__(app_callback, user_data, parser)

    def _ensure_pose_setup(self):
        """Resolve pose resources. Deferred until pipeline-build time because it
        needs self.arch / self.frame_rate, which the parent __init__ sets."""
        if not self.pose_enabled or self.pose_hef_path:
            return

        try:
            self.pose_hef_path = resolve_hef_path(
                POSE_OPTIONS.get("hef") or POSE_HEF_NAME,
                app_name='pose_estimation',
                arch=self.arch,
            )
            self.pose_post_process_so = get_resource_path(
                'pose_estimation', 'so', self.arch, POSE_POSTPROCESS_SO_NAME
            )
            print(f"[Pose] HEF: {self.pose_hef_path}")
            print(f"[Pose] Post-process: {self.pose_post_process_so}")
        except Exception as e:
            print(f"[Pose] Could not resolve pose resources ({e}) — pose branch disabled")
            self.pose_enabled = False
            return

        # Pose branch frame rate: default to the source rate (no decimation).
        pose_fps = POSE_OPTIONS.get("fps")
        self.pose_frame_rate = pose_fps if pose_fps else self.frame_rate
        if self.pose_mode == 'overlay':
            print("[Pose] Mode: overlay — skeletons drawn by hailooverlay on the video stream")
            if pose_fps:
                print("[Pose] Note: --pose-fps only applies to --pose-mode parallel; ignoring")
            self.pose_frame_rate = self.frame_rate
        else:
            print("[Pose] Mode: parallel — isolated branch, no skeleton overlay")
            if self.pose_frame_rate < self.frame_rate:
                print(f"[Pose] Decimating pose branch to {self.pose_frame_rate} fps "
                      f"(source is {self.frame_rate} fps)")

    def _ensure_clip_setup(self):
        """Resolve CLIP resources. Deferred for the same reason as pose setup."""
        if not self.clip_enabled or self.clip_hef_path:
            return

        try:
            from hailo_apps.python.core.common.core import resolve_hef_paths
            # resources_config.yaml orders clip's models [encoder, detector].
            models = resolve_hef_paths(hef_paths=None, app_name='clip', arch=self.arch)
            self.clip_hef_path = models[0].path
            self.clip_detection_hef_path = models[1].path

            self.clip_post_process_so = get_resource_path(
                None, 'so', self.arch, CLIP_POSTPROCESS_SO_NAME)
            self.clip_cropper_so = get_resource_path(
                None, 'so', self.arch, CLIP_CROPPER_SO_NAME)
            self.clip_detection_post_process_so = get_resource_path(
                None, 'so', self.arch, CLIP_DETECTION_POSTPROCESS_SO_NAME)

            self.clip_labels_json = None
            if self.clip_use_own_detector:
                from hailo_apps.python.core.common.hef_utils import get_hef_labels_json
                self.clip_labels_json = get_hef_labels_json(self.clip_detection_hef_path)

            print(f"[CLIP] Encoder HEF: {self.clip_hef_path}")
            if self.clip_use_own_detector:
                print(f"[CLIP] Detector HEF: {self.clip_detection_hef_path}")
            else:
                print("[CLIP] Reusing pose person detections (no second detector)")
        except Exception as e:
            print(f"[CLIP] Could not resolve CLIP resources ({e}) — CLIP disabled")
            self.clip_enabled = False
            return

        if init_clip_text_matcher() is None:
            print("[CLIP] Warning: running without text prompts — no labels will be produced")

    def _clip_overlay_stage(self):
        """CLIP spliced into the main chain, upstream of hailooverlay.

        Person boxes come either from CLIP's own small detector, or (in
        pose+clip) from the pose model that already ran. Each person crop is
        embedded by the CLIP encoder; clip_app_callback then attaches the
        matching text label, which hailooverlay renders.
        """
        clip_inference = INFERENCE_PIPELINE(
            hef_path=self.clip_hef_path,
            post_process_so=self.clip_post_process_so,
            post_function_name=CLIP_POSTPROCESS_FUNCTION,
            batch_size=CLIP_ENCODER_BATCH_SIZE,
            name='clip_inference',
            scheduler_priority=CLIP_SCHEDULER_PRIORITY,
            scheduler_timeout_ms=CLIP_SCHEDULER_TIMEOUT_MS,
        )

        detector_stage = ''
        if self.clip_use_own_detector:
            clip_detection = INFERENCE_PIPELINE(
                hef_path=self.clip_detection_hef_path,
                post_process_so=self.clip_detection_post_process_so,
                post_function_name=CLIP_DETECTION_POSTPROCESS_FUNCTION,
                batch_size=CLIP_DETECTION_BATCH_SIZE,
                config_json=self.clip_labels_json,
                name='clip_detection_inference',
                scheduler_priority=POSE_SCHEDULER_PRIORITY,
                scheduler_timeout_ms=POSE_SCHEDULER_TIMEOUT_MS,
            )
            detector_stage = (
                f"{INFERENCE_PIPELINE_WRAPPER(clip_detection, name='clip_detection_wrapper', bypass_max_size_buffers=PIPELINE_BYPASS_BUFFERS)} ! "
                f"{TRACKER_PIPELINE(class_id=CLIP_PERSON_CLASS_ID, keep_past_metadata=True, name='hailo_clip_tracker')} ! "
            )

        clip_cropper = CROPPER_PIPELINE(
            inner_pipeline=clip_inference,
            so_path=self.clip_cropper_so,
            function_name=CLIP_CROPPER_FUNCTION,
            name='kiki_clip_cropper',
            bypass_max_size_buffers=PIPELINE_BYPASS_BUFFERS,
        )

        return (
            f"{QUEUE(name='clip_overlay_in_q')} ! "
            f"{detector_stage}"
            f"{clip_cropper} ! "
            f"identity name=clip_callback"
        )

    def _pose_inference_stage(self):
        """The pose inference + tracker stages, shared by both pose modes.

        Element names all carry a `pose_` prefix — the face chain uses the
        helper defaults (`inference*`, `inference_wrapper*`), and duplicate
        element names are the most common way Gst.parse_launch fails to link.
        """
        pose_inference = INFERENCE_PIPELINE(
            hef_path=self.pose_hef_path,
            post_process_so=self.pose_post_process_so,
            post_function_name=POSE_POSTPROCESS_FUNCTION,
            batch_size=POSE_BATCH_SIZE,
            name='pose_inference',
            scheduler_priority=POSE_SCHEDULER_PRIORITY,
            scheduler_timeout_ms=POSE_SCHEDULER_TIMEOUT_MS,
        )
        # The wrapper's hailocropper does the letterboxing that
        # filter_letterbox expects, and preserves the original frame.
        pose_inference_wrapper = INFERENCE_PIPELINE_WRAPPER(
            pose_inference, name='pose_inference_wrapper',
            bypass_max_size_buffers=PIPELINE_BYPASS_BUFFERS,
        )
        pose_tracker = TRACKER_PIPELINE(class_id=0, name='hailo_pose_tracker')

        return (
            f"{pose_inference_wrapper} ! "
            f"{pose_tracker} ! "
            f"identity name=pose_callback"
        )

    def _pose_overlay_stage(self):
        """Pose spliced into the main chain, upstream of hailooverlay.

        Placed *after* the face user-callback so app_callback still sees only
        face metadata, and no tracker downstream ever sees the "person" boxes —
        the face tracker's associations are untouched. hailooverlay then draws
        the skeletons onto the frame that feeds both the display window and the
        MJPEG appsink.
        """
        return f"{QUEUE(name='pose_overlay_in_q')} ! {self._pose_inference_stage()}"

    def _pose_branch_pipeline(self):
        """Pose as an isolated branch off the source tee (no overlay).

        Cannot slow the face chain down, and supports --pose-fps decimation,
        but its metadata never reaches hailooverlay.
        """
        # Only pay for videorate when actually decimating.
        rate_stage = ""
        if self.pose_frame_rate < self.frame_rate:
            rate_stage = (
                f"videorate name=pose_videorate drop-only=true ! "
                f"video/x-raw,framerate={int(self.pose_frame_rate)}/1 ! "
            )

        # The leaky queue right after the tee is essential: a tee stalls *all*
        # of its branches if any one of them blocks, so the pose branch must
        # never be able to back-pressure the face branch.
        return (
            f"{QUEUE(name='pose_branch_q', max_size_buffers=2, leaky='downstream')} ! "
            f"{rate_stage}"
            f"{self._pose_inference_stage()} ! "
            f"{QUEUE(name='pose_sink_q', max_size_buffers=2, leaky='downstream')} ! "
            f"fakesink name=pose_fakesink sync=false async=false"
        )

    def get_pipeline_string(self):
        """Override to add tee for MJPEG streaming after hailooverlay."""
        # Get the original pipeline string from parent
        from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
            SOURCE_PIPELINE, INFERENCE_PIPELINE, INFERENCE_PIPELINE_WRAPPER,
            TRACKER_PIPELINE, USER_CALLBACK_PIPELINE, CROPPER_PIPELINE
        )
        from hailo_apps.python.core.common.core import get_resource_path
        from hailo_apps.python.core.common.defines import RESOURCES_JSON_DIR_NAME, FACE_DETECTION_JSON_NAME
        
        # Build source pipeline
        source_pipeline = SOURCE_PIPELINE(
            self.video_source, self.video_width, self.video_height,
            frame_rate=self.frame_rate, sync=self.sync
        )
        
        # Build detection pipeline
        detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path_detection,
            post_process_so=self.post_process_so_scrfd,
            post_function_name=self.detection_func,
            batch_size=self.batch_size,
            config_json=get_resource_path(
                pipeline_name=None,
                resource_type=RESOURCES_JSON_DIR_NAME,
                arch=self.arch,
                model=FACE_DETECTION_JSON_NAME
            )
        )
        detection_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(
            detection_pipeline, bypass_max_size_buffers=PIPELINE_BYPASS_BUFFERS)
        
        # Build tracker pipeline
        tracker_pipeline = TRACKER_PIPELINE(
            class_id=-1, kalman_dist_thr=0.7, iou_thr=0.8, init_iou_thr=0.9,
            keep_new_frames=2, keep_tracked_frames=6, keep_lost_frames=8,
            keep_past_metadata=True, name='hailo_face_tracker'
        )
        
        # Build face recognition pipeline (cropper with recognition)
        mobile_facenet_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path_recognition,
            post_process_so=self.post_process_so_face_recognition,
            post_function_name=self.recognition_func,
            batch_size=self.batch_size,
            config_json=None,
            name='face_recognition_inference'
        )
        cropper_pipeline = CROPPER_PIPELINE(
            inner_pipeline=(
                f'hailofilter so-path={self.post_process_so_face_align} '
                f'name=face_align_hailofilter use-gst-buffer=true qos=false ! '
                f'{QUEUE(name="detector_pos_face_align_q")} ! '
                f'{mobile_facenet_pipeline}'
            ),
            so_path=self.post_process_so_cropper,
            function_name=self.cropper_func,
            internal_offset=True,
            bypass_max_size_buffers=PIPELINE_BYPASS_BUFFERS
        )
        
        # Vector DB callback pipeline
        vector_db_callback_pipeline = USER_CALLBACK_PIPELINE(name=self.vector_db_callback_name)
        
        # User callback pipeline
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        
        # Custom display pipeline with tee for MJPEG streaming
        # After hailooverlay, we split: one branch to display, one to appsink for MJPEG
        display_with_mjpeg = (
            f"{OVERLAY_PIPELINE(name='hailo_display_overlay')} ! "
            f"{QUEUE(name='hailo_display_videoconvert_q')} ! "
            f"videoconvert name=hailo_display_videoconvert n-threads=2 qos=false ! "
            f"tee name=display_tee ! "
            # Branch 1: Display
            f"{QUEUE(name='display_branch_q')} ! "
            f"fpsdisplaysink name=hailo_display video-sink={self.video_sink} sync={self.sync} text-overlay={self.show_fps} signal-fps-measurements=true "
            # Branch 2: MJPEG appsink
            f"display_tee. ! {QUEUE(name='mjpeg_branch_q')} ! "
            f"videoconvert name=mjpeg_convert ! video/x-raw,format=BGR ! "
            f"appsink name=mjpeg_sink emit-signals=true drop=true max-buffers=2 sync=false"
        )
        
        self._ensure_pose_setup()
        self._ensure_clip_setup()

        # Pose and CLIP in overlay mode sit between the face callback and
        # hailooverlay. Order matters: pose runs first so that in pose+clip the
        # CLIP cropper has person boxes to crop.
        pose_overlay_stage = ''
        if self.pose_enabled and self.pose_mode == 'overlay':
            pose_overlay_stage = f'{self._pose_overlay_stage()} ! '

        clip_overlay_stage = ''
        if self.clip_enabled:
            clip_overlay_stage = f'{self._clip_overlay_stage()} ! '

        # Sits between the tracker and the cropper: strips boxes that would make
        # hailocropper hand an empty cv::Mat to cv::resize and abort the process.
        face_sanitizer_pipeline = USER_CALLBACK_PIPELINE(name='face_sanitizer')

        face_chain = (
            f'{detection_pipeline_wrapper} ! '
            f'{tracker_pipeline} ! '
            f'{face_sanitizer_pipeline} ! '
            f'{cropper_pipeline} ! '
            f'{vector_db_callback_pipeline} ! '
            f'{user_callback_pipeline} ! '
            f'{pose_overlay_stage}'
            f'{clip_overlay_stage}'
            f'{display_with_mjpeg}'
        )

        # Leak at the source so the pipeline can never fall far behind: if the
        # inference stages are momentarily slower than the camera, old frames
        # are discarded rather than queued. Without this, queues fill once and
        # stay full, which is fixed latency with no benefit for live use.
        if SOURCE_LEAKY:
            source_pipeline = (
                f'{source_pipeline} ! '
                f'{QUEUE(name="source_leaky_q", max_size_buffers=SOURCE_QUEUE_BUFFERS, leaky="downstream")}'
            )

        if not self.pose_enabled or self.pose_mode == 'overlay':
            # With pose disabled this is byte-identical to the pre-pose
            # pipeline, so --no-pose doubles as the regression control.
            return f'{source_pipeline} ! {face_chain}'

        return (
            f'{source_pipeline} ! tee name=kiki_src_tee '
            f'kiki_src_tee. ! {QUEUE(name="face_branch_q")} ! {face_chain} '
            f'kiki_src_tee. ! {self._pose_branch_pipeline()}'
        )
    
    def run(self):
        """
        Override run to:
        1. Connect MJPEG appsink callback
        2. Store references for external control
        3. NOT call sys.exit() so we can restart the pipeline
        """
        global pipeline_control
        
        # Connect MJPEG appsink before running
        mjpeg_sink = self.pipeline.get_by_name("mjpeg_sink")
        if mjpeg_sink:
            mjpeg_sink.connect("new-sample", self._on_mjpeg_sample)
            print("[MJPEG] Connected appsink for streaming")
        else:
            print("[MJPEG] Warning: mjpeg_sink not found in pipeline")

        # Must be connected before the pipeline plays — it is what keeps
        # hailocropper from aborting on a degenerate face box.
        sanitizer = self.pipeline.get_by_name("face_sanitizer")
        if sanitizer:
            sanitizer.set_property("signal-handoffs", True)
            sanitizer.connect("handoff", face_sanitize_callback, self)
            print("[Sanitize] Face bbox guard active (protects hailocropper)")
        else:
            print("[Sanitize] WARNING: face_sanitizer element not found — "
                  "hailocropper may abort on a degenerate face box")

        # GStreamerApp._connect_callback() only binds the element literally named
        # "identity_callback", so the pose branch's callback is wired by hand.
        if self.pose_enabled:
            pose_state.clear()
            pose_identity = self.pipeline.get_by_name("pose_callback")
            if pose_identity:
                pose_identity.set_property("signal-handoffs", True)
                pose_identity.connect("handoff", pose_app_callback, self)
                print("[Pose] Connected pose callback")
            else:
                print("[Pose] Warning: pose_callback element not found in pipeline")

        if self.clip_enabled:
            clip_state.clear()
            clip_identity = self.pipeline.get_by_name("clip_callback")
            if clip_identity:
                clip_identity.set_property("signal-handoffs", True)
                clip_identity.connect("handoff", clip_app_callback, self)
                print("[CLIP] Connected CLIP callback")
            else:
                print("[CLIP] Warning: clip_callback element not found in pipeline")

        # Store reference for external control
        pipeline_control["app"] = self
        pipeline_control["loop"] = self.loop
        pipeline_control["running"] = True
        print(f"[Pipeline] Stored pipeline control references (loop={self.loop})")
        
        # === Custom run implementation (based on GStreamerApp.run but without sys.exit) ===
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)
        
        self._connect_callback()
        
        # Disable QoS 
        from hailo_apps.python.core.gstreamer.gstreamer_common import disable_qos
        disable_qos(self.pipeline)
        
        # Start pipeline
        self.pipeline.set_state(Gst.State.PAUSED)
        self.pipeline.set_latency(self.pipeline_latency * Gst.MSECOND)
        self.pipeline.set_state(Gst.State.PLAYING)
        print("[Pipeline] Pipeline set to PLAYING")
        
        # Run the main loop (blocks until loop.quit() or error)
        try:
            self.loop.run()
        except Exception as e:
            print(f"[Pipeline] Main loop error: {e}")
        
        print("[Pipeline] Main loop exited")
        
        # Cleanup (but DON'T call sys.exit!)
        try:
            self.user_data.running = False
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
                bus = self.pipeline.get_bus()
                if bus:
                    bus.remove_signal_watch()
        except Exception as e:
            print(f"[Pipeline] Cleanup error: {e}")
        
        pipeline_control["running"] = False
        pipeline_control["loop"] = None
        pipeline_control["app"] = None
        print("[Pipeline] run() completed - ready for restart")
    
    def stop_pipeline(self):
        """Stop the GStreamer pipeline and free resources."""
        print("[Pipeline] Stopping pipeline...")
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            print("[Pipeline] Pipeline set to NULL state")
    
    def _on_mjpeg_sample(self, sink):
        """Callback when a new sample is available for MJPEG streaming."""
        sample = sink.emit("pull-sample")
        if sample:
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            
            # Get frame dimensions
            structure = caps.get_structure(0)
            width = structure.get_value("width")
            height = structure.get_value("height")
            
            # Extract frame data
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                # Convert to numpy array (BGR format from videoconvert)
                frame = np.ndarray(
                    shape=(height, width, 3),
                    dtype=np.uint8,
                    buffer=map_info.data
                )
                # This numpy array is a VIEW over the GStreamer buffer, which is
                # unmapped and returned to the pool the moment we leave this
                # callback. Encoding now happens on another thread, so the view
                # must be copied — otherwise the encoder reads recycled memory
                # and serves torn or stale frames.
                hands = (hand_state.get_hands()
                         if (HAND_OPTIONS.get("enabled") and hand_state is not None)
                         else None)
                if hands:
                    # draw_hands writes into the array, so copy first regardless.
                    frame = draw_hands(frame.copy(), hands)
                else:
                    frame = frame.copy()

                if (POSE_RECOGNITION_OPTIONS.get("enabled")
                        and POSE_RECOGNITION_OPTIONS.get("overlay")
                        and pose_recognition_state is not None):
                    results = pose_recognition_state.get()
                    if results:
                        frame = draw_pose_labels(
                            frame, pose_state.get_persons(), results)

                # Stamp capture time so latency can be measured end to end.
                mjpeg_frame_buffer.set_frame(frame, captured_at=time.time())
                buffer.unmap(map_info)
        
        return Gst.FlowReturn.OK


# --- BODY DETECTION PIPELINE WITH MJPEG STREAMING ---
class MJPEGStreamingBodyDetectionApp(GStreamerDetectionSimpleApp):
    """Extended Body Detection App with MJPEG streaming capability for full body mode."""
    
    def run(self):
        """Override run to connect MJPEG appsink and store references."""
        global pipeline_control
        
        # Store reference for external control
        pipeline_control["app"] = self
        pipeline_control["loop"] = self.loop
        pipeline_control["running"] = True
        print("[BodyPipeline] Stored pipeline control references")
        
        # === Custom run implementation ===
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)
        
        self._connect_callback()
        
        # Disable QoS
        from hailo_apps.python.core.gstreamer.gstreamer_common import disable_qos
        disable_qos(self.pipeline)
        
        # Start pipeline
        self.pipeline.set_state(Gst.State.PAUSED)
        self.pipeline.set_latency(self.pipeline_latency * Gst.MSECOND)
        self.pipeline.set_state(Gst.State.PLAYING)
        print("[BodyPipeline] Pipeline set to PLAYING")
        
        # Run the main loop
        try:
            self.loop.run()
        except Exception as e:
            print(f"[BodyPipeline] Main loop error: {e}")
        
        print("[BodyPipeline] Main loop exited")
        
        # Cleanup
        try:
            self.user_data.running = False
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
                bus = self.pipeline.get_bus()
                if bus:
                    bus.remove_signal_watch()
        except Exception as e:
            print(f"[BodyPipeline] Cleanup error: {e}")
        
        pipeline_control["running"] = False
        pipeline_control["loop"] = None
        pipeline_control["app"] = None
        print("[BodyPipeline] run() completed - ready for restart")


# --- BODY DETECTION CALLBACK ---
class BodyFollowerCallback(app_callback_class):
    """Callback data for body detection mode."""
    def __init__(self, cmd_queue):
        super().__init__()
        self.cmd_queue = cmd_queue
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0.0
        
        # Search state variables
        self.last_seen_time = time.time()
        self.last_direction = "LEFT"
        
        # Turn delay variables
        self.turn_wait_start = 0.0
        self.last_raw_turn = "CENTER"
        
        # Continuous Forward Movement tracking
        self.forward_start_time = 0.0
        self.is_moving_forward = False


def body_detection_callback(element, buffer, user_data):
    """Callback for body detection - controls robot chassis motors."""
    global mjpeg_frame_buffer
    
    user_data.frame_count += 1
    current_time = time.time()
    if current_time - user_data.last_time >= 1:
        user_data.fps = user_data.frame_count / (current_time - user_data.last_time)
        user_data.frame_count = 0
        user_data.last_time = current_time
    
    if buffer is None:
        return
    
    # Get Video Frame
    pad = element.get_static_pad("src")
    format_type, width, height = get_caps_from_pad(pad)
    
    frame = None
    if user_data.use_frame and format_type is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format_type, width, height)
    
    # Get ROI and Detections
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    # Filter for 'person' with confidence > 60%
    person_detections = []
    for detection in detections:
        if detection.get_label() == "person" and detection.get_confidence() > 0.6:
            person_detections.append(detection)
        else:
            roi.remove_object(detection)
    
    action = "STOP"
    log_info = ""
    current_dead_zone = BODY_DEAD_ZONE_MAX
    
    if person_detections:
        # Found a person
        user_data.last_seen_time = current_time
        
        # Find best target (Largest width -> Closest)
        target = max(person_detections, key=lambda d: d.get_bbox().width())
        bbox = target.get_bbox()
        
        # Normalized to pixels
        cx = (bbox.xmin() + bbox.xmax()) / 2.0
        width_norm = bbox.width()
        
        cx_px = cx * FRAME_WIDTH
        width_px = width_norm * FRAME_WIDTH
        
        # Update last direction for search
        if cx_px < CENTER_X:
            user_data.last_direction = "LEFT"
        else:
            user_data.last_direction = "RIGHT"
        
        # --- DYNAMIC DEAD ZONE CALCULATION ---
        denom = max(1.0, RANGE_WIDTH_CLOSE - RANGE_WIDTH_FAR)
        w_ratio = (width_px - RANGE_WIDTH_FAR) / denom
        w_ratio = max(0.0, min(1.0, w_ratio))
        
        current_dead_zone = int(BODY_DEAD_ZONE_MIN + (w_ratio * (BODY_DEAD_ZONE_MAX - BODY_DEAD_ZONE_MIN)))
        current_turn_speed = int(SPEED_TURN_MIN + (w_ratio * (SPEED_TURN_MAX - SPEED_TURN_MIN)))
        
        # --- CONTROL LOGIC ---
        raw_turn = "CENTER"
        move = "STOP"
        
        # Steering
        if cx_px < (CENTER_X - current_dead_zone):
            raw_turn = "RIGHT"
        elif cx_px > (CENTER_X + current_dead_zone):
            raw_turn = "LEFT"
        
        # Turn delay logic
        if raw_turn != user_data.last_raw_turn:
            user_data.turn_wait_start = current_time
            user_data.last_raw_turn = raw_turn
        
        turn = "CENTER"
        if raw_turn == "CENTER":
            turn = "CENTER"
        else:
            if (current_time - user_data.turn_wait_start) >= TURN_DELAY:
                turn = raw_turn
            else:
                turn = "CENTER"
        
        # Distance (Throttle)
        if width_px < TARGET_WIDTH_MIN:
            move = "FORWARD"
        elif width_px > TARGET_WIDTH_MAX:
            move = "BACKWARD"
        
        # Forward movement limit
        if move == "FORWARD":
            if not user_data.is_moving_forward:
                user_data.is_moving_forward = True
                user_data.forward_start_time = current_time
            elif (current_time - user_data.forward_start_time) > 0.5:
                move = "STOP"
                if turn == "CENTER":
                    user_data.is_moving_forward = False
        else:
            user_data.is_moving_forward = False
        
        # Combine movement and turn
        if move == "FORWARD":
            if turn == "LEFT": action = "FORWARD_LEFT"
            elif turn == "RIGHT": action = "FORWARD_RIGHT"
            else: action = "FORWARD"
        elif move == "BACKWARD":
            if turn == "LEFT": action = "BACKWARD_LEFT"
            elif turn == "RIGHT": action = "BACKWARD_RIGHT"
            else: action = "BACKWARD"
        else:
            if turn == "LEFT": action = f"TURN_LEFT:{current_turn_speed}"
            elif turn == "RIGHT": action = f"TURN_RIGHT:{current_turn_speed}"
            else: action = "STOP"
        
        log_info = f"Tgt: W_RAT={int(w_ratio*100)} W={int(width_px)} DZ={current_dead_zone} | Act: {action}"
    else:
        # Person Lost - Search Logic
        elapsed = current_time - user_data.last_seen_time
        user_data.last_raw_turn = "CENTER"
        
        if elapsed < 5:
            action = "STOP"
            log_info = f"Lost. Waiting... ({elapsed:.1f}s)"
        elif elapsed < 7.0:
            action = f"SEARCH_{user_data.last_direction}"
            log_info = f"Searching {user_data.last_direction}... ({elapsed:.1f}s)"
        else:
            action = "STOP"
            log_info = "Lost. Timeout."
    
    # Update display
    if frame is not None:
        cv2.putText(frame, f"FPS: {user_data.fps:.1f}  ACTION: {action}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        if person_detections:
            target = max(person_detections, key=lambda d: d.get_bbox().width())
            bbox = target.get_bbox()
            x1 = int(bbox.xmin() * FRAME_WIDTH)
            y1 = int(bbox.ymin() * height)
            x2 = int(bbox.xmax() * FRAME_WIDTH)
            y2 = int(bbox.ymax() * height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            
            # Draw dead zone lines
            dz_left = int(CENTER_X - current_dead_zone)
            dz_right = int(CENTER_X + current_dead_zone)
            cv2.line(frame, (dz_left, 0), (dz_left, height), (255, 0, 0), 2)
            cv2.line(frame, (dz_right, 0), (dz_right, height), (255, 0, 0), 2)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        mjpeg_frame_buffer.set_frame(frame)
    
    # Send command to motor process
    user_data.cmd_queue.put(action)
    
    # Throttled print
    if user_data.frame_count % 5 == 0:
        print(f"[Body] FPS: {user_data.fps:.1f} | {log_info}")
    
    return


# --- ROBOT MOTOR CONTROL PROCESS (for body mode) ---
def get_ultrasonic_distance(request, trig_pin, echo_pin):
    """Get distance from ultrasonic sensor."""
    if not request:
        return -1
    
    # Trigger Pulse
    request.set_value(trig_pin, Value.INACTIVE)
    time.sleep(0.000002)
    request.set_value(trig_pin, Value.ACTIVE)
    time.sleep(0.00001)
    request.set_value(trig_pin, Value.INACTIVE)
    
    timeout = time.time() + 0.04
    
    # Wait for Echo Start
    pulse_start = time.time()
    while request.get_value(echo_pin) == Value.INACTIVE:
        pulse_start = time.time()
        if pulse_start > timeout:
            return -1
    
    # Wait for Echo End
    pulse_end = time.time()
    while request.get_value(echo_pin) == Value.ACTIVE:
        pulse_end = time.time()
        if pulse_end > timeout:
            return -1
    
    duration = pulse_end - pulse_start
    distance = duration * 17150
    return round(distance, 2)


def motor_control_server(body_cmd_queue, stop_event):
    """
    THE single owner of all chassis GPIO (motors + ultrasonic sensors).
    Runs for the entire lifetime of the program as a separate process.

    Two command sources, serialised via _motor_lock:
      1. body_cmd_queue  — body-detection pipeline (active when body mode is on)
      2. ZMQ port 5557   — external clients: tools.py move() / dance() / music safety

    ZMQ protocol (all REQ/REP, JSON):
      execute_step  {cmd, action, speed, duration}
                    → {status, front_dist, rear_dist}
                       status: "ok" | "collision_front" | "collision_rear" | "error"
      stop          {cmd}                → {status}
      get_sensors   {cmd}               → {status, front_dist, rear_dist}
      set_body_mode {cmd, active: bool} → {status}
    """
    setproctitle.setproctitle("kiki_motor_server")
    print("[MotorServer] Starting...")

    # ── If hardware unavailable, still bind socket so clients get clean errors ─
    if not MOTOR_CONTROL_AVAILABLE:
        ctx  = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.RCVTIMEO, 200)
        sock.bind(f"tcp://*:{ZMQ_MOTOR_PORT}")
        print("[MotorServer] Motor hardware unavailable — serving error responses only")
        while not stop_event.is_set():
            try:
                sock.recv_json()
                sock.send_json({"status": "error", "message": "Motor control not available"})
            except zmq.Again:
                pass
            except Exception:
                pass
        sock.close()
        ctx.term()
        return

    # ── Hardware init ─────────────────────────────────────────────────────────
    mc.init()

    # ── Ultrasonic GPIO (this process is the ONLY one that touches these pins) ─
    sensor_cfg = {
        TRIG_PIN:      gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE),
        ECHO_PIN:      gpiod.LineSettings(direction=Direction.INPUT),
        BACK_TRIG_PIN: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE),
        BACK_ECHO_PIN: gpiod.LineSettings(direction=Direction.INPUT),
    }
    ultrasonic_req = None
    try:
        ultrasonic_req = gpiod.request_lines(
            "/dev/gpiochip4", consumer="motor_server", config=sensor_cfg
        )
        print("[MotorServer] Ultrasonic sensors acquired")
    except Exception as e:
        print(f"[MotorServer] Warning — ultrasonic unavailable: {e}")

    # ── Sensor cache — updated every 50 ms by background thread ─────────────
    _s_lock  = threading.Lock()
    _sensors = {"front": -1, "rear": -1}

    def _sensor_poll():
        while not stop_event.is_set():
            f = get_ultrasonic_distance(ultrasonic_req, TRIG_PIN,      ECHO_PIN)      if ultrasonic_req else -1
            r = get_ultrasonic_distance(ultrasonic_req, BACK_TRIG_PIN, BACK_ECHO_PIN) if ultrasonic_req else -1
            with _s_lock:
                _sensors["front"] = f
                _sensors["rear"]  = r
            time.sleep(0.05)

    threading.Thread(target=_sensor_poll, daemon=True, name="sensor_poll").start()

    def _get_sensors():
        with _s_lock:
            return _sensors["front"], _sensors["rear"]

    # ── Motor lock — prevents ZMQ thread and body-mode loop from fighting ────
    _motor_lock = threading.Lock()

    # ── Body mode flag ────────────────────────────────────────────────────────
    _body_mode = threading.Event()   # set → body mode active

    # ── Complete action map — ALL mc wheel states supported ──────────────────
    _ACTION_MAP = {
        # Linear
        "forward":               mc.forward,
        "backward":              mc.backward,
        "stop":                  mc.stop,
        # Tank turns
        "turn_left":             mc.turn_left,
        "turn_right":            mc.turn_right,
        # Strafing (holonomic)
        "strafe_left":           mc.strafe_left,
        "strafe_right":          mc.strafe_right,
        # Diagonals
        "diagonal_front_left":   mc.diagonal_front_left,
        "diagonal_front_right":  mc.diagonal_front_right,
        "diagonal_back_left":    mc.diagonal_back_left,
        "diagonal_back_right":   mc.diagonal_back_right,
        # Curved forward/backward
        "forward_left":          mc.forward_left,
        "forward_right":         mc.forward_right,
        "backward_left":         mc.backward_left,
        "backward_right":        mc.backward_right,
        # Pivot — rear axis anchor
        "turn_rear_axis_left":   mc.turn_rear_axis_left,
        "turn_rear_axis_right":  mc.turn_rear_axis_right,
        # Pivot — front axis anchor
        "turn_front_axis_left":  mc.turn_front_axis_left,
        "turn_front_axis_right": mc.turn_front_axis_right,
        # Swing turns — side anchor, forward
        "swing_turn_right":      mc.swing_turn_right,
        "swing_turn_left":       mc.swing_turn_left,
        # Swing turns — side anchor, backward
        "swing_turn_back_right": mc.swing_turn_back_right,
        "swing_turn_back_left":  mc.swing_turn_back_left,
        # Body-mode search aliases
        "search_left":           mc.turn_rear_axis_left,
        "search_right":          mc.turn_front_axis_right,
    }
    _BODY_ACTION_MAP = {
        "FORWARD":        mc.forward,
        "BACKWARD":       mc.backward,
        "STOP":           mc.stop,
        # These MUST use the original asymmetric pivot functions — using the
        # generic turn_left/turn_right produces reversed physical rotation on
        # this robot's specific motor wiring.
        "TURN_LEFT":      mc.turn_rear_axis_left,    # ← original
        "TURN_RIGHT":     mc.turn_front_axis_right,  # ← original
        "FORWARD_LEFT":   mc.diagonal_front_left,    # ← original
        "FORWARD_RIGHT":  mc.diagonal_front_right,   # ← original
        # BACKWARD_LEFT/RIGHT were unhandled in original (fell through to stop)
        # Map to closest diagonal equivalents for graceful behaviour
        "BACKWARD_LEFT":  mc.diagonal_back_left,
        "BACKWARD_RIGHT": mc.diagonal_back_right,
        "SEARCH_LEFT":    mc.turn_rear_axis_left,    # ← original
        "SEARCH_RIGHT":   mc.turn_front_axis_right,  # ← original
    }
    # ── ZMQ thread — serves tools.py clients ─────────────────────────────────
    def _zmq_loop():
        ctx  = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.RCVTIMEO, 100)
        sock.bind(f"tcp://*:{ZMQ_MOTOR_PORT}")
        print(f"[MotorServer] ZMQ listening on port {ZMQ_MOTOR_PORT}")

        while not stop_event.is_set():
            # ── Receive ───────────────────────────────────────────────────
            try:
                msg = sock.recv_json()
            except zmq.Again:
                continue
            except Exception as e:
                print(f"[MotorServer] ZMQ recv error: {e}")
                continue

            cmd = msg.get("cmd", "")

            try:
                # ── execute_step ──────────────────────────────────────────
                if cmd == "execute_step":
                    action   = msg.get("action", "stop").lower().strip().replace(" ", "_")
                    speed    = max(30, min(100, int(msg.get("speed", 50))))
                    duration = max(0.1, min(15.0, float(msg.get("duration", 0.5))))
                    move_fn  = _ACTION_MAP.get(action)

                    if move_fn is None:
                        sock.send_json({"status": "error",
                                        "message": f"Unknown action '{action}'. "
                                                   f"Valid: {sorted(_ACTION_MAP.keys())}"})
                        continue

                    collision = None
                    with _motor_lock:
                        mc.update_speed(speed)
                        move_fn()
                        t0 = time.time()
                        while (time.time() - t0) < duration and not stop_event.is_set():
                            f, r = _get_sensors()

                            # Forward collision
                            if "forward" in action and 0 < f < MIN_DIST_FRONT:
                                mc.stop()
                                if f < BACKOFF_DIST and (r == -1 or r > MIN_DIST_REAR):
                                    mc.update_speed(SPEED_SLOW)
                                    mc.backward()
                                    time.sleep(0.3)
                                mc.stop()
                                collision = "front"
                                break

                            # Backward collision
                            if "backward" in action and 0 < r < MIN_DIST_REAR:
                                mc.stop()
                                mc.update_speed(SPEED_SLOW)
                                mc.forward()
                                time.sleep(0.3)
                                mc.stop()
                                collision = "rear"
                                break

                            time.sleep(0.05)

                        mc.stop()

                    f, r = _get_sensors()
                    sock.send_json({
                        "status":     f"collision_{collision}" if collision else "ok",
                        "front_dist": f,
                        "rear_dist":  r,
                    })
# ── direct_move: fire-and-forget for live BT control ──────
                # Non-blocking: starts the movement and returns immediately.
                # The next command (or a stop) cancels it.
                elif cmd == "direct_move":
                    action  = msg.get("action", "stop").lower().strip().replace(" ", "_")
                    move_fn = _ACTION_MAP.get(action)

                    if move_fn is None:
                        sock.send_json({
                            "status":  "error",
                            "message": f"Unknown action '{action}'. "
                                       f"Valid: {sorted(_ACTION_MAP.keys())}"
                        })
                        continue

                    with _motor_lock:
                        move_fn()

                    sock.send_json({"status": "ok"})

                # ── set_speed: persistent speed update for live control ────
                elif cmd == "set_speed":
                    speed = max(0, min(100, int(msg.get("speed", 50))))
                    with _motor_lock:
                        mc.update_speed(speed)
                    sock.send_json({"status": "ok"})
                # ── stop ──────────────────────────────────────────────────
                elif cmd == "stop":
                    with _motor_lock:
                        mc.stop()
                    sock.send_json({"status": "ok"})

                # ── get_sensors ───────────────────────────────────────────
                elif cmd == "get_sensors":
                    f, r = _get_sensors()
                    sock.send_json({"status": "ok", "front_dist": f, "rear_dist": r})

                # ── set_body_mode ─────────────────────────────────────────
                elif cmd == "set_body_mode":
                    if msg.get("active", False):
                        _body_mode.set()
                        print("[MotorServer] Body mode ENABLED")
                    else:
                        _body_mode.clear()
                        with _motor_lock:
                            mc.stop()
                        print("[MotorServer] Body mode DISABLED")
                    sock.send_json({"status": "ok"})

                else:
                    sock.send_json({"status": "error",
                                    "message": f"Unknown cmd '{cmd}'"})

            except Exception as e:
                print(f"[MotorServer] Handler error: {e}")
                try:
                    sock.send_json({"status": "error", "message": str(e)})
                except Exception:
                    pass

        sock.close()
        ctx.term()
        print("[MotorServer] ZMQ loop stopped")

    threading.Thread(target=_zmq_loop, daemon=True, name="motor_zmq").start()

    # ── Main loop — body-mode execution (runs when _body_mode is set) ────────
    last_body_cmd      = "STOP"
    last_body_cmd_time = time.time()
    CMD_TIMEOUT        = 1.0

    print("[MotorServer] Ready")

    try:
        while not stop_event.is_set():
            if not _body_mode.is_set():
                time.sleep(0.02)
                continue

            # Drain body_cmd_queue, keep only the latest command
            body_cmd = None
            try:
                while True:
                    body_cmd = body_cmd_queue.get_nowait()
            except queue.Empty:
                pass

            if body_cmd:
                last_body_cmd      = body_cmd
                last_body_cmd_time = time.time()

            # Timeout guard
            current = last_body_cmd if (time.time() - last_body_cmd_time) < CMD_TIMEOUT else "STOP"

            # ── Safety overrides (same logic as old robot_motor_control_loop) ─
            f, r = _get_sensors()
            if f != -1 and f < MIN_DIST_FRONT:
                if "FORWARD" in current:
                    print(f"[MotorServer] OBSTACLE FRONT ({f}cm) — stopping")
                    current = "STOP"
                if f < BACKOFF_DIST:
                    if r == -1 or r > MIN_DIST_REAR:
                        current = "BACKWARD"
                        print(f"[MotorServer] TOO CLOSE ({f}cm) — backing off")
                    else:
                        current = "STOP"
                        print(f"[MotorServer] TRAPPED (f={f}cm r={r}cm)")
            if r != -1 and r < MIN_DIST_REAR and "BACKWARD" in current:
                print(f"[MotorServer] OBSTACLE REAR ({r}cm) — stopping")
                current = "STOP"

            # ── Parse embedded speed  e.g. "TURN_LEFT:45" ────────────────
            target_speed = SPEED_NORMAL
            if ":" in current:
                parts   = current.split(":", 1)
                current = parts[0]          # e.g. "TURN_LEFT" (still uppercase)
                try:
                    target_speed = int(parts[1])
                except ValueError:
                    pass

            if "SEARCH" in current:
                target_speed = SPEED_SEARCH
            elif "TURN" in current:
                target_speed = SPEED_TURN_MAX

            # Use _BODY_ACTION_MAP (UPPERCASE keys) — NOT _ACTION_MAP ──────────
            if _motor_lock.acquire(blocking=False):
                try:
                    mc.update_speed(target_speed)
                    _BODY_ACTION_MAP.get(current, mc.stop)()
                finally:
                    _motor_lock.release()

            time.sleep(0.05)


    except Exception as e:
        print(f"[MotorServer] Main loop error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        mc.stop()
        mc.cleanup()
        if ultrasonic_req:
            ultrasonic_req.release()
        print("[MotorServer] Fully stopped and GPIO released")


# --- WEBCAM CONTROL PROCESS (Stepper Motor via GPIO) ---
def webcam_control_loop(cmd_queue, stop_event):
    setproctitle.setproctitle("kiki_webcam_ctrl")
    print("[NeckStepper] Starting Neck Stepper Motor Control Process...")

    request = None
    motor_enabled = True  # Start with motor enabled (matches old relay-active-low default)

    try:
        request = gpiod.request_lines(
            NECK_STEPPER_CHIP,
            consumer="kiki-neck-stepper",
            config={
                NECK_STEPPER_PIN_1: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE),
                NECK_STEPPER_PIN_2: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE),
                NECK_STEPPER_PIN_3: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE),
                NECK_STEPPER_PIN_4: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE),
            }
        )
        print(f"[NeckStepper] GPIO lines acquired on {NECK_STEPPER_CHIP} "
              f"(pins {NECK_STEPPER_LINE_OFFSETS}). Ready.")

        def set_step(step_values):
            """Apply a single step of the full-step sequence to the motor coils."""
            line_values = {}
            for i, pin in enumerate(NECK_STEPPER_LINE_OFFSETS):
                line_values[pin] = Value.ACTIVE if step_values[i] == 1 else Value.INACTIVE
            request.set_values(line_values)

        def step_motor(direction, steps=NECK_STEPPER_STEPS_PER_CMD):
            """Drive the stepper `steps` steps in the given direction.
            direction: 'forward' for one way, 'reverse' for the other."""
            seq = NECK_STEPPER_STEP_SEQUENCE if direction == "forward" else list(reversed(NECK_STEPPER_STEP_SEQUENCE))
            for _ in range(steps):
                for step in seq:
                    set_step(step)
                    time.sleep(NECK_STEPPER_STEP_DELAY)

        def deenergize():
            """Turn off all coils to save power and prevent overheating."""
            set_step([0, 0, 0, 0])

        # Cumulative neck position in steps from center (+ = right/forward,
        # - = left/reverse). Updated by BOTH auto-tracking turns and explicit
        # looks so that an explicit LOOK_CENTER returns to neutral.
        neck_offset = 0

        def _move(direction, steps):
            """Energize-move-release, returning the signed step delta applied."""
            step_motor(direction, steps)
            deenergize()
            return steps if direction == "forward" else -steps

        while not stop_event.is_set():
            # Drain ALL queued commands this tick. Auto-tracking (TURN_*) is
            # keep-latest (only the freshest matters), but explicit expressive
            # LOOK_* commands must never be dropped, so we separate them.
            pending = []
            try:
                while True:
                    pending.append(cmd_queue.get_nowait())
            except queue.Empty:
                pass

            explicit = [c for c in pending
                        if isinstance(c, str) and c.startswith("LOOK_")]
            tracking = [c for c in pending
                        if not (isinstance(c, str) and c.startswith("LOOK_"))]

            # 1) Explicit expressive looks — run every one, in order.
            for look in explicit:
                if not motor_enabled:
                    continue
                name, _, arg = look.partition(":")
                try:
                    n = int(arg) if arg else NECK_STEPPER_STEPS_PER_CMD
                except ValueError:
                    n = NECK_STEPPER_STEPS_PER_CMD
                if name == "LOOK_LEFT":
                    print(f"[NeckStepper] LOOK LEFT ({n} steps)")
                    neck_offset += _move("reverse", n)
                elif name == "LOOK_RIGHT":
                    print(f"[NeckStepper] LOOK RIGHT ({n} steps)")
                    neck_offset += _move("forward", n)
                elif name == "LOOK_CENTER":
                    if neck_offset != 0:
                        print(f"[NeckStepper] LOOK CENTER (return {abs(neck_offset)} steps)")
                        neck_offset += _move("reverse" if neck_offset > 0 else "forward",
                                             abs(neck_offset))

            # 2) Auto-tracking — apply only the latest command.
            cmd = tracking[-1] if tracking else None
            if cmd:
                if cmd == "TURN_LEFT" and motor_enabled:
                    print(f"[NeckStepper] Turning LEFT ({NECK_STEPPER_STEPS_PER_CMD} steps)")
                    neck_offset += _move("reverse", NECK_STEPPER_STEPS_PER_CMD)
                elif cmd == "TURN_RIGHT" and motor_enabled:
                    print(f"[NeckStepper] Turning RIGHT ({NECK_STEPPER_STEPS_PER_CMD} steps)")
                    neck_offset += _move("forward", NECK_STEPPER_STEPS_PER_CMD)
                elif cmd == "RELAY_ON":
                    motor_enabled = True
                    print("[NeckStepper] Motor ENABLED")
                elif cmd == "RELAY_OFF":
                    motor_enabled = False
                    deenergize()
                    print("[NeckStepper] Motor DISABLED (coils de-energized)")

            time.sleep(0.01)

    except Exception as e:
        print(f"[NeckStepper] Error: {e}")
    finally:
        print("[NeckStepper] Stopping...")
        if request:
            try:
                # De-energize all coils before releasing
                line_values = {}
                for pin in NECK_STEPPER_LINE_OFFSETS:
                    line_values[pin] = Value.INACTIVE
                request.set_values(line_values)
                request.release()
                print("[NeckStepper] GPIO lines released, coils off.")
            except Exception as e:
                print(f"[NeckStepper] Cleanup error: {e}")

# --- VISION CALLBACK ---
class FollowerCallback(app_callback_class):
    def __init__(self, cmd_queue):
        super().__init__()
        self.cmd_queue = cmd_queue
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0.0
        self.telegram_enabled = False 
        self.locked_track_id = None

# Callback function for tracking logic (motor control)
def app_callback(element, buffer, user_data):
    global event_publisher, face_tracker, raw_frame_buffer, pipeline_state
    
    if buffer is None:
        return

    # Get frame for raw buffer (before detection boxes are drawn)
    # Note: This gets the frame with ROI metadata but before overlay
    pad = element.get_static_pad("sink")
    if pad:
        caps = pad.get_current_caps()
        if caps:
            structure = caps.get_structure(0)
            format_str = structure.get_value("format")
            width = structure.get_value("width")
            height = structure.get_value("height")
            
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                try:
                    frame = np.ndarray(
                        shape=(height, width, 3),
                        dtype=np.uint8,
                        buffer=map_info.data
                    ).copy()
                    # The pipeline delivers RGB (see the MJPEG path, which does the
                    # same RGB->BGR). cv2.imwrite / cv2.imencode expect BGR, so
                    # without this the saved training photos (and the Gemini/OLED
                    # crop) come out with red/blue swapped — blue faces & fingers.
                    if format_str == "RGB":
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    raw_frame_buffer.set_frame(frame)
                except:
                    pass
                finally:
                    buffer.unmap(map_info)

    # Get ROI and Detections
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    face_detections = [d for d in detections if d.get_label() == "face"]

    # --- FACE EVENT TRACKING ---
    current_faces_seen = set()
    enroll_faces = []  # lightweight per-face info for the auto-enroller
    for face in face_detections:
        track = face.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if track:
            track_id = track[0].get_id()
            current_faces_seen.add(track_id)

            # Get name from classification
            classifications = face.get_objects_typed(hailo.HAILO_CLASSIFICATION)
            name = "Unknown"
            confidence = 0.0
            for c in classifications:
                if c.get_label() != "Unknown":
                    name = c.get_label()
                    confidence = c.get_confidence()
                    break

            # Update tracker and check for confirmed face (after 1 second)
            event_type = face_tracker.update_face(track_id, name, confidence)
            if event_type == "confirmed" and event_publisher:
                event_publisher.publish_event("face_detected", {
                    "track_id": track_id,
                    "name": name,
                    "confidence": confidence
                })

            # Collect info for the unknown-face auto-enroller, and crop the face
            # so we can (a) enroll unknowns and (b) cache a crop for the OLED.
            try:
                bbox = face.get_bbox()
                wfrac = float(bbox.width())
                crop = None
                src = raw_frame_buffer.get_frame()
                if src is not None:
                    crop = crop_face_from_frame(src, bbox)
                enroll_faces.append({
                    "track_id": track_id, "name": name, "confidence": confidence,
                    "bbox_wfrac": wfrac, "crop": crop,
                })
            except Exception:
                pass

    # Cache the largest face's crop (for the on-demand get face_crop command).
    try:
        if enroll_faces:
            biggest = max(enroll_faces, key=lambda f: f["bbox_wfrac"])
            if biggest.get("crop") is not None:
                cache_latest_face_crop(biggest["name"], biggest["crop"])
    except Exception:
        pass

    # Auto-enroll a lone, clear stranger (dispatches to a bg thread; cheap here).
    try:
        unknown_enroller.evaluate(enroll_faces, None, pipeline_state, event_publisher)
    except Exception as e:
        print(f"[AutoEnroll] evaluate error: {e}")

    # --- LOCKING LOGIC ---
    target_detection = None
    
    # Get current target person from state
    current_target = pipeline_state.target_person

    # 1. Check existing lock
    if user_data.locked_track_id is not None:
        found_lock = False
        for face in face_detections:
            track = face.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if track and track[0].get_id() == user_data.locked_track_id:
                target_detection = face
                found_lock = True
                break
        if not found_lock:
            user_data.locked_track_id = None
    
    # 2. Acquire new lock if needed - use current target from state
    if user_data.locked_track_id is None:
        for face in face_detections:
            classifications = face.get_objects_typed(hailo.HAILO_CLASSIFICATION)
            for c in classifications:
                if c.get_label() == current_target:
                    target_detection = face
                    track = face.get_objects_typed(hailo.HAILO_UNIQUE_ID)
                    if track:
                        user_data.locked_track_id = track[0].get_id()
                    break
            if target_detection:
                break
        
        # Auto-follow lone person if specific target not found
        # if not target_detection and len(face_detections) == 1:
        #     target_detection = face_detections[0]

    # --- TRACKING LOGIC ---
    action = "STOP"
    
    # Only track if neck movement is enabled
    if pipeline_state.neck_movement_enabled and target_detection:
        bbox = target_detection.get_bbox()
        cx = (bbox.xmin() + bbox.xmax()) / 2.0
        cx_px = cx * FRAME_WIDTH
        
        if cx_px < (CENTER_X - DEAD_ZONE):
            action = "TURN_LEFT"
        elif cx_px > (CENTER_X + DEAD_ZONE):
            action = "TURN_RIGHT"
        else:
            action = "STOP"

    # Send command to Motor Process
    user_data.cmd_queue.put(action)
    return


def run_pipeline(cmd_queue, stop_event, zmq_stop_event):
    """Run the combined face-recognition + pose pipeline. Returns when it stops."""
    global pipeline_control

    try:
        user_data = FollowerCallback(cmd_queue)
        app = MJPEGStreamingFaceRecognitionApp(app_callback, user_data,
                                               parser=build_arg_parser())

        # The face-recognition pipeline __init__ overwrites the process title to
        # "Hailo Face Recognition App"; re-assert ours so the running server is
        # identifiable as kiki_follower_main (and not mistaken for a stray
        # training subprocess, which uses that same generic title).
        try:
            setproctitle.setproctitle("kiki_follower_main")
        except Exception:
            pass

        # Store main loop reference for external control
        pipeline_control["main_loop"] = app.main_loop if hasattr(app, 'main_loop') else None

        app.run()
        
    except Exception as e:
        print(f"[Pipeline] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clear buffers when pipeline stops
        mjpeg_frame_buffer.clear()
        raw_frame_buffer.clear()
        face_tracker.clear()
        pose_state.clear()
        clip_state.clear()
        print("[Pipeline] Pipeline stopped and resources freed")


def run_body_pipeline(cmd_queue, stop_event, zmq_stop_event):
    """Run the body detection vision pipeline. Returns when pipeline stops."""
    global pipeline_control
    
    try:
        user_data = BodyFollowerCallback(cmd_queue)
        # Force enable use_frame for OpenCV display
        user_data.use_frame = True
        
        app = MJPEGStreamingBodyDetectionApp(body_detection_callback, user_data)
        
        # Force enable use_frame option
        if hasattr(app, 'options_menu'):
            app.options_menu.use_frame = True
        
        # Store main loop reference for external control
        pipeline_control["main_loop"] = app.main_loop if hasattr(app, 'main_loop') else None
        
        app.run()
        
    except Exception as e:
        print(f"[BodyPipeline] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clear buffers when pipeline stops
        mjpeg_frame_buffer.clear()
        print("[BodyPipeline] Pipeline stopped and resources freed")
def _notify_motor_body_mode(active: bool, retries: int = 8, retry_delay: float = 0.8):
    """
    Tell motor_control_server to enter/exit body-follow mode.
    Retries up to `retries` times with `retry_delay` second gaps so that
    a slow process startup doesn't silently drop the notification.
    """
    for attempt in range(1, retries + 1):
        ctx  = None
        sock = None
        try:
            ctx  = zmq.Context()
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.RCVTIMEO, 1500)
            sock.setsockopt(zmq.SNDTIMEO, 1500)
            sock.connect(f"tcp://127.0.0.1:{ZMQ_MOTOR_PORT}")
            sock.send_json({"cmd": "set_body_mode", "active": active})
            resp = sock.recv_json()
            print(f"[Main] Motor server body_mode={'ON' if active else 'OFF'} "
                  f"(attempt {attempt}): {resp}")
            return True
        except Exception as e:
            print(f"[Main] Motor server notify attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(retry_delay)
        finally:
            try:
                if sock:  sock.close()
                if ctx:   ctx.term()
            except Exception:
                pass

    print(f"[Main] WARNING: Could not notify motor server after {retries} attempts — "
          f"body mode may not activate correctly")
    return False

def build_arg_parser():
    """The stock hailo pipeline parser plus this app's own flags.

    GStreamerFaceRecognitionApp.__init__ accepts a parser, so extra options go
    here rather than through the sys.argv rewriting used for --hef-path.
    Called once per pipeline build (the restart loop rebuilds the app).
    """
    parser = get_pipeline_parser()
    parser.add_argument("--pose-fps", type=int, default=None,
                        help="Frame rate for the pose branch. Defaults to --frame-rate. "
                             "Lower it (e.g. 15) to buy CPU headroom for face recognition.")
    parser.add_argument("--pose-hef", default=POSE_HEF_NAME,
                        help=f"Pose model name or HEF path (default: {POSE_HEF_NAME}).")
    parser.add_argument("--no-pose", action="store_true",
                        help="Disable the pose branch — face recognition only.")
    parser.add_argument("--hands", action="store_true",
                        help="Enable hand landmark + gesture recognition. Requires pose "
                             "(hand regions are derived from pose wrist/elbow keypoints).")
    parser.add_argument("--hands-fps", type=int, default=10,
                        help="Target rate for hand inference (default 10).")
    parser.add_argument("--max-hands", type=int, default=2,
                        help="Maximum hands to track per frame (default 2).")
    parser.add_argument("--hands-hef", default=None,
                        help="Hand landmark model: a full path, or a bare name such as "
                             "'hand_landmark_full' / 'hand_landmark_lite' (default: lite). "
                             "Note: the 'full' palm detector measured WORSE than lite here, "
                             "so try tiling before swapping models.")
    parser.add_argument("--vision-mode",
                        choices=("pose", "clip", "pose+clip", "all", "no-clip", "none"),
                        default="pose",
                        help="What runs alongside face recognition (face is always on). "
                             "pose (default): +yolov8s_pose. clip: +CLIP person description. "
                             "pose+clip: both. no-clip: pose + hand gestures, no CLIP. "
                             "all: pose + hands + CLIP, everything at once. "
                             "none: face recognition only.")
    parser.add_argument("--pose-recognition", dest="pose_recognition",
                        action="store_true", default=True,
                        help="Recognize posture, arm configuration, gestures and falls "
                             "from the pose keypoints (default: on when pose is active).")
    parser.add_argument("--no-pose-recognition", dest="pose_recognition",
                        action="store_false",
                        help="Disable pose recognition; keep raw pose estimation only.")
    parser.add_argument("--no-pose-recognition-overlay", dest="pose_recognition_overlay",
                        action="store_false", default=True,
                        help="Recognize poses but do not draw labels on the video stream.")
    parser.add_argument("--pose-mode", choices=("overlay", "parallel"), default="overlay",
                        help="overlay (default): pose runs in the main chain so hailooverlay "
                             "draws skeletons on the video stream. parallel: pose runs in an "
                             "isolated branch that cannot slow face recognition and honours "
                             "--pose-fps, but draws nothing.")
    return parser


def main():
    global event_publisher, pipeline_control, pose_recognizer

    # A C++ abort (e.g. an uncaught cv::Exception from inside an OpenCV worker
    # thread) kills the process via SIGABRT with no Python traceback at all —
    # it just prints "terminate called ...". faulthandler turns that into a
    # per-thread Python stack dump so the culprit is identifiable.
    faulthandler.enable()
    crash_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "crash_traceback.log")
    try:
        # Kept open for the process lifetime on purpose; faulthandler writes to
        # this fd from a signal handler.
        _crash_fd = open(crash_log_path, "a")
        faulthandler.enable(file=_crash_fd, all_threads=True)
        globals()["_CRASH_FD"] = _crash_fd
        print(f"[Main] Crash tracebacks will be appended to {crash_log_path}")
    except Exception as e:
        print(f"[Main] Could not open crash log: {e}")

    setproctitle.setproctitle("kiki_follower_main")
    print("Starting Kiki Webcam Tracker with MJPEG Streaming and ZeroMQ Control...")

    # ── Strip --hef-path overrides (unchanged) ────────────────────────────────
    new_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ('--hef-path', '-n'):
            print("WARNING: Ignoring provided --hef-path (forcing defaults for Face Recognition)")
            i += 2
        else:
            new_argv.append(arg)
            i += 1
    sys.argv = new_argv

    # ── Resolve pose options up front so pipeline builds can read them ────────
    known_args, _ = build_arg_parser().parse_known_args()
    mode = known_args.vision_mode
    # "all" and "no-clip" are conveniences that also switch hands on, so expand
    # them into the underlying feature set before anything reads the mode.
    want_hands = known_args.hands
    if mode == "all":
        mode = "pose+clip"
        want_hands = True
    elif mode == "no-clip":
        mode = "pose"
        want_hands = True

    if known_args.no_pose:   # backwards-compatible alias
        mode = "clip" if "clip" in mode else "none"
    VISION_MODE["value"] = mode

    POSE_OPTIONS["enabled"] = "pose" in mode
    POSE_OPTIONS["hef"] = known_args.pose_hef
    POSE_OPTIONS["fps"] = known_args.pose_fps
    POSE_OPTIONS["mode"] = known_args.pose_mode

    print(f"[Vision] Mode: {mode}")
    if POSE_OPTIONS["enabled"]:
        print(f"[Pose] Enabled (model={POSE_OPTIONS['hef']}, "
              f"mode={POSE_OPTIONS['mode']}, "
              f"fps={POSE_OPTIONS['fps'] or 'source rate'})")
    if "clip" in mode:
        print("[CLIP] Enabled — zero-shot descriptions of detected people")
    if mode == "none":
        print("[Vision] Face recognition only")

    # Hand gestures need pose keypoints to locate the hands.
    HAND_OPTIONS["fps"] = known_args.hands_fps
    HAND_OPTIONS["max_hands"] = known_args.max_hands
    HAND_OPTIONS["hef"] = known_args.hands_hef
    if want_hands:
        if not HAND_GESTURES_AVAILABLE:
            print("[Hand] --hands requested but hand_gestures.py is unavailable — disabled")
        else:
            # With palm detection, hands are found directly and pose is no
            # longer a prerequisite. Only the legacy pose-wrist fallback needs it.
            import hand_gestures as _hg
            if not _hg.BLAZE_AVAILABLE and not POSE_OPTIONS["enabled"]:
                print("[Hand] Palm detector unavailable and pose is off; the "
                      "wrist fallback needs pose. Use --vision-mode pose. Disabled.")
            else:
                HAND_OPTIONS["enabled"] = True
                print(f"[Hand] Enabled (~{HAND_OPTIONS['fps']} fps, "
                      f"max {HAND_OPTIONS['max_hands']} hands)")

    # Pose recognition rides on the pose branch's keypoints, so it is only
    # meaningful when pose is actually running.
    POSE_RECOGNITION_OPTIONS["overlay"] = known_args.pose_recognition_overlay
    if known_args.pose_recognition and POSE_OPTIONS["enabled"]:
        if not POSE_RECOGNITION_AVAILABLE:
            print("[PoseRec] pose_recognition.py unavailable — disabled")
        else:
            pose_recognizer = PoseRecognizer(state=pose_recognition_state)
            if pose_recognizer.enabled:
                pose_recognizer.start()
                POSE_RECOGNITION_OPTIONS["enabled"] = True
            else:
                print(f"[PoseRec] Disabled: "
                      f"{pose_recognizer.classifier.load_error or 'no model'}"
                      " — run train_pose_classifier.py to build pose_classifier.npz")
                pose_recognizer = None

    # ── IPC primitives ────────────────────────────────────────────────────────
    cmd_queue      = multiprocessing.Queue()   # neck (serial) motor commands
    body_cmd_queue = multiprocessing.Queue()   # body-detection → motor server
    stop_event     = multiprocessing.Event()
    zmq_stop_event = threading.Event()

    # Store cmd_queue for relay control from ZMQ commands
    pipeline_control["cmd_queue"] = cmd_queue

    # ── Webcam neck motor process (serial, unchanged) ─────────────────────────
    webcam_process = multiprocessing.Process(
        target=webcam_control_loop, args=(cmd_queue, stop_event)
    )
    webcam_process.start()
    print("[Main] Webcam neck process started")

    # ── Chassis motor server — ALWAYS running, owns ALL chassis GPIO ──────────
    motor_server_proc = multiprocessing.Process(
        target=motor_control_server, args=(body_cmd_queue, stop_event)
    )
    motor_server_proc.start()
    print(f"[Main] Motor control server started (ZMQ port {ZMQ_MOTOR_PORT})")

    # ── ZeroMQ event publisher ────────────────────────────────────────────────
    event_publisher = ZMQEventPublisher(ZMQ_EVENT_PORT)

    # ── ZeroMQ command server thread ──────────────────────────────────────────
    zmq_thread = threading.Thread(
        target=zmq_command_server,
        args=(zmq_stop_event, pipeline_state, event_publisher, raw_frame_buffer),
        daemon=True
    )
    zmq_thread.start()
    print(f"[Main] ZeroMQ command server on port {ZMQ_CMD_PORT}")
    print(f"[Main] ZeroMQ event publisher on port {ZMQ_EVENT_PORT}")

    # ── Face exit checker thread ──────────────────────────────────────────────
    exit_checker_thread = threading.Thread(
        target=face_exit_checker,
        args=(zmq_stop_event, face_tracker, event_publisher),
        daemon=True
    )
    exit_checker_thread.start()

    # ── MJPEG server thread ───────────────────────────────────────────────────
    mjpeg_thread = threading.Thread(
        target=mjpeg_server_thread, args=(stop_event,), daemon=True
    )
    mjpeg_thread.start()
    print(f"[Main] MJPEG stream at http://localhost:{MJPEG_PORT}/mjpeg")

    # ── Keep-alive frame server (separate from Flask; see frame_server_thread) ──
    frame_thread = threading.Thread(
        target=frame_server_thread, args=(stop_event,), daemon=True
    )
    frame_thread.start()

    # ── Hand gesture worker ───────────────────────────────────────────────────
    # Must live in this process: its Python VDevice joins the same SHARED group
    # as the GStreamer hailonets, which only works in-process.
    hand_worker = None
    if HAND_OPTIONS.get("enabled") and hand_state is not None:
        hand_worker = HandGestureWorker(
            pose_state=pose_state,
            frame_source=raw_frame_buffer.get_frame,
            hand_state=hand_state,
            stop_event=zmq_stop_event,
            max_hands=HAND_OPTIONS["max_hands"],
            target_fps=HAND_OPTIONS["fps"],
            landmark_hef=HAND_OPTIONS.get("hef"),
            gesture_callback=lambda data: event_publisher.publish_event(
                "hand_gesture", data),
        )
        hand_worker.start()

    # ── Graceful SIGTERM ──────────────────────────────────────────────────────
    # Python's default SIGTERM disposition kills the interpreter outright, so
    # the finally block below never runs and the motor/neck children are left
    # orphaned still holding port 5557 — which makes the *next* startup fail
    # with "Address already in use".
    #
    # Raising an exception from the handler is not enough: the main thread
    # spends its life inside GLib's loop.run(), a C call that never returns, so
    # a pending Python exception is never delivered and the process hangs in
    # shutdown instead of exiting. The loop has to be told to quit explicitly,
    # from the loop's own thread via idle_add.
    def _handle_sigterm(signum, frame):
        print(f"\n[Main] Received signal {signum} — shutting down gracefully")
        # Stop the restart loop from bringing the pipeline back up.
        pipeline_state.webcam_enabled = False
        stop_event.set()
        zmq_stop_event.set()

        app = pipeline_control.get("app")
        loop = pipeline_control.get("loop")

        def _quit_loop():
            try:
                if app is not None and getattr(app, "pipeline", None):
                    app.pipeline.set_state(Gst.State.NULL)
                if loop is not None:
                    loop.quit()
            except Exception as e:
                print(f"[Main] Error quitting GStreamer loop: {e}")
            return False

        if loop is not None:
            GLib.idle_add(_quit_loop)
        else:
            # No pipeline running (e.g. parked waiting for webcam ON); the
            # restart loop polls stop_event and will fall through on its own.
            pass

    try:
        signal.signal(signal.SIGTERM, _handle_sigterm)
        signal.signal(signal.SIGHUP, _handle_sigterm)
    except Exception as e:
        print(f"[Main] Could not install signal handlers: {e}")

    # Face-DB housekeeping BEFORE the pipeline claims the Hailo device: purge
    # leftover guest_* identities and re-enroll any named person missing an
    # embedding. Runs once per process (not on pipeline bounces/restarts), while
    # the device is still free so the trainer can actually acquire it.
    try:
        reconcile_face_db_on_start(event_publisher)
    except Exception as e:
        print(f"[Reconcile] error (continuing): {e}")

    # ── Pipeline restart loop ─────────────────────────────────────────────────
    try:
        while True:
            if pipeline_state.webcam_enabled:

                if LEGACY_BODY_PIPELINE and pipeline_state.full_body_mode_enabled:
                    # ── LEGACY FULL BODY MODE (escape hatch) ──────────────
                    print("[Main] Starting FULL BODY mode with robot chassis control...")

                    if not MOTOR_CONTROL_AVAILABLE:
                        print("[Main] WARNING: Motor control not available — body mode degraded")

                    # Activate body-mode in the always-running motor server
                    _notify_motor_body_mode(True)

                    # Run body detection pipeline (blocks until it stops)
                    run_body_pipeline(body_cmd_queue, stop_event, zmq_stop_event)

                    # Deactivate body-mode in motor server (motors idle, ZMQ still served)
                    _notify_motor_body_mode(False)

                else:
                    # ── COMBINED FACE RECOGNITION + POSE ──────────────────
                    # Both models share the one Hailo-8 via the HailoRT
                    # scheduler; full-body mode no longer swaps pipelines, it
                    # only arms the chassis motors.
                    print("[Main] Starting FACE RECOGNITION"
                          f"{' + POSE' if POSE_OPTIONS.get('enabled') else ''} mode...")
                    run_pipeline(cmd_queue, stop_event, zmq_stop_event)

                # After any pipeline exits, decide whether to restart
                if pipeline_state.webcam_enabled:
                    print("[Main] Pipeline stopped unexpectedly — restarting in 2 s...")
                    time.sleep(2)
                else:
                    print("[Main] Pipeline stopped by request (webcam OFF)")

            else:
                # Webcam is OFF — park here until re-enabled
                print("[Main] Webcam OFF — waiting for restart signal...")
                while not pipeline_state.webcam_enabled:
                    time.sleep(0.5)
                    if stop_event.is_set():
                        break

                if pipeline_state.webcam_enabled:
                    print("[Main] Webcam ON requested — restarting pipeline...")
                    pipeline_state.clear_restart_request()
                    time.sleep(1)   # brief settle before restart

            if stop_event.is_set():
                break

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user")
    except Exception as e:
        print(f"[Main] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # ── Pet mode ─────────────────────────────────────────────────────
        if PET_MODE_AVAILABLE and pet_mode_module._running:
            print("[Main] Stopping pet mode on shutdown...")
            pet_mode_module.stop_pet_mode()

        print("[Main] Shutting down all processes...")
        stop_event.set()
        zmq_stop_event.set()

        if pose_recognizer is not None:
            pose_recognizer.stop()
            print("[PoseRec] Worker stopped")

        mjpeg_frame_buffer.stop_encoder()

        if event_publisher:
            event_publisher.close()

        def _reap(proc, name, join_timeout=3.0):
            """Stop a child for certain.

            The old code called terminate() and returned without joining, so a
            child that ignored SIGTERM survived as an orphan — still holding the
            motor server's port 5557, which made the next startup fail with
            "Address already in use". Escalate to SIGKILL and confirm.
            """
            if proc is None:
                return
            proc.join(timeout=join_timeout)
            if proc.is_alive():
                print(f"[Main] {name} still alive — terminating")
                proc.terminate()
                proc.join(timeout=2.0)
            if proc.is_alive():
                print(f"[Main] {name} ignored SIGTERM — killing")
                proc.kill()
                proc.join(timeout=2.0)
            if proc.is_alive():
                print(f"[Main] WARNING: {name} (pid {proc.pid}) could not be stopped")
            else:
                print(f"[Main] {name} stopped")

        _reap(webcam_process, "Neck motor process", join_timeout=2.0)
        _reap(motor_server_proc, "Chassis motor server", join_timeout=3.0)

        if hand_worker is not None and hand_worker.is_alive():
            hand_worker.join(timeout=3.0)

        print("[Main] Shutdown complete")


main()
