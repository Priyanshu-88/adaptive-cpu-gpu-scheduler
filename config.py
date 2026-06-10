"""
Configuration module for the Adaptive CPU-GPU Hybrid Scheduler.

Centralizes all hyperparameters, thresholds, and system settings
so they can be tuned from a single location.
"""


# ---------- Video Capture ----------
VIDEO_SOURCE = "sample.mp4"       # 0 = webcam, or path to video file
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ---------- Feature Extraction ----------
CANNY_LOW = 50
CANNY_HIGH = 150

# ---------- Scaling ----------
# Initial min/max bounds for min-max scaling (updated online)
FEATURE_MIN = 0.0
FEATURE_MAX = 255.0

# ---------- LSTM ----------
SEQUENCE_LENGTH = 10              # Number of past feature vectors to feed
FEATURE_DIM = 3                   # motion, edges, brightness
LSTM_HIDDEN_SIZE = 32
LSTM_NUM_LAYERS = 1
LSTM_LR = 1e-3

# ---------- Temporal Smoothing ----------
SMOOTHING_ALPHA = 0.3             # Exponential smoothing factor (0–1)

# ---------- State Builder ----------
STATE_DIM = 4                     # complexity, cpu, gpu, queue_len

# ---------- DQN Scheduler ----------
NUM_ACTIONS = 3                   # 0=CPU, 1=GPU, 2=HYBRID
DQN_HIDDEN_SIZE = 128             # Increased capacity for better scheduling decisions
DQN_LR = 1.5e-3                   # Slightly faster learning rate
DQN_GAMMA = 0.99                  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 250               # Faster decay (250 instead of 500 frames)

# ---------- Fallback Thresholds ----------
COMPLEXITY_CPU_THRESHOLD = 0.33
COMPLEXITY_GPU_THRESHOLD = 0.66
FALLBACK_CONFIDENCE = 40          # Start trusting DQN earlier (40 steps instead of 50)

# ---------- Replay Buffer ----------
REPLAY_CAPACITY = 5000
BATCH_SIZE = 64                   # Larger batch size for more stable gradients

# ---------- Training ----------
TRAIN_INTERVAL = 10               # Train DQN every N frames
TARGET_UPDATE_INTERVAL = 25       # Sync target network more frequently (every 25 train steps)

# ---------- Paths ----------
CHECKPOINT_DIR = "models/checkpoint"
OUTPUT_DIR = "output"
LOG_FILE = "output/scheduler.log"

# ---------- Display ----------
PRINT_INTERVAL = 1                # Print stats every N frames
MAX_FRAMES = 0                    # 0 = unlimited

# ---------- Processing Mode ----------
PROCESSING_MODE = "video"         # "video" or "image"

# ---------- Image Processing ----------
IMAGE_SOURCE = "images/"          # path to image file or directory
IMAGE_OUTPUT_DIR = "output/images"
IMAGE_FEATURE_DIM = 4             # resolution, edges, colour_var, brightness
DEFAULT_IMAGE_TASK = "sharpen"    # default processing task
IMAGE_RESIZE_TARGET = (640, 480)
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

