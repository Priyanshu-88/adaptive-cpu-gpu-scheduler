# adaptive-cpu-gpu-scheduler
Adaptive task scheduling framework that dynamically balances workloads between CPU and GPU based on runtime performance metrics.
<![CDATA[# 🧠 Adaptive CPU–GPU Hybrid Scheduler for Real-Time Video & Image Processing

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

<p align="center">
  An intelligent, real-time processing pipeline that <b>dynamically schedules</b> each frame/image across <b>CPU</b>, <b>GPU</b>, or <b>hybrid</b> execution — powered by deep reinforcement learning.
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [How It Works](#-how-it-works)
- [Image Processing Tasks](#-image-processing-tasks)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#%EF%B8%8F-configuration)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Future Improvements](#-future-improvements)

---

## 🔍 Overview

This project implements an **adaptive scheduling system** that intelligently routes computational workloads between CPU and GPU resources in real time. Instead of statically assigning tasks to a fixed device, the scheduler learns the optimal routing policy through online reinforcement learning.

The scheduling decision is driven by three components:

| Component | Role |
|---|---|
| **LSTM Predictor** | Predicts frame/image complexity from lightweight features using temporal sequences |
| **DQN Agent** | Deep Q-Network that learns the optimal CPU/GPU/Hybrid scheduling policy online |
| **Rule-Based Fallback** | Ensures safe operation via threshold-based decisions before the RL agent has trained sufficiently |

### Processing Modes

- **🎬 Video Mode** — Real-time frame-by-frame processing from webcam or video file
- **🖼️ Image Mode** — Batch/single image processing with 7 built-in tasks
- **🌐 Web Dashboard** — Flask-based UI with drag-and-drop upload, live stats, and before/after comparison

> **Everything runs locally** — no cloud APIs, no external datasets, no internet required.

---

## ✨ Key Features

| Category | Capability | Details |
|---|---|---|
| **Input** | Video capture | Webcam or video file via OpenCV |
| | Image processing | Single images or batch directory processing |
| **Processing** | 7 image tasks | Resize, blur, sharpen, edge detection, denoise, histogram equalisation, style transfer |
| **Intelligence** | Feature extraction | Motion / edges / brightness (video); resolution / edges / colour / brightness (image) |
| | Complexity prediction | Lightweight LSTM over a sliding window of features |
| | Temporal smoothing | Exponential moving average on complexity scores |
| | System monitoring | Real-time CPU / GPU utilisation and queue tracking |
| **Scheduling** | RL scheduling | DQN with experience replay buffer and target network |
| | Rule-based fallback | Guarantees safe decisions when DQN is untrained |
| **Execution** | Workers | Separate CPU, GPU, and hybrid processing workers |
| **Observability** | Live metrics | Per-frame latency, FPS, complexity, and ε-greedy stats |
| | Online learning | DQN trains continuously as frames/images arrive |
| | Web dashboard | Flask UI with video/image upload, live SSE stats, and before/after comparison |

---

## 🏗️ Architecture

```
┌────────────┐    ┌──────────────────┐    ┌────────────┐
│  Webcam /  │───>│ Feature Extraction│───>│   Scaling  │
│  Video /   │    │ (motion/edges/   │    │ (Min-Max)  │
│  Images    │    │  brightness/     │    └─────┬──────┘
└────────────┘    │  colour/res)     │          │
                  └──────────────────┘          v
                  ┌──────────────────┐    ┌────────────┐
                  │  LSTM Complexity │<───│  Sequence  │
                  │  Predictor       │    │  Buffer    │
                  └───────┬──────────┘    └────────────┘
                          │
                          v
                  ┌──────────────────┐
                  │    Temporal      │
                  │    Smoothing     │
                  └───────┬──────────┘
                          │
                          v
         ┌────────────────────────────────┐
         │         State Builder          │
         │ [complexity, cpu, gpu, queue]  │
         └───────────────┬────────────────┘
                         │
                         v
              ┌─────────────────────┐
              │   DQN Scheduler     │
              │ (or fallback rules) │
              └──────┬──────────────┘
                     │
          ┌──────────┼──────────┐
          v          v          v
      ┌───────┐ ┌───────┐ ┌────────┐
      │  CPU  │ │  GPU  │ │ Hybrid │
      │Worker │ │Worker │ │ Worker │
      └───┬───┘ └───┬───┘ └───┬────┘
          └──────────┼─────────┘
                     v
              ┌─────────────┐
              │   Metrics   │──> Reward ──> DQN Training
              │ (latency,   │
              │  FPS)       │
              └─────────────┘
```

---

## ⚙️ How It Works

### 🎬 Video Mode Pipeline

1. **Capture** — A frame is grabbed from the webcam or video file
2. **Feature Extraction** — Motion, Canny edge density, and brightness computed
3. **Scaling** — Features normalised to `[0, 1]` with online min-max scaling
4. **LSTM Prediction** — Complexity score `∈ [0, 1]` predicted from temporal sequence
5. **Temporal Smoothing** — Exponential moving average applied to reduce noise
6. **State Construction** — Complexity + system metrics → RL state vector
7. **Scheduling Decision** — DQN selects CPU / GPU / Hybrid (or fallback rules if untrained)
8. **Execution** — Selected worker processes the frame
9. **Metrics & Reward** — Latency/FPS measured, reward computed
10. **Learning** — DQN trained via mini-batch gradient descent from replay buffer

### 🖼️ Image Mode Pipeline

1. **Load** — Image(s) loaded from file or directory
2. **Feature Extraction** — Resolution complexity, edge density, colour variance, brightness
3. **Scheduling** — Same LSTM → DQN pipeline selects the optimal worker
4. **Task Execution** — Selected worker runs the chosen task (sharpen, blur, etc.)
5. **Save** — Processed images saved to `output/images/`
6. **Learning** — Same RL training loop continues online

---

## 🖼️ Image Processing Tasks

| Task | Description | CPU Implementation | GPU Implementation |
|---|---|---|---|
| **Resize** | Scale to target dimensions | `cv2.resize` | `F.interpolate` |
| **Blur** | Gaussian blur | `cv2.GaussianBlur` | Separable conv2d |
| **Sharpen** | Enhance edges | Laplacian kernel | Unsharp mask |
| **Edge Detection** | Extract edges | Canny | Sobel via conv2d |
| **Denoise** | Reduce noise | Non-local means | Averaging + blend |
| **Histogram Eq** | Normalise contrast | Per-channel equalise | CDF-based tensor ops |
| **Style Transfer** | Artistic effect | Bilateral + quantise | Quantise + edge overlay |

---

## 🚀 Installation

### Prerequisites

- **Python 3.8+**
- **pip** (Python package manager)
- A **webcam** (optional, for live video mode)
- **NVIDIA GPU + CUDA** (optional, for GPU acceleration — falls back to CPU automatically)

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/adaptive-cpu-gpu-scheduler.git
cd adaptive-cpu-gpu-scheduler

# Create a virtual environment (recommended)
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Video/image capture and processing |
| `torch` | Deep learning (LSTM, DQN) |
| `numpy` | Numerical operations |
| `psutil` | System monitoring (CPU/memory) |
| `flask` | Web dashboard server |
| `Pillow` | Image format support |

---

## 🎯 Usage

### 1. Video Mode (Default)

Process video frames in real time with adaptive scheduling:

```bash
python main.py
```

- Press **`q`** in the OpenCV window or **`Ctrl+C`** in the terminal to stop
- Console outputs per-frame stats: action, complexity, latency, FPS, ε, and training steps

To use a video file instead of the webcam, edit `config/config.py`:

```python
VIDEO_SOURCE = "path/to/your/video.mp4"   # or 0 for webcam
```

### 2. Image Mode

Process single images or batch directories:

```python
# In config/config.py:
PROCESSING_MODE = "image"
IMAGE_SOURCE = "images/"             # path to image file or directory
DEFAULT_IMAGE_TASK = "sharpen"       # resize | blur | sharpen | edge_detect | denoise | histogram_eq | style_transfer
```

```bash
python main.py
```

Processed images are saved to `output/images/`.

### 3. Web Dashboard

Launch the interactive web interface:

```bash
python dashboard.py
```

Open **http://localhost:5000** in your browser. The dashboard supports:

| Feature | Description |
|---|---|
| 📤 **Upload Video** | Drag & drop MP4 for processing with live frame-by-frame stats |
| 📹 **Live Webcam** | Real-time webcam feed with adaptive scheduling |
| 🖼️ **Image Processing** | Upload an image, select a task, view before/after comparison |
| 📊 **Live Stats** | Real-time SSE streaming of scheduling metrics |
| 🏗️ **Pipeline View** | Visual architecture diagram at `/pipeline` |

---

## 🛠️ Configuration

All hyperparameters are centralized in [`config/config.py`](config/config.py):

| Parameter | Default | Description |
|---|---|---|
| `VIDEO_SOURCE` | `"sample.mp4"` | `0` for webcam, or path to video file |
| `PROCESSING_MODE` | `"video"` | `"video"` or `"image"` |
| `SEQUENCE_LENGTH` | `10` | LSTM temporal window size |
| `SMOOTHING_ALPHA` | `0.3` | Exponential smoothing factor |
| `NUM_ACTIONS` | `3` | CPU / GPU / Hybrid |
| `DQN_HIDDEN_SIZE` | `128` | DQN network capacity |
| `DQN_LR` | `1.5e-3` | DQN learning rate |
| `DQN_GAMMA` | `0.99` | Discount factor |
| `EPSILON_DECAY` | `250` | Exploration decay rate |
| `REPLAY_CAPACITY` | `5000` | Experience replay buffer size |
| `BATCH_SIZE` | `64` | Training mini-batch size |
| `FALLBACK_CONFIDENCE` | `40` | Steps before trusting DQN over rules |
| `DEFAULT_IMAGE_TASK` | `"sharpen"` | Default image processing task |

---

## 📁 Project Structure

```
adaptive-cpu-gpu-scheduler/
│
├── config/
│   └── config.py                  # All hyperparameters and settings
│
├── pipeline/                      # Core video processing pipeline
│   ├── frame_capture.py           # Video frame source (webcam / file)
│   ├── image_capture.py           # Image file / directory source
│   ├── feature_extraction.py      # Video feature extraction (motion, edges, brightness)
│   ├── scaling.py                 # Online min-max normalisation
│   ├── smoothing.py               # Exponential temporal smoothing
│   ├── state_builder.py           # RL state vector construction
│   └── scheduler.py               # DQN + fallback rule-based scheduler
│
├── image_processing/              # Image-specific processing
│   ├── tasks.py                   # 7 image tasks (CPU + GPU implementations)
│   └── feature_extraction.py      # Image-specific feature extraction
│
├── execution/                     # Processing workers
│   ├── cpu_worker.py              # CPU processing worker
│   ├── gpu_worker.py              # GPU processing worker (PyTorch)
│   └── hybrid_worker.py           # Hybrid CPU+GPU worker
│
├── models/                        # Neural network models
│   ├── lstm_model.py              # LSTM complexity predictor
│   ├── dqn_model.py               # DQN Q-network
│   └── checkpoint/                # Saved model weights
│
├── rl/                            # Reinforcement learning
│   ├── train.py                   # DQN trainer with target network sync
│   ├── reward.py                  # Reward function (latency + FPS)
│   └── memory.py                  # Experience replay buffer
│
├── utils/                         # Utilities
│   ├── metrics.py                 # FPS / latency tracking
│   ├── monitor.py                 # System resource monitoring (CPU/GPU)
│   └── logger.py                  # Logging configuration
│
├── templates/                     # Web dashboard templates
│   ├── dashboard.html             # Main dashboard UI
│   └── pipeline.html              # Pipeline architecture visualisation
│
├── main.py                        # CLI entry point
├── dashboard.py                   # Flask web server
├── requirements.txt               # Python dependencies
├── sample.mp4                     # Sample video for testing
└── .gitignore
```

---

## 🧰 Tech Stack

| Technology | Usage |
|---|---|
| **Python 3.8+** | Core language |
| **PyTorch** | LSTM complexity predictor, DQN agent, GPU tensor operations |
| **OpenCV** | Video capture, image I/O, CPU-based image processing |
| **NumPy** | Feature computation, array operations |
| **Flask** | Web dashboard with SSE streaming |
| **psutil** | Real-time CPU / memory monitoring |
| **Pillow** | Extended image format support |

---

## 🔮 Future Improvements

- [ ] **GPU utilisation via `pynvml`** — Real NVIDIA GPU load monitoring for better state representation
- [ ] **Multi-stream support** — Schedule across multiple concurrent video feeds
- [ ] **Prioritised experience replay** — Faster DQN convergence by sampling high-error transitions
- [ ] **Model checkpointing** — Automatic save/resume on restart with best-model tracking
- [ ] **A3C / PPO** — Alternative RL algorithms for potentially better scheduling policies
- [ ] **Adaptive frame skipping** — Skip redundant frames based on predicted complexity
- [ ] **Batch image processing** — Parallel image processing via the web dashboard

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  Made with ❤️ using PyTorch, OpenCV, and Reinforcement Learning
</p>
]]>
