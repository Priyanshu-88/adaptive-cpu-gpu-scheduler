# 🧠 Adaptive CPU–GPU Hybrid Scheduler

### Intelligent Real-Time Video & Image Processing Using Adaptive Resource Scheduling

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red" />
  <img src="https://img.shields.io/badge/OpenCV-4.x-green" />
  <img src="https://img.shields.io/badge/Flask-3.x-black" />
  <img src="https://img.shields.io/badge/License-MIT-brightgreen" />
</p>

---

## 📌 Project Overview

The **Adaptive CPU-GPU Hybrid Scheduler** is an intelligent scheduling framework designed for real-time video and image processing applications.

The system dynamically decides whether a task should run on the **CPU**, **GPU**, or a **Hybrid CPU-GPU execution mode** based on workload complexity and current system resource utilization.

Instead of statically assigning tasks to a fixed processor, the scheduler continuously analyzes system performance and automatically selects the most efficient execution strategy.

---

## 🎯 Problem Statement

Traditional processing systems typically execute workloads either on the CPU or GPU regardless of task complexity.

This often leads to:

- Inefficient resource utilization
- Increased processing latency
- Reduced throughput
- Performance bottlenecks during peak workloads

This project addresses these issues by implementing an adaptive scheduling mechanism that intelligently distributes workloads across available computing resources.

---

## 🚀 Key Features

- ✅ Adaptive CPU/GPU task scheduling
- ✅ Real-time video processing
- ✅ Image processing support
- ✅ Deep Reinforcement Learning (DQN)
- ✅ LSTM-based complexity prediction
- ✅ Live CPU/GPU monitoring
- ✅ Dynamic workload balancing
- ✅ Flask-based web dashboard
- ✅ Online learning and optimization
- ✅ CPU, GPU, and Hybrid execution modes

---

## 🔄 Workflow

```text
Input Video/Image
        │
        ▼
Feature Extraction
        │
        ▼
Complexity Prediction (LSTM)
        │
        ▼
Resource Monitoring
        │
        ▼
DQN Scheduler
   ┌────┼────┐
   ▼    ▼    ▼
 CPU   GPU Hybrid
   │    │    │
   └────┴────┘
        ▼
Processed Output
        │
        ▼
Reward Calculation
        │
        ▼
Continuous Learning
```

---

## 🏗️ System Architecture

```text
┌──────────────┐
│ Video/Image  │
│    Input     │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ Feature Extraction│
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ LSTM Complexity  │
│    Predictor     │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Resource Monitor │
│ CPU / GPU Status │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  DQN Scheduler   │
└───┬────┬─────┬───┘
    │    │     │
    ▼    ▼     ▼
  CPU   GPU  Hybrid
 Worker Worker Worker
    │    │     │
    └────┴─────┘
          │
          ▼
     Processed Data
          │
          ▼
    Reward Function
          │
          ▼
      DQN Training
```

---

## 🎬 Processing Modes

### 1. Video Mode

Real-time frame processing from:

- Webcam
- Video files

Features:

- Frame-by-frame scheduling
- Live latency tracking
- FPS monitoring
- Dynamic CPU/GPU allocation

---

### 2. Image Mode

Supports:

- Single image processing
- Batch image processing

Available operations:

- Resize
- Blur
- Sharpen
- Edge Detection
- Denoise
- Histogram Equalization
- Style Transfer

---

### 3. Web Dashboard

Features:

- Upload videos and images
- Live monitoring
- Resource statistics
- Before/After image comparison
- Processing analytics

---

## 🖼️ Supported Image Processing Tasks

| Task | Description |
|--------|-------------|
| Resize | Scale image dimensions |
| Blur | Gaussian blur filtering |
| Sharpen | Enhance image details |
| Edge Detection | Detect object boundaries |
| Denoise | Remove image noise |
| Histogram Equalization | Improve contrast |
| Style Transfer | Apply artistic effects |

---

## 🛠️ Technology Stack

| Technology | Purpose |
|------------|----------|
| Python | Core Development |
| PyTorch | Deep Learning Models |
| OpenCV | Video & Image Processing |
| Flask | Web Dashboard |
| NumPy | Numerical Computation |
| Pillow | Image Handling |
| psutil | System Monitoring |

---

## 📂 Project Structure

```text
adaptive-cpu-gpu-scheduler/
│
├── config/
│   └── config.py
│
├── pipeline/
│   ├── feature_extraction.py
│   ├── scheduler.py
│   ├── state_builder.py
│   └── smoothing.py
│
├── execution/
│   ├── cpu_worker.py
│   ├── gpu_worker.py
│   └── hybrid_worker.py
│
├── models/
│   ├── lstm_model.py
│   └── dqn_model.py
│
├── rl/
│   ├── train.py
│   ├── reward.py
│   └── memory.py
│
├── dashboard/
│   └── dashboard.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/adaptive-cpu-gpu-scheduler.git

cd adaptive-cpu-gpu-scheduler
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Video Mode

```bash
python main.py
```

### Web Dashboard

```bash
python dashboard.py
```

Open:

```text
http://localhost:5000
```

---

## 📈 Future Improvements

- Multi-GPU Scheduling
- PPO-Based Reinforcement Learning
- Distributed Resource Scheduling
- Kubernetes Integration
- Cloud-Edge Hybrid Scheduling
- Advanced Workload Prediction
- Model Checkpointing

---

## 📊 Expected Benefits

- Improved resource utilization
- Reduced processing latency
- Better workload balancing
- Increased throughput
- Adaptive real-time optimization

---

## 📄 License

This project is licensed under the MIT License.

---

<p align="center">
  Made with ❤️ using Python, PyTorch, OpenCV, and Reinforcement Learning
</p>
