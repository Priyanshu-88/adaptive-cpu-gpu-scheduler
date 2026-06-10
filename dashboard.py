"""
Dashboard — Flask web UI for the Adaptive CPU-GPU Hybrid Scheduler.

Supports:
  • Upload MP4 for processing
  • Live webcam feed processing
  • Stop button to halt processing at any time
  • Per-frame live stats via SSE
  • Full summary table after completion / stop
  • Pipeline architecture page
"""

import os
import uuid
import json
import time
import collections
import threading

import cv2
import numpy as np
import torch
import base64
from flask import Flask, render_template, request, jsonify, Response

# ── project imports ──────────────────────────────────────────────────────────
from config import config
from pipeline.feature_extraction import FeatureExtractor
from pipeline.scaling import MinMaxScaler
from pipeline.smoothing import ExponentialSmoother
from pipeline.state_builder import StateBuilder
from pipeline.scheduler import Scheduler
from models.lstm_model import LSTMComplexityPredictor
from execution.cpu_worker import CPUWorker
from execution.gpu_worker import GPUWorker
from execution.hybrid_worker import HybridWorker
from rl.train import DQNTrainer
from rl.reward import compute_reward
from utils.metrics import MetricsTracker
from image_processing.feature_extraction import ImageFeatureExtractor
from image_processing.tasks import get_available_tasks

app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_FOLDER = os.path.join("output", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

# ── Shared state for SSE streaming ──────────────────────────────────────────
jobs: dict = {}


# ═══════════════════════════════════════════════════════════════════════════
#  Core processing function — works for both file and webcam
# ═══════════════════════════════════════════════════════════════════════════

def _process_source(job_id: str, source):
    """Run the full scheduling pipeline on a video source.

    Args:
        job_id:  Unique job identifier.
        source:  int (webcam index) or str (path to video file).
    """
    job = jobs[job_id]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_live = isinstance(source, int)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        job["events"].append({"type": "error", "msg": f"Cannot open video source: {source}"})
        job["done"] = True
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_live else 0
    video_fps_native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Pipeline components
    extractor = FeatureExtractor()
    scaler = MinMaxScaler(feature_dim=config.FEATURE_DIM)
    smoother = ExponentialSmoother()
    state_builder = StateBuilder()

    lstm = LSTMComplexityPredictor().to(device)
    lstm.eval()

    scheduler = Scheduler(device=device)
    trainer = DQNTrainer(policy_net=scheduler.policy_net, device=device)

    cpu_worker = CPUWorker()
    gpu_worker = GPUWorker(device=device)
    hybrid_worker = HybridWorker(device=device)
    workers = {0: cpu_worker, 1: gpu_worker, 2: hybrid_worker}

    metrics = MetricsTracker()
    seq_buffer = collections.deque(maxlen=config.SEQUENCE_LENGTH)

    all_rows = []
    action_counts = {"CPU": 0, "GPU": 0, "HYBRID": 0}
    total_latency = 0.0
    prev_state = None
    prev_action = None
    frame_idx = 0

    # Send metadata
    job["events"].append({
        "type": "meta",
        "total_frames": total_frames,
        "native_fps": round(video_fps_native, 1),
        "resolution": f"{width}x{height}",
        "device": str(device),
        "mode": "live" if is_live else "file",
    })

    while True:
        # ── Check stop signal ────────────────────────────────────────────
        if job.get("stop"):
            break

        ret, frame = cap.read()
        if not ret:
            break

        metrics.start_frame()

        # Feature extraction + scaling
        raw_features = extractor.extract(frame)
        scaled_features = scaler.transform(raw_features)
        seq_buffer.append(scaled_features)

        # LSTM complexity
        if len(seq_buffer) >= config.SEQUENCE_LENGTH:
            seq = np.array(seq_buffer, dtype=np.float32)
            seq_t = torch.tensor(seq, device=device).unsqueeze(0)
            with torch.no_grad():
                complexity = lstm(seq_t).item()
        else:
            complexity = float(np.mean(scaled_features))

        smoothed = smoother.smooth(complexity)
        state = state_builder.build(smoothed_complexity=smoothed, queue_length=0)
        action = scheduler.select_action(state)

        # Execute
        if action == 1:
            processed = gpu_worker.process(frame)
        elif action == 2:
            processed = hybrid_worker.process(frame)
        else:
            processed = cpu_worker.process(frame)

        # Store the latest frame for video streaming
        ret_jpg, buffer = cv2.imencode('.jpg', processed)
        if ret_jpg:
            job["latest_frame"] = buffer.tobytes()

        latency = metrics.end_frame()
        fps = metrics.fps

        # RL learning
        reward = compute_reward(latency, fps)
        if prev_state is not None:
            trainer.step(prev_state, prev_action, reward, state, False)
            scheduler.train_steps = trainer.train_steps
        prev_state = state
        prev_action = action

        frame_idx += 1
        action_name = Scheduler.action_name(action)
        action_counts[action_name] += 1
        total_latency += latency

        row = {
            "frame": frame_idx,
            "action": action_name,
            "complexity": round(smoothed, 4),
            "latency_ms": round(latency * 1000, 1),
            "fps": round(fps, 1),
            "epsilon": round(scheduler._current_epsilon(), 4),
            "trained": trainer.train_steps,
            "reward": round(reward, 2),
            "motion": round(float(raw_features[0]), 2),
            "edges": round(float(raw_features[1]), 2),
            "brightness": round(float(raw_features[2]), 2),
        }
        all_rows.append(row)

        # Push live event (throttle to every 2 frames for performance)
        if frame_idx % 2 == 0 or frame_idx <= 5:
            job["events"].append({"type": "frame", **row})

    cap.release()

    # ── Summary ──────────────────────────────────────────────────────────
    avg_latency = (total_latency / frame_idx * 1000) if frame_idx else 0
    avg_fps = (1.0 / (total_latency / frame_idx)) if frame_idx and total_latency else 0
    latencies = [r["latency_ms"] for r in all_rows] if all_rows else []
    sigma_latency = float(np.std(latencies)) if len(latencies) > 1 else 0.0

    stopped = job.get("stop", False)

    summary = {
        "type": "summary",
        "total_frames": frame_idx,
        "avg_latency_ms": round(avg_latency, 2),
        "sigma_latency_ms": round(sigma_latency, 2),
        "avg_fps": round(avg_fps, 1),
        "native_fps": round(video_fps_native, 1),
        "resolution": f"{width}x{height}",
        "device": str(device),
        "mode": "Live Webcam" if is_live else "Video File",
        "status": "Stopped by user" if stopped else "Completed",
        "action_counts": action_counts,
        "total_train_steps": trainer.train_steps,
        "final_epsilon": round(scheduler._current_epsilon(), 4),
        "total_rewards": round(sum(r["reward"] for r in all_rows), 2) if all_rows else 0,
        "min_latency_ms": round(min(r["latency_ms"] for r in all_rows), 1) if all_rows else 0,
        "max_latency_ms": round(max(r["latency_ms"] for r in all_rows), 1) if all_rows else 0,
        "avg_complexity": round(sum(r["complexity"] for r in all_rows) / len(all_rows), 4) if all_rows else 0,
        "all_rows": all_rows,
    }
    job["events"].append(summary)

    # Save checkpoint
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "dqn_policy.pth")
    torch.save(scheduler.policy_net.state_dict(), ckpt_path)

    job["done"] = True


# ═══════════════════════════════════════════════════════════════════════════
#  Routes
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/pipeline")
def pipeline():
    return render_template("pipeline.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Accept an MP4 upload, start processing, return a job ID."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    job_id = uuid.uuid4().hex[:12]
    save_path = os.path.join(UPLOAD_FOLDER, f"{job_id}.mp4")
    file.save(save_path)

    jobs[job_id] = {"events": [], "done": False, "stop": False}
    thread = threading.Thread(target=_process_source, args=(job_id, save_path), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/start_live", methods=["POST"])
def start_live():
    """Start processing the live webcam feed."""
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"events": [], "done": False, "stop": False}
    thread = threading.Thread(target=_process_source, args=(job_id, 0), daemon=True)
    thread.start()
    return jsonify({"job_id": job_id})


@app.route("/stop/<job_id>", methods=["POST"])
def stop(job_id):
    """Signal a running job to stop."""
    if job_id in jobs:
        jobs[job_id]["stop"] = True
        return jsonify({"status": "stopping"})
    return jsonify({"error": "Job not found"}), 404


@app.route("/stream/<job_id>")
def stream(job_id):
    """SSE endpoint — streams processing events to the browser."""
    if job_id not in jobs:
        return "Job not found", 404

    def generate():
        job = jobs[job_id]
        cursor = 0
        while True:
            while cursor < len(job["events"]):
                evt = job["events"][cursor]
                yield f"data: {json.dumps(evt)}\n\n"
                cursor += 1
            if job["done"] and cursor >= len(job["events"]):
                break
            time.sleep(0.05)

    return Response(generate(), mimetype="text/event-stream")


@app.route("/video_feed/<job_id>")
def video_feed(job_id):
    """Multipart JPEG stream for the processed video frames."""
    if job_id not in jobs:
        return "Job not found", 404

    def generate():
        job = jobs[job_id]
        last_yielded = None
        while True:
            if job["done"] and job.get("latest_frame") is None:
                break
                
            frame = job.get("latest_frame")
            if frame and frame != last_yielded:
                last_yielded = frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.03)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ═══════════════════════════════════════════════════════════════════════════
#  Image processing routes
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/upload_image", methods=["POST"])
def upload_image():
    """Accept an image upload, process it with the selected task, return results."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    task_name = request.form.get("task", config.DEFAULT_IMAGE_TASK)
    available = get_available_tasks()
    if task_name not in available:
        return jsonify({"error": f"Unknown task '{task_name}'. Available: {available}"}), 400

    # Save uploaded image
    img_id = uuid.uuid4().hex[:12]
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    save_path = os.path.join(UPLOAD_FOLDER, f"{img_id}{ext}")
    file.save(save_path)

    # Process the image
    result = _process_single_image(save_path, task_name)
    return jsonify(result)


@app.route("/image_tasks")
def image_tasks():
    """Return the list of available image processing tasks."""
    return jsonify({"tasks": get_available_tasks()})


def _process_single_image(image_path: str, task_name: str) -> dict:
    """Run a single image through the scheduling pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Could not read image"}

    # Keep original for before/after
    _, orig_buf = cv2.imencode(".jpg", frame)
    original_b64 = base64.b64encode(orig_buf).decode("utf-8")

    # Feature extraction
    extractor = ImageFeatureExtractor()
    raw_features = extractor.extract(frame)

    # Scaling + complexity
    scaler = MinMaxScaler(feature_dim=config.IMAGE_FEATURE_DIM)
    scaled_features = scaler.transform(raw_features)
    complexity = float(np.mean(scaled_features))

    # State + scheduling
    state_builder = StateBuilder()
    state = state_builder.build(smoothed_complexity=complexity, queue_length=0)
    scheduler = Scheduler(device=device)
    action = scheduler.select_action(state)

    # Workers
    cpu_worker = CPUWorker()
    gpu_worker = GPUWorker(device=device)
    hybrid_worker = HybridWorker(device=device)
    workers = {0: cpu_worker, 1: gpu_worker, 2: hybrid_worker}

    # Execute
    metrics = MetricsTracker()
    metrics.start_frame()
    worker = workers[action]
    processed = worker.process_image(frame, task_name)
    latency = metrics.end_frame()

    # Encode processed image
    _, proc_buf = cv2.imencode(".jpg", processed)
    processed_b64 = base64.b64encode(proc_buf).decode("utf-8")

    h, w = frame.shape[:2]

    return {
        "original": original_b64,
        "processed": processed_b64,
        "task": task_name,
        "action": Scheduler.action_name(action),
        "complexity": round(complexity, 4),
        "latency_ms": round(latency * 1000, 1),
        "resolution": f"{w}x{h}",
        "device": str(device),
        "features": {
            "resolution_complexity": round(float(raw_features[0]), 4),
            "edge_density": round(float(raw_features[1]), 4),
            "colour_variance": round(float(raw_features[2]), 4),
            "brightness": round(float(raw_features[3]), 4),
        },
    }


if __name__ == "__main__":
    print("\n  Dashboard running at  http://localhost:5000")
    print("  Pipeline page at     http://localhost:5000/pipeline\n")
    app.run(debug=False, port=5000, threaded=True)

