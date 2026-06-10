"""
main.py — Adaptive CPU–GPU Hybrid Scheduler for Real-Time Processing.

Supports two processing modes:
  • VIDEO — capture → extract features → scale → predict complexity → smooth →
            build state → schedule (DQN / fallback) → execute → measure →
            compute reward → train DQN
  • IMAGE — load image(s) → extract features → scale → predict complexity →
            build state → schedule → execute task → save result →
            compute reward → train DQN
"""

import os
import sys
import collections

import cv2
import numpy as np
import torch

# ── project imports ──────────────────────────────────────────────────────────
from config import config
from pipeline.frame_capture import FrameCapture
from pipeline.image_capture import ImageCapture
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
from utils.logger import get_logger
from image_processing.feature_extraction import ImageFeatureExtractor
from image_processing.tasks import get_available_tasks


# ═══════════════════════════════════════════════════════════════════════════
#  Video processing mode (original pipeline)
# ═══════════════════════════════════════════════════════════════════════════

def run_video_mode():
    """Run the adaptive scheduling pipeline for video frames."""

    logger = get_logger("main")

    # Ensure output dirs exist
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # ── GUI availability check ───────────────────────────────────────────
    gui_available = True
    try:
        test_img = np.zeros((2, 2, 3), dtype=np.uint8)
        cv2.imshow("__test__", test_img)
        cv2.destroyWindow("__test__")
        cv2.waitKey(1)
    except cv2.error:
        gui_available = False
        logger.info("OpenCV GUI unavailable — running in headless mode. "
                    "Install 'pip install opencv-contrib-python' for GUI.")

    # ── device setup ─────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── pipeline components ──────────────────────────────────────────────
    capture = FrameCapture()
    extractor = FeatureExtractor()
    scaler = MinMaxScaler(feature_dim=config.FEATURE_DIM)
    smoother = ExponentialSmoother()
    state_builder = StateBuilder()

    # ── models ───────────────────────────────────────────────────────────
    lstm = LSTMComplexityPredictor().to(device)
    lstm.eval()  # Inference-only; no LSTM training in this demo

    scheduler = Scheduler(device=device)
    trainer = DQNTrainer(policy_net=scheduler.policy_net, device=device)

    # ── workers ──────────────────────────────────────────────────────────
    cpu_worker = CPUWorker()
    gpu_worker = GPUWorker(device=device)
    hybrid_worker = HybridWorker(device=device)
    workers = {0: cpu_worker, 1: gpu_worker, 2: hybrid_worker}

    # ── metrics ──────────────────────────────────────────────────────────
    metrics = MetricsTracker()

    # ── sequence buffer for LSTM ─────────────────────────────────────────
    seq_buffer = collections.deque(maxlen=config.SEQUENCE_LENGTH)

    frame_idx = 0
    prev_state = None
    prev_action = None

    logger.info("Starting VIDEO pipeline — press 'q' in the window or Ctrl+C to stop.")

    try:
        while True:
            # ── 1. Capture ───────────────────────────────────────────────
            frame = capture.read()
            if frame is None:
                logger.info("End of video stream.")
                break

            metrics.start_frame()

            # ── 2. Feature extraction ────────────────────────────────────
            raw_features = extractor.extract(frame)

            # ── 3. Scaling ───────────────────────────────────────────────
            scaled_features = scaler.transform(raw_features)
            seq_buffer.append(scaled_features)

            # ── 4. LSTM complexity prediction ────────────────────────────
            if len(seq_buffer) >= config.SEQUENCE_LENGTH:
                seq = np.array(seq_buffer, dtype=np.float32)          # (SEQ, FEAT)
                seq_t = torch.tensor(seq, device=device).unsqueeze(0) # (1, SEQ, FEAT)
                with torch.no_grad():
                    complexity = lstm(seq_t).item()
            else:
                # Not enough history yet — use mean of scaled features
                complexity = float(np.mean(scaled_features))

            # ── 5. Temporal smoothing ────────────────────────────────────
            smoothed = smoother.smooth(complexity)

            # ── 6. State construction ────────────────────────────────────
            state = state_builder.build(smoothed_complexity=smoothed,
                                         queue_length=0)

            # ── 7. Scheduling decision ───────────────────────────────────
            action = scheduler.select_action(state)

            # ── 8. Execution ─────────────────────────────────────────────
            worker = workers[action]
            if action == 1 and isinstance(worker, GPUWorker):
                processed = worker.process(frame)
            elif action == 2 and isinstance(worker, HybridWorker):
                processed = worker.process(frame)
            else:
                processed = cpu_worker.process(frame)

            # ── 9. Metrics ───────────────────────────────────────────────
            latency = metrics.end_frame()
            fps = metrics.fps

            # ── 10. Reward & learning ────────────────────────────────────
            reward = compute_reward(latency, fps)

            if prev_state is not None:
                trainer.step(
                    state=prev_state,
                    action=prev_action,
                    reward=reward,
                    next_state=state,
                    done=False,
                )
                # Keep scheduler in sync with trainer's progress
                scheduler.train_steps = trainer.train_steps

            prev_state = state
            prev_action = action

            # ── Display ─────────────────────────────────────────────────
            if gui_available:
                cv2.imshow("Adaptive Scheduler", processed)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User pressed 'q' — stopping.")
                    break

            # ── Console stats ────────────────────────────────────────────
            frame_idx += 1
            if frame_idx % config.PRINT_INTERVAL == 0:
                label = Scheduler.action_name(action)
                print(
                    f"Frame {frame_idx:>5d} | "
                    f"Action: {label:<6s} | "
                    f"Complexity: {smoothed:.3f} | "
                    f"Latency: {latency*1000:.1f} ms | "
                    f"FPS: {fps:.1f} | "
                    f"ε: {scheduler._current_epsilon():.3f} | "
                    f"Trained: {trainer.train_steps}"
                )

            if 0 < config.MAX_FRAMES <= frame_idx:
                logger.info(f"Reached MAX_FRAMES ({config.MAX_FRAMES}).")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")

    finally:
        # ── Cleanup ──────────────────────────────────────────────────────
        capture.release()
        if gui_available:
            cv2.destroyAllWindows()

        # Save checkpoint
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, "dqn_policy.pth")
        torch.save(scheduler.policy_net.state_dict(), ckpt_path)
        logger.info(f"Model saved to {ckpt_path}")

        logger.info(
            f"Session complete — processed {frame_idx} frames, "
            f"avg FPS: {metrics.fps:.1f}"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Image processing mode (new pipeline)
# ═══════════════════════════════════════════════════════════════════════════

def run_image_mode():
    """Run the adaptive scheduling pipeline for image processing tasks."""

    logger = get_logger("main")

    # Ensure output dirs exist
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.IMAGE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # ── device setup ─────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    task_name = config.DEFAULT_IMAGE_TASK
    available = get_available_tasks()
    if task_name not in available:
        logger.error(f"Unknown task '{task_name}'. Available: {available}")
        return
    logger.info(f"Image processing task: {task_name}")

    # ── pipeline components ──────────────────────────────────────────────
    capture = ImageCapture()
    extractor = ImageFeatureExtractor()
    scaler = MinMaxScaler(feature_dim=config.IMAGE_FEATURE_DIM)
    smoother = ExponentialSmoother()
    state_builder = StateBuilder()

    # ── models ───────────────────────────────────────────────────────────
    lstm = LSTMComplexityPredictor(
        input_dim=config.IMAGE_FEATURE_DIM
    ).to(device)
    lstm.eval()

    scheduler = Scheduler(device=device)
    trainer = DQNTrainer(policy_net=scheduler.policy_net, device=device)

    # ── workers ──────────────────────────────────────────────────────────
    cpu_worker = CPUWorker()
    gpu_worker = GPUWorker(device=device)
    hybrid_worker = HybridWorker(device=device)
    workers = {0: cpu_worker, 1: gpu_worker, 2: hybrid_worker}

    # ── metrics ──────────────────────────────────────────────────────────
    metrics = MetricsTracker()

    # ── sequence buffer for LSTM ─────────────────────────────────────────
    seq_buffer = collections.deque(maxlen=config.SEQUENCE_LENGTH)

    img_idx = 0
    prev_state = None
    prev_action = None

    logger.info(
        f"Starting IMAGE pipeline — source: {config.IMAGE_SOURCE} | "
        f"Total images: {capture.total}"
    )

    try:
        while True:
            # ── 1. Load image ────────────────────────────────────────────
            frame = capture.read()
            if frame is None:
                logger.info("All images processed.")
                break

            filename = capture.current_filename
            metrics.start_frame()

            # ── 2. Feature extraction (image-specific) ───────────────────
            raw_features = extractor.extract(frame)

            # ── 3. Scaling ───────────────────────────────────────────────
            scaled_features = scaler.transform(raw_features)
            seq_buffer.append(scaled_features)

            # ── 4. LSTM complexity prediction ────────────────────────────
            if len(seq_buffer) >= config.SEQUENCE_LENGTH:
                seq = np.array(seq_buffer, dtype=np.float32)
                seq_t = torch.tensor(seq, device=device).unsqueeze(0)
                with torch.no_grad():
                    complexity = lstm(seq_t).item()
            else:
                complexity = float(np.mean(scaled_features))

            # ── 5. Temporal smoothing ────────────────────────────────────
            smoothed = smoother.smooth(complexity)

            # ── 6. State construction ────────────────────────────────────
            state = state_builder.build(smoothed_complexity=smoothed,
                                         queue_length=0)

            # ── 7. Scheduling decision ───────────────────────────────────
            action = scheduler.select_action(state)

            # ── 8. Execute image task ────────────────────────────────────
            worker = workers[action]
            processed = worker.process_image(frame, task_name)

            # ── 9. Save result ───────────────────────────────────────────
            out_path = os.path.join(config.IMAGE_OUTPUT_DIR, f"processed_{filename}")
            cv2.imwrite(out_path, processed)

            # ── 10. Metrics ──────────────────────────────────────────────
            latency = metrics.end_frame()
            fps = metrics.fps

            # ── 11. Reward & learning ────────────────────────────────────
            reward = compute_reward(latency, fps)

            if prev_state is not None:
                trainer.step(
                    state=prev_state,
                    action=prev_action,
                    reward=reward,
                    next_state=state,
                    done=False,
                )
                scheduler.train_steps = trainer.train_steps

            prev_state = state
            prev_action = action

            # ── Console stats ────────────────────────────────────────────
            img_idx += 1
            label = Scheduler.action_name(action)
            print(
                f"Image {img_idx:>4d} | "
                f"{filename:<30s} | "
                f"Task: {task_name:<14s} | "
                f"Action: {label:<6s} | "
                f"Complexity: {smoothed:.3f} | "
                f"Latency: {latency*1000:.1f} ms | "
                f"Saved: {out_path}"
            )

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")

    finally:
        capture.release()

        # Save checkpoint
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, "dqn_policy.pth")
        torch.save(scheduler.policy_net.state_dict(), ckpt_path)
        logger.info(f"Model saved to {ckpt_path}")

        logger.info(
            f"Session complete — processed {img_idx} images, "
            f"avg FPS: {metrics.fps:.1f} | "
            f"Output: {config.IMAGE_OUTPUT_DIR}"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Run the adaptive scheduling pipeline in the configured mode."""
    if config.PROCESSING_MODE == "image":
        run_image_mode()
    else:
        run_video_mode()


if __name__ == "__main__":
    main()
