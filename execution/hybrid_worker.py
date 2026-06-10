"""
Hybrid worker.

Splits processing between CPU and GPU for frames of moderate complexity.
Supports both video-frame processing and image-task processing.
"""

import numpy as np
import cv2
import torch

from execution.cpu_worker import CPUWorker
from execution.gpu_worker import GPUWorker
from image_processing.tasks import run_task_cpu, run_task_gpu


class HybridWorker:
    """Combines CPU pre-processing with GPU post-processing."""

    def __init__(self, device: torch.device):
        self.cpu_worker = CPUWorker()
        self.gpu_worker = GPUWorker(device)
        self.device = device

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Run CPU pre-processing, then GPU enhancement.

        Pipeline:
          1. CPU — noise reduction (Gaussian blur)
          2. GPU — sharpening (unsharp mask via tensors)

        Args:
            frame: BGR image (H x W x 3).

        Returns:
            Processed BGR image.
        """
        # Stage 1: CPU – denoise
        denoised = cv2.GaussianBlur(frame, (3, 3), 0)

        # Stage 2: GPU – sharpen
        result = self.gpu_worker.process(denoised)
        return result

    def process_image(self, frame: np.ndarray, task_name: str,
                      params: dict = None) -> np.ndarray:
        """Run image task with CPU denoising + GPU processing.

        Pipeline:
          1. CPU — denoise the input image
          2. GPU — run the requested image task

        Args:
            frame: BGR image (H x W x 3).
            task_name: Name of the task (e.g. 'sharpen', 'blur').
            params: Optional dict of task-specific parameters.

        Returns:
            Processed BGR image.
        """
        params = params or {}
        # Stage 1: CPU – denoise
        denoised = run_task_cpu(frame, "denoise", {"denoise_strength": 5})
        # Stage 2: GPU – run the actual task
        result = run_task_gpu(denoised, task_name, self.device, params)
        return result
