"""
GPU worker.

Performs frame processing on the GPU using PyTorch tensor operations
when CUDA is available; falls back to CPU otherwise.
Supports both video-frame processing and image-task processing.
"""

import numpy as np
import torch

from image_processing.tasks import run_task_gpu


class GPUWorker:
    """Processes a frame using GPU tensor operations."""

    def __init__(self, device: torch.device):
        self.device = device

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Apply a GPU-accelerated sharpening kernel.

        If CUDA is not available the same operation runs on CPU tensors,
        so the pipeline never crashes.

        Args:
            frame: BGR image (H x W x 3), dtype uint8.

        Returns:
            Processed BGR image, dtype uint8.
        """
        # Convert to float tensor on the target device
        tensor = torch.from_numpy(frame.astype(np.float32)).to(self.device)  # (H, W, 3)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

        # Simple sharpening via unsharp-mask approach:
        #   sharpened = 2 * original - blurred
        # We approximate "blurred" with an average-pool
        blurred = torch.nn.functional.avg_pool2d(
            tensor, kernel_size=5, stride=1, padding=2
        )
        sharpened = torch.clamp(2.0 * tensor - blurred, 0, 255)

        # Move back to CPU / numpy
        result = sharpened.squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
        return result

    def process_image(self, frame: np.ndarray, task_name: str,
                      params: dict = None) -> np.ndarray:
        """Run an image-processing task on the GPU.

        Args:
            frame: BGR image (H x W x 3).
            task_name: Name of the task (e.g. 'sharpen', 'blur').
            params: Optional dict of task-specific parameters.

        Returns:
            Processed BGR image.
        """
        return run_task_gpu(frame, task_name, self.device, params or {})

