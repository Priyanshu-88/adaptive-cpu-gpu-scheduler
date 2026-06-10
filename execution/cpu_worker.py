"""
CPU worker.

Performs lightweight image processing on the CPU using OpenCV.
Supports both video-frame processing and image-task processing.
"""

import cv2
import numpy as np

from image_processing.tasks import run_task_cpu


class CPUWorker:
    """Processes a frame entirely on the CPU."""

    @staticmethod
    def process(frame: np.ndarray) -> np.ndarray:
        """Apply a simple blur + edge enhancement on the CPU.

        Args:
            frame: BGR image (H x W x 3).

        Returns:
            Processed BGR image.
        """
        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        # Convert to grayscale, detect edges, convert back
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        # Blend edges back onto the original for a stylised look
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        result = cv2.addWeighted(frame, 0.7, edges_bgr, 0.3, 0)
        return result

    @staticmethod
    def process_image(frame: np.ndarray, task_name: str,
                      params: dict = None) -> np.ndarray:
        """Run an image-processing task on the CPU.

        Args:
            frame: BGR image (H x W x 3).
            task_name: Name of the task (e.g. 'sharpen', 'blur').
            params: Optional dict of task-specific parameters.

        Returns:
            Processed BGR image.
        """
        return run_task_cpu(frame, task_name, params or {})
