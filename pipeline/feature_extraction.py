"""
Feature extraction module.

Extracts lightweight, real-time features from video frames:
  - Motion  (absolute difference between consecutive frames)
  - Edges   (Canny edge density)
  - Brightness (mean pixel intensity)
"""

import cv2
import numpy as np
from config import config


class FeatureExtractor:
    """Extracts a compact feature vector from a BGR frame."""

    def __init__(self):
        self.prev_gray = None

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """Compute features for a single frame.

        Args:
            frame: BGR image (H x W x 3).

        Returns:
            np.ndarray of shape (3,) — [motion, edge_density, brightness].
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Motion (mean absolute frame difference) ---
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            motion = float(np.mean(diff))
        else:
            motion = 0.0
        self.prev_gray = gray.copy()

        # --- Edge density (fraction of Canny edge pixels) ---
        edges = cv2.Canny(gray, config.CANNY_LOW, config.CANNY_HIGH)
        edge_density = float(np.mean(edges))  # 0-255 scale

        # --- Brightness (mean pixel value) ---
        brightness = float(np.mean(gray))

        return np.array([motion, edge_density, brightness], dtype=np.float32)
