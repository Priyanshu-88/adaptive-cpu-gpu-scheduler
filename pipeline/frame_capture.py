"""
Frame capture module.

Provides a class that wraps OpenCV's VideoCapture to grab frames
from a webcam or video file in a consistent way.
"""

import cv2
from config import config


class FrameCapture:
    """Captures frames from a webcam or video file."""

    def __init__(self, source=None):
        """
        Args:
            source: int (webcam index) or str (path to video file).
                    Defaults to config.VIDEO_SOURCE.
        """
        self.source = source if source is not None else config.VIDEO_SOURCE
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        # Attempt to set resolution (webcam only; ignored for files)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        self.prev_frame = None  # Stored for motion detection

    def read(self):
        """Read a single frame.

        Returns:
            frame (np.ndarray | None): BGR frame, or None if stream ended.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        """Release the underlying capture device."""
        if self.cap is not None:
            self.cap.release()

    def __del__(self):
        self.release()
