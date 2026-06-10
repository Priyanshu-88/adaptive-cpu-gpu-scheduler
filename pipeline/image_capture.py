"""
Image capture module.

Loads images from a single file or iterates over a directory,
exposing the same `.read()` / `.release()` interface as FrameCapture
so the main loop can be reused.
"""

import os
import cv2
import numpy as np
from config import config


class ImageCapture:
    """Loads images from a file or directory, one at a time."""

    def __init__(self, source: str = None):
        """
        Args:
            source: Path to a single image file or a directory of images.
                    Defaults to ``config.IMAGE_SOURCE``.
        """
        self.source = source or config.IMAGE_SOURCE
        self._files: list = []
        self._index: int = 0
        self._prepare()

    # ── public API (mirrors FrameCapture) ────────────────────────────────

    def read(self) -> np.ndarray | None:
        """Return the next image as a BGR ndarray, or None when exhausted."""
        if self._index >= len(self._files):
            return None

        path = self._files[self._index]
        self._index += 1
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            # Skip unreadable files and try the next one
            return self.read()

        # Optionally resize to a consistent working resolution
        h, w = frame.shape[:2]
        max_dim = max(config.FRAME_WIDTH, config.FRAME_HEIGHT)
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        return frame

    def release(self):
        """No-op for compatibility with FrameCapture."""
        pass

    @property
    def total(self) -> int:
        """Total number of images discovered."""
        return len(self._files)

    @property
    def current_filename(self) -> str:
        """Filename of the most recently read image."""
        if 0 < self._index <= len(self._files):
            return os.path.basename(self._files[self._index - 1])
        return ""

    # ── internal ─────────────────────────────────────────────────────────

    def _prepare(self):
        """Build the list of image file paths."""
        exts = tuple(config.SUPPORTED_IMAGE_FORMATS)

        if os.path.isfile(self.source):
            if self.source.lower().endswith(exts):
                self._files = [self.source]
        elif os.path.isdir(self.source):
            self._files = sorted(
                os.path.join(self.source, f)
                for f in os.listdir(self.source)
                if f.lower().endswith(exts)
            )
        # else: empty list → read() will return None immediately
