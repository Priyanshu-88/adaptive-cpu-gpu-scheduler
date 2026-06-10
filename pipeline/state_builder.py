"""
State builder module.

Constructs the RL state vector consumed by the DQN scheduler:
  [smoothed_complexity, cpu_usage, gpu_usage, normalised_queue_length]
"""

import numpy as np
import psutil

from config import config


def _gpu_usage() -> float:
    """Return GPU utilisation in [0, 1].

    Falls back to 0.0 when no NVIDIA GPU / pynvml is unavailable.
    """
    try:
        import torch
        if torch.cuda.is_available():
            # Simple heuristic: fraction of allocated memory
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_mem
            return min(allocated / max(total, 1), 1.0)
    except Exception:
        pass
    return 0.0


class StateBuilder:
    """Builds a fixed-size state vector for the RL agent."""

    def __init__(self, max_queue: int = 20):
        """
        Args:
            max_queue: Maximum expected queue length, used for normalisation.
        """
        self.max_queue = max_queue

    def build(self, smoothed_complexity: float, queue_length: int = 0) -> np.ndarray:
        """Assemble the state vector.

        Args:
            smoothed_complexity: EMA-smoothed complexity score in [0, 1].
            queue_length: Number of frames currently queued for processing.

        Returns:
            np.ndarray of shape (STATE_DIM,).
        """
        cpu_usage = psutil.cpu_percent(interval=None) / 100.0  # [0, 1]
        gpu_usage = _gpu_usage()
        norm_queue = min(queue_length / max(self.max_queue, 1), 1.0)

        state = np.array(
            [smoothed_complexity, cpu_usage, gpu_usage, norm_queue],
            dtype=np.float32,
        )
        return state
