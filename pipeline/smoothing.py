"""
Temporal smoothing module.

Applies exponential moving average (EMA) to a scalar signal to reduce
high-frequency noise / jitter.
"""

from config import config


class ExponentialSmoother:
    """Exponential moving average smoother for a scalar value."""

    def __init__(self, alpha: float = None):
        """
        Args:
            alpha: Smoothing factor in (0, 1].  Larger values track the raw
                   signal more closely.  Defaults to config.SMOOTHING_ALPHA.
        """
        self.alpha = alpha if alpha is not None else config.SMOOTHING_ALPHA
        self.value = None  # Will be initialised on the first call

    def smooth(self, new_value: float) -> float:
        """Apply EMA and return the smoothed value.

        Args:
            new_value: Latest observation.

        Returns:
            Smoothed scalar.
        """
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        """Reset internal state."""
        self.value = None
