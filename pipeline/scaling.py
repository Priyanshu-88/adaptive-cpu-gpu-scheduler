"""
Online Min-Max scaler.

Normalises feature vectors to [0, 1] by tracking running min / max
values.  No external libraries required.
"""

import numpy as np


class MinMaxScaler:
    """Maintains per-feature running min and max for normalisation."""

    def __init__(self, feature_dim: int):
        """
        Args:
            feature_dim: Number of features in each vector.
        """
        self.feature_dim = feature_dim
        # Start with extremes so the first real observation defines the range
        self.min_vals = np.full(feature_dim, float("inf"), dtype=np.float32)
        self.max_vals = np.full(feature_dim, float("-inf"), dtype=np.float32)

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Update running stats and return scaled features.

        Args:
            features: Raw feature vector of shape (feature_dim,).

        Returns:
            Scaled feature vector in [0, 1].
        """
        # Update running min / max
        self.min_vals = np.minimum(self.min_vals, features)
        self.max_vals = np.maximum(self.max_vals, features)

        # Avoid division by zero
        range_vals = self.max_vals - self.min_vals
        range_vals[range_vals == 0] = 1.0

        scaled = (features - self.min_vals) / range_vals
        return scaled.astype(np.float32)
