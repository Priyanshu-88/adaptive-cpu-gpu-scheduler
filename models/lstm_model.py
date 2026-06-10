"""
LSTM complexity predictor.

Takes a sequence of feature vectors (last N frames) and outputs a
scalar complexity score in [0, 1].
"""

import torch
import torch.nn as nn

from config import config


class LSTMComplexityPredictor(nn.Module):
    """Lightweight LSTM that maps a sequence of features → complexity score."""

    def __init__(
        self,
        input_dim: int = config.FEATURE_DIM,
        hidden_dim: int = config.LSTM_HIDDEN_SIZE,
        num_layers: int = config.LSTM_NUM_LAYERS,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, input_dim).

        Returns:
            Tensor of shape (batch,) with values in [0, 1].
        """
        # LSTM output: (batch, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(x)
        # Take the output of the last time step
        last_hidden = lstm_out[:, -1, :]
        score = self.sigmoid(self.fc(last_hidden)).squeeze(-1)
        return score
