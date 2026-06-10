"""
DQN (Deep Q-Network) model.

Simple feedforward network that maps a state vector to Q-values
for each possible action (CPU / GPU / HYBRID).
"""

import torch
import torch.nn as nn

from config import config


class DQNNetwork(nn.Module):
    """Two-hidden-layer Q-network."""

    def __init__(
        self,
        state_dim: int = config.STATE_DIM,
        hidden_dim: int = config.DQN_HIDDEN_SIZE,
        num_actions: int = config.NUM_ACTIONS,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: State tensor of shape (batch, state_dim).

        Returns:
            Q-values tensor of shape (batch, num_actions).
        """
        return self.net(x)
