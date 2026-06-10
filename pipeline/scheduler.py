"""
Scheduler module.

Wraps the DQN model with an ε-greedy policy and a mandatory
rule-based fallback for when the agent is not yet trained.
"""

import random
import math

import torch
import numpy as np

from config import config
from models.dqn_model import DQNNetwork

# Human-readable action labels
ACTION_NAMES = {0: "CPU", 1: "GPU", 2: "HYBRID"}


class Scheduler:
    """DQN-based scheduler with rule-based fallback."""

    def __init__(self, device: torch.device):
        """
        Args:
            device: Torch device to place the policy network on.
        """
        self.device = device
        self.policy_net = DQNNetwork(
            state_dim=config.STATE_DIM,
            hidden_dim=config.DQN_HIDDEN_SIZE,
            num_actions=config.NUM_ACTIONS,
        ).to(device)

        self.steps_done = 0      # Counts scheduling calls for ε decay
        self.train_steps = 0     # Counts gradient updates (set externally)

    # ----- public API -----

    def select_action(self, state: np.ndarray) -> int:
        """Choose an action using ε-greedy or fallback.

        Args:
            state: State vector of shape (STATE_DIM,).

        Returns:
            Action index (0=CPU, 1=GPU, 2=HYBRID).
        """
        # If the DQN has not been trained enough, use the fallback
        if self.train_steps < config.FALLBACK_CONFIDENCE:
            return self._rule_based(state)

        # ε-greedy exploration
        epsilon = self._current_epsilon()
        if random.random() < epsilon:
            return random.randrange(config.NUM_ACTIONS)

        # Greedy action from the policy network
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return int(q_values.argmax(dim=1).item())

    @staticmethod
    def action_name(action: int) -> str:
        """Return a human-readable name for an action index."""
        return ACTION_NAMES.get(action, "UNKNOWN")

    # ----- internal helpers -----

    def _current_epsilon(self) -> float:
        """Compute ε with exponential decay."""
        eps = config.EPSILON_END + (config.EPSILON_START - config.EPSILON_END) * \
              math.exp(-self.steps_done / config.EPSILON_DECAY)
        self.steps_done += 1
        return eps

    @staticmethod
    def _rule_based(state: np.ndarray) -> int:
        """Simple threshold-based fallback.

        Uses the smoothed complexity (first element of the state vector).
        """
        complexity = state[0]
        if complexity < config.COMPLEXITY_CPU_THRESHOLD:
            return 0  # CPU
        elif complexity > config.COMPLEXITY_GPU_THRESHOLD:
            return 1  # GPU
        else:
            return 2  # HYBRID
