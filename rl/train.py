"""
DQN trainer.

Provides a `DQNTrainer` that owns the policy and target networks,
the optimiser, and the replay memory.  Call `step()` to store a
transition and (periodically) run a training update.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import config
from models.dqn_model import DQNNetwork
from rl.memory import ReplayMemory


class DQNTrainer:
    """Manages DQN training: replay buffer, gradient steps, target sync."""

    def __init__(self, policy_net: DQNNetwork, device: torch.device):
        """
        Args:
            policy_net: The policy network (shared with the Scheduler).
            device:     Torch device (cpu or cuda).
        """
        self.device = device
        self.policy_net = policy_net
        self.target_net = copy.deepcopy(policy_net).to(device)
        self.target_net.eval()

        self.optimiser = optim.Adam(self.policy_net.parameters(), lr=config.DQN_LR)
        self.memory = ReplayMemory(config.REPLAY_CAPACITY)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        self.frame_count = 0
        self.train_steps = 0

    # ----- public API -----

    def step(self, state, action, reward, next_state, done):
        """Store a transition and optionally train.

        Args:
            state:      np.ndarray (STATE_DIM,)
            action:     int
            reward:     float
            next_state: np.ndarray (STATE_DIM,)
            done:       bool
        """
        self.memory.push(state, action, reward, next_state, done)
        self.frame_count += 1

        # Only train periodically and when enough samples exist
        if (
            self.frame_count % config.TRAIN_INTERVAL == 0
            and len(self.memory) >= config.BATCH_SIZE
        ):
            self._train_batch()

    # ----- internal -----

    def _train_batch(self):
        """Sample a mini-batch and perform one gradient step."""
        transitions = self.memory.sample(config.BATCH_SIZE)

        # Unpack batch
        states = torch.tensor(
            np.array([t.state for t in transitions]), dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            [t.action for t in transitions], dtype=torch.long, device=self.device
        ).unsqueeze(1)
        rewards = torch.tensor(
            [t.reward for t in transitions], dtype=torch.float32, device=self.device
        )
        next_states = torch.tensor(
            np.array([t.next_state for t in transitions]), dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(
            [t.done for t in transitions], dtype=torch.float32, device=self.device
        )

        # Current Q-values for chosen actions
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # Target Q-values (Bellman target)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            target = rewards + config.DQN_GAMMA * next_q * (1 - dones)

        loss = self.loss_fn(q_values, target)

        self.optimiser.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimiser.step()

        self.train_steps += 1

        # Periodically sync target network
        if self.train_steps % config.TARGET_UPDATE_INTERVAL == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
