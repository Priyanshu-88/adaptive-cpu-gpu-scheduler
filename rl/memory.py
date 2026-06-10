"""
Replay memory for DQN training.

Stores (state, action, reward, next_state, done) transitions and
provides uniform random sampling.
"""

import random
from collections import deque, namedtuple

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayMemory:
    """Fixed-capacity circular buffer of transitions."""

    def __init__(self, capacity: int):
        """
        Args:
            capacity: Maximum number of transitions to store.
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store a transition.

        Args:
            state:      np.ndarray
            action:     int
            reward:     float
            next_state: np.ndarray
            done:       bool
        """
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Sample a random mini-batch.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            List[Transition]
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
