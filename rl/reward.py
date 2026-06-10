"""
Reward function for the RL scheduler.

Computes a scalar reward from latency and FPS metrics to guide the
DQN toward low-latency, high-throughput scheduling decisions.
"""


def compute_reward(latency: float, fps: float, penalty: float = 0.0) -> float:
    """Calculate the reward for a single frame.

    Formula:
        reward = (1 / latency) + fps - penalty

    The inverse-latency term encourages fast processing; the FPS term
    rewards sustained throughput; the penalty term can be used to
    discourage undesirable actions (e.g. choosing GPU when none exists).

    Args:
        latency: Processing time for the current frame (seconds).
                 Clamped to a minimum of 1e-6 to avoid division by zero.
        fps:     Current frames-per-second estimate.
        penalty: Optional penalty (e.g. failed GPU allocation).

    Returns:
        Scalar reward value.
    """
    latency = max(latency, 1e-6)
    reward = (1.0 / latency) + fps - penalty
    return reward
