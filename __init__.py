"""
gridworld_env — Gymnasium environment for reward-misspecification research.

Importing this package registers the environment with Gymnasium so you can use:

    gym.make("RewardMisspecGridWorld-v0")

Reward modes
------------
    "true"        — aligned objective
    "proxy"       — farmable proxy tile (+0.40/step at P)
    "misleading"  — misleading tile valued over goal
    "delayed"     — reward hidden until episode end
"""

from __future__ import annotations

import gymnasium as gym

from .env import RewardMisspecGridWorldEnv, REWARD_MODES, ACTIONS, TILE_EMPTY

# Register the environment
gym.register(
    id="RewardMisspecGridWorld-v0",
    entry_point="gridworld_env.env:RewardMisspecGridWorldEnv",
    kwargs={
        "size": 5,
        "max_steps": 30,
        "reward_mode": "true",
        "render_mode": None,
    },
    max_episode_steps=30,
)

__all__ = ["RewardMisspecGridWorldEnv", "REWARD_MODES", "ACTIONS", "TILE_EMPTY"]
__version__ = "1.0.0"
