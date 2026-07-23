"""Definitions for neural-network components for RL-agents."""

from .actor_critic import ActorCritic
from .actor_critic_humanoid_transformer import ActorCriticHumanoidTransformer

__all__ = [
    "ActorCritic",
    "ActorCriticHumanoidTransformer",
]
