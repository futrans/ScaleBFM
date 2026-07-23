# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from torch import Tensor
from typing import List
from collections.abc import Generator
from tensordict import TensorDict

class RolloutStorage:
    class Transition:
        def __init__(self) -> None:
            self.observations: TensorDict | None = None
            self.actions: torch.Tensor | None = None
            self.rewards: torch.Tensor | None = None
            self.dones: torch.Tensor | None = None
            self.values: torch.Tensor | None = None
            self.actions_log_prob: torch.Tensor
            self.action_mean: torch.Tensor | None = None
            self.action_sigma: torch.Tensor | None = None

        def clear(self) -> None:
            self.__init__()

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
        device: str = "cpu",
    ) -> None:
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.actions_shape = actions_shape

        # Core
        self.observations = TensorDict(
            {key: torch.zeros(num_transitions_per_env, *value.shape, device=device) for key, value in obs.items()},
            batch_size=[num_transitions_per_env, num_envs],
            device=self.device,
        )
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For reinforcement learning
        if training_type == "rl":
            self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        else:
            raise NotImplementedError

        # Counter for the number of transitions stored
        self.step = 0

    def add_transitions(self, transition: Transition) -> None:
        # Check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # For reinforcement learning
        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)

        # Increment the counter
        self.step += 1

    def clear(self) -> None:
        self.step = 0

    def compute_returns(
        self, last_values: torch.Tensor, gamma: float, lam: float, gpu_global_rank: int = 0, gpu_world_size: int = 1
    ) -> None:
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            # If we are at the last step, bootstrap the return value
            next_values = last_values if step == self.num_transitions_per_env - 1 else self.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            self.returns[step] = advantage + self.values[step]

        # Compute the advantages
        self.advantages = self.returns - self.values

        mean = self.advantages.mean()
        var = self.advantages.var()
        count = torch.tensor(self.advantages.numel(), device=mean.device)

        if gpu_world_size > 1:
            all_means = [torch.zeros_like(mean, device=mean.device) for _ in range(gpu_world_size)]
            all_vars = [torch.zeros_like(var, device=var.device) for _ in range(gpu_world_size)]
            all_counts = [torch.zeros_like(count, device=count.device) for _ in range(gpu_world_size)]

            torch.distributed.all_gather(all_means,mean)
            torch.distributed.all_gather(all_vars,var)
            torch.distributed.all_gather(all_counts, count)

            if gpu_global_rank == 0:
                mean, var, count = self.combine_moments(
                    all_means,
                    all_vars,
                    all_counts
                )

            torch.distributed.broadcast(mean, src=0)
            torch.distributed.broadcast(var, src=0)

        self.advantages = (self.advantages - mean) / (torch.sqrt(var) + 1e-8)

    def generator(self) -> Generator:
        raise ValueError("This function is deprecated! Please use mini_batch_generator or recurrent mini batch generator for explicit assignment!")

    # For reinforcement learning with feedforward networks
    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator:
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Core
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        # For PPO
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                # Create the mini-batch
                obs_batch = observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                # Yield the mini-batch
                yield (
                    obs_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                )

    def combine_moments(self, means: List[Tensor], vars: List[Tensor], counts: List[Tensor]):
        """
        Combine moments from multiple processes robustly using a pairwise algorithm.
        """
        if not isinstance(counts, torch.Tensor):
            counts = torch.tensor(counts)

        # Convert all inputs to a compatible type for accumulation
        counts = counts.float()

        while len(means) > 1:
            new_means, new_vars, new_counts = [], [], []

            # Iteratively combine pairs of means, variances, and counts
            # We use non-sequential pairwise combination to minimize combinations across different magnitudes
            for i in range(0, len(means), 2):
                if i + 1 < len(means):
                    # Combine a pair of moments
                    mean_a, var_a, count_a = means[i], vars[i], counts[i]
                    mean_b, var_b, count_b = means[i + 1], vars[i + 1], counts[i + 1]

                    total_count = count_a + count_b
                    delta = mean_b - mean_a

                    # Combine means
                    combined_mean = mean_a + delta * (count_b / total_count)

                    # Combine variances (numerically stable formula)
                    m_2_a = var_a * count_a
                    m_2_b = var_b * count_b
                    m_2_combined = (
                        m_2_a + m_2_b + (delta**2) * (count_a * count_b / total_count)
                    )
                    combined_var = m_2_combined / total_count

                    new_means.append(combined_mean)
                    new_vars.append(combined_var)
                    new_counts.append(total_count)
                else:
                    # If there's an odd number of batches, just carry the last one over
                    new_means.append(means[i])
                    new_vars.append(vars[i])
                    new_counts.append(counts[i])

            means = new_means
            vars = new_vars
            counts = new_counts

        combined_mean = means[0]
        combined_var = torch.clamp(vars[0], min=0.0)  # Ensure non-negative variance
        total_count = counts[0].long()

        return combined_mean, combined_var, total_count