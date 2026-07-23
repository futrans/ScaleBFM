# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from my_rsl_rl.networks import MLP


class ActorCritic(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        std_clamp_range: tuple[float] | list[float] = [0.001, 1.0],
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs])
            )
        super().__init__()

        # Get the observation dimensions
        self.obs_groups = obs_groups

        num_actor_obs = obs["policy"].flatten(start_dim=1).shape[-1] + obs["policy_task"].flatten(start_dim=1).shape[-1] + obs["action"].flatten(start_dim=1).shape[-1]
        num_critic_obs = obs["critic"].flatten(start_dim=1).shape[-1] + obs["critic_task"].flatten(start_dim=1).shape[-1] + obs["action"].flatten(start_dim=1).shape[-1]

        if "mode" in obs:
            assert "mode_mapping" in obs, f"You have to ensure that the mapping is defined between mode and task observation."
            assert obs["mode_mapping"].shape[-1] == obs['policy_task'].shape[-1], f"The mode mapping does not have the aligned dimension as task observation."
            print(f"Mode Strategy Activated!")
            num_mode_obs = obs['mode'].shape[-1]
            num_actor_obs += num_mode_obs

        self.state_dependent_std = state_dependent_std
        self.std_clamp_range = std_clamp_range

        if self.state_dependent_std:
            self.actor = MLP(num_actor_obs, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        print(f"Actor MLP: {self.actor}")

        # Critic
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, obs: TensorDict) -> None:
        if self.state_dependent_std:
            # Compute mean and standard deviation
            mean_and_std = self.actor(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
                std = torch.clamp(std, min=self.std_clamp_range[0], max=self.std_clamp_range[1])
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                log_std = torch.clamp(log_std, min=self.std_clamp_range[0], max=self.std_clamp_range[1])
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            # Compute mean
            mean = self.actor(obs)
            # Compute standard deviation
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_actor_obs(obs, inference=True)
        if self.state_dependent_std:
            return self.actor(obs)[..., 0, :]
        else:
            return self.actor(obs)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_critic_obs(obs)
        return self.critic(obs)

    def get_actor_obs(self, obs: TensorDict, inference: bool = False) -> torch.Tensor:
        prop_obs = obs['policy']
        if prop_obs.dim() == 2:
            prop_obs = prop_obs.unsqueeze(-2)
        task_obs = obs['policy_task']
        if task_obs.dim() == 2:
            task_obs = task_obs.unsqueeze(-2)
        action_obs = obs["action"]
        if action_obs.dim() == 2:
            action_obs = action_obs.unsqueeze(-2)

        if 'mode' in obs:
            mode_mapping = obs['mode_mapping']
            if mode_mapping.dim() == 2:
                    mode_mapping = mode_mapping.unsqueeze(-2).repeat(1, task_obs.shape[-2], 1)
            mode_obs = obs["mode"]
            
            if not inference:
                task_obs *= mode_mapping
            else:
                mode_obs = torch.ones_like(mode_obs, device=mode_obs.device)
            
            actor_obs = torch.cat([
                prop_obs.flatten(start_dim=1),
                task_obs.flatten(start_dim=1),
                action_obs.flatten(start_dim=1),
                mode_obs.flatten(start_dim=1)
            ], dim=-1)
        else:
            actor_obs = torch.cat([
                prop_obs.flatten(start_dim=1),
                task_obs.flatten(start_dim=1),
                action_obs.flatten(start_dim=1),
            ], dim=-1)
        
        return actor_obs

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        prop_obs = obs['critic']
        task_obs = obs['critic_task']
        action_obs = obs["action"]
        return torch.cat([
            prop_obs.flatten(start_dim=1), 
            task_obs.flatten(start_dim=1), 
            action_obs.flatten(start_dim=1)
        ], dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the actor-critic model.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        super().load_state_dict(state_dict, strict=strict)
        return True

    @property
    def actor_parameters(self):
        params = list(self.actor.parameters())

        if not self.state_dependent_std:
            if self.noise_std_type == "scalar":
                params.append(self.std)
            elif self.noise_std_type == "log":
                params.append(self.log_std)
        
        return params
    
    @property
    def critic_parameters(self):
        params = list(self.critic.parameters())
        
        return params