from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from scaletrack.tasks.tracking.mdp.commands import MotionCommand
from scaletrack.tasks.tracking.mdp.rewards import _get_body_indexes
from scaletrack.tasks.tracking.four_arm import specialized_tracking_mask

def time_out_customized(env: ManagerBasedRLEnv, disable_flag: bool = False) -> torch.Tensor:
    if disable_flag:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    return env.episode_length_buf >= env.max_episode_length

def motion_time_out_customized(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    return (command.time_steps >= command.time_totals.gather(0, command.motion_ids.clone()) - 1).to(env.device)

def bad_global_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None, disable_flag: bool = False
) -> torch.Tensor:
    if disable_flag:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def four_arm_bad_global_body_pos(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    minimum_pelvis_height: float = 0.35,
    body_names: list[str] | None = None,
    disable_flag: bool = False,
) -> torch.Tensor:
    baseline = bad_global_body_pos(env, command_name, threshold, body_names, disable_flag)
    if disable_flag:
        return baseline
    command: MotionCommand = env.command_manager.get_term(command_name)
    if command.cfg.four_arm_variant == "b0":
        return baseline
    pelvis_xy_error = torch.linalg.vector_norm(
        command.anchor_pos_w[..., :2] - command.robot_anchor_pos_w[..., :2], dim=-1
    )
    conditioned_failure = (pelvis_xy_error > threshold) | (
        command.robot_anchor_pos_w[..., 2] < minimum_pelvis_height
    )
    mask = specialized_tracking_mask(
        variant=command.cfg.four_arm_variant,
        conditioned=command.four_arm_conditioned,
        active=command.four_arm_constraint_active,
    )
    return torch.where(mask, conditioned_failure, baseline)
