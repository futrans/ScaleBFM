from __future__ import annotations

import torch
from typing import TYPE_CHECKING, List

from isaaclab.utils.math import copysign, matrix_from_quat, subtract_frame_transforms, quat_apply, quat_mul, quat_conjugate, euler_xyz_from_quat, quat_apply_inverse, quat_inv

from scaletrack.tasks.tracking.four_arm import (
    condition_observation,
    heading_tan_norm_wxyz,
    mask_conditioned_task_features,
)
from scaletrack.tasks.tracking.mdp.commands import MotionCommand
from scaletrack.utils.torch_utils import calc_heading_quat_inv, quat_to_tan_norm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def local_body_pos(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)

    heading_inv_rot = calc_heading_quat_inv(command.robot_anchor_quat_w)
    heading_inv_rot_extend = heading_inv_rot.unsqueeze(-2).repeat(1, num_bodies, 1)

    local_body_pos = quat_apply(heading_inv_rot_extend, command.robot_body_pos_w - command.robot_anchor_pos_w.unsqueeze(-2))

    anchor_index = command.cfg.body_names.index(command.cfg.anchor_body_name)
    mask = torch.ones(len(command.cfg.body_names), dtype=torch.bool, device=local_body_pos.device)
    mask[anchor_index] = False
    local_body_pos = local_body_pos[..., mask, :]

    return local_body_pos.view(env.num_envs, -1)

def local_body_rot(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)

    heading_inv_rot = calc_heading_quat_inv(command.robot_anchor_quat_w)
    heading_inv_rot_extend = heading_inv_rot.unsqueeze(-2).repeat(1, num_bodies, 1)

    local_body_rot = quat_mul(heading_inv_rot_extend, command.robot_body_quat_w)
    local_body_rot = quat_to_tan_norm(local_body_rot)

    return local_body_rot.view(env.num_envs, -1)

def local_body_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)

    heading_inv_rot = calc_heading_quat_inv(command.robot_anchor_quat_w)
    heading_inv_rot_extend = heading_inv_rot.unsqueeze(-2).repeat(1, num_bodies, 1)

    local_body_vel = quat_apply(heading_inv_rot_extend, command.robot_body_lin_vel_w)

    return local_body_vel.view(env.num_envs, -1)

def local_body_ang_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)

    heading_inv_rot = calc_heading_quat_inv(command.robot_anchor_quat_w)
    heading_inv_rot_extend = heading_inv_rot.unsqueeze(-2).repeat(1, num_bodies, 1)

    local_body_ang_vel = quat_apply(heading_inv_rot_extend, command.robot_body_ang_vel_w)

    return local_body_ang_vel.view(env.num_envs, -1)

def ref_motion_time_offsets_future_manual(env: ManagerBasedEnv, command_name: str, future_idx: List[int]) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.time_offsets_future_manual(future_idx)
    
def base_height(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.robot_anchor_pos_w[..., 2:]

def mode(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.mode

def mode_mapping(
    env: ManagerBasedEnv,
    command_name: str,
    feature_dims_per_link: List[int],
    with_time: bool = False,
    extra_dims: int = 0,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mode = command.mode
    
    mappings = []
    for dim in feature_dims_per_link:
        mappings.append(
            (torch.ones(mode.shape[0], mode.shape[1], dim, device=mode.device, dtype=torch.float) * mode.unsqueeze(-1)).view(mode.shape[0], -1)
        )
    
    if with_time:
        mappings.append(
            torch.ones(mode.shape[0], 1, device=mode.device, dtype=torch.float)
        )
    if extra_dims:
        mappings.append(torch.ones(mode.shape[0], extra_dims, device=mode.device, dtype=torch.float))

    return torch.cat(mappings, dim=-1)


def four_arm_condition(env: ManagerBasedEnv, command_name: str, future_count: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return condition_observation(
        variant=command.cfg.four_arm_variant,
        visible=command.four_arm_condition_visible,
        active=command.four_arm_constraint_active,
        height=command.four_arm_height,
        robot_top=command.four_arm_robot_top,
        future_count=future_count,
    )


def four_arm_target_body_pos_future_to_robot_base_manual(
    env: ManagerBasedEnv, command_name: str, future_idx: List[int]
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    values = target_body_pos_future_to_robot_base_manual(env, command_name, future_idx)
    return mask_conditioned_task_features(
        values,
        command.four_arm_conditioned,
        body_count=len(command.cfg.body_names),
        features_per_body=3,
        kept_body_features={command.motion_anchor_body_index: (0, 1)},
    )


def four_arm_target_body_pos_future_rel_to_robot_base_manual(
    env: ManagerBasedEnv, command_name: str, future_idx: List[int]
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    values = target_body_pos_future_rel_to_robot_base_manual(env, command_name, future_idx)
    return mask_conditioned_task_features(
        values,
        command.four_arm_conditioned,
        body_count=len(command.cfg.body_names),
        features_per_body=3,
        kept_body_features={},
    )


def four_arm_target_body_rot_future_to_robot_base_manual(
    env: ManagerBasedEnv, command_name: str, future_idx: List[int]
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    num_frames = len(future_idx)
    num_bodies = len(command.cfg.body_names)
    values = target_body_rot_future_to_robot_base_manual(env, command_name, future_idx)
    pelvis_yaw = heading_tan_norm_wxyz(
        command.body_quat_w_future_manual(future_idx)[:, :, command.motion_anchor_body_index],
        command.robot_anchor_quat_w[:, None, :].expand(-1, num_frames, -1),
    )
    shaped = values.view(env.num_envs, num_frames, num_bodies, 6)
    allowed = torch.zeros_like(shaped)
    allowed[:, :, command.motion_anchor_body_index] = pelvis_yaw
    return torch.where(command.four_arm_conditioned[:, None, None, None], allowed, shaped).flatten(start_dim=2)


def four_arm_target_body_rot_future_rel_to_robot_base_manual(
    env: ManagerBasedEnv, command_name: str, future_idx: List[int]
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    values = target_body_rot_future_rel_to_robot_base_manual(env, command_name, future_idx)
    return mask_conditioned_task_features(
        values,
        command.four_arm_conditioned,
        body_count=len(command.cfg.body_names),
        features_per_body=6,
        kept_body_features={},
    )

def target_body_pos_future_to_robot_base_manual(env: ManagerBasedEnv, command_name: str, future_idx: List[int]) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    num_frames = len(future_idx)
    
    robot_anchor_quat_inv = quat_inv(command.robot_anchor_quat_w)[:, None, None, :].expand(-1, num_frames, num_bodies, -1)
    body_pos_w_future = command.body_pos_w_future_manual(future_idx)
    target_body_pos_future_to_robot_base = quat_apply(robot_anchor_quat_inv, body_pos_w_future - command.robot_anchor_pos_w[:, None, None, :])

    return target_body_pos_future_to_robot_base.view(env.num_envs, num_frames, -1)

def target_body_pos_future_rel_to_robot_base_manual(env: ManagerBasedEnv, command_name: str, future_idx: List[int]) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    num_frames = len(future_idx)

    robot_anchor_quat_inv = quat_inv(command.robot_anchor_quat_w)[:, None, None, :].expand(-1, num_frames, num_bodies, -1)
    body_pos_w_future = command.body_pos_w_future_manual(future_idx)
    target_body_pos_future_rel_to_robot_base = quat_apply(robot_anchor_quat_inv, body_pos_w_future - command.robot_body_pos_w[:, None, :, :])

    return target_body_pos_future_rel_to_robot_base.view(env.num_envs, num_frames, -1)

def target_body_rot_future_to_robot_base_manual(env: ManagerBasedEnv, command_name: str, future_idx: List[int]) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    num_frames = len(future_idx)

    robot_anchor_quat_inv = quat_inv(command.robot_anchor_quat_w)[:, None, None, :].expand(-1, num_frames, num_bodies, -1)
    
    target_body_rot_future_to_robot_base = quat_mul(
        robot_anchor_quat_inv,
        command.body_quat_w_future_manual(future_idx)
    )
    target_body_rot_future_to_robot_base = quat_to_tan_norm(target_body_rot_future_to_robot_base)

    return target_body_rot_future_to_robot_base.view(env.num_envs, num_frames, -1)

def target_body_rot_future_rel_to_robot_base_manual(env: ManagerBasedEnv, command_name: str, future_idx: List[int]) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    num_frames = len(future_idx)

    robot_anchor_quat_inv = quat_inv(command.robot_anchor_quat_w)[:, None, None, :].expand(-1, num_frames, num_bodies, -1)
    body_quat_w_future = command.body_quat_w_future_manual(future_idx)

    target_body_rot_future_rel_to_robot_base = quat_mul(
        body_quat_w_future, quat_conjugate(command.robot_body_quat_w[:, None, :, :].expand(-1, num_frames, -1, -1))
    )
    target_body_rot_future_rel_to_robot_base = quat_mul(
        quat_mul(
            robot_anchor_quat_inv, target_body_rot_future_rel_to_robot_base
        ),
        command.robot_anchor_quat_w[:, None, None, :].expand(-1, num_frames, num_bodies, -1),
    )
    target_body_rot_future_rel_to_robot_base = quat_to_tan_norm(target_body_rot_future_rel_to_robot_base)

    return target_body_rot_future_rel_to_robot_base.view(env.num_envs, num_frames, -1)

def target_body_vel_future_rel_to_robot_base_manual(env: ManagerBasedEnv, command_name: str, future_idx: List[int]) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    num_frames = len(future_idx)

    robot_anchor_quat_inv = quat_inv(command.robot_anchor_quat_w)[:, None, None, :].expand(-1, num_frames, num_bodies, -1)
    body_lin_vel_w_future = command.body_lin_vel_w_future_manual(future_idx)
    target_body_vel_future_rel_to_robot_base = quat_apply(robot_anchor_quat_inv, body_lin_vel_w_future - command.robot_body_lin_vel_w[:, None, :, :])

    return target_body_vel_future_rel_to_robot_base.view(env.num_envs, num_frames, -1)

def target_body_ang_vel_future_rel_to_robot_base_manual(env: ManagerBasedEnv, command_name: str, future_idx: List[int]) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    num_frames = len(future_idx)

    robot_anchor_quat_inv = quat_inv(command.robot_anchor_quat_w)[:, None, None, :].expand(-1, num_frames, num_bodies, -1)
    body_ang_vel_w_future = command.body_ang_vel_w_future_manual(future_idx)
    target_body_ang_vel_future_rel_to_robot_base = quat_apply(robot_anchor_quat_inv, body_ang_vel_w_future - command.robot_body_ang_vel_w[:, None, :, :])

    return target_body_ang_vel_future_rel_to_robot_base.view(env.num_envs, num_frames, -1)
