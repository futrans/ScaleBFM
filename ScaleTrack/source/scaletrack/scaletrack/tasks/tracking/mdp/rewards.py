from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from scaletrack.tasks.tracking.four_arm import condition_reward_value, specialized_tracking_mask
from scaletrack.tasks.tracking.mdp.commands import MotionCommand
from scaletrack.utils.torch_utils import calc_heading_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def global_body_height(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.square(command.anchor_pos_w[..., 2] - command.robot_anchor_pos_w[..., 2])
    return torch.exp(-error / std**2)

def global_body_pos(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)

def global_body_rot(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)

def global_body_lin_vel(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)

def global_body_ang_vel(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def _four_arm_blend(command: MotionCommand, baseline: torch.Tensor, conditioned: torch.Tensor) -> torch.Tensor:
    mask = specialized_tracking_mask(
        variant=command.cfg.four_arm_variant,
        conditioned=command.four_arm_conditioned,
        active=command.four_arm_constraint_active,
    )
    return torch.where(mask, conditioned, baseline)


def four_arm_global_body_height(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    baseline = global_body_height(env, command_name, std)
    if command.cfg.four_arm_variant in {"c_eq", "c_ub"}:
        return _four_arm_blend(command, baseline, torch.zeros_like(baseline))
    return baseline


def four_arm_global_body_pos(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    baseline = global_body_pos(env, command_name, std, body_names)
    if command.cfg.four_arm_variant == "b1":
        specialized = global_body_pos(env, command_name, std, [command.cfg.anchor_body_name])
    elif command.cfg.four_arm_variant in {"c_eq", "c_ub"}:
        error = torch.sum(torch.square(command.anchor_pos_w[..., :2] - command.robot_anchor_pos_w[..., :2]), dim=-1)
        specialized = torch.exp(-error / std**2)
    else:
        specialized = baseline
    return _four_arm_blend(command, baseline, specialized)


def four_arm_global_body_rot(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    baseline = global_body_rot(env, command_name, std, body_names)
    if command.cfg.four_arm_variant == "b1":
        specialized = global_body_rot(env, command_name, std, [command.cfg.anchor_body_name])
    elif command.cfg.four_arm_variant in {"c_eq", "c_ub"}:
        error = quat_error_magnitude(
            calc_heading_quat(command.anchor_quat_w),
            calc_heading_quat(command.robot_anchor_quat_w),
        ) ** 2
        specialized = torch.exp(-error / std**2)
    else:
        specialized = baseline
    return _four_arm_blend(command, baseline, specialized)


def four_arm_global_body_lin_vel(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    baseline = global_body_lin_vel(env, command_name, std, body_names)
    if command.cfg.four_arm_variant == "b1":
        specialized = global_body_lin_vel(env, command_name, std, [command.cfg.anchor_body_name])
    elif command.cfg.four_arm_variant in {"c_eq", "c_ub"}:
        error = torch.sum(
            torch.square(command.anchor_lin_vel_w[..., :2] - command.robot_anchor_lin_vel_w[..., :2]), dim=-1
        )
        specialized = torch.exp(-error / std**2)
    else:
        specialized = baseline
    return _four_arm_blend(command, baseline, specialized)


def four_arm_global_body_ang_vel(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    baseline = global_body_ang_vel(env, command_name, std, body_names)
    if command.cfg.four_arm_variant == "b1":
        specialized = global_body_ang_vel(env, command_name, std, [command.cfg.anchor_body_name])
    elif command.cfg.four_arm_variant in {"c_eq", "c_ub"}:
        error = torch.square(command.anchor_ang_vel_w[..., 2] - command.robot_anchor_ang_vel_w[..., 2])
        specialized = torch.exp(-error / std**2)
    else:
        specialized = baseline
    return _four_arm_blend(command, baseline, specialized)


def four_arm_condition_reward(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return condition_reward_value(
        variant=command.cfg.four_arm_variant,
        robot_top=command.four_arm_robot_top,
        height=command.four_arm_height,
        active=command.four_arm_constraint_active,
        std=std,
    )


def four_arm_nonfoot_contact(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    contacts = torch.amax(torch.linalg.vector_norm(forces, dim=-1), dim=1) > threshold
    penalty = contacts.to(torch.float32).sum(dim=-1)
    return penalty * command.four_arm_constraint_active.to(penalty.dtype)

def foot_slip_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Penalize foot planar (xy) slip when in contact with the ground"""
    asset = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    foot_planar_velocity = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)

    reward = is_contact * foot_planar_velocity
    return torch.sum(reward, dim=1)

def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward
