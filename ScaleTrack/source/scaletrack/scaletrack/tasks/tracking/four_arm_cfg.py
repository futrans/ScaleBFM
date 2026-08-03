from __future__ import annotations

from pathlib import Path

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

import scaletrack.tasks.tracking.mdp as mdp
from scaletrack.tasks.tracking.four_arm import FOUR_ARM_VARIANTS


def configure_four_arm_env(
    env_cfg,
    *,
    variant: str,
    schedule_file: str,
    geometry_file: str,
    conditioned_fraction: float = 0.5,
    bones_seed_max_fraction: float = 0.6,
    pelvis_height_offset_m: float = 0.49,
    condition_reward_weight: float = 1.0,
    condition_reward_std: float = 0.05,
    nonfoot_contact_weight: float = -0.1,
    contact_threshold: float = 1.0,
    sampler_seed: int = 42,
    sampler_rank: int = 0,
) -> None:
    if variant not in FOUR_ARM_VARIANTS[1:]:
        raise ValueError(f"four-arm training requires one of {FOUR_ARM_VARIANTS[1:]}, got {variant!r}")
    for label, value in (("schedule", schedule_file), ("geometry", geometry_file)):
        if not value or not Path(value).is_file():
            raise ValueError(f"four-arm {label} file does not exist: {value}")

    command = env_cfg.commands.motion
    command.four_arm_variant = variant
    command.four_arm_schedule_file = str(Path(schedule_file).resolve())
    command.four_arm_geometry_file = str(Path(geometry_file).resolve())
    command.four_arm_conditioned_fraction = conditioned_fraction
    command.four_arm_bones_seed_max_fraction = bones_seed_max_fraction
    command.four_arm_pelvis_height_offset_m = pelvis_height_offset_m
    command.four_arm_sampler_seed = sampler_seed
    command.four_arm_sampler_rank = sampler_rank

    policy_task = env_cfg.observations.policy_task
    critic_task = env_cfg.observations.critic_task
    policy_task.four_arm_condition = ObsTerm(
        func=mdp.four_arm_condition,
        params={"command_name": "motion", "future_count": 6},
    )
    critic_task.four_arm_condition = ObsTerm(
        func=mdp.four_arm_condition,
        params={"command_name": "motion", "future_count": 7},
    )
    env_cfg.observations.mode_mapping.mode_mapping.params["extra_dims"] = 3

    if variant in {"c_eq", "c_ub"}:
        policy_task.target_body_pos.func = mdp.four_arm_target_body_pos_future_to_robot_base_manual
        policy_task.target_body_pos_rel.func = mdp.four_arm_target_body_pos_future_rel_to_robot_base_manual
        policy_task.target_body_rot.func = mdp.four_arm_target_body_rot_future_to_robot_base_manual
        policy_task.target_body_rot_rel.func = mdp.four_arm_target_body_rot_future_rel_to_robot_base_manual

    rewards = env_cfg.rewards
    rewards.motion_body_height.func = mdp.four_arm_global_body_height
    rewards.motion_body_pos.func = mdp.four_arm_global_body_pos
    rewards.motion_body_rot.func = mdp.four_arm_global_body_rot
    rewards.motion_body_lin_vel.func = mdp.four_arm_global_body_lin_vel
    rewards.motion_body_ang_vel.func = mdp.four_arm_global_body_ang_vel

    rewards.four_arm_condition = None
    rewards.four_arm_nonfoot_contact = None
    if variant in {"c_eq", "c_ub"}:
        rewards.four_arm_condition = RewTerm(
            func=mdp.four_arm_condition_reward,
            weight=condition_reward_weight,
            params={"command_name": "motion", "std": condition_reward_std},
        )
        rewards.four_arm_nonfoot_contact = RewTerm(
            func=mdp.four_arm_nonfoot_contact,
            weight=nonfoot_contact_weight,
            params={
                "command_name": "motion",
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$).*$"],
                ),
                "threshold": contact_threshold,
            },
        )
    env_cfg.terminations.body_pos.func = mdp.four_arm_bad_global_body_pos
    env_cfg.terminations.body_pos.params["minimum_pelvis_height"] = 0.35
