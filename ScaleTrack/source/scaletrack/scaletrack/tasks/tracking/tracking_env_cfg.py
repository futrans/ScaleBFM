from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import scaletrack.tasks.tracking.mdp as mdp

##
# Scene definition
##

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


## Scene Config

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    # robots
    robot: ArticulationCfg = MISSING
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis/.*", history_length=3,
    )

## Command Config

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
    )

## Action Config

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)

## Observation Config

BFM_CONTEXT_SIZE = 3

@configclass
class BFMMaskObservationCfg:
    
    @configclass
    class PolicyCfg(ObsGroup):
        
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), history_length=BFM_CONTEXT_SIZE, flatten_history_dim=False)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), history_length=BFM_CONTEXT_SIZE, flatten_history_dim=False)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=BFM_CONTEXT_SIZE, flatten_history_dim=False)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), history_length=BFM_CONTEXT_SIZE, flatten_history_dim=False, scale=0.05)

        def __post_init__(self):
            self.enable_corruption = True

    @configclass
    class PolicyTaskCfg(ObsGroup):

        target_body_pos = ObsTerm(func=mdp.target_body_pos_future_to_robot_base_manual, params={"command_name": "motion","future_idx": [0,1,2,3,4,-1]}, noise=Unoise(n_min=-0.05, n_max=0.05))
        target_body_pos_rel = ObsTerm(func=mdp.target_body_pos_future_rel_to_robot_base_manual, params={"command_name": "motion","future_idx": [0,1,2,3,4,-1]}, noise=Unoise(n_min=-0.05, n_max=0.05))
        target_body_rot = ObsTerm(func=mdp.target_body_rot_future_to_robot_base_manual, params={"command_name": "motion","future_idx": [0,1,2,3,4,-1]}, noise=Unoise(n_min=-0.05, n_max=0.05))
        target_body_rot_rel = ObsTerm(func=mdp.target_body_rot_future_rel_to_robot_base_manual, params={"command_name": "motion","future_idx": [0,1,2,3,4,-1]}, noise=Unoise(n_min=-0.05, n_max=0.05))
        timestamp = ObsTerm(func=mdp.ref_motion_time_offsets_future_manual, params={"command_name": "motion","future_idx": [0,1,2,3,4,-1]})
        four_arm_condition = None

        def __post_init__(self):
            self.enable_corruption = True

    @configclass
    class PrivilegedCfg(ObsGroup):

        root_height = ObsTerm(func=mdp.base_height, params={"command_name": "motion"}, history_length=BFM_CONTEXT_SIZE,flatten_history_dim=False) # weishuai: This is only applicable when no terrain;
        local_body_pos = ObsTerm(func=mdp.local_body_pos, params={"command_name": "motion"},history_length=BFM_CONTEXT_SIZE,flatten_history_dim=False)
        local_body_rot = ObsTerm(func=mdp.local_body_rot, params={"command_name": "motion"},history_length=BFM_CONTEXT_SIZE,flatten_history_dim=False)
        local_body_vel = ObsTerm(func=mdp.local_body_vel, params={"command_name": "motion"},history_length=BFM_CONTEXT_SIZE,flatten_history_dim=False)
        local_body_ang_vel = ObsTerm(func=mdp.local_body_ang_vel, params={"command_name": "motion"},history_length=BFM_CONTEXT_SIZE,flatten_history_dim=False)

        joint_pos = ObsTerm(func=mdp.joint_pos_rel,history_length=BFM_CONTEXT_SIZE,flatten_history_dim=False)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel,history_length=BFM_CONTEXT_SIZE,flatten_history_dim=False, scale=0.05)
    

    @configclass
    class PrivilegedTaskCfg(ObsGroup):
        
        target_body_pos = ObsTerm(func=mdp.target_body_pos_future_to_robot_base_manual, params={"command_name": "motion","future_idx": [0,1,2,4,8,16,32]})
        target_body_pos_rel = ObsTerm(func=mdp.target_body_pos_future_rel_to_robot_base_manual, params={"command_name": "motion","future_idx": [0,1,2,4,8,16,32]})
        target_body_rot = ObsTerm(func=mdp.target_body_rot_future_to_robot_base_manual, params={"command_name": "motion","future_idx": [0,1,2,4,8,16,32]})
        target_body_rot_rel = ObsTerm(func=mdp.target_body_rot_future_rel_to_robot_base_manual, params={"command_name": "motion","future_idx": [0,1,2,4,8,16,32]})
        target_body_vel_rel = ObsTerm(func=mdp.target_body_vel_future_rel_to_robot_base_manual, params={"command_name": "motion","future_idx": [0,1,2,4,8,16,32]})
        target_body_ang_vel_rel = ObsTerm(func=mdp.target_body_ang_vel_future_rel_to_robot_base_manual, params={"command_name": "motion","future_idx": [0,1,2,4,8,16,32]})
        timestamp = ObsTerm(func=mdp.ref_motion_time_offsets_future_manual, params={"command_name": "motion","future_idx": [0,1,2,4,8,16,32]})
        four_arm_condition = None

    @configclass
    class ActionCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action,history_length=BFM_CONTEXT_SIZE,flatten_history_dim=False)

    @configclass
    class ModeCfg(ObsGroup):
        mode = ObsTerm(func=mdp.mode, params={"command_name": "motion"})
    
    @configclass
    class ModeMappingCfg(ObsGroup):
        mode_mapping = ObsTerm(func=mdp.mode_mapping, params={"command_name": "motion", "feature_dims_per_link": [3,3,6,6], "with_time": True})
    
    policy: PolicyCfg = PolicyCfg()
    policy_task: PolicyTaskCfg = PolicyTaskCfg()
    critic: PrivilegedCfg = PrivilegedCfg()
    critic_task: PrivilegedTaskCfg = PrivilegedTaskCfg()
    action: ActionCfg = ActionCfg()
    mode: ModeCfg = ModeCfg()
    mode_mapping: ModeMappingCfg = ModeMappingCfg()

## Domain Randomization Config

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    hand_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"]),
            "mass_distribution_params": (0.0, 1.0),
            "operation": "add",
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity_customized,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE, "disable_flag": False},
    )

## Reward Config

@configclass
class RewardsCfg:

    motion_body_height = RewTerm(
        func=mdp.global_body_height,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )

    motion_body_pos = RewTerm(
        func=mdp.global_body_pos,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_rot = RewTerm(
        func=mdp.global_body_rot,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )

    motion_body_lin_vel = RewTerm(
        func=mdp.global_body_lin_vel,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.global_body_ang_vel,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    survival = RewTerm(
        func=mdp.is_alive,
        weight=1.0,
    )
    four_arm_condition = None
    four_arm_nonfoot_contact = None

## Termination Config

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out_customized, time_out=True, params={"disable_flag": False})
    motion_time_out = DoneTerm(func=mdp.motion_time_out_customized, time_out=True, params={"command_name": "motion"})
    body_pos = DoneTerm(
        func=mdp.bad_global_body_pos,
        params={"command_name": "motion", "threshold": 0.5, "disable_flag": False},
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass


##
# Environment configuration
##

@configclass
class BFMTrackingEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=8192, env_spacing=2.5)
    # Basic settings
    observations: BFMMaskObservationCfg = BFMMaskObservationCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
