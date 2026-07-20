import mink
import time
import numpy as np
import mujoco as mj
from omegaconf import OmegaConf
from loguru import logger
from rich.progress import track
from scipy.spatial.transform import Rotation as R
from scaleretarget.retargeter.base_retargeter import BaseRetargeter


class GmrRetargeter(BaseRetargeter):
    def __init__(self, config):
        super().__init__(config)

        self.xml_file = self.config.robot.robot_xml_path
        logger.debug(f"[Retargeter] Loading robot model from {self.xml_file}")
        self.model = mj.MjModel.from_xml_path(self.xml_file)

        logger.debug(f"[Retargeter] Robot Degrees of Freedom names and their order:")
        self.robot_dof_names = {}
        for i in range(self.model.nv):
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            logger.debug(f"[Retargeter]: Dof {i}: {dof_name}")

        logger.debug("[Retargeter] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            logger.debug(f"[Retargeter] Body ID {i}: {body_name}")
        
        logger.debug("[Retargeter] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            logger.debug(f"[Retargeter] Motor ID {i}: {motor_name}")

        ik_config = self.config.ik_config
        ik_config = OmegaConf.load(ik_config)
        logger.debug(f"[Retargeter] Using IK config: {ik_config}")

        self.human_height_assumption = ik_config['human_height_assumption']
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]

        # used for retargeting
        self.ik_match_table = ik_config["ik_match_table"]
        self.human_scale_table_orig = ik_config['human_scale_table']

        # self.use_damping = ik_config.get('use_damping', False)
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])

        self.max_iter = 10

        self.solver = self.config.solver
        self.damping = self.config.damping

        self.human_body_to_task = {}
        self.pos_offsets = {}
        self.rot_offsets = {}

        self.task_errors = {}

        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if self.config.use_velocity_limit:
            logger.info(f"[Retargeter] Velocity limit activated!")
            VELOCITY_LIMITS = {k: np.pi/4 for k in self.robot_motor_names.keys()}
            # if 'velocity_limit' in ik_config:
            #     for joint in ik_config['velocity_limit']:
            #         VELOCITY_LIMITS[joint] = ik_config['velocity_limit'][joint]
            logger.debug(f"[Retargeter] Velocity limits: {VELOCITY_LIMITS}")
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS)) 
            
        self._setup_retarget_configuration()
        
        self.ground_offset = 0.0

    def _setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)
    
        self.tasks = []
        
        for frame_name, entry in self.ik_match_table.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                if body_name in self.human_body_to_task:
                    self.human_body_to_task[body_name].append(task)
                else:
                    self.human_body_to_task[body_name] = [task]
                
                if body_name in self.pos_offsets or body_name in self.rot_offsets:
                    logger.warning(f"{body_name} exists in current offset dict! Overwriting with pos_offset {pos_offset}; rot_offset {rot_offset}")
                self.pos_offsets[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks.append(task)
                self.task_errors[task] = []    

        # self.damping_task = mink.DampingTask(self.model,cost=5.0) 

    def update(self, extras):
        super().update(extras)

        # adjust the human scale table
        self.human_scale_table = {}

        if 'actual_human_height' in extras:
            ratio = extras['actual_human_height'] / self.human_height_assumption
        elif 'ratio' in extras: # weishuai: sometimes the height is hard to get
            ratio = extras['ratio']
        else:
            ratio = 1.0

        if extras.get('disable_scale_table', False):
            logger.info(f'[Retargeter] You have enabled optimized shape; The defined scale table is disabled!')
            for key in self.human_scale_table_orig.keys():
                if key == self.human_root_name:
                    self.human_scale_table[key] = self.human_scale_table_orig[key] * ratio
                else:
                    self.human_scale_table[key] = 1.0
            
            if extras.get('disable_init_height_anchor', False):
                self.init_height_anchor = False
            else:
                self.init_height_anchor = True
        
        else:
            self.init_height_anchor = False
            for key in self.human_scale_table_orig.keys():
                self.human_scale_table[key] = self.human_scale_table_orig[key] * ratio

    def retarget(self, frames):
        qpos_list = []

        self.init_frame_height = frames[0][self.human_root_name][0][2]

        init_human_data = frames[0]
        init_human_data = self.to_numpy(init_human_data)
        init_human_data = self.scale_human_data(init_human_data, self.human_root_name, self.human_scale_table)
        init_human_data = self.offset_human_data(init_human_data, self.pos_offsets, self.rot_offsets)
        init_human_data = self.apply_ground_offset(init_human_data)

        init_qpos = np.zeros_like(self.configuration.data.qpos)
        init_qpos[:3] = init_human_data[self.human_root_name][0]
        init_qpos[3:7] = init_human_data[self.human_root_name][1]
        self.configuration.update(q=init_qpos)

        for frame_idx in track(range(len(frames)), total=len(frames)):
            frame = frames[frame_idx]
            qpos = self.retarget_frame(frame)
            if self.enable_viewer:
                self.viewer.step(
                    root_pos=qpos[:3],
                    root_rot=qpos[3:7],
                    dof_pos=qpos[7:],
                    human_motion_data=self.scaled_human_data,
                )
            qpos_list.append(qpos.copy())
            while self.paused:
                time.sleep(0.05)
        qpos_list = np.array(qpos_list)
        return qpos_list
            

    def update_targets(self, human_data, offset_to_ground=False):
        # scale human data in local frame
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)
        human_data = self.offset_human_data(human_data, self.pos_offsets, self.rot_offsets)
        human_data = self.apply_ground_offset(human_data)
        if offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)
        self.scaled_human_data = human_data
        
        for body_name in self.human_body_to_task.keys():
            tasks = self.human_body_to_task[body_name]
            for task in tasks:
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
            
            
    def retarget_frame(self, human_data, offset_to_ground=False):
        # Update the task targets
        self.update_targets(human_data, offset_to_ground)

        curr_error = self.error()
        dt = self.configuration.model.opt.timestep
        vel = mink.solve_ik(
            self.configuration, self.tasks, dt, self.solver, self.damping, self.ik_limits
        )
        self.configuration.integrate_inplace(vel, dt)
        next_error = self.error()
        num_iter = 0
        while curr_error - next_error > 0.001 and num_iter < self.max_iter:
            curr_error = next_error
            # Solve the IK problem with the second task
            dt = self.configuration.model.opt.timestep
            # if self.use_damping:
            #     vel = mink.solve_ik(
            #         self.configuration, [*self.tasks, self.damping_task], dt, self.solver, self.damping, self.ik_limits
            #     )
            vel = mink.solve_ik(
                self.configuration, self.tasks, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel, dt)
                
            next_error = self.error()
            num_iter += 1
                
            
        return self.configuration.data.qpos.copy()


    def error(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks]
            )
        )


    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [np.asarray(human_data[body_name][0]), np.asarray(human_data[body_name][1])]
        return human_data


    def scale_human_data(self, human_data, human_root_name, human_scale_table):
        
        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]
        
        # scale root
        if self.init_height_anchor:
            scaled_root_pos = root_pos.copy()
            scaled_root_pos[..., :2] *= human_scale_table[human_root_name]
            scaled_root_pos[..., 2] = self.init_frame_height + (scaled_root_pos[..., 2] - self.init_frame_height) * human_scale_table[human_root_name]
        else:
            scaled_root_pos = human_scale_table[human_root_name] * root_pos.copy()

        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (human_data[body_name][0] - root_pos) * human_scale_table[body_name]
            
        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1])

        return human_data_global
    
    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        """the pos offsets are applied in the local frame"""
        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            # apply rotation offset first
            updated_quat = (R.from_quat(quat, scalar_first=True) * rot_offsets[body_name]).as_quat(scalar_first=True)
            offset_human_data[body_name][1] = updated_quat
            
            local_offset = pos_offsets[body_name]
            # compute the global position offset using the updated rotation
            global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)
            
            offset_human_data[body_name][0] = pos + global_pos_offset
           
        return offset_human_data
            
    def offset_human_data_to_ground(self, human_data):
        """find the lowest point of the human data and offset the human data to the ground"""
        offset_human_data = {}
        ground_offset = 0.1
        lowest_pos = np.inf

        for body_name in human_data.keys():
            # only consider the foot/Foot
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
                lowest_body_name = body_name
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])
        return offset_human_data

    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data):
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset])
        return human_data
