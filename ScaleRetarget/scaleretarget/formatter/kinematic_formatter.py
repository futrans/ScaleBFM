import torch
from loguru import logger
from scaleretarget.formatter.base_formatter import BaseFormatter
from scaleretarget.utils.kinematic_model.kinematics_model import KinematicsModel

class KinematicFormatter(BaseFormatter):
    def __init__(self, config):
        super().__init__(config)
        self.kinematic_model_device = self.config.kinematic_model_device
        self.height_adjust = self.config.height_adjust
        logger.info(f"[Formatter] height_adjust: {self.height_adjust}")
        self.kinematic_model = None
        if self.height_adjust:
            self.kinematic_model = KinematicsModel(
                file_path=self.config.robot.robot_xml_path,
                device=self.kinematic_model_device
            )
        self.root_offset = self.config.root_offset
        logger.info(f"[Formatter] root offset: {self.root_offset}")


    def format(self, qpos_list, extras):
        root_pos = qpos_list[:, :3]
        root_rot = qpos_list[:, 3:7] # wxyz
        if self.quat_order == 'xyzw':
            root_rot[:, [0,1,2,3]] = root_rot[:, [1,2,3,0]]
        dof_pos = qpos_list[:, 7:]
        # num_frames = root_pos.shape[0]
        
        if self.height_adjust:
            with torch.inference_mode():
                body_pos, _ = self.kinematic_model.forward_kinematics(
                    torch.from_numpy(root_pos).float().to(self.kinematic_model_device),
                    torch.from_numpy(root_rot).float().to(self.kinematic_model_device),
                    torch.from_numpy(dof_pos).float().to(self.kinematic_model_device)
                )
            ground_offset = 0.0
            lowest_height = torch.min(body_pos[..., 2]).item()
            root_pos[:, 2] = root_pos[:, 2] - lowest_height + ground_offset

        if self.root_offset:
            root_pos[:, :2] -= root_pos[0, :2]
 
        motion_data = {
            "fps": extras['fps'],
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
        }
        
        return motion_data
