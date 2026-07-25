import os
import json
import torch
import torch_tensorrt # weishuai: tensorrt should be imported to ensure successful loading
from loguru import logger
from scalebridge.agent.base_agent import BaseAgent

CONTROL_MODE_DICT = {
    0: "Pelvis",
    1: "Double Hands",
    2: "Pelvis and Double Hands",
    3: "Double Hands and Feet",
    4: "Pelvis and Double Hands and Feet",
    5: "Left and Right Shoulders, Elbows and Hands",
    6: "Pelvis and Left and Right Shoulders, Elbows and Hands",
    7: "Pelvis, Torso, Left and Right Shoulders, Elbows and Hands, Left and Right Hips, Knees and Feet"
}

class BFMAgent(BaseAgent):
    def __init__(self, config, device):

        super().__init__(config, device)

        assert self.device == "cuda", f"Compiled tensorrt model should only run on CUDA devices!"
        self.control_mode = torch.tensor([self.config.control_mode], dtype=torch.long, device=self.device)
        logger.info(f"[Agent] Activating control mode {self.config.control_mode}: {CONTROL_MODE_DICT[self.config.control_mode]}")

    def _load_policy(self):

        logger.info(f"[Agent] Loading checkpoint from {self.checkpoint}")
        self.policy = torch.jit.load(self.checkpoint).to(self.device)
        self.policy.eval()

        path_base, _ = os.path.splitext(self.checkpoint)
        meta_path = path_base + "_metadata.json"
        logger.info(f"[Agent] loading order dict from: {meta_path}")

        with open(meta_path, "r") as f:
            self.meta_data_dict = json.load(f)

        for key, item in self.meta_data_dict.items():
            logger.debug(f"Metadata {key}: {item}")
        
    def get_meta_data(self):
        return self.meta_data_dict
    
    def get_action(self, obs_dict):
        return self.policy(
            *[
                obs_dict["root_quat_buffer"],
                obs_dict["base_ang_vel_buffer"],
                obs_dict["dof_pos_buffer"],
                obs_dict["dof_vel_buffer"],
                obs_dict["actions_buffer"],
                obs_dict["target_body_pos_future_to_robot_base"],
                obs_dict["target_body_rot_future_to_robot_base"],
                self.control_mode,
                obs_dict["future_time_offsets"]
            ]
        )