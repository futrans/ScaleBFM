import os
import numpy as np
import torch
from natsort import natsorted
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES
from loguru import logger

from scaleretarget.loader.base_loader import BaseLoader, Mode
from scaleretarget.utils.resampling import resample_smplx_motion
from scaleretarget.utils.shape_optimizer import optimize_smplx_shape
from scaleretarget.utils.smplx import SmplxModelCache, run_smplx_inference

class OmomoLoader(BaseLoader):

    def __init__(self, config):
        super().__init__(config)
        self.body_models = SmplxModelCache(self.config.body_model_path)
        self._optimize_shape()

    def load(self, data_path):
        self.data_path = data_path

        if os.path.isdir(self.data_path):
            self.mode = Mode.dir

            self.data_list = []
            for dir_path, _, filenames in os.walk(self.data_path):
                for filename in natsorted(filenames):
                    if filename.endswith((".pkl", ".npz")):
                        self.data_list.append(os.path.join(dir_path, filename))
            
            if self.config.exclude_content:
                logger.info(f"[Loader] Removing motions with keywords: {self.config.exclude_content}")
                self.data_list = [
                    path for path in self.data_list
                    if not any(content in path for content in self.config.exclude_content)
                ]

            self.data_num = len(self.data_list)
        elif os.path.isfile(self.data_path):
            suffix = os.path.splitext(self.data_path)[-1]
            assert suffix == self.format, f"You are using the data loader for {self.format} format but the data you give is {suffix} format!"
            self.mode = Mode.file
            self.data_list = [self.data_path]
            self.data_num = 1
        else:
            raise Exception(f"Check your data path: {self.data_path}")
        
        logger.info(f"[Loader] Total number of motions: {self.data_num}")

    def _load_sample(self, sample_path):
        smplx_data = np.load(sample_path, allow_pickle=True)    
        num_frames = smplx_data['pose_body'].shape[0]

        if self.use_optimized_shape:
            betas = self.shape
            body_model = self.body_models.get('neutral')
        else:
            betas = torch.tensor(smplx_data["betas"]).float().view(1, -1)
            body_model = self.body_models.get(smplx_data['gender'])
        smplx_output = run_smplx_inference(
            body_model,
            betas=betas, # (16,)
            global_orient=torch.tensor(smplx_data["root_orient"]).float(), # (N, 3)
            body_pose=torch.tensor(smplx_data["pose_body"]).float(), # (N, 63)
            transl=torch.tensor(smplx_data["trans"]).float(), # (N, 3)
            left_hand_pose=torch.zeros(num_frames, 45).float(),
            right_hand_pose=torch.zeros(num_frames, 45).float(),
            jaw_pose=torch.zeros(num_frames, 3).float(),
            leye_pose=torch.zeros(num_frames, 3).float(),
            reye_pose=torch.zeros(num_frames, 3).float(),
            # expression=torch.zeros(num_frames, 10).float(),
            return_full_pose=True,
        )

        if len(smplx_data["betas"].shape)==1:
            human_height = 1.66 + 0.1 * smplx_data["betas"][0]
        else:
            human_height = 1.66 + 0.1 * smplx_data["betas"][0, 0]

        src_fps = smplx_data['mocap_frame_rate'].item()
        global_orient = smplx_output.global_orient.detach().cpu().numpy().reshape(num_frames, 3)
        full_body_pose = smplx_output.full_pose.detach().cpu().numpy().reshape(num_frames, -1, 3)
        joints = smplx_output.joints.detach().cpu().numpy().reshape(num_frames, -1, 3)
        joint_names = JOINT_NAMES[: len(body_model.parents)]
        parents = body_model.parents

        global_orient, full_body_pose, joints, aligned_fps = resample_smplx_motion(
            global_orient, full_body_pose, joints, src_fps, self.target_fps
        )

        frames = []
        for curr_frame in range(len(global_orient)):
            result = {}
            single_global_orient = global_orient[curr_frame]
            single_full_body_pose = full_body_pose[curr_frame]
            single_joints = joints[curr_frame]
            joint_orientations = []
            for i, joint_name in enumerate(joint_names):
                if i == 0:
                    rot = R.from_rotvec(single_global_orient)
                else:
                    rot = joint_orientations[parents[i]] * R.from_rotvec(
                        single_full_body_pose[i].squeeze()
                    )
                joint_orientations.append(rot)
                result[joint_name] = (single_joints[i], rot.as_quat(scalar_first=True))


            frames.append(result)
        
        extras = {
            "fps": aligned_fps,
            "actual_human_height": human_height,
            "disable_scale_table": self.use_optimized_shape
        }
        
        return frames, extras

    def _optimize_shape(self):
        self.use_optimized_shape = self.config.use_optimized_shape
        if self.use_optimized_shape:
            logger.info('[Loader] Using optimized shape for retargeting.')
            self.shape = optimize_smplx_shape(
                robot_type=self.config.robot.robot_type,
                robot_xml_path=self.config.robot.robot_xml_path,
                body_model_path=self.config.body_model_path,
                optim_joint_matches=self.config.optim_joint_matches,
                optim_iterations=self.config.optim_iterations,
            )
