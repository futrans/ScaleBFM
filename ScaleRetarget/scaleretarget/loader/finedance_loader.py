import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES
from loguru import logger

from scaleretarget.loader.base_loader import BaseLoader, Mode
from scaleretarget.utils.resampling import resample_smplx_motion
from scaleretarget.utils.shape_optimizer import optimize_smplx_shape
from scaleretarget.utils.smplx import SmplxModelCache, run_smplx_inference

def tan_norm_to_axis_angle(tan_norm_vec):
    """
    Convert a rotation in tan-norm representation into axis angle representation
    """

    tan, norm =  tan_norm_vec[..., :3], tan_norm_vec[..., 3:]
    
    u1 = tan / np.linalg.norm(tan, axis=-1, keepdims=True)
    
    u2 = norm - np.sum(u1 * norm, axis=-1, keepdims=True) * u1

    u2 = u2 / np.linalg.norm(u2, axis=-1, keepdims=True)

    u3 = np.cross(u1, u2, axis=-1)

    rot_matrix = np.stack([u1, u2, u3], axis=-2).reshape(-1, 3, 3)

    axis_angle = R.from_matrix(rot_matrix).as_rotvec().reshape(*tan_norm_vec.shape[:-1], 3)

    return axis_angle

class FineDanceLoader(BaseLoader):

    def __init__(self, config):
        super().__init__(config)
        self.body_models = SmplxModelCache(self.config.body_model_path)
        self._optimize_shape()

    def _load_sample(self, sample_path):
        orig_data = np.load(sample_path)
        num_frames = orig_data.shape[0]
        
        root_pos = orig_data[:, :3]
        body_pos_tan_norm = orig_data[:, 3:].reshape(num_frames, 52, 6) # (num_frames, 312) -> (num_frames, 52, 6)
        body_pos = tan_norm_to_axis_angle(body_pos_tan_norm).reshape(num_frames, 156)

        rot_offset = R.from_quat([0.5,0.5,0.5,0.5], scalar_first=True)
        body_pos[:, :3] = (rot_offset * R.from_rotvec(body_pos[:, :3])).as_rotvec()
        root_pos = rot_offset.apply(root_pos)

        if self.use_optimized_shape:
            betas = self.shape
        else:
            betas = torch.zeros(1, 16, dtype=torch.float)
        body_model = self.body_models.get('neutral')

        smplx_output = run_smplx_inference(
            body_model,
            betas=betas, # (16,)
            global_orient=torch.tensor(body_pos[:, :3]).float(), # (N, 3)
            body_pose=torch.tensor(body_pos[:, 3:66]).float(), # (N, 63)
            transl=torch.tensor(root_pos).float(), # (N, 3)
            left_hand_pose=torch.tensor(body_pos[:, 66:66+45]).float(),
            right_hand_pose=torch.tensor(body_pos[:, 66+45:]).float(),
            jaw_pose=torch.zeros(num_frames, 3).float(),
            leye_pose=torch.zeros(num_frames, 3).float(),
            reye_pose=torch.zeros(num_frames, 3).float(),
            # expression=torch.zeros(num_frames, 10).float(),
            return_full_pose=True,
        )

        human_height = 1.66 + 0.1 * betas[0, 0].item()

        src_fps = 30 # FineDance does not include fps, default to 30
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
            logger.info(f'[Loader] Using optimized shape for retargeting.')
            self.shape = optimize_smplx_shape(
                robot_type=self.config.robot.robot_type,
                robot_xml_path=self.config.robot.robot_xml_path,
                body_model_path=self.config.body_model_path,
                optim_joint_matches=self.config.optim_joint_matches,
                optim_iterations=self.config.optim_iterations,
            )
