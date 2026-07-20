import numpy as np
from scipy.spatial.transform import Rotation as R

from scaleretarget.loader.base_loader import BaseLoader
from scaleretarget.utils.lafan_vendor.extract import read_bvh
from scaleretarget.utils.lafan_vendor.utils import quat_fk, quat_mul
from scaleretarget.utils.resampling import resample_bvh_motion

from loguru import logger
from scaleretarget.utils.shape_optimizer import optimize_bvh_shape

class XsensLoader(BaseLoader):

    def __init__(self, config):
        super().__init__(config)

    def load(self, data_path):
        super().load(data_path)
        self._optimize_shape(self.data_list[0])

    def _load_sample(self, sample_path):
        
        bvh_data = read_bvh(sample_path)

        root_height = self._get_root_height(bvh_data)

        bvh_data.pos = bvh_data.pos[1:]
        bvh_data.quats = bvh_data.quats[1:]

        aligned_fps = resample_bvh_motion(bvh_data, self.target_fps)

        if self.use_optimized_shape:
            if np.any(np.linalg.norm(bvh_data.offsets[1:] - self.cached_offsets[1:], axis=-1) > 0.1):
                self._optimize_shape(sample_path)
            bvh_data.pos[:, 1:] *= self.scale[..., None]

        global_data = quat_fk(bvh_data.quats, bvh_data.pos, bvh_data.parents)

        rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

        frames = []
        for frame in range(bvh_data.pos.shape[0]):
            result = {}
            for i, bone in enumerate(bvh_data.bones):
                orientation = quat_mul(rotation_quat, global_data[0][frame, i])
                position = global_data[1][frame, i] @ rotation_matrix.T / 100  # cm to m
                result[bone] = (position, orientation)

            # Add modified foot pose
            result["LeftFootMod"] = (result["LeftAnkle"][0], result["LeftToe"][1])
            result["RightFootMod"] = (result["RightAnkle"][0], result["RightToe"][1])
            
            frames.append(result)
        

        extras = {
            "fps": aligned_fps,
            "disable_scale_table": self.use_optimized_shape,
            "ratio": self.config.robot.robot_root_height / root_height,
            "disable_init_height_anchor": True
        }

        return frames, extras
    
    def _optimize_shape(self, bvh_file):
        self.use_optimized_shape = self.config.use_optimized_shape
        if self.use_optimized_shape:
            logger.info(f"[Loader] Using optimized shape for retargeting.")
            bvh_data = read_bvh(bvh_file)
            self.cached_offsets = bvh_data.offsets
            self.scale = optimize_bvh_shape(
                robot_type=self.config.robot.robot_type,
                robot_xml_path=self.config.robot.robot_xml_path,
                bvh_data=bvh_data,
                optim_joint_matches=self.config.optim_joint_matches,
                optim_iterations=self.config.optim_iterations
            )
        else:
            raise NotImplementedError(f"Currently we only support optimization-based shape attainment!")
    
    def _get_root_height(self, bvh_data):

        bvh_offset = bvh_data.offsets
        bvh_offset[0, [0,2]] = 0
        bvh_joint_names = bvh_data.bones
        rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)

        rest_positions = np.zeros((len(bvh_joint_names), 3), dtype=np.float32)
        for i in range(len(bvh_joint_names)):
            parent = bvh_data.parents[i]
            if parent == -1:
                assert i == 0
                rest_positions[i] = np.zeros((3), dtype=np.float32)
            else:
                rest_positions[i] = rest_positions[parent] + bvh_offset[i]
        
        rest_positions = rest_positions @ rotation_matrix.T / 100

        root_z = rest_positions[0, 2]
        
        foot_index = [bvh_joint_names.index(j) for j in ['LeftAnkle', 'LeftToe', 'RightAnkle', 'RightToe']]
        foot_z = rest_positions[foot_index, 2]
        
        root_height = root_z - foot_z.min()

        return root_height
