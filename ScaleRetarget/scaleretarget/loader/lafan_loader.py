import numpy as np
from scipy.spatial.transform import Rotation as R

from scaleretarget.loader.base_loader import BaseLoader
from scaleretarget.utils.lafan_vendor.extract import read_bvh
from scaleretarget.utils.lafan_vendor.utils import quat_fk, quat_mul
from scaleretarget.utils.resampling import resample_bvh_motion

class LafanLoader(BaseLoader):

    def _load_sample(self, sample_path):
        
        bvh_data = read_bvh(sample_path)

        aligned_fps = resample_bvh_motion(bvh_data, self.target_fps)

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
            result["LeftFootMod"] = (result["LeftFoot"][0], result["LeftToe"][1])
            result["RightFootMod"] = (result["RightFoot"][0], result["RightToe"][1])
            
            frames.append(result)
        
        human_height = result["Head"][0][2] - min(result["LeftFootMod"][0][2], result["RightFootMod"][0][2])
        # human_height = human_height + 0.2  # cm to m
        human_height = 1.75  # cm to m

        extras = {
            "fps": aligned_fps,
            "actual_human_height": human_height
        }

        return frames, extras
