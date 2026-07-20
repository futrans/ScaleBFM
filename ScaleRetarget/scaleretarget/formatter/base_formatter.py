from loguru import logger

class BaseFormatter:
    def __init__(self, config):
        self.config = config
        self.quat_order = config.get('quat_order', 'xyzw')
        logger.info(f"[Formatter] Formatter has quat order: {self.quat_order}")


    def format(self, qpos_list, extras):
        root_pos = qpos_list[:, :3]
        root_rot = qpos_list[:, 3:7] # wxyz
        if self.quat_order == 'xyzw':
            root_rot[:, [0,1,2,3]] = root_rot[:, [1,2,3,0]]
        dof_pos = qpos_list[:, 7:]
        # num_frames = root_pos.shape[0]
        
        motion_data = {
            "fps": extras['fps'],
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
        }
        
        return motion_data