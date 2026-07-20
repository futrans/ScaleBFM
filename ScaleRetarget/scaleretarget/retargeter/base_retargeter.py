import os
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from scaleretarget.viewer.robot_motion_viewer import RobotMotionViewer

class BaseRetargeter:
    def __init__(self, config: DictConfig):
        self.config = config
        self._init_viewer()
        self.paused = False

    def _init_viewer(self):
        self.enable_viewer = self.config.viewer.enable_viewer
        if self.enable_viewer:
            self.viewer = RobotMotionViewer(
                robot_config=self.config.robot,
                camera_follow=self.config.viewer.camera_follow,
                record_video=self.config.viewer.record_video,
                video_path=os.path.join(HydraConfig.get().runtime.output_dir, 'recording.mp4'),
                rate_limit=self.config.viewer.rate_limit,
                key_callback=self.key_callback
            )

    def retarget(self, frames):
        # weishuai: we do not apply loop here as some retargeting methods may not need frame-specific operation
        raise NotImplementedError
    
    def update(self, extras):
        self.motion_fps = extras['fps']
        if self.enable_viewer:
            self.viewer.update(self.motion_fps)
        

    def finish(self):
        if self.enable_viewer:
            self.viewer.close()

    def key_callback(self, keycode):
        if chr(keycode) == " ":
            self.paused = not self.paused
    