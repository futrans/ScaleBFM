import os
import glob
import typer
import joblib
from omegaconf import OmegaConf
from easydict import EasyDict
from scaleretarget.viewer.robot_motion_viewer import RobotMotionViewer

def key_call_back( keycode):
    global frame_idx, motion_idx, motion_list, motion_data, viewer, paused, camera_follow
    if chr(keycode) == "R":
        print("Reset")
        frame_idx = 0
    elif chr(keycode) == "S":
        motion_idx = (motion_idx + 1) % len(motion_list)
        motion_data = load_motion_data(motion_list[motion_idx])
        viewer.update(fps=motion_data['fps'])
        frame_idx = 0
        print(f"Switch to {motion_list[motion_idx]}")
    elif chr(keycode) == " ":
        print("Paused")
        paused = not paused
    elif chr(keycode) == "C":
        print("Camera State Updated")
        camera_follow = not camera_follow
        viewer.camera_follow = camera_follow
    else:
        print("not mapped", chr(keycode))

def load_motion_data(motion_file, quat_order = "xyzw"):
    motion_dict = joblib.load(motion_file)
    if quat_order == 'xyzw':
        motion_dict['root_rot'] = motion_dict['root_rot'][:, [3,0,1,2]]
    return motion_dict
    

def main(
    motion_path: str, robot_type: str, record_video: bool = False, video_path: str = "videos/example.mp4", quat_order: str = "xyzw"
) -> None:
    global frame_idx, motion_idx, motion_list, motion_data, viewer, paused, camera_follow
    frame_idx, motion_idx, paused, camera_follow= 0, 0, False, True
    if os.path.isfile(motion_path):
        motion_list = [motion_path]
    elif os.path.isdir(motion_path):
        motion_list = glob.glob(f"{motion_path}/**/*.pkl",recursive=True)
    else:
        raise Exception(f"Check motion path: {motion_path}")

    robot_args = OmegaConf.load(f"config/robot/{robot_type}.yaml")
    
    motion_data = load_motion_data(motion_list[motion_idx])

    

    viewer = RobotMotionViewer(
        robot_config=robot_args.robot,
        camera_follow=True if record_video else camera_follow,
        record_video=record_video,
        video_path=video_path,
        key_callback=key_call_back
    )
    viewer.update(fps=motion_data['fps'])

    frame_idx = 0
    while True:
        if not paused:
            viewer.step(
                root_pos=motion_data['root_pos'][frame_idx],
                root_rot=motion_data['root_rot'][frame_idx],
                dof_pos=motion_data['dof_pos'][frame_idx],
            )
            
            frame_idx += 1
            if frame_idx >= len(motion_data['root_pos']):
                if record_video:
                    break
                frame_idx = 0

    viewer.close()

if __name__ == "__main__":
    typer.run(main)