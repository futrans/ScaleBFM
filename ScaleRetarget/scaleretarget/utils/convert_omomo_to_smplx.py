import os
import joblib
import numpy as np
import pickle
import typer


def main(
    omomo_dir: str, save_dir: str
) -> None:
    train_file = os.path.join(omomo_dir, "train_diffusion_manip_seq_joints24.p")
    test_file = os.path.join(omomo_dir, "test_diffusion_manip_seq_joints24.p")
    
    train_data = joblib.load(train_file)
    test_data = joblib.load(test_file)

    os.makedirs(save_dir, exist_ok=True)
    
    for motion_data in [train_data, test_data]:
        for data_name in motion_data.keys():
            
            smpl_data = motion_data[data_name]
            seq_name = smpl_data['seq_name']
            
            num_frames = smpl_data['pose_body'].shape[0]

            mocap_frame_rate = 30

            poses = np.concatenate([smpl_data["pose_body"], np.zeros((num_frames, 102))], axis=1)

            smpl_data["poses"] = poses
            smpl_data['mocap_frame_rate'] = np.array(mocap_frame_rate)

            with open(f"{save_dir}/{seq_name}.pkl", "wb") as f:
                pickle.dump(smpl_data, f)


if __name__ == "__main__":
    typer.run(main)