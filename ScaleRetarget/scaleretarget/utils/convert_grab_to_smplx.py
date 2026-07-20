import os
import glob
import typer
import joblib
import numpy as np

def construct_smplx_data(smplx_file):
    data = dict(np.load(smplx_file, allow_pickle=True))
    body_dict = data["body"].item()["params"]
    lhand_dict = data["lhand"].item()["params"]
    rhand_dict = data["rhand"].item()["params"]
    
    smplx_data = {}
    smplx_data["mocap_frame_rate"] = data["framerate"]
    smplx_data["gender"] = data["gender"]
    smplx_data["betas"] = np.zeros((1, 16))
    smplx_data["pose_body"] = body_dict["body_pose"]
    smplx_data["root_orient"] = body_dict["global_orient"]
    smplx_data["trans"] = body_dict["transl"]
    smplx_data["left_hand_pose"] = lhand_dict["fullpose"]
    smplx_data["right_hand_pose"] = rhand_dict["fullpose"]
   
    return smplx_data



def main(
    input_path: str, output_path: str
) -> None:
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")

    assert os.path.isdir(input_path)
    os.makedirs(output_path, exist_ok=True)

    data_save_dir = os.path.join(output_path, "data")
    os.makedirs(data_save_dir, exist_ok=True)
    files = glob.glob(f"{os.path.join(input_path, 'grab')}/**/*.npz", recursive=True)
    for file in files:
        subject = file.split("/")[-2]
        seq_name = file.split("/")[-1][:-4]
        smplx_data = construct_smplx_data(file)
        
        save_path = os.path.join(data_save_dir, f"{subject}_{seq_name}.pkl")
        joblib.dump(smplx_data, save_path)
        print(f"Subject {seq_name} Sequence {seq_name} saved to {save_path}")

if __name__ == "__main__":
    typer.run(main)