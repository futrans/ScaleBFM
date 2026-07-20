import os

import joblib
import numpy as np
import typer


SMPLX_COMPONENT_DIRS = {
    "pose_body": "smplx_mesh_body_pose",
    "root_orient": "smplx_mesh_global_orient",
    "left_hand_pose": "smplx_mesh_left_hand_pose",
    "right_hand_pose": "smplx_mesh_right_hand_pose",
    "trans": "smplx_mesh_transl",
}


def load_component(subject_path: str, component_dir: str) -> np.ndarray:
    component_path = os.path.join(subject_path, component_dir)
    component_files = sorted(
        filename
        for filename in os.listdir(component_path)
        if filename.endswith(".npy")
    )
    if not component_files:
        raise FileNotFoundError(f"No .npy files found in {component_path}")
    return np.load(os.path.join(component_path, component_files[0]))


def list_subdirectories(path: str) -> list[str]:
    return sorted(
        name
        for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
    )


def main(input_path: str, output_path: str) -> None:
    print(f"Input Path: {input_path}")
    print(f"Output Path: {output_path}")
    assert os.path.isdir(input_path)
    os.makedirs(output_path, exist_ok=True)

    required_component_dirs = set(SMPLX_COMPONENT_DIRS.values())

    for subset_name in list_subdirectories(input_path):
        subset_path = os.path.join(input_path, subset_name)

        for capture_name in list_subdirectories(subset_path):
            capture_path = os.path.join(subset_path, capture_name)

            for subject_name in list_subdirectories(capture_path):
                subject_path = os.path.join(capture_path, subject_name)
                subject_dirs = set(list_subdirectories(subject_path))
                if not required_component_dirs.issubset(subject_dirs):
                    print(f"Skipping {subject_path}: missing SMPL-X components")
                    continue

                try:
                    smplx_data = {
                        field: load_component(subject_path, component_dir)
                        for field, component_dir in SMPLX_COMPONENT_DIRS.items()
                    }

                    capture_output_path = os.path.join(
                        output_path,
                        subset_name,
                        capture_name,
                    )
                    os.makedirs(capture_output_path, exist_ok=True)
                    subject_output_path = os.path.join(
                        capture_output_path,
                        f"{subject_name}.pkl",
                    )

                    joblib.dump(smplx_data, subject_output_path)
                    print(f"Saved {subject_output_path}")
                except Exception as e:
                    print(f"Error {e} when processing {subject_path}; Continue...")


if __name__ == "__main__":
    typer.run(main)
